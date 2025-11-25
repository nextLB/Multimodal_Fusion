import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 自定义数据集类（修复路径问题）
class NYUDepthDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, is_train=True, img_size=(640, 480)):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.img_size = img_size

        if is_train:
            self.samples = self._build_train_samples()
        else:
            self.data_frame = pd.read_csv(csv_file, header=None)
            print(f"Loaded {len(self.data_frame)} test samples from CSV")

    def _build_train_samples(self):
        samples = []
        train_dir = os.path.join(self.data_dir, 'nyu2_train')

        for subdir in os.listdir(train_dir):
            subdir_path = os.path.join(train_dir, subdir)
            if os.path.isdir(subdir_path):
                jpg_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
                for jpg_file in jpg_files:
                    base_name = os.path.splitext(jpg_file)[0]
                    png_file = base_name + '.png'

                    jpg_path = os.path.join(subdir_path, jpg_file)
                    png_path = os.path.join(subdir_path, png_file)

                    if os.path.exists(png_path):
                        samples.append((jpg_path, png_file))

        print(f"Found {len(samples)} training samples")
        return samples

    def __len__(self):
        if self.is_train:
            return len(self.samples)
        else:
            return len(self.data_frame)

    def __getitem__(self, idx):
        if self.is_train:
            rgb_path, depth_path = self.samples[idx]
            depth_path = os.path.join(os.path.dirname(rgb_path), depth_path)
        else:
            rgb_path, depth_path = self.data_frame.iloc[idx, 0], self.data_frame.iloc[idx, 1]

            # 修复路径问题：检查不同的可能路径格式
            possible_rgb_paths = [
                os.path.join(self.data_dir, rgb_path),
                os.path.join(self.data_dir, 'data', rgb_path),
                os.path.join(self.data_dir, rgb_path.replace('data/', '')),
                rgb_path  # 直接使用CSV中的路径
            ]

            possible_depth_paths = [
                os.path.join(self.data_dir, depth_path),
                os.path.join(self.data_dir, 'data', depth_path),
                os.path.join(self.data_dir, depth_path.replace('data/', '')),
                depth_path  # 直接使用CSV中的路径
            ]

            # 找到存在的RGB路径
            rgb_path = None
            for path in possible_rgb_paths:
                if os.path.exists(path):
                    rgb_path = path
                    break

            # 找到存在的深度图路径
            depth_path = None
            for path in possible_depth_paths:
                if os.path.exists(path):
                    depth_path = path
                    break

            if rgb_path is None or depth_path is None:
                print(f"Warning: Could not find files for sample {idx}")
                print(f"RGB paths tried: {possible_rgb_paths}")
                print(f"Depth paths tried: {possible_depth_paths}")
                # 返回空数据，后续会跳过
                return torch.zeros(3, *self.img_size), torch.zeros(1, *self.img_size), "", ""

        # 检查文件是否存在
        if not os.path.exists(rgb_path):
            print(f"RGB file not found: {rgb_path}")
            return torch.zeros(3, *self.img_size), torch.zeros(1, *self.img_size), "", ""

        if not os.path.exists(depth_path):
            print(f"Depth file not found: {depth_path}")
            return torch.zeros(3, *self.img_size), torch.zeros(1, *self.img_size), "", ""

        try:
            # 加载RGB图像
            rgb_image = Image.open(rgb_path).convert('RGB')

            # 加载深度图
            depth_image = Image.open(depth_path)

            # 转换为numpy数组进行处理
            rgb_array = np.array(rgb_image)
            depth_array = np.array(depth_image)

            # 调整尺寸
            rgb_array = cv2.resize(rgb_array, (self.img_size[1], self.img_size[0]))
            depth_array = cv2.resize(depth_array, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

            # 归一化RGB图像到[0,1]
            rgb_array = rgb_array.astype(np.float32) / 255.0

            # 处理深度图
            if len(depth_array.shape) == 3:
                depth_array = depth_array[:, :, 0]  # 取第一个通道

            # 归一化深度图
            depth_array = depth_array.astype(np.float32)
            depth_array = depth_array / 5000.0  # 使用与训练相同的归一化
            depth_array = np.clip(depth_array, 0.001, 1.0)

            # 转换为tensor
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)  # HWC to CHW
            depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)  # 添加通道维度

            return rgb_tensor, depth_tensor, rgb_path, depth_path

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return torch.zeros(3, *self.img_size), torch.zeros(1, *self.img_size), "", ""

# 模型定义（与训练时相同）
class DepthEstimationModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationModel, self).__init__()

        backbone = resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = self._make_decoder_block(256 + 512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = self._make_decoder_block(128 + 256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = self._make_decoder_block(64 + 64, 64)

        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder0 = self._make_decoder_block(32, 32)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x

        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.upconv4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, conv1_feat], dim=1)
        d1 = self.decoder1(d1)

        d0 = self.upconv0(d1)
        d0 = self.decoder0(d0)

        output = self.final_conv(d0)
        return output

# 推理和可视化函数
def inference_and_visualize(model_path, data_dir, test_csv, output_dir, num_samples=50):
    """
    加载训练好的模型并对测试集进行推理和可视化

    Args:
        model_path: 训练好的模型路径
        data_dir: 数据目录
        test_csv: 测试集CSV文件路径
        output_dir: 输出目录
        num_samples: 要可视化的样本数量
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model = DepthEstimationModel(pretrained=False)

    # 修复权重加载警告
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except:
        # 如果上面的方法失败，尝试传统方法
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # 创建测试数据集
    test_dataset = NYUDepthDataset(test_csv, data_dir, is_train=False, img_size=(640, 480))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)  # 设置num_workers=0以避免多进程问题

    print(f"Total test samples: {len(test_dataset)}")
    print(f"Will visualize {min(num_samples, len(test_dataset))} samples")

    # 进行推理和可视化
    valid_count = 0
    with torch.no_grad():
        for i, (rgb_images, depth_maps, rgb_paths, depth_paths) in enumerate(test_loader):
            if valid_count >= num_samples:
                break

            # 跳过无效数据
            if rgb_paths[0] == "" or depth_paths[0] == "":
                continue

            rgb_images = rgb_images.to(device)

            # 推理
            pred_depth = model(rgb_images)

            # 转换为numpy用于可视化
            rgb_np = rgb_images.cpu().numpy()[0].transpose(1, 2, 0)
            true_depth_np = depth_maps.cpu().numpy()[0, 0]
            pred_depth_np = pred_depth.cpu().numpy()[0, 0]

            # 创建可视化图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # RGB图像
            axes[0].imshow(rgb_np)
            axes[0].set_title('Input RGB Image', fontsize=14)
            axes[0].axis('off')

            # 真实深度图
            im1 = axes[1].imshow(true_depth_np, cmap='plasma')
            axes[1].set_title('Ground Truth Depth', fontsize=14)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            # 预测深度图
            im2 = axes[2].imshow(pred_depth_np, cmap='plasma')
            axes[2].set_title('Predicted Depth', fontsize=14)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            # 添加文件信息
            rgb_filename = os.path.basename(rgb_paths[0])
            depth_filename = os.path.basename(depth_paths[0])
            plt.figtext(0.5, 0.01, f"RGB: {rgb_filename} | Depth: {depth_filename}",
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

            plt.suptitle(f'Sample {valid_count+1} - Depth Estimation Results', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)

            # 保存结果
            output_path = os.path.join(output_dir, f'result_{valid_count+1:03d}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            valid_count += 1
            print(f"Saved result {valid_count}/{min(num_samples, len(test_dataset))} to {output_path}")

    print(f"All results saved to {output_dir}")
    print(f"Successfully processed {valid_count} samples")

    # 创建汇总图（显示前6个样本）
    if valid_count > 0:
        create_summary_plot(model, test_loader, output_dir, device, num_samples=min(6, valid_count))

def create_summary_plot(model, test_loader, output_dir, device, num_samples=6):
    """
    创建包含多个样本的汇总图 - 修复索引错误版本
    """
    model.eval()
    with torch.no_grad():
        # 使用2行3列布局，每个样本占一行中的两个位置（RGB和深度）
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))

        # 如果只有一个样本，将axes转换为2D数组
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        count = 0
        for i, (rgb_images, depth_maps, rgb_paths, depth_paths) in enumerate(test_loader):
            if count >= num_samples:
                break

            # 跳过无效数据
            if rgb_paths[0] == "" or depth_paths[0] == "":
                continue

            rgb_images = rgb_images.to(device)
            pred_depth = model(rgb_images)

            rgb_np = rgb_images.cpu().numpy()[0].transpose(1, 2, 0)
            pred_depth_np = pred_depth.cpu().numpy()[0, 0]

            # 显示RGB图像
            axes[count, 0].imshow(rgb_np)
            axes[count, 0].set_title(f'Sample {count+1} - RGB')
            axes[count, 0].axis('off')

            # 显示预测深度
            im = axes[count, 1].imshow(pred_depth_np, cmap='plasma')
            axes[count, 1].set_title(f'Sample {count+1} - Predicted Depth')
            axes[count, 1].axis('off')
            plt.colorbar(im, ax=axes[count, 1], fraction=0.046)

            count += 1

        plt.suptitle('Depth Estimation Summary - Multiple Samples', fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 为标题留出空间
        plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Summary plot saved to {os.path.join(output_dir, 'summary.png')}")

# 检查数据集的函数
def check_dataset_structure(data_dir, test_csv):
    """检查数据集结构和CSV文件内容"""
    print("Checking dataset structure...")

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist!")
        return False

    # 检查测试CSV文件是否存在
    if not os.path.exists(test_csv):
        print(f"Test CSV file {test_csv} does not exist!")
        return False

    # 读取CSV文件的前几行
    try:
        df = pd.read_csv(test_csv, header=None)
        print(f"CSV file has {len(df)} rows and {len(df.columns)} columns")
        print("First 5 rows of CSV file:")
        print(df.head())

        # 检查第一行的路径
        first_rgb_path = df.iloc[0, 0]
        first_depth_path = df.iloc[0, 1]
        print(f"First RGB path in CSV: {first_rgb_path}")
        print(f"First Depth path in CSV: {first_depth_path}")

        # 尝试找到实际文件
        possible_paths = [
            os.path.join(data_dir, first_rgb_path),
            os.path.join(data_dir, 'data', first_rgb_path),
            os.path.join(data_dir, first_rgb_path.replace('data/', '')),
            first_rgb_path
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found RGB file at: {path}")
                found = True
                break

        if not found:
            print("Could not find the RGB file. Please check your data structure.")
            print("Available directories in data folder:")
            for item in os.listdir(data_dir):
                print(f"  - {item}")

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    return True

# 主函数
def main():
    # 参数设置
    data_dir = './data'
    test_csv = os.path.join(data_dir, 'nyu2_test.csv')
    output_dir = './inference_results'
    num_samples = 50  # 要可视化的样本数量

    # 检查数据集结构
    if not check_dataset_structure(data_dir, test_csv):
        print("Dataset structure check failed. Please fix the data paths.")
        return

    # 查找可用的模型文件
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not model_files:
        print("No model files found! Please train the model first.")
        return

    print("Available model files:")
    for i, file in enumerate(model_files):
        print(f"  {i+1}. {file}")

    # 让用户选择模型或使用默认
    if len(model_files) == 1:
        model_path = model_files[0]
    else:
        try:
            choice = int(input(f"Select model file (1-{len(model_files)}): ")) - 1
            model_path = model_files[choice]
        except:
            # 默认选择epoch最多的模型
            epoch_numbers = []
            for file in model_files:
                try:
                    epoch_num = int(file.split('_')[-1].split('.')[0])
                    epoch_numbers.append((epoch_num, file))
                except:
                    pass

            if epoch_numbers:
                model_path = max(epoch_numbers)[1]
            else:
                model_path = model_files[0]

    print(f"Using model: {model_path}")

    # 进行推理和可视化
    inference_and_visualize(model_path, data_dir, test_csv, output_dir, num_samples)

if __name__ == '__main__':
    main()
