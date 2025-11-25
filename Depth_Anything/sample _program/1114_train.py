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

# 自定义数据集类
class NYUDepthDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, is_train=True, img_size=(640, 480)):
        """
        Args:
            csv_file (string): CSV文件的路径
            data_dir (string): 数据目录的路径
            transform (callable, optional): 可选的变换
            is_train (bool): 是否是训练集
            img_size (tuple): 图像尺寸 (height, width)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.img_size = img_size

        if is_train:
            # 训练集：从文件夹结构构建数据对
            self.samples = self._build_train_samples()
        else:
            # 测试集：从CSV文件读取
            self.data_frame = pd.read_csv(csv_file, header=None)

    def _build_train_samples(self):
        """构建训练集样本列表"""
        samples = []
        train_dir = os.path.join(self.data_dir, 'nyu2_train')

        # 遍历所有子文件夹
        for subdir in os.listdir(train_dir):
            subdir_path = os.path.join(train_dir, subdir)
            if os.path.isdir(subdir_path):
                # 获取文件夹中的所有jpg文件
                jpg_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
                for jpg_file in jpg_files:
                    # 对应的png文件（深度图）
                    base_name = os.path.splitext(jpg_file)[0]
                    png_file = base_name + '.png'

                    jpg_path = os.path.join(subdir_path, jpg_file)
                    png_path = os.path.join(subdir_path, png_file)

                    if os.path.exists(png_path):
                        samples.append((jpg_path, png_path))

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
        else:
            rgb_path, depth_path = self.data_frame.iloc[idx, 0], self.data_frame.iloc[idx, 1]
            rgb_path = os.path.join(self.data_dir, rgb_path)
            depth_path = os.path.join(self.data_dir, depth_path)

        # 加载RGB图像
        rgb_image = Image.open(rgb_path).convert('RGB')

        # 加载深度图 - 使用PIL直接打开，保持原始精度
        depth_image = Image.open(depth_path)

        # 转换为numpy数组进行处理
        rgb_array = np.array(rgb_image)
        depth_array = np.array(depth_image)

        # 调整尺寸
        rgb_array = cv2.resize(rgb_array, (self.img_size[1], self.img_size[0]))
        depth_array = cv2.resize(depth_array, (self.img_size[1], self.img_size[0]),
                               interpolation=cv2.INTER_NEAREST)

        # 归一化RGB图像到[0,1]
        rgb_array = rgb_array.astype(np.float32) / 255.0

        # 处理深度图
        if len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0]  # 取第一个通道

        # 深度图处理 - 更准确的归一化
        depth_array = depth_array.astype(np.float32)

        # NYU深度图的真实范围通常是0.7m到10m
        # 深度图存储为16位PNG，实际深度值需要除以1000或5000
        # 尝试不同的归一化方式
        depth_array = depth_array / 5000.0  # 尝试5000而不是10000
        depth_array = np.clip(depth_array, 0.001, 1.0)  # 避免除零

        # 转换为tensor
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)  # HWC to CHW
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)  # 添加通道维度

        return rgb_tensor, depth_tensor

# 深度估计模型
class DepthEstimationModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationModel, self).__init__()

        # 使用预训练的ResNet作为编码器
        backbone = resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # 编码器部分 - 分离初始层以获取中间特征
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1  # 1/4
        self.encoder2 = backbone.layer2  # 1/8
        self.encoder3 = backbone.layer3  # 1/16
        self.encoder4 = backbone.layer4  # 1/32

        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = self._make_decoder_block(256 + 512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = self._make_decoder_block(128 + 256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = self._make_decoder_block(64 + 64, 64)

        # 额外的上采样层
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder0 = self._make_decoder_block(32, 32)

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 输出在0-1范围内
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
        # 编码器 - 保存所有中间特征
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x  # 1/2 (112x112)

        x = self.maxpool(x)  # 1/4 (56x56)

        # 残差块
        e1 = self.encoder1(x)  # 1/4 (56x56)
        e2 = self.encoder2(e1)  # 1/8 (28x28)
        e3 = self.encoder3(e2)  # 1/16 (14x14)
        e4 = self.encoder4(e3)  # 1/32 (7x7)

        # 解码器（带有跳跃连接）
        d4 = self.upconv4(e4)  # 1/16 (14x14)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)  # 1/8 (28x28)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)  # 1/4 (56x56)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)  # 1/2 (112x112)
        d1 = torch.cat([d1, conv1_feat], dim=1)
        d1 = self.decoder1(d1)

        # 额外的上采样到原始尺寸
        d0 = self.upconv0(d1)  # 原始尺寸 (224x224)
        d0 = self.decoder0(d0)

        # 最终输出
        output = self.final_conv(d0)
        return output

# 改进的损失函数，添加更多诊断信息
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def gradient_loss(self, pred, target):
        # 计算梯度差异
        grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        grad_loss_x = torch.abs(grad_pred_x - grad_target_x).mean()
        grad_loss_y = torch.abs(grad_pred_y - grad_target_y).mean()

        return grad_loss_x + grad_loss_y

    def forward(self, pred, target):
        # 确保预测和目标尺寸相同
        if pred.size() != target.size():
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

        # 添加诊断信息
        if torch.rand(1) < 0.001:  # 随机采样0.1%的批次进行诊断
            print(f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")

        l1_loss = F.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)

        # 组合损失 - 调整权重
        total_loss = l1_loss + 0.1 * grad_loss  # 降低梯度损失的权重

        # 添加诊断信息
        if torch.rand(1) < 0.001:
            print(f"L1 Loss: {l1_loss.item():.6f}, Grad Loss: {grad_loss.item():.6f}, Total: {total_loss.item():.6f}")

        return total_loss

# 改进的训练函数，添加更多监控
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []

    # 初始验证，检查数据范围
    print("Initial data check:")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        rgb, depth = sample_batch
        rgb, depth = rgb.to(device), depth.to(device)
        output = model(rgb)
        print(f"Sample RGB range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
        print(f"Sample depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}]")
        print(f"Sample output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"Initial loss: {criterion(output, depth).item():.6f}")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        batch_count = 0

        for i, (rgb_images, depth_maps) in enumerate(train_loader):
            rgb_images = rgb_images.to(device)
            depth_maps = depth_maps.to(device)

            # 前向传播
            outputs = model(rgb_images)
            loss = criterion(outputs, depth_maps)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            if i % 100 == 99:
                avg_loss_so_far = running_loss / batch_count
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Avg Loss: {avg_loss_so_far:.6f}')

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_images, depth_maps in val_loader:
                rgb_images = rgb_images.to(device)
                depth_maps = depth_maps.to(device)

                outputs = model(rgb_images)
                loss = criterion(outputs, depth_maps)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')

        # 每个epoch保存一次模型
        torch.save(model.state_dict(), f'depth_model_epoch_{epoch+1}.pth')

        # 每5个epoch可视化一次结果
        if (epoch + 1) % 5 == 0:
            visualize_results(model, val_loader, device, epoch+1)

    return train_losses, val_losses

# 可视化结果函数
def visualize_results(model, val_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        rgb_images, depth_maps = next(iter(val_loader))
        rgb_images = rgb_images.to(device)
        depth_maps = depth_maps.to(device)

        # 预测
        pred_depth = model(rgb_images)

        # 转换为numpy用于可视化
        rgb_np = rgb_images.cpu().numpy()[0].transpose(1, 2, 0)
        true_depth_np = depth_maps.cpu().numpy()[0, 0]
        pred_depth_np = pred_depth.cpu().numpy()[0, 0]

        # 创建可视化图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RGB图像
        axes[0].imshow(rgb_np)
        axes[0].set_title('Input RGB Image')
        axes[0].axis('off')

        # 真实深度图
        im1 = axes[1].imshow(true_depth_np, cmap='plasma')
        axes[1].set_title('Ground Truth Depth')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # 预测深度图
        im2 = axes[2].imshow(pred_depth_np, cmap='plasma')
        axes[2].set_title('Predicted Depth')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.suptitle(f'Epoch {epoch} - Depth Estimation Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'results_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

# 主函数
def main():
    # 参数设置
    data_dir = './data'
    train_csv = os.path.join(data_dir, 'nyu2_train.csv')
    test_csv = os.path.join(data_dir, 'nyu2_test.csv')
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.0001  # 降低学习率
    img_size = (640, 480)

    # 创建数据加载器
    train_dataset = NYUDepthDataset(train_csv, data_dir, is_train=True, img_size=img_size)
    test_dataset = NYUDepthDataset(test_csv, data_dir, is_train=False, img_size=img_size)

    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 创建模型
    model = DepthEstimationModel(pretrained=True).to(device)

    # 打印模型结构以验证输出尺寸
    print("Testing model with sample input...")

    # 测试模型输出尺寸
    test_input = torch.randn(1, 3, 640, 480).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

    # 损失函数和优化器
    criterion = DepthLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 训练模型
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 最终测试
    print("Testing on test set...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for rgb_images, depth_maps in test_loader:
            rgb_images = rgb_images.to(device)
            depth_maps = depth_maps.to(device)

            outputs = model(rgb_images)
            loss = criterion(outputs, depth_maps)
            test_loss += loss.item()

    print(f'Test Loss: {test_loss / len(test_loader):.6f}')

    # 保存最终模型
    torch.save(model.state_dict(), 'final_depth_model.pth')
    print("Training completed!")

if __name__ == '__main__':
    main()
