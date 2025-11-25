# train_depth_anything.py
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

class ResNetBackbone(nn.Module):
    """ResNet34骨干网络用于特征提取"""

    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()

        # 使用torchvision的ResNet34预训练模型
        from torchvision.models import resnet34, ResNet34_Weights

        # 加载预训练的ResNet34模型
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        # 提取ResNet的不同卷积层作为特征提取器
        # 输入: [batch_size, 3, H, W]
        self.conv1 = resnet.conv1      # 输出: [batch_size, 64, H/2, W/2]
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 输出: [batch_size, 64, H/4, W/4]

        # ResNet的四个层级，每个层级包含多个残差块
        self.layer1 = resnet.layer1    # 输出: [batch_size, 64, H/4, W/4]
        self.layer2 = resnet.layer2    # 输出: [batch_size, 128, H/8, W/8]
        self.layer3 = resnet.layer3    # 输出: [batch_size, 256, H/16, W/16]
        self.layer4 = resnet.layer4    # 输出: [batch_size, 512, H/32, W/32]

    def forward(self, x):
        """
        前向传播
        输入: x [batch_size, 3, H, W]
        输出: features list of 4 feature maps at different scales
        """
        # 初始卷积层
        # [batch_size, 3, H, W] -> [batch_size, 64, H/2, W/2]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # [batch_size, 64, H/2, W/2] -> [batch_size, 64, H/4, W/4]
        x = self.maxpool(x)

        # 通过四个层级的残差网络提取多尺度特征
        # [batch_size, 64, H/4, W/4] -> [batch_size, 64, H/4, W/4]
        feature1 = self.layer1(x)
        # [batch_size, 64, H/4, W/4] -> [batch_size, 128, H/8, W/8]
        feature2 = self.layer2(feature1)
        # [batch_size, 128, H/8, W/8] -> [batch_size, 256, H/16, W/16]
        feature3 = self.layer3(feature2)
        # [batch_size, 256, H/16, W/16] -> [batch_size, 512, H/32, W/32]
        feature4 = self.layer4(feature3)

        # 返回四个不同尺度的特征图
        features = [feature1, feature2, feature3, feature4]
        return features

class FeatureFusionBlock(nn.Module):
    """特征融合模块，用于融合不同层级的特征"""

    def __init__(self, features, activation=nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        # 输出卷积层，保持特征维度不变
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

        # 第一个残差卷积单元
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True),
            activation,
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # 第二个残差卷积单元
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True),
            activation,
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, *xs, size=None):
        """
        前向传播
        输入: xs - 一个或多个特征图
        输出: 融合后的特征图
        """
        # 第一个输入作为基础特征
        output = xs[0]

        # 如果有第二个输入，进行残差连接
        if len(xs) == 2:
            residual = self.res_conv1(xs[1])
            output = output + residual  # 残差连接

        # 通过第二个残差卷积单元
        output = self.res_conv2(output)

        # 上采样到指定尺寸或2倍上采样
        if size is not None:
            # 上采样到指定尺寸
            output = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)
        else:
            # 2倍上采样
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)

        # 最终输出卷积
        output = self.out_conv(output)
        return output

class DptHead(nn.Module):
    """DPT头部网络，用于深度估计"""

    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[64, 128, 256, 512]):
        super(DptHead, self).__init__()

        # 投影层，将不同层级的特征映射到统一维度
        self.projects = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])

        # 上采样/下采样层，将所有特征调整到相同分辨率
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),  # 4x上采样
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),  # 2x上采样
            nn.Identity(),  # 保持原样
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)  # 2x下采样
        ])

        # 特征重映射层，将特征映射到统一维度
        self.scratch = nn.ModuleDict({
            'layer1_rn': nn.Conv2d(out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False),
            'layer2_rn': nn.Conv2d(out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False),
            'layer3_rn': nn.Conv2d(out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False),
            'layer4_rn': nn.Conv2d(out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        })

        # 特征融合网络，逐步融合多尺度特征
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features)

        # 输出卷积层
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)

        # 最终输出层，生成深度图
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)  # 确保深度值为正
        )

    def forward(self, features):
        """
        前向传播
        输入: features - 四个不同尺度的特征图列表
        输出: depth - 深度图 [batch_size, 1, H, W]
        """
        processed_features = []

        # 对每个特征层进行投影和尺寸调整
        for i, (feature, project, resize) in enumerate(zip(features, self.projects, self.resize_layers)):
            # 1x1卷积投影到指定通道数
            x = project(feature)
            # 调整特征图尺寸到统一分辨率 (1/4输入尺寸)
            x = resize(x)
            processed_features.append(x)

        # 解包处理后的特征
        layer_1, layer_2, layer_3, layer_4 = processed_features

        # 通过重映射卷积调整特征维度
        layer_1_rn = self.scratch.layer1_rn(layer_1)  # [B, 256, H/4, W/4]
        layer_2_rn = self.scratch.layer2_rn(layer_2)  # [B, 256, H/4, W/4]
        layer_3_rn = self.scratch.layer3_rn(layer_3)  # [B, 256, H/4, W/4]
        layer_4_rn = self.scratch.layer4_rn(layer_4)  # [B, 256, H/4, W/4]

        # 自底向上的特征融合路径
        # 从最深层的特征开始，逐步与浅层特征融合
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # 通过输出卷积层生成深度图
        out = self.scratch.output_conv1(path_1)  # [B, 128, H/4, W/4]
        out = self.scratch.output_conv2(out)     # [B, 1, H/4, W/4]

        return out

class DepthAnythingResNet(nn.Module):
    """完整的深度估计模型，结合ResNet34骨干和DPT头部"""

    def __init__(self, pretrained=True):
        super(DepthAnythingResNet, self).__init__()

        # ResNet34骨干网络用于特征提取
        self.backbone = ResNetBackbone(pretrained=pretrained)

        # DPT头部网络用于深度估计
        # ResNet34各层输出通道数: [64, 128, 256, 512]
        in_channels = [64, 128, 256, 512]
        self.dpt_head = DptHead(in_channels)

    def forward(self, x):
        """
        前向传播
        输入: x [batch_size, 3, H, W] - 输入图像
        输出: depth [batch_size, H, W] - 深度图
        """
        # 通过骨干网络提取多尺度特征
        # 输出: 4个特征图，分辨率分别为 1/4, 1/8, 1/16, 1/32
        features = self.backbone(x)

        # 通过DPT头部生成深度图
        # 输出: [batch_size, 1, H/4, W/4]
        depth = self.dpt_head(features)

        # 上采样到输入图像尺寸
        if depth.shape[2:] != x.shape[2:]:
            depth = F.interpolate(depth, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 移除通道维度，返回 [batch_size, H, W]
        return depth.squeeze(1)

class NyuDepthDataset(Dataset):
    """NYU Depth V2数据集加载器 - 修正版本"""

    def __init__(self, data_path, split='train', transform=None, max_samples=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.max_samples = max_samples

        # 根据split确定使用哪个CSV文件和文件夹
        if split == 'train':
            csv_file = os.path.join(data_path, 'nyu2_train.csv')
            data_folder = 'nyu2_train'
        else:  # 'val' 或 'test'
            csv_file = os.path.join(data_path, 'nyu2_test.csv')
            data_folder = 'nyu2_test'

        # 读取CSV文件
        try:
            self.data_frame = pd.read_csv(csv_file, header=None)
        except FileNotFoundError:
            print(f"CSV file not found: {csv_file}")
            self.data_frame = pd.DataFrame()

        # 构建完整的文件路径
        self.image_paths = []
        self.depth_paths = []

        # 如果CSV文件为空，尝试自动发现数据
        if len(self.data_frame) == 0:
            self._discover_data_automatically(data_folder)
        else:
            self._load_from_csv(data_folder)

        # 限制样本数量（用于调试）
        if self.max_samples is not None and len(self.image_paths) > self.max_samples:
            self.image_paths = self.image_paths[:self.max_samples]
            self.depth_paths = self.depth_paths[:self.max_samples]

        print(f"Loaded {len(self.image_paths)} {split} samples")

    def _discover_data_automatically(self, data_folder):
        """自动发现数据文件（如果CSV文件不可用）"""
        data_dir = os.path.join(self.data_path, data_folder)

        if self.split == 'test':
            # 测试集：查找所有_colors.png和对应的_depth.png文件
            color_files = glob.glob(os.path.join(data_dir, '*_colors.png'))
            for color_file in color_files:
                base_name = color_file.replace('_colors.png', '')
                depth_file = base_name + '_depth.png'

                if os.path.exists(depth_file):
                    self.image_paths.append(color_file)
                    self.depth_paths.append(depth_file)
        else:
            # 训练集：查找所有子文件夹中的jpg和对应的png文件
            subdirs = [d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]

            for subdir in subdirs:
                subdir_path = os.path.join(data_dir, subdir)
                jpg_files = glob.glob(os.path.join(subdir_path, '*.jpg'))

                for jpg_file in jpg_files:
                    base_name = os.path.splitext(jpg_file)[0]
                    png_file = base_name + '.png'

                    if os.path.exists(png_file):
                        self.image_paths.append(jpg_file)
                        self.depth_paths.append(png_file)

    def _load_from_csv(self, data_folder):
        """从CSV文件加载数据路径"""
        for idx, row in self.data_frame.iterrows():
            if len(row) >= 2:
                rgb_rel_path = row[0].strip()
                depth_rel_path = row[1].strip()

                # 移除可能的"data/nyu2_train/"或"data/nyu2_test/"前缀
                rgb_rel_path = self._clean_path(rgb_rel_path, data_folder)
                depth_rel_path = self._clean_path(depth_rel_path, data_folder)

                # 构建完整路径
                rgb_full_path = os.path.join(self.data_path, data_folder, rgb_rel_path)
                depth_full_path = os.path.join(self.data_path, data_folder, depth_rel_path)

                # 检查文件是否存在
                if os.path.exists(rgb_full_path) and os.path.exists(depth_full_path):
                    self.image_paths.append(rgb_full_path)
                    self.depth_paths.append(depth_full_path)
                else:
                    print(f"Warning: File not found - RGB: {rgb_full_path}, Depth: {depth_full_path}")

    def _clean_path(self, path, data_folder):
        """清理路径，移除多余的前缀"""
        # 移除常见的路径前缀
        prefixes = [
            f"data/{data_folder}/",
            f"data/nyu2_{self.split}/",
            "data/",
            f"{data_folder}/"
        ]

        for prefix in prefixes:
            if path.startswith(prefix):
                path = path[len(prefix):]
                break

        return path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像和深度图路径
        img_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]

        # 读取图像 (RGB格式)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取深度图
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(f"Depth map not found: {depth_path}")

        # 根据深度图的数据类型进行处理
        if depth.dtype == np.uint16:
            # 16位深度图，通常单位为毫米
            depth = depth.astype(np.float32) / 1000.0  # 转换为米
        elif depth.dtype == np.uint8:
            # 8位深度图，归一化到0-10米
            depth = depth.astype(np.float32) / 255.0 * 10.0
        else:
            # 其他类型，直接转换为float32
            depth = depth.astype(np.float32)

        # 检查深度图中是否有无效值
        if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
            print(f"Warning: Invalid depth values in {depth_path}")
            depth = np.nan_to_num(depth, nan=0.0, posinf=10.0, neginf=0.0)

        # 确保深度值在合理范围内
        depth = np.clip(depth, 0.1, 10.0)  # 限制在0.1米到10米之间

        # 应用数据变换
        if self.transform:
            image, depth = self.transform(image, depth)

        return image, depth

class DepthTransform:
    """数据预处理和增强变换"""

    def __init__(self, size=(480, 640), train=True):
        self.size = size  # (height, width)
        self.train = train

    def __call__(self, image, depth):
        """
        应用数据变换
        输入: image [H, W, 3], depth [H, W]
        输出: image_tensor [3, H, W], depth_tensor [1, H, W]
        """
        # 调整图像和深度图尺寸
        image = cv2.resize(image, (self.size[1], self.size[0]))  # OpenCV使用 (width, height)
        depth = cv2.resize(depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

        # 训练时的数据增强
        if self.train:
            # 随机水平翻转 (50%概率)
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)  # 水平翻转
                depth = cv2.flip(depth, 1)

            # 随机亮度调整 (50%概率)
            if np.random.random() > 0.5:
                # 亮度缩放因子 [0.9, 1.1]
                alpha = 1.0 + 0.2 * (np.random.random() - 0.5)
                image = np.clip(image * alpha, 0, 255).astype(np.uint8)

            # 随机对比度调整 (50%概率)
            if np.random.random() > 0.5:
                # 对比度缩放因子 [0.9, 1.1]
                alpha = 1.0 + 0.2 * (np.random.random() - 0.5)
                mean = np.mean(image, axis=(0, 1), keepdims=True)
                image = np.clip(alpha * (image - mean) + mean, 0, 255).astype(np.uint8)

        # 转换为PyTorch张量并归一化
        # 图像: [H, W, 3] -> [3, H, W], 值域 [0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 深度图: [H, W] -> [1, H, W]
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)

        # 图像标准化 (ImageNet统计量)
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])(image_tensor)

        return image_tensor, depth_tensor

class DepthLoss(nn.Module):
    """深度估计损失函数，结合尺度不变损失和L1损失"""

    def __init__(self, epsilon=1e-6):
        super(DepthLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        计算深度估计损失
        输入: pred [B, H, W], target [B, 1, H, W]
        输出: 损失值
        """
        # 确保目标张量维度正确
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        # 创建有效深度掩码 (深度值大于0的区域)
        mask = (target > 0.1)  # 使用0.1作为阈值，因为我们裁剪到0.1-10.0

        # 检查是否有有效的深度值
        if mask.sum() == 0:
            print("Warning: No valid depth points in batch!")
            # 返回一个小的损失值而不是0，避免梯度消失
            return torch.tensor(1.0, device=pred.device, requires_grad=True)

        # 只计算有效深度区域的损失
        pred_masked = pred[mask]
        target_masked = target[mask]

        # 添加小值避免对数计算中的零
        pred_masked = torch.clamp(pred_masked, min=self.epsilon)
        target_masked = torch.clamp(target_masked, min=self.epsilon)

        # 尺度不变对数损失 (Scale-Invariant Logarithmic Loss)
        # 减少尺度模糊的影响
        diff_log = torch.log(pred_masked) - torch.log(target_masked)
        silog_loss = torch.sqrt(torch.mean(diff_log ** 2) - 0.85 * torch.mean(diff_log) ** 2 + self.epsilon)

        # L1损失 (平均绝对误差)
        l1_loss = F.l1_loss(pred_masked, target_masked)

        # 组合损失
        total_loss = silog_loss + 0.1 * l1_loss

        # 检查损失是否为NaN
        if torch.isnan(total_loss):
            print("Warning: NaN loss detected!")
            print(f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
            return torch.tensor(1.0, device=pred.device, requires_grad=True)

        return total_loss

def validate_dataset_loading(data_path="./data"):
    """验证数据集加载是否正确"""
    print("Testing dataset loading...")

    # 测试训练集
    try:
        train_dataset = NyuDepthDataset(data_path, 'train', max_samples=10)
        print(f"Train dataset: {len(train_dataset)} samples")

        # 显示前几个样本的路径
        for i in range(min(3, len(train_dataset))):
            print(f"  Sample {i}:")
            print(f"    Image: {train_dataset.image_paths[i]}")
            print(f"    Depth: {train_dataset.depth_paths[i]}")

    except Exception as e:
        print(f"Error loading train dataset: {e}")

    # 测试测试集
    try:
        test_dataset = NyuDepthDataset(data_path, 'test', max_samples=10)
        print(f"Test dataset: {len(test_dataset)} samples")

        # 显示前几个样本的路径
        for i in range(min(3, len(test_dataset))):
            print(f"  Sample {i}:")
            print(f"    Image: {test_dataset.image_paths[i]}")
            print(f"    Depth: {test_dataset.depth_paths[i]}")

    except Exception as e:
        print(f"Error loading test dataset: {e}")

def debug_data_sample(dataset, idx=0):
    """调试数据样本"""
    print(f"Debugging sample {idx}:")
    image, depth = dataset[idx]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}, range: [{depth.min():.3f}, {depth.max():.3f}]")

    # 检查深度图中有效像素的比例
    valid_ratio = (depth > 0.1).float().mean()
    print(f"Valid depth pixels ratio: {valid_ratio:.3f}")

def train_model(args):
    """模型训练主函数"""
    # 设备设置 (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建模型
    model = DepthAnythingResNet(pretrained=True)
    model = model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')

    # 数据加载和预处理
    train_transform = DepthTransform(size=args.input_size, train=True)
    val_transform = DepthTransform(size=args.input_size, train=False)

    # 限制样本数量用于调试
    train_dataset = NyuDepthDataset(args.data_path, 'train', train_transform, max_samples=args.max_samples)

    # 修复：检查max_samples是否为None
    if args.max_samples is not None:
        val_max_samples = args.max_samples // 10
    else:
        val_max_samples = None

    val_dataset = NyuDepthDataset(args.data_path, 'test', val_transform, max_samples=val_max_samples)

    # 调试数据样本
    print("Debugging training data sample:")
    debug_data_sample(train_dataset, 0)
    print("Debugging validation data sample:")
    debug_data_sample(val_dataset, 0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    # 损失函数和优化器
    criterion = DepthLoss(epsilon=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 梯度裁剪
    grad_clip = args.grad_clip

    # 训练记录
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Starting training for {args.epochs} epochs...')
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # 训练循环
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')

        for batch_idx, (images, depths) in enumerate(train_bar):
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 调试：检查模型输出
            if batch_idx == 0 and epoch == 0:
                print(f"First batch - Model output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"First batch - Target depth range: [{depths.min().item():.4f}, {depths.max().item():.4f}]")

            loss = criterion(outputs, depths)

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # 更新统计信息
            epoch_train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # 每100个batch打印一次调试信息
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: loss = {loss.item():.4f}")

        # 计算平均训练损失
        if len(train_loader) > 0:
            avg_train_loss = epoch_train_loss / len(train_loader)
        else:
            avg_train_loss = 0.0
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')

        with torch.no_grad():
            for images, depths in val_bar:
                images = images.to(device, non_blocking=True)
                depths = depths.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, depths)

                epoch_val_loss += loss.item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 计算平均验证损失
        if len(val_loader) > 0:
            avg_val_loss = epoch_val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
        val_losses.append(avg_val_loss)

        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        scheduler.step()

        # 打印epoch结果
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  Best model saved with val_loss: {avg_val_loss:.4f}')

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_path}')

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
        'args': vars(args)
    }, final_model_path)

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, learning_rates, args.output_dir)

    print(f'Training completed! Best validation loss: {best_val_loss:.4f}')
    print(f'All models saved in: {args.output_dir}')

def plot_training_curves(train_losses, val_losses, learning_rates, output_dir):
    """绘制训练和验证损失曲线以及学习率曲线"""

    # 创建2x1的子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制学习率曲线
    ax2.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')  # 对数尺度更好地显示学习率变化
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    loss_curve_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Training curves saved: {loss_curve_path}')

def main():
    """主函数：解析参数并启动训练"""
    parser = argparse.ArgumentParser(description='Train Depth Anything with ResNet34 backbone on NYU Depth V2')

    # 数据参数
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to NYU Depth V2 dataset directory')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for models and logs')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                       help='Weight decay for optimizer')
    parser.add_argument('--step-size', type=int, default=20,
                       help='StepLR scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='StepLR scheduler gamma')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (for debugging)')

    # 模型参数
    parser.add_argument('--input-size', type=int, nargs=2, default=[480, 640],
                       help='Input image size (height width)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # 转换输入尺寸为元组
    args.input_size = tuple(args.input_size)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存训练参数
    args_file = os.path.join(args.output_dir, 'training_arguments.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 验证数据集加载
    validate_dataset_loading(args.data_path)

    # 开始训练
    train_model(args)

if __name__ == '__main__':
    main()
