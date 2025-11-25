import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet101
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca

        # 空间注意力
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)

        return x_ca * sa

# 深度估计模型
class DepthEstimationModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationModel, self).__init__()

        # 使用预训练的ResNet101作为编码器
        backbone = resnet101(weights='IMAGENET1K_V1' if pretrained else None)

        # 编码器部分
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 1024)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = self._make_decoder_block(512 + 512, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = self._make_decoder_block(256 + 256, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = self._make_decoder_block(128 + 64, 128)

        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder0 = self._make_decoder_block(64, 64)

        # 细化层
        self.refine1 = self._make_refinement_block(64, 32)
        self.refine2 = self._make_refinement_block(32, 16)

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 注意力机制
        self.attention1 = SpatialAttention(1024)
        self.attention2 = SpatialAttention(512)
        self.attention3 = SpatialAttention(256)

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_refinement_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x

        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 应用注意力机制
        e3_att = self.attention1(e3)
        e2_att = self.attention2(e2)
        e1_att = self.attention3(e1)

        # 解码器
        d4 = self.upconv4(e4)
        d4 = torch.cat([d4, e3 * e3_att], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2 * e2_att], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1 * e1_att], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, conv1_feat], dim=1)
        d1 = self.decoder1(d1)

        d0 = self.upconv0(d1)
        d0 = self.decoder0(d0)

        # 细化阶段
        r1 = self.refine1(d0)
        r2 = self.refine2(r1)

        # 最终输出
        output = self.final_conv(r2)
        return output

# 统一的数据集类
class NYUDepthDataset(Dataset):
    def __init__(self, csv_file, data_dir, is_train=True, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train

        # 读取CSV文件
        self.data_frame = pd.read_csv(csv_file, header=None)
        print(f"Loaded {len(self.data_frame)} samples from CSV")

        # 验证文件是否存在
        self.valid_samples = []
        for i in range(len(self.data_frame)):
            rgb_path, depth_path = self.data_frame.iloc[i, 0], self.data_frame.iloc[i, 1]

            rgb_found = self._find_file(rgb_path)
            depth_found = self._find_file(depth_path)

            if rgb_found and depth_found:
                self.valid_samples.append((rgb_found, depth_found))
            else:
                print(f"Warning: Could not find files for sample {i}: RGB={rgb_path}, Depth={depth_path}")

        print(f"Found {len(self.valid_samples)} valid samples")

    def _find_file(self, file_path):
        """尝试在不同位置查找文件"""
        possible_paths = [
            file_path,
            os.path.join(self.data_dir, file_path),
            os.path.join(self.data_dir, 'data', file_path),
            os.path.join(self.data_dir, file_path.replace('data/', '')),
            os.path.join(self.data_dir, 'nyu2_train', file_path),
            os.path.join(self.data_dir, 'nyu2_test', file_path),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.valid_samples[idx]

        try:
            # 加载RGB图像
            rgb_image = Image.open(rgb_path).convert('RGB')
            depth_image = Image.open(depth_path)

            # 转换为numpy数组
            rgb_array = np.array(rgb_image)
            depth_array = np.array(depth_image)

            # 调整尺寸
            rgb_array = cv2.resize(rgb_array, (self.img_size[1], self.img_size[0]))
            depth_array = cv2.resize(depth_array, (self.img_size[1], self.img_size[0]),
                                   interpolation=cv2.INTER_NEAREST)

            # 归一化RGB图像
            rgb_array = rgb_array.astype(np.float32) / 255.0

            # 处理深度图
            if len(depth_array.shape) == 3:
                depth_array = depth_array[:, :, 0]

            # 深度图归一化 - 使用与NYU数据集一致的5000比例
            depth_array = depth_array.astype(np.float32)
            depth_array = depth_array / 5000.0
            depth_array = np.clip(depth_array, 0.001, 1.0)

            # 转换为tensor
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)
            depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)

            return rgb_tensor, depth_tensor

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return (torch.zeros(3, *self.img_size),
                    torch.zeros(1, *self.img_size))

# 改进的损失函数
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def gradient_loss(self, pred, target):
        grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        grad_loss_x = torch.abs(grad_pred_x - grad_target_x).mean()
        grad_loss_y = torch.abs(grad_pred_y - grad_target_y).mean()

        return grad_loss_x + grad_loss_y

    def forward(self, pred, target):
        if pred.size() != target.size():
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

        l1_loss = F.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)

        total_loss = l1_loss + 0.1 * grad_loss

        return total_loss

# 改进的可视化函数 - 修复深度图显示
def visualize_results(model, val_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        rgb_images, depth_maps = next(iter(val_loader))
        rgb_images = rgb_images.to(device)
        depth_maps = depth_maps.to(device)

        pred_depth = model(rgb_images)

        # 反归一化深度图用于显示
        rgb_np = rgb_images.cpu().numpy()[0].transpose(1, 2, 0)
        true_depth_np = depth_maps.cpu().numpy()[0, 0] * 5000.0  # 反归一化
        pred_depth_np = pred_depth.cpu().numpy()[0, 0] * 5000.0  # 反归一化

        # 创建可视化图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RGB图像
        axes[0].imshow(rgb_np)
        axes[0].set_title('Input RGB Image')
        axes[0].axis('off')

        # 真实深度图 - 使用正确的颜色范围
        vmax = max(true_depth_np.max(), pred_depth_np.max())
        im1 = axes[1].imshow(true_depth_np, cmap='plasma', vmin=0, vmax=vmax)
        axes[1].set_title('Ground Truth Depth')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # 预测深度图 - 使用相同的颜色范围
        im2 = axes[2].imshow(pred_depth_np, cmap='plasma', vmin=0, vmax=vmax)
        axes[2].set_title('Predicted Depth')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.suptitle(f'Epoch {epoch} - Depth Estimation Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'results_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization for epoch {epoch}")

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []

    # 初始验证
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

            if i % 100 == 99:
                avg_loss_so_far = running_loss / (i + 1)
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

        # 保存模型
        torch.save(model.state_dict(), f'depth_model_epoch_{epoch+1}.pth')

        # 每5个epoch可视化一次
        if (epoch + 1) % 5 == 0:
            visualize_results(model, val_loader, device, epoch+1)

    return train_losses, val_losses

# 主函数
def main():
    # 参数设置
    data_dir = './data'
    train_csv = os.path.join(data_dir, 'nyu2_train.csv')
    test_csv = os.path.join(data_dir, 'nyu2_test.csv')
    batch_size = 16
    num_epochs = 80
    learning_rate = 0.0001
    img_size = (224, 224)

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

    # 测试模型
    test_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    criterion = DepthLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 训练模型
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # 绘制损失曲线
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
    torch.save({
        'state_dict': model.state_dict(),
        'epoch': num_epochs,
        'loss': test_loss / len(test_loader)
    }, 'final_depth_model.pth')
    print("Training completed!")

if __name__ == '__main__':
    main()
