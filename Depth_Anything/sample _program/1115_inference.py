import os
import torch
import torch.nn as nn
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

# 空间注意力模块（与训练代码一致）
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
        ca = self.channel_attention(x)
        x_ca = x * ca
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        return x_ca * sa

# 深度估计模型（与训练代码一致）
class DepthEstimationModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationModel, self).__init__()
        backbone = resnet101(weights='IMAGENET1K_V1' if pretrained else None)

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

def load_model(model_path, device):
    """加载训练好的模型"""
    model = DepthEstimationModel(pretrained=False).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"成功加载模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return None

    model.eval()
    return model

def preprocess_image(image_path, img_size=(224, 224)):
    """预处理输入图像"""
    try:
        # 加载RGB图像
        rgb_image = Image.open(image_path).convert('RGB')

        # 转换为numpy数组并调整尺寸
        rgb_array = np.array(rgb_image)
        original_size = rgb_array.shape[:2]  # 保存原始尺寸

        rgb_array = cv2.resize(rgb_array, (img_size[1], img_size[0]))

        # 归一化
        rgb_array = rgb_array.astype(np.float32) / 255.0

        # 转换为tensor
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0)

        return rgb_tensor, original_size, rgb_array

    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return None, None, None

def postprocess_depth(pred_depth, original_size):
    """后处理深度图"""
    # 将深度图转换为numpy
    depth_np = pred_depth.cpu().numpy()[0, 0]

    # 反归一化到实际深度值
    depth_np = depth_np * 5000.0

    # 调整回原始尺寸
    if original_size:
        depth_np = cv2.resize(depth_np, (original_size[1], original_size[0]),
                             interpolation=cv2.INTER_LINEAR)

    return depth_np

def visualize_and_save(rgb_image, depth_map, output_path, title="Depth Estimation"):
    """可视化并保存结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 显示RGB图像
    axes[0].imshow(rgb_image)
    axes[0].set_title('Input RGB Image')
    axes[0].axis('off')

    # 显示深度图
    vmax = depth_map.max()
    im = axes[1].imshow(depth_map, cmap='plasma', vmin=0, vmax=vmax)
    axes[1].set_title('Predicted Depth')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存结果: {output_path}")

def inference_single_image(model, image_path, output_dir, img_size=(224, 224)):
    """对单张图像进行推理"""
    # 预处理
    rgb_tensor, original_size, rgb_array = preprocess_image(image_path, img_size)
    if rgb_tensor is None:
        return

    # 推理
    with torch.no_grad():
        rgb_tensor = rgb_tensor.to(device)
        pred_depth = model(rgb_tensor)

    # 后处理
    depth_map = postprocess_depth(pred_depth, original_size)

    # 准备输出路径
    image_name = os.path.basename(image_path).replace('_colors.png', '')
    output_path = os.path.join(output_dir, f"{image_name}_depth_result.png")

    # 可视化并保存
    visualize_and_save(rgb_array, depth_map, output_path,
                      f"Depth Estimation - {image_name}")

def inference_test_set(model, test_csv, data_dir, output_dir, img_size=(224, 224)):
    """对整个测试集进行推理"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取测试CSV文件
    try:
        test_data = pd.read_csv(test_csv, header=None)
        print(f"找到 {len(test_data)} 个测试样本")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    # 处理每个测试样本
    successful_count = 0
    for i in range(len(test_data)):
        rgb_path = test_data.iloc[i, 0]

        # 尝试在不同位置查找文件
        possible_paths = [
            rgb_path,
            os.path.join(data_dir, rgb_path),
            os.path.join(data_dir, 'data', rgb_path),
            os.path.join(data_dir, rgb_path.replace('data/', '')),
            os.path.join(data_dir, 'nyu2_test', os.path.basename(rgb_path)),
        ]

        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break

        if found_path:
            print(f"处理 {i+1}/{len(test_data)}: {os.path.basename(found_path)}")
            inference_single_image(model, found_path, output_dir, img_size)
            successful_count += 1
        else:
            print(f"未找到文件: {rgb_path}")

    print(f"成功处理 {successful_count}/{len(test_data)} 个样本")

def main():
    # 参数设置
    data_dir = './data'
    test_csv = os.path.join(data_dir, 'nyu2_test.csv')
    model_path = './depth_model_epoch_37.pth'  # 或者使用其他训练好的模型
    output_dir = './inference_results'
    img_size = (224, 224)

    # 检查文件是否存在
    if not os.path.exists(test_csv):
        print(f"测试CSV文件不存在: {test_csv}")
        return

    # 加载模型
    model = load_model(model_path, device)
    if model is None:
        print("无法加载模型，请检查模型路径")
        return

    # 对整个测试集进行推理
    print("开始对测试集进行推理...")
    inference_test_set(model, test_csv, data_dir, output_dir, img_size)
    print("推理完成！")

    # 可选：对单张图像进行推理（取消注释以下代码）
    """
    single_image_path = "path/to/your/image.jpg"  # 替换为您的图像路径
    if os.path.exists(single_image_path):
        print(f"对单张图像进行推理: {single_image_path}")
        inference_single_image(model, single_image_path, output_dir, img_size)
    else:
        print(f"图像文件不存在: {single_image_path}")
    """

if __name__ == '__main__':
    main()
