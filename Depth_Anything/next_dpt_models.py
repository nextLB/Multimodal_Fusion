"""
    自主构建的深度估计网络模型
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
import torch.nn.functional as F


# 深度估计模型 基于resnet50的
class DepthEstimationResnet50Model(nn.Module):
    def __init__(self, pretrained):
        super(DepthEstimationResnet50Model, self).__init__()

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


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # 平均池化分支
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_out).view(b, c, 1, 1)

        # 最大池化分支
        max_out = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_out).view(b, c, 1, 1)

        # 合并
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度的平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class MLPBlock(nn.Module):
    """多层感知机块"""

    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super(MLPBlock, self).__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.mlp(x)


class UpsampleBlock(nn.Module):
    """上采样块 (替代转置卷积)"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode='bilinear', align_corners=True)
        return self.conv(x)


class DepthEstimationResnet101Model(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationResnet101Model, self).__init__()

        # 使用预训练的ResNet101作为编码器
        backbone = resnet101(weights='IMAGENET1K_V1' if pretrained else None)

        # 编码器部分
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1  # 1/4, 256 channels
        self.encoder2 = backbone.layer2  # 1/8, 512 channels
        self.encoder3 = backbone.layer3  # 1/16, 1024 channels
        self.encoder4 = backbone.layer4  # 1/32, 2048 channels

        # 在编码器输出后加入注意力机制
        self.attention4 = CBAM(2048)
        self.attention3 = CBAM(1024)
        self.attention2 = CBAM(512)
        self.attention1 = CBAM(256)

        # 解码器部分 - 使用上采样替代转置卷积
        # 第一级上采样
        self.up4 = UpsampleBlock(2048, 1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.mlp4 = MLPBlock(512, 256, 512)

        # 第二级上采样
        self.up3 = UpsampleBlock(512, 256)
        self.decoder3 = self._make_decoder_block(256 + 512, 256)
        self.mlp3 = MLPBlock(256, 128, 256)

        # 第三级上采样
        self.up2 = UpsampleBlock(256, 128)
        self.decoder2 = self._make_decoder_block(128 + 256, 128)
        self.mlp2 = MLPBlock(128, 64, 128)

        # 第四级上采样
        self.up1 = UpsampleBlock(128, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 64)
        self.mlp1 = MLPBlock(64, 32, 64)

        # 额外的上采样层到原始尺寸
        self.up0 = UpsampleBlock(64, 32)
        self.decoder0 = self._make_decoder_block(32, 32)
        self.mlp0 = MLPBlock(32, 16, 32)

        # 全局上下文模块
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid()
        )

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 输出在0-1范围内
        )

        # 初始化权重
        self._initialize_weights()

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码器 - 保存所有中间特征
        # 初始卷积层
        x_init = self.conv1(x)
        x_init = self.bn1(x_init)
        x_init = self.relu(x_init)
        conv1_feat = x_init  # 1/2

        x = self.maxpool(x_init)  # 1/4

        # 残差块
        e1 = self.encoder1(x)  # 1/4
        e1 = self.attention1(e1)

        e2 = self.encoder2(e1)  # 1/8
        e2 = self.attention2(e2)

        e3 = self.encoder3(e2)  # 1/16
        e3 = self.attention3(e3)

        e4 = self.encoder4(e3)  # 1/32
        e4 = self.attention4(e4)

        # 全局上下文信息
        global_context = self.global_context(e4)
        e4 = e4 * global_context

        # 解码器（带有跳跃连接和注意力机制）
        # 第一级解码
        d4 = self.up4(e4)  # 1/16
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        d4 = self.mlp4(d4)

        # 第二级解码
        d3 = self.up3(d4)  # 1/8
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        d3 = self.mlp3(d3)

        # 第三级解码
        d2 = self.up2(d3)  # 1/4
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        d2 = self.mlp2(d2)

        # 第四级解码
        d1 = self.up1(d2)  # 1/2
        d1 = torch.cat([d1, conv1_feat], dim=1)
        d1 = self.decoder1(d1)
        d1 = self.mlp1(d1)

        # 额外的上采样到原始尺寸
        d0 = self.up0(d1)  # 原始尺寸
        d0 = self.decoder0(d0)
        d0 = self.mlp0(d0)

        # 最终输出
        output = self.final_conv(d0)
        return output




def main(select):

    if select == 'train':
        # 加载模型架构
        dptModel = DepthEstimationResnet50Model(pretrained=True)
    elif select == 'inference':
        dptModel = DepthEstimationResnet50Model(pretrained=False)
    else:
        dptModel = DepthEstimationResnet50Model(pretrained=True)


    return dptModel

def main_1(select):
    if select == 'train':
        # 加载模型架构
        dptModel = DepthEstimationResnet101Model(pretrained=True)
    elif select == 'inference':
        dptModel = DepthEstimationResnet101Model(pretrained=False)
    else:
        dptModel = DepthEstimationResnet101Model(pretrained=True)


    return dptModel



if __name__ == '__main__':
    # main('')
    main_1('')

