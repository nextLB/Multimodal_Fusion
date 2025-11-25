"""
    此文件用于模型构建时的所需要用到的基础模块
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import config_parameters
from timm.models.layers import DropPath



class LayerNorm(nn.Module):
    """自定义层归一化，支持两种数据格式"""

    def __init__(self, normalizedShape, eps, dataFormat):
        """
        Args:
            normalizedShape: 要归一化的形状（通常是特征维度）
            eps: 防止除零的小常数
            dataFormat: 数据格式，"channels_last" 或 "channels_first"
        """
        super().__init__()

        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(normalizedShape))
        # 可学习的偏置参数
        self.bias = nn.Parameter(torch.zeros(normalizedShape))

        self.eps = eps
        self.dataFormat = dataFormat

        # 验证数据格式是否支持
        if self.dataFormat not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        self.normalizedShape = (normalizedShape,)

    def forward(self, x):
        if self.dataFormat == "channels_last":
            # 使用PyTorch内置的layer_norm，适用于通道在最后的格式
            # 例如: [N, L, C] 或 [N, H, W, C]
            return F.layer_norm(x, self.normalizedShape, self.weight, self.bias, self.eps)

        elif self.dataFormat == "channels_first":
            # 手动实现，适用于通道在前的格式
            # 例如: [N, C, H, W]

            # 计算均值：沿通道维度求平均，保持维度用于广播
            u = x.mean(1, keepdim=True)  # [N, 1, H, W]

            # 计算方差：先求平方差，再求平均
            s = (x - u).pow(2).mean(1, keepdim=True)  # [N, 1, H, W]

            # 归一化：(x - μ) / sqrt(σ² + ε)
            x = (x - u) / torch.sqrt(s + self.eps)  # [N, C, H, W]

            # 可学习的缩放和偏移
            # weight和bias需要扩展维度以匹配x的形状
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # [N, C, H, W]

            return x


class GRN(nn.Module):
    """全局响应归一化，来自ConvNeXt V2等现代架构"""

    def __init__(self, dim):
        """
        Args:
            dim: 特征维度
        """
        super().__init__()

        # 可学习的缩放参数，初始为0
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        # 可学习的偏置参数，初始为0
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # 计算每个特征图的L2范数（沿空间维度H和W）
        # Gx形状: [N, 1, 1, C] - 每个通道有一个范数值
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)

        # 归一化：每个通道的范数除以所有通道范数的均值
        # Nx形状: [N, 1, 1, C]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)

        # 应用可学习参数并添加残差连接
        # self.gamma * (x * Nx) + self.beta + x
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    def __init__(self, dim, dropPath):
        super().__init__()

        # 深度可分离卷积 (Depthwise Convolution)
        # 使用7x7大核卷积，分组数=dim，每个通道独立卷积
        # 注释中提到原本使用MinkowskiDepthwiseConvolution(用于3D点云)，这里用2D卷积替代
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, groups=dim, padding=3)

        # 层归一化，使用channels_last格式
        self.norm = LayerNorm(dim, config_parameters.LAYER_NORM_EPS, "channels_last")

        # 第一个点卷积(Pointwise Convolution)，相当于全连接层
        # 将维度扩展4倍
        self.pwconv1 = nn.Linear(dim, 4 * dim)

        # 激活函数
        self.act = nn.GELU()

        # 第二个点卷积，将维度还原
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # 全局响应归一化
        self.grn = GRN(4 * dim)

        # 随机深度丢弃，用于正则化
        self.dropPath = DropPath(dropPath)

    def forward(self, x):
        # 保存输入用于残差连接
        input = x

        # 1. 深度可分离卷积
        x = self.dwconv(x)  # [N, C, H, W]

        # 2. 调整维度顺序为channels_last以适应LayerNorm
        x = x.permute(0, 2, 3, 1)  # [N, H, W, C]

        # 3. 层归一化
        x = self.norm(x)  # [N, H, W, C]

        # 4. 第一个点卷积(扩展维度)
        x = self.pwconv1(x)  # [N, H, W, 4*C]

        # 5. 激活函数
        x = self.act(x)  # [N, H, W, 4*C]

        # 6. 全局响应归一化
        x = self.grn(x)  # [N, H, W, 4*C]

        # 7. 第二个点卷积(压缩维度)
        x = self.pwconv2(x)  # [N, H, W, C]

        # 8. 调整维度顺序回channels_first
        x = x.permute(0, 3, 1, 2)  # [N, C, H, W]

        # 9. 残差连接 + 随机深度丢弃
        x = input + self.dropPath(x)  # [N, C, H, W]

        return x



class SparseConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2.
    用于处理带掩码输入的ConvNeXt V2变体

    Args:
        inChannels (int): 输入图像通道数
        numClasses (int): 分类头类别数
        depths (tuple(int)): 每个阶段的块数
        dims (int): 每个阶段的特征维度
        dropPathRate (float): 随机深度丢弃率
        D: 维度参数（可能用于稀疏卷积）
    """

    def __init__(self, inChannels, numClasses, depths, dims, dropPathRate, D):
        super().__init__()
        self.depths = depths
        self.numClasses = numClasses

        # 下采样层模块列表
        self.downsampleLayers = nn.ModuleList()

        # ============================================================ #
        # 构建stem（第一层）
        # 使用4x4卷积，步长4，进行patch embedding
        stem = nn.Sequential(
            nn.Conv2d(inChannels, dims[0], kernel_size=4, stride=4),  # 下采样4倍
            LayerNorm(dims[0], config_parameters.LAYER_NORM_EPS, "channels_first")
        )
        self.downsampleLayers.append(stem)
        # ============================================================ #

        # ============================================================ #
        # 构建后续3个下采样层
        for i in range(3):
            downsampleLayer = nn.Sequential(
                # 先进行层归一化
                LayerNorm(dims[i], config_parameters.LAYER_NORM_EPS, "channels_first"),
                # 2x2卷积，步长2，下采样2倍
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsampleLayers.append(downsampleLayer)
        # ============================================================ #

        # ============================================================ #
        # 构建4个阶段的残差块
        self.stages = nn.ModuleList()

        # 生成随机深度丢弃率列表
        # 从0线性增加到dropPathRate，总和为depths的长度
        dpRates = [x.item() for x in torch.linspace(0, dropPathRate, sum(depths))]
        cur = 0  # 当前深度索引

        for i in range(4):
            # 创建第i个阶段，包含depths[i]个Block
            stage = nn.Sequential(
                *[Block(dim=dims[i], dropPath=dpRates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]  # 更新深度索引
        # ============================================================ #

    def upsample_mask(self, mask, scale):
        """上采样掩码以匹配特征图尺寸

        Args:
            mask: 原始掩码，形状为 [N, L]，L是patch数量
            scale: 上采样尺度

        Returns:
            上采样后的掩码，形状为 [N, H, W]
        """
        assert len(mask.shape) == 2
        # 计算原始patch网格大小
        p = int(mask.shape[1] ** .5)
        # 重塑为2D并重复插值上采样
        return mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)

    def forward(self, x, mask):
        """前向传播

        Args:
            x: 输入图像 [N, C, H, W]
            mask: 二进制掩码 [N, L]，0表示保留，1表示掩码

        Returns:
            处理后的特征
        """
        numStages = len(self.stages)

        # 上采样掩码以匹配最终特征图尺寸
        # 2^(numStages-1) = 2^3 = 8倍上采样
        mask = self.upsample_mask(mask, 2 ** (numStages - 1))
        # 增加通道维度并转换为与x相同的数据类型
        mask = mask.unsqueeze(1).type_as(x)  # [N, 1, H, W]

        # patch embedding (stem层)
        x = self.downsampleLayers[0](x)  # 下采样4倍

        # 应用掩码：将掩码区域置零
        # 注意：mask中1表示要掩码的区域，所以用1-mask
        x *= (1. - mask)

        # 依次通过4个阶段
        for i in range(4):
            # 第0阶段已经应用了stem，所以i>0时才应用下采样
            x = self.downsampleLayers[i](x) if i > 0 else x
            # 通过第i个阶段的多个Block
            x = self.stages[i](x)

        return x











