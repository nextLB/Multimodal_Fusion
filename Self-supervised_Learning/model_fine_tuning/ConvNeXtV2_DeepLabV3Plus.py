

"""
    基于ConvNextV2_DeepLabV3Plus模型架构进行训练的程序文件
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import numpy as np
import os
import config_parameters
import config_path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import utils
from tqdm import tqdm
import cv2
from timm.models.layers import DropPath
import torch.nn.functional as F
import torch
import torch.nn as nn



################################################################
#           预训练数据集的构建过程如下                               #
################################################################


# 预训练集的数据增强
trainTransform = A.Compose(
    [
        A.RandomResizedCrop(config_parameters.PRETRAINED_RESIZED_HEIGHT, config_parameters.PRETRAINED_RESIZED_WIDTH,
                                scale=config_parameters.CROPS_SCALE, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5),  # 作用是对图像的颜色属性（亮度、对比度、饱和度、色调）进行随机调整
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=0.1),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()

    ])




# 预训练时的数据加载类
class PretrainedDataset(Dataset):
    def __init__(self, rootDir, transform):
        self.rootDir = rootDir
        self.transform = transform
        self.imagePaths = []

        self.baseTransform = A.Compose([
            A.Resize(config_parameters.PRETRAINED_RESIZED_HEIGHT,
                     config_parameters.PRETRAINED_RESIZED_WIDTH,
                     interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for subdir, _, files in os.walk(rootDir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    self.imagePaths.append(os.path.join(subdir, file))

        if len(self.imagePaths) == 0:
            raise ValueError(f"No images found in {rootDir}")

        print(f"Found {len(self.imagePaths)} images in dataset")

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        img_path = self.imagePaths[idx]

        # 使用 OpenCV 读取图像 (Albumentations 推荐方式)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        originalImage = self.baseTransform(image=image)["image"]
        augmentationImage = self.transform(image=image)["image"]


        return originalImage, augmentationImage



def load_pretrained_datasets():
    fullDatasets = PretrainedDataset(config_path.SELF_SUPERVISED_DATA_PATH, trainTransform)

    fullDataLoader = DataLoader(
        fullDatasets,
        batch_size=config_parameters.PRETRAINED_BATCH_SIZE,
        shuffle=True,
        num_workers=config_parameters.PRETRAINED_NUM_WORKERS
    )


    # # 创建一下可视化存储路径
    # os.makedirs(config_path.SAVE_TRANSFORM_DATASETS, exist_ok=True)
    # utils.clear_folder(config_path.SAVE_TRANSFORM_DATASETS)
    # os.makedirs(config_path.SAVE_TRANSFORM_DATASETS, exist_ok=True)
    # # # 可视化一下加载的数据集
    # with tqdm(total=len(fullDataLoader), desc="数据集可视化中") as pbarDataloader:
    #     for batchIndex, (originalImage, augmentationImage) in enumerate(fullDataLoader):
    #         for i in range(config_parameters.PRETRAINED_BATCH_SIZE):
    #             utils.save_transform_datasets(originalImage[i], augmentationImage[i], config_path.SAVE_TRANSFORM_DATASETS, 'trainData', batchIndex, i, 0)
    #         pbarDataloader.update(1)

    return fullDatasets





################################################################
#           预训练数据集的构建过程如上                               #
################################################################





################################################################
#           预训练模型架构的构建过程如下                               #
################################################################

class LayerNorm(nn.Module):
    """自定义层归一化，支持两种数据格式"""

    def __init__(self, normalizedShape, eps, dataFormat):
        """

        :param normalizedShape: 要归一化的形状 (通常是特征维度)
        :param eps: 防止除零的小常数
        :param dataFormat: 数据格式，“channels_last” 或 "channels_first"
        """
        super(LayerNorm, self).__init__()

        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(normalizedShape))
        # 可学习的偏置参数
        self.bias = nn.Parameter(torch.zeros(normalizedShape))

        self.eps = eps
        self.dataFormat = dataFormat

        # 验证数据格式是否支持
        if self.dataFormat not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        self.normalizedShape = (normalizedShape, )

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

        :param dim: 特征维度
        """
        super(GRN, self).__init__()

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



################################################################
#           预训练模型架构的构建过程如上                               #
################################################################




def main(saveModelPath, saveLogFilePath, saveFeatureMapsPath, logger, writer, MyFeatureMapHook):

    # TODO: 加载预训练数据集
    logger.info('开始加载本次预训练所用数据集')
    pretrainedDataset = load_pretrained_datasets()
    logger.info('预训练数据集加载完成')

    # TODO: 加载CovNeXtV2预训练模型架构
    logger.info('开始加载本次预训练所用的ConvNeXtV2预训练模型架构')

    logger.info('ConvNeXtV2预训练模型架构加载完毕')



    # TODO: 加载模型微调数据集



