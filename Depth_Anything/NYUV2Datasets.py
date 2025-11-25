"""
    对于NYUV2训练集构建的程序文件
"""
import os
import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import CONFIG
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# 自定义NYUV2的数据集构建文件
class NYUDepthDataset(Dataset):
    def __init__(self, csvFile, dataDir, transform, isTrain, imgSize):

        self.dataDir = dataDir
        self.transform = transform
        self.isTrain = isTrain
        self.imgSize = imgSize

        # 训练集：从文件夹结构构建数据对
        if isTrain:
            self.samples = self._build_train_samples()
        # 测试集：从CSV文件读取
        else:
            self.dataFrame = pd.read_csv(csvFile, header=None)

    def _build_train_samples(self):
        """构建训练集样本列表"""
        samples = []
        trainDir = os.path.join(self.dataDir, 'nyu2_train')

        # 遍历所有子文件夹
        for subdir in os.listdir(trainDir):
            sudirPath = os.path.join(trainDir, subdir)
            if os.path.isdir(sudirPath):
                # 获取文件夹中的所有jpg文件
                jpgFile = [f for f in os.listdir(sudirPath) if f.endswith('.jpg')]
                for jpgFile in jpgFile:
                    # 对应的png文件（深度图）
                    baseName = os.path.splitext(jpgFile)[0]
                    pngFile = baseName + '.png'
                    jpgPath = os.path.join(sudirPath, jpgFile)
                    pngPath = os.path.join(sudirPath, pngFile)

                    if os.path.exists(pngPath):
                        samples.append((jpgPath, pngPath))

        print(f"Found {len(samples)} training samples")
        return samples

    def __len__(self):
        if self.isTrain:
            return len(self.samples)
        else:
            return len(self.dataFrame)

    def __getitem__(self, idx):
        if self.isTrain:
            rgbPath, depthPath = self.samples[idx]
        else:
            rgbPath, depthPath = self.dataFrame.iloc[idx, 0], self.dataFrame.iloc[idx, 1]
            # 去掉路径开头的 "data/" 前缀（如果有的话）
            rgbPath = rgbPath.lstrip("data/")  # 关键：删除多余的"data/"
            depthPath = depthPath.lstrip("data/")  # 关键：删除多余的"data/"
            rgbPath = os.path.join(self.dataDir, rgbPath)
            depthPath = os.path.join(self.dataDir, depthPath)


        # 加载RGB图像
        rgbImage = Image.open(rgbPath)

        # 加载深度图 - 使用PIL直接打开，保持原始精度
        depthImage = Image.open(depthPath)

        # 转换为numpy数组进行处理
        rgbArray = np.array(rgbImage)
        depthArray = np.array(depthImage)

        # 调整尺寸
        rgbArray = cv2.resize(rgbArray, (self.imgSize[1], self.imgSize[0]))
        depthArray = cv2.resize(depthArray, (self.imgSize[1], self.imgSize[0]), interpolation=cv2.INTER_NEAREST)

        # 归一化RGB图像到[0, 1]
        rgbArray = rgbArray.astype(np.float32) / 255.0

        # 处理深度图
        if len(depthArray.shape) == 3:
            depthArray = depthArray[:, :, 0]    # 取第一个通道

        # 深度图处理 - 更精准的归一化
        depthArray = depthArray.astype(np.float32)

        # NYU深度图的真实范围通常是0.7m到10m
        # 深度图存储为16为PNG,实际深度值需要除以1000或5000
        depthArray = depthArray / 5000.0
        depthArray = np.clip(depthArray, 0.001, 1.0)    # 避免除以零

        # 转换为tensor
        rgbTensor = torch.from_numpy(rgbArray).permute(2, 0, 1)     # HWC to CHW
        depthTensor = torch.from_numpy(depthArray).unsqueeze(0)     # 添加通道维度

        return rgbTensor, depthTensor




def main():
    # 进行数据集的构建与可视化
    # 创建数据加载器
    trainDataset = NYUDepthDataset(CONFIG.TRAIN_CSV_PATH, CONFIG.DATA_DIR_PATH, transform=None, isTrain=True, imgSize=CONFIG.IMAGE_SIZE)
    testDataset = NYUDepthDataset(CONFIG.TEST_CSV_PATH, CONFIG.DATA_DIR_PATH, transform=None, isTrain=False, imgSize=CONFIG.IMAGE_SIZE)

    # 分割训练集和验证集
    trainSize = int(0.8 * len(trainDataset))
    valSize = len(trainDataset) - trainSize
    trainSubset, valSubset = torch.utils.data.random_split(trainDataset, [trainSize, valSize])

    trainLoader = DataLoader(trainSubset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=CONFIG.NUMBER_WORKERS)
    valLoader = DataLoader(valSubset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUMBER_WORKERS)
    testLoader = DataLoader(testDataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUMBER_WORKERS)
    print(f"Training samples: {len(trainLoader)}")
    print(f"Validation samples: {len(valLoader)}")
    print(f"Test samples: {len(testLoader)}")


    # # 可视化数据集
    # os.makedirs(CONFIG.VISUAL_DATASETS, exist_ok=True)
    # with tqdm(total=len(trainLoader) + len(valLoader) + len(testLoader), desc="数据集可视化") as pbar:
    #
    #     for batchIndex, (rgbTensor, depthTensor) in enumerate(trainLoader):
    #         for i in range(CONFIG.BATCH_SIZE):
    #
    #             # 转换为numpy数组
    #             rgbNp = rgbTensor[i].permute(1, 2, 0).numpy()   # CHW to HWC
    #             depthNp = depthTensor[i].squeeze().numpy()      # 移除通道维度
    #
    #             # 反归一化RGB
    #             rgbNp = (rgbNp * 255.0).astype(np.uint8)
    #
    #             # 创建可视化图
    #             fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #
    #             # RGB图像
    #             axes[0].imshow(rgbNp)
    #             axes[0].set_title('Input RGB Image', fontsize=14)
    #             axes[0].axis('off')
    #
    #             # 真实深度图
    #             im1 = axes[1].imshow(depthNp, cmap='plasma')
    #             axes[1].set_title('Ground Truth Depth', fontsize=14)
    #             axes[1].axis('off')
    #             plt.colorbar(im1, ax=axes[1], fraction=0.046)
    #
    #             plt.tight_layout()
    #             plt.subplots_adjust(bottom=0.1)
    #
    #             # 保存结果
    #             outputPath = os.path.join(CONFIG.VISUAL_DATASETS, f'sample_{batchIndex}_{i}.png')
    #             plt.savefig(outputPath, dpi=150, bbox_inches='tight')
    #             plt.close()
    #
    #         pbar.update(1)
    #

    return trainLoader, valLoader, testLoader



if __name__ == '__main__':
    main()


