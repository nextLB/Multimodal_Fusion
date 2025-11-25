
"""
        项目实现过程中自己自定义实现的一些工具
"""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import gc
import torch
import re
from typing import List
import config_path
import config_parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def clear_folder(folderPath: str):

    # 遍历文件夹内的所有内容
    for item in os.listdir(folderPath):
        itemPath = os.path.join(folderPath, item)
        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(itemPath) or os.path.islink(itemPath):
                os.unlink(itemPath)
                print(f'已删除文件夹: {itemPath}')
            elif os.path.isdir(itemPath):
                # 使用 shutil.rmtree 删除文件夹及其内容
                shutil.rmtree(itemPath)
                print(f"已删除文件夹： {itemPath}")

        except Exception as e:
            print(f"删除 {itemPath} 时出错： {e}")

    print(f"文件夹 {folderPath} 内容已清空")




# 实现对于使用pytorch变换后数据集的可视化函数
def save_transform_datasets(data, target, savePath, name, batchIndex, i, j) -> None:

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))


    # 转换为numpy形式
    npData = data.permute(1, 2, 0).detach().cpu().numpy()
    if target.dim() == 3:
        npTarget = target.permute(1, 2, 0).detach().cpu().numpy()
    else:
        npTarget = target.squeeze().cpu().numpy()


    # 计算图像统计信息
    dataHeight, dataWidth, dataChannels = data.shape
    if target.dim() == 3:
        targetHeight, targetWidth, targetChannels = target.shape
    else:
        targetHeight, targetWidth = target.shape
        targetChannels = 1

    dataMin, dataMax = np.min(npData), np.max(npData)
    targetMin, targetMax = np.min(npTarget), np.max(npTarget)

    dataMean = np.mean(npData)
    targetMean = np.mean(npTarget)


    # 显示图片1
    im1 = ax1.imshow(npData)
    ax1.set_title('Data', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 显示图片2
    im2 = ax2.imshow(npTarget)
    ax2.set_title('Target', fontsize=14, fontweight='bold')
    ax2.axis('off')


    # 调整布局
    plt.tight_layout()
    # 为标题和信息文本留出空间
    plt.subplots_adjust(top=0.88, bottom=0.2)
    # 准备统计信息文本
    stats_text = (
        f"Data Info:\n"
        f"  Size: {dataHeight}x{dataWidth}x{dataChannels}\n"
        f"  Range: [{dataMin:.2f}, {dataMax:.2f}]\n"
        f"  Mean: {dataMean:.2f}\n\n"
        f"Target Info:\n"
        f"  Size: {targetHeight}x{targetWidth}x{targetChannels}\n"
        f"  Range: [{targetMin}, {targetMax}]\n"
        f"  Mean: {targetMean:.2f}\n\n"
        f"Sample Info: {name} | Batch Index: {batchIndex} | Position: ({i},{j})"
    )

    # 在图像下方添加统计信息
    plt.figtext(0.5, 0.01, stats_text,
                fontsize=10, ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='gray', pad=8))

    # 保存图像
    fullPath = os.path.join(savePath, f'name_{name}_batchIndex_{batchIndex}_{i}_{j}.png')
    plt.savefig(fullPath, dpi=300, bbox_inches='tight')


    # 释放内存
    plt.close(fig)
    gc.collect()





# 构建存储路径
def create_all_path():
    saveModelPath = os.path.join(config_path.MODEL_PATH, f'maxEpochs_{config_parameters.MAX_EPOCHS}_learningRate_{config_parameters.LEARNING_RATE}')
    saveLogFilePath = os.path.join(config_path.LOG_FILE_PATH, f'maxEpochs_{config_parameters.MAX_EPOCHS}_learningRate_{config_parameters.LEARNING_RATE}')
    saveFeatureMapsPath = os.path.join(config_path.FEATURE_MAPS_PATH, f'maxEpochs_{config_parameters.MAX_EPOCHS}_learningRate_{config_parameters.LEARNING_RATE}')
    os.makedirs(saveModelPath, exist_ok=True)
    clear_folder(saveModelPath)
    os.makedirs(saveModelPath, exist_ok=True)

    os.makedirs(saveLogFilePath, exist_ok=True)
    clear_folder(saveLogFilePath)
    os.makedirs(saveLogFilePath, exist_ok=True)

    os.makedirs(saveFeatureMapsPath, exist_ok=True)
    clear_folder(saveFeatureMapsPath)
    os.makedirs(saveFeatureMapsPath, exist_ok=True)

    return saveModelPath, saveLogFilePath, saveFeatureMapsPath






# 绘制训练图像
def draw_train_picture(trainLosses, valLosses, valIous, savePath) -> None:
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label='train loss')
    plt.plot(valLosses, label='val loss')
    plt.title('train and val losses')
    plt.xlabel('Epoch')
    plt.ylabel('losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valIous, label='valIou', color='green')
    plt.title('valIou')
    plt.xlabel('Epoch')
    plt.ylabel('Iou')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(savePath, 'train_metric.png'))
    print("save picture over")













