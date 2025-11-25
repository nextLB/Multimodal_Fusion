
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
import albumentations as A
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 推理无人机图像数据前进行的数据转换器
padTransform = A.Compose([
    A.PadIfNeeded(
        min_height=config_parameters.HEIGHT,
        min_width=config_parameters.WIDTH,
        border_mode=0,
        value=0,
        mask_value=0,
        position="center"
    )

])

# 目前对于无人机数据进行语义推理的类别与颜色映射设定
CLASS_COLORS = {
    0: (0, 0, 0),   # 背景 - 黑色
    1: (0, 255, 0), # 水稻行(rice_row) - 绿色
    2: (0, 0, 255), # 田梗(ridge) - 蓝色
    3: (255, 0, 0), # 坟头(grave) - 红色
    4: (255, 0, 255),   # 杆子(pole) -
}




# 将类别掩码转换为彩色图像
def apply_color_map(maskArray):
    height, width = maskArray.shape
    colorMask = np.zeros((height, width, 3), dtype=np.uint8)

    for classId, color in CLASS_COLORS.items():
        colorMask[maskArray == classId] = color

    return colorMask

def drone_draw(tempImage, predMask, tempWidth, tempHeight, outputPath, number):
    # 调整回原始尺寸
    resizedMask = cv2.resize(
        predMask,
        (tempWidth, tempHeight),
        interpolation=cv2.INTER_NEAREST  # 保持类别标签不变
    )

    # 彩色掩码覆盖图
    colorMask = apply_color_map(resizedMask)
    overlay = cv2.addWeighted(tempImage, 0.7, colorMask, 0.3, 0)

    # 轮廓图
    contourImage = overlay.copy()
    riceRowContour = []
    ridgeContour = []
    graveContour = []
    poleContour = []
    for classId in range(1, config_parameters.NUM_CLASSES):  # 跳过背景
        classMask = (resizedMask == classId).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            classMask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS  # 更智能的压缩算法
        )

        # save picture
        cv2.imwrite(
            os.path.join(outputPath, f"results_{number}.jpg"),
            cv2.cvtColor(contourImage, cv2.COLOR_RGB2BGR)
        )


        # # save mask
        # cv2.imwrite(
        #     os.path.join(outputPath, f"masks_{number}.png"),
        #     resizedMask
        # )
        # print(resizedMask.shape)

        if classId == 1:
            for i in range(len(contours)):
                riceRowContour.append([])
                for j in range(len(contours[i])):
                    riceRowContour[i].append([int(contours[i][j][0][0]), int(contours[i][j][0][1])])

        if classId == 2:
            for i in range(len(contours)):
                ridgeContour.append([])
                for j in range(len(contours[i])):
                    ridgeContour[i].append([int(contours[i][j][0][0]), int(contours[i][j][0][1])])

        if classId == 3:
            for i in range(len(contours)):
                graveContour.append([])
                for j in range(len(contours[i])):
                    graveContour[i].append([int(contours[i][j][0][0]), int(contours[i][j][0][1])])

        if classId == 4:
            for i in range(len(contours)):
                poleContour.append([])
                for j in range(len(contours[i])):
                    poleContour[i].append([int(contours[i][j][0][0]), int(contours[i][j][0][1])])

    return contourImage, riceRowContour, ridgeContour, graveContour, poleContour




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

def inference_create_all_path():
    saveInferenceImagesPath = config_path.INFERENCE_RESULT_PATH
    saveInferenceContoursPath = config_path.INFERENCE_CONTOURS_PATH

    os.makedirs(saveInferenceImagesPath, exist_ok=True)
    clear_folder(saveInferenceImagesPath)
    os.makedirs(saveInferenceImagesPath, exist_ok=True)

    os.makedirs(saveInferenceContoursPath, exist_ok=True)
    clear_folder(saveInferenceContoursPath)
    os.makedirs(saveInferenceContoursPath, exist_ok=True)

    return saveInferenceImagesPath, saveInferenceContoursPath



# 计算语义分割多类别的Iou
def calculate_iou(preds, targets, numClasses):
    ious = []
    preds = torch.argmax(preds, dim=1)

    # 忽略背景类
    presentClasses = 0
    for cls in range(numClasses):
        predInds = (preds == cls)
        targetInds = (targets == cls)

        # 如果目标中没有该类，则跳过
        if targetInds.long().sum().item() == 0:
            continue

        intersection = (predInds & targetInds).long().sum().float()
        union = (predInds | targetInds).long().sum().float()

        if union > 0:
            ious.append(intersection / union)
            presentClasses += 1

    if presentClasses == 0:
        return torch.tensor(0.0).to(device)
    return torch.sum(torch.stack(ious)) / presentClasses




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


# 对指定文件夹下的图片按文件名中的数字进行排序，以便后续的遍历操作
def sort_images_by_number(folderPath: str) -> List[str]:
    # 支持的图片文件扩展名
    imageExtensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # 获取文件夹中所有的图片文件
    imageFiles = [
        f for f in os.listdir(folderPath)
        if f.lower().endswith(imageExtensions) and os.path.isfile(os.path.join(folderPath, f))
    ]

    def extract_number(filename: str) -> int:
        # 使用正则表达式找到所有数字序列
        numbers = re.findall(r'\d+', filename)
        # 如果找到数字，返回第一个数字的整数形式
        if numbers:
            return int(numbers[0])
        # 如果没有找到数字，返回一个很大的数放在最后
        return float('inf')

    # 按提取的数字排序
    sortedFiles = sorted(imageFiles, key=extract_number)

    return sortedFiles



