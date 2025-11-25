
"""
    调用模型进行推理检测的程序
"""

import models
import utils
import config_parameters
import config_path
import torch
import os
import numpy as np
from PIL import Image
import cv2
import datasets
import WYT_Kalman_V1_9_3



USE_INFERENCE_MODEL_PATH = '/home/next_lb/桌面/WYT_S_S/code/robot_deeplabv3/results/pytorch_models/maxEpochs_110_learningRate_0.001/deepLabV3_low_loss.pth'
OUTPUT_IMAGES_PATH = '/home/next_lb/桌面/WYT_S_S/code/robot_deeplabv3/results/inference_images'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别颜色映射 (可根据实际类别修改)
CLASS_COLORS = {
    0: (0, 0, 0),  # 背景 - 黑色
    1: (0, 255, 0),  # 水稻行 - 绿色
    2: (0, 0, 255),  # 田埂 - 蓝色
    3: (255, 0, 0),  #  - 红色
    4: (255, 0, 255)
}

horizontal_ys = [config_parameters.layerUpper, config_parameters.layerMiddle, config_parameters.layerLower]


def apply_color_map(mask_array):
    """将类别掩码转换为彩色图像"""
    height, width = mask_array.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask_array == class_id] = color

    return color_mask



def point_in_contour(contour, point):
    """
    优化版：判断点是否在轮廓内部（边界不算内部）
    专门针对格式 [[[x1,y1]], [[x2,y2]], ...] 进行优化
    :param contour: 轮廓点集，格式为 [[[x1,y1]], [[x2,y2]], ...]
    :param point: 待判断点 [x, y]
    :return: True（内部）或 False（外部或边界）
    """
    # 预提取所有点坐标，避免多次索引
    n = len(contour)
    if n < 3:  # 至少3个点才能形成多边形
        return False

    # 直接提取所有点坐标
    points = [pt[0] for pt in contour]
    px, py = point
    inside = False
    x1, y1 = points[0]
    x1, y1 = int(x1), int(y1)

    # 主要优化：减少循环内计算和类型转换
    for i in range(n + 1):
        # 使用索引技巧避免取模运算
        x2, y2 = points[i % n]
        x2, y2 = int(x2), int(y2)

        # 检查点是否在边上（边界不算内部）
        if (y1 == y2 == py) and (min(x1, x2) <= px <= max(x1, x2)):
            return False
        elif min(y1, y2) < py <= max(y1, y2):
            # 计算交点横坐标（避免除零错误）
            if y1 != y2:
                # 使用整数运算避免浮点开销
                x_intersect = x1 + (x2 - x1) * (py - y1) // (y2 - y1)
                if px < x_intersect:
                    inside = not inside

        # 移动到下一个点
        x1, y1 = x2, y2

    return inside




# 根据三条线与轮廓的交点初步计算得到跟踪点位
def process_contour_intersections(contours, horizontalYs):
    """优化后的交点处理函数，减少无效计算"""
    lineIntersections = {y: [] for y in horizontalYs}

    CONTOURSIZE = 10
    tempIndex = []
    allPoints = []
    for i in range(len(contours)):
        if len(contours[i]) <= CONTOURSIZE:
            tempIndex.append(i)

    contours = list(contours)
    for i in range(len(tempIndex)):
        del contours[tempIndex[i] - i]


    # 先遍历水平直线，再遍历轮廓
    for yLine in horizontalYs:
        for contour in contours:
            # 提前过滤不相交的线段
            for i in range(len(contour)):
                x1, y1 = contour[i][0]
                x2, y2 = contour[(i + 1) % len(contour)][0]

                # 线段两端均在y_line上方或下方，无交点
                if (y1 > yLine and y2 > yLine) or (y1 < yLine and y2 < yLine):
                    continue

                # 计算交点(仅处理可能相交的线段)
                if y1 == y2:    # 水平线段，跳过
                    continue

                t = (yLine - y1) / (y2 - y1)
                xInter = x1 + t * (x2 - x1)
                lineIntersections[yLine].append((int(xInter), yLine))

        intersections = lineIntersections[yLine]
        intersections.sort(key=lambda p: p[0])
        # calculate all middle points
        middlePoints = []
        for i in range(len(intersections)-1):
            middleX = int((intersections[i][0] + intersections[i+1][0]) / 2)
            middleY = int((intersections[i][1] + intersections[i+1][1]) / 2)
            tempPoint = [middleX, middleY]
            for contour in contours:
                tempFlag = point_in_contour(contour, tempPoint)
                if tempFlag == True:
                    middlePoints.append(tempPoint)
                    break
        allPoints.append(middlePoints)


    return contours, allPoints





def draw(originalImage, predMask, originalWidth, originalHeight, baseName, outputPath):
    # 调整回原始尺寸
    resizedMask = cv2.resize(
        predMask,
        (originalWidth, originalHeight),
        interpolation=cv2.INTER_NEAREST  # 保持类别标签不变
    )

    # 彩色掩码覆盖图
    colorMask = apply_color_map(resizedMask)
    overlay = cv2.addWeighted(originalImage, 0.7, colorMask, 0.3, 0)

    # 轮廓图
    contourImage = overlay.copy()
    for classId in range(1, config_parameters.NUM_CLASSES):  # 跳过背景
        classMask = (resizedMask == classId).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            classMask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS  # 更智能的压缩算法
        )


        # ￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥ #
        #                      这里计算跟踪点位                           #
        # ￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥ #
        if classId == 1 and len(contours) != 0:
            contours, middlePoints = process_contour_intersections(contours, horizontal_ys)

            for i in range(len(middlePoints)):
                for j in range(len(middlePoints[i])):
                    cv2.circle(contourImage, (int(middlePoints[i][j][0]), int(middlePoints[i][j][1])), 3, (0, 0, 255), -1)

            contourImage, XLine = WYT_Kalman_V1_9_3.new_ulfd_find_line(contourImage, middlePoints)
            print(XLine)


        cv2.drawContours(
            contourImage,
            contours,
            -1,
            CLASS_COLORS[classId],
            2
        )

        # save picture
        cv2.imwrite(
            os.path.join(outputPath, f"{baseName}_overlay.jpg"),
            cv2.cvtColor(contourImage, cv2.COLOR_RGB2BGR)
        )


    print(f"处理完成: {baseName}")




def main():
    # deepLabV3Model = models.SimpleDeepLabV3
    # deepLabV3Model.load_state_dict(torch.load(USE_INFERENCE_MODEL_PATH, map_location=device, weights_only=True))
    # deepLabV3Model.eval()


    deepLabV3Model = models.MobileNetV2DeepLabV3().to(device)
    deepLabV3Model.load_state_dict(torch.load(USE_INFERENCE_MODEL_PATH, map_location=device, weights_only=True))
    deepLabV3Model.eval()



    testImages = utils.sort_images_by_number(config_path.INFERENCE_IMAGES_PATH)
    os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)


    # ======================================================================= #
    for imageName in testImages:
        if not imageName.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        imagePath = os.path.join(config_path.INFERENCE_IMAGES_PATH, imageName)
        print(f"处理图像: {imagePath}")

        # 读取原始图像
        originalImage = np.array(Image.open(imagePath).convert("RGB"))

        tempOriginalImage = cv2.imread(imagePath)

        originalImage = originalImage[0:280, 100:380]
        originalHeight, originalWidth = originalImage.shape[:2]

        # 预处理
        transformed = datasets.valTransform(image=originalImage)
        inputTensor = transformed["image"].unsqueeze(0).to(device)


        # 模型推理
        with torch.no_grad():

            output = deepLabV3Model(inputTensor)

            # 多类别使用argmax获取预测类别
            predMask = torch.argmax(output, dim=1)
            # 去除batch维度
            predMask = predMask.squeeze(0).cpu().numpy().astype(np.uint8)

        baseName = os.path.splitext(imageName)[0]


        draw(originalImage, predMask, originalWidth, originalHeight, baseName, OUTPUT_IMAGES_PATH)




if __name__ == '__main__':
    main()


