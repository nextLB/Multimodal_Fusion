

"""
    处理images与jsons文件生成对应的masks文件的程序
"""


import os
import json
import numpy as np
import cv2
from tqdm import tqdm
# 可视化透明度
ALPHA = 0.5



# 类别到像素值的映射 (可根据需要扩展)
class_mapping = {
    "rice_row": 1,
    "ridge": 2,
    "grave": 3,
    "pole": 4
    # 添加更多类别...
}

# 类别到颜色的映射 (可根据需要扩展)
class_colors = {
    1: (0, 255, 0),  # rice_row: 绿色
    2: (0, 0, 255),  # ridge: 红色
    3: (255, 0, 0),
    4: (255, 0, 255)
    # 添加更多类别...
}


def create_mask_from_json(jsonPath, imgDir, outputDir) -> None:
    # 从JSON文件名获取基础名称
    baseName = os.path.splitext(os.path.basename(jsonPath))[0]
    # 构建对应的图片路径
    imgPath = os.path.join(imgDir, baseName + '.png')

    if not os.path.exists(imgPath):
        print(f"警告: 对应的图片文件不存在 {imgPath}")
        return

    # 读取图片获取尺寸
    img = cv2.imread(imgPath)
    if img is None:
        print(f"错误: 无法读取图片 {imgPath}")
        return

    height, width = img.shape[:2]

    # 创建空白掩码 (单通道)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 读取JSON文件
    with open(jsonPath, 'r') as f:
        data = json.load(f)
    
    for shape in data['shapes']:
        label = shape['label']

        # 跳过未定义的类别
        if label not in class_mapping:
            print(f"警告: 未定义的类别 '{label}' 在文件 {jsonPath}")
            continue

        # 获取类别对应的像素值
        classValue = class_mapping[label]

        # 提取多边形点
        points = np.array(shape['points'], dtype=np.int32)

        # 在掩码上绘制多边形
        cv2.fillPoly(mask, [points], color=classValue)

    maskOutputPath = os.path.join(outputDir, baseName + '_mask.png')
    cv2.imwrite(maskOutputPath, mask)
    print(f"已保存掩码: {maskOutputPath}")



def visualize_mask(imgPath, maskPath, outputPath):
    # 检查文件是否存在
    if not os.path.exists(imgPath):
        print(f"警告: 图片文件不存在 {imgPath}")
        return

    if not os.path.exists(maskPath):
        print(f"警告: 掩码文件不存在 {maskPath}")
        return

    # 读取图片和掩码
    img = cv2.imread(imgPath)
    mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"错误: 无法读取图片 {imgPath}")
        return

    if mask is None:
        print(f"错误: 无法读取掩码 {maskPath}")
        return

    # 确保图片和掩码尺寸一致
    if img.shape[:2] != mask.shape:
        print(f"错误: 图片和掩码尺寸不匹配 {imgPath}")
        return

    # 创建彩色掩码
    color_mask = np.zeros_like(img)

    # 根据掩码值为每个类别填充颜色
    for class_value, color in class_colors.items():
        # 找到该类别的区域
        class_region = (mask == class_value)

        # 在彩色掩码上填充对应颜色
        color_mask[class_region] = color

    # 将彩色掩码叠加到原图上
    overlay = cv2.addWeighted(img, 1, color_mask, ALPHA, 0)

    # 保存可视化结果
    cv2.imwrite(outputPath, overlay)
    print(f"已保存可视化结果: {outputPath}")






def convert_to_masks(imagesPath, jsonsPath, masksPath, visualizationsPath) -> None:

    # 处理所有JSON文件
    jsonFiles = [f for f in os.listdir(jsonsPath) if f.endswith('.json')]

    for jsonFile in tqdm(jsonFiles, desc="处理JSON文件"):
        jsonPath = os.path.join(jsonsPath, jsonFile)
        create_mask_from_json(jsonPath, imagesPath, masksPath)
    

    # 获取所有图片文件
    imgFiles = [f for f in os.listdir(imagesPath) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 处理每张图片
    for imgFile in tqdm(imgFiles, desc="可视化掩码"):
        baseName = os.path.splitext(imgFile)[0]

        # 构建对应的掩码路径
        maskFile = baseName + '_mask.png'
        maskPath = os.path.join(masksPath, maskFile)

        # 构建输出路径
        outputFile = baseName + '_visualization.png'
        outputPath = os.path.join(visualizationsPath, outputFile)

        # 可视化并保存
        imgPath = os.path.join(imagesPath, imgFile)
        visualize_mask(imgPath, maskPath, outputPath)




def main():
    imagesPath = './images'
    jsonsPath = './jsons'
    masksPath = './masks'
    visualizationsPath = './visualizations'
    convert_to_masks(imagesPath, jsonsPath, masksPath, visualizationsPath)


if __name__ == '__main__':
    main()
