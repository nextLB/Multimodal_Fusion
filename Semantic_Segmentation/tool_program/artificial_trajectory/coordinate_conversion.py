

import json
import cv2
import os
from pyproj import Transformer

imagePath = '/home/next_lb/桌面/WYT_S_S/code/tool_program/artificial_trajectory/raw_image_and_data/yanjiaqiao_2.png'
jsonPath = '/home/next_lb/桌面/WYT_S_S/code/tool_program/artificial_trajectory/raw_jsons/yanjiaqiao_2.json'
RESOLUTION_X = 0.00790000000000125
DISTORTION_COEFFICIENT_X = 0
DISTORTION_COEFFICIENT_Y = 0
RESOLUTION_Y = -0.00790000000001095
START_COORDINATES_X = 267529.743424
START_COORDINATES_Y = 3505668.493377

def utm_to_wgs84(utm_x, utm_y, utm_zone, northern_hemisphere=True):
    """
    将UTM坐标转换为WGS84坐标

    参数:
    utm_x: UTM东坐标
    utm_y: UTM北坐标
    utm_zone: UTM区域编号
    northern_hemisphere: 布尔值，表示是否在北半球

    返回:
    包含纬度和经度的元组 (lat, lon)
    """
    # 确定UTM坐标系的半球参数
    if northern_hemisphere:
        utm_crs = f"EPSG:326{utm_zone}"
    else:
        utm_crs = f"EPSG:327{utm_zone}"

    wgs84_crs = "EPSG:4326"  # WGS84坐标系
    transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    # 转换坐标
    lon, lat = transformer.transform(utm_x, utm_y)

    return lat, lon



def convert_pixel_to_utm(pixelX, pixelY, originX, originY, xRes, yRes):
    """
    将像素坐标转换为UTM坐标

    参数:
    pixel_x: 像素x坐标
    pixel_y: 像素y坐标
    origin_x: 左上角UTM东坐标
    origin_y: 左上角UTM北坐标
    x_res: X方向分辨率(米/像素)
    y_res: Y方向分辨率(米/像素)，通常为负值

    返回:
    包含UTM坐标(x, y)的元组
    """

    utmX = originX + pixelX * xRes
    utmY = originY + pixelY * yRes  # y_res为负值，所以这里用加法

    return utmX, utmY


def main():

    ############################################
    # resized images and jsons
    original_image = cv2.imread(imagePath)

    # 获取原始图像尺寸
    original_height, original_width = original_image.shape[:2]

    # 目标尺寸
    target_width, target_height = 3000, 3000

    # 计算缩放比例
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # 调整图像尺寸
    resized_image = cv2.resize(original_image, (target_width, target_height))

    # 读取json文件
    with open(jsonPath, 'r') as f:
        data = json.load(f)

    # 提取原始标注中的shape_type（默认用polyline，可根据实际情况调整）
    # 优先从原始数据中获取shape_type，没有则默认polyline
    shape_type = data['shapes'][0].get('shape_type', 'polyline') if data['shapes'] else 'polyline'


    # 提取需要的数据
    extractedData = {}

    # 构建LabelMe格式的resized数据
    resized_trajectories = []  # 缩放后像素坐标：[{"label": "...", "points": [...]}]

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        # print(label)
        utmPoints = []
        tempPoints = []
        for i, point in enumerate(points):
            # print(i, point)
            pixelX, pixelY = point

            # 将像素坐标转换为UTM坐标
            utmX, utmY = convert_pixel_to_utm(pixelX, pixelY, START_COORDINATES_X, START_COORDINATES_Y, RESOLUTION_X, RESOLUTION_Y)
            # print(utmX, utmY)
            utmPoints.append((utmX, utmY))
            # # 将UTM坐标转换为WGS84坐标
            # lat, lon = utm_to_wgs84(utmX, utmY, '51', True)
            # print(lat, lon)

            pixelX = pixelX * scale_x
            pixelY = pixelY * scale_y
            tempPoints.append((pixelX, pixelY))

        # print(utmPoints)
        extractedData[f'{label}'] = utmPoints

        # 将单条轨迹添加到列表（可视化工具要求的格式）
        resized_trajectories.append({"label": label, "points": tempPoints})




    outputJsonPath = '/home/next_lb/桌面/WYT_S_S/code/tool_program/artificial_trajectory/wgs84_jsons/wgs84_ridge_coordinate.json'

    # 保存结果到新的JSON文件
    with open(outputJsonPath, 'w') as f:
        json.dump(extractedData, f, indent=4)

    print(f"转换完成，结果已保存到: {outputJsonPath}")


    resizedImagePath = '/home/next_lb/桌面/WYT_S_S/code/tool_program/artificial_trajectory/resized_images/resized_area_1.png'
    cv2.imwrite(resizedImagePath, resized_image)

    resizedJsonPath = '/home/next_lb/桌面/WYT_S_S/code/tool_program/artificial_trajectory/resized_jsons/resized_area_1.json'
    with open(resizedJsonPath, 'w') as f:
        json.dump(resized_trajectories, f, indent=4)

    print("图像和JSON文件已成功调整并保存")



if __name__ == '__main__':
    main()





