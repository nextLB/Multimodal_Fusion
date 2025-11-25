

import json
import cv2
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import torch
import time
import math
from typing import List, Tuple
import copy




RESIZED_IMAGE_WIDTH = 1024
RESIZED_IMAGE_HEIGHT = 1024
BLOCK_HEIGHT = 320
BLOCK_WIDTH = 320
DISPLAY_DURATION = 0.01
SAVE_RESULTS_BOUNDARY_PATH = '/home/next_lb/桌面/WYT_S_S/code/tool_program/predicted_boundary/boundary_results/result_boundary.png'



def euclidean_distance(point1, point2):
    """计算两个点之间的欧几里得距离。

    参数:
    point1: tuple, 第一个点的坐标 (x1, y1)
    point2: tuple, 第二个点的坐标 (x2, y2)

    返回:
    float, 欧几里得距离

    """
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance




def load_contour_data(json_file_path):
    """读取轮廓数据JSON文件并返回解析后的数据"""
    try:
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"文件不存在: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as f:
            contour_data = json.load(f)

        required_keys = ['ridge', 'rice_row', 'grave', 'pole', 'image_info']
        for key in required_keys:
            if key not in contour_data:
                raise KeyError(f"JSON文件缺少必要的键: {key}")

        return contour_data

    except Exception as e:
        print(f"读取JSON文件失败: {str(e)}")
        return None

# The minimum value is selected and its subscript in the original array is returned
def get_min_value(array):
    length = len(array)
    if length == 0:
        # print('It is not valid for the array length to be empty.')
        return -1
    elif length == 1:
        return 0
    else:
        minValue = float('inf')
        minIndex = -1
        for i in range(length):
            if array[i] <= minValue:
                minValue = array[i]
                minIndex = i
        return minIndex






def generate_rectangle_edge_points(up_left: List[float],
                                 down_right: List[float],
                                 points_per_side: int) -> Tuple[List[List[float]],
                                                              List[List[float]],
                                                              List[List[float]],
                                                              List[List[float]]]:
    """
    生成矩形边界上的均匀分布点，并分别返回四条边上的点

    参数:
    up_left: 矩形左上角坐标 [x, y]
    down_right: 矩形右下角坐标 [x, y]
    points_per_side: 每条边上要生成的点数

    返回:
    包含四个列表的元组，分别表示:
    (上边点列表, 右边点列表, 下边点列表, 左边点列表)
    所有点按顺时针方向排列
    """
    x1, y1 = up_left
    x2, y2 = down_right

    # 计算矩形的宽度和高度
    width = x2 - x1
    height = y2 - y1

    # 计算每条边上点的间隔
    x_interval = width / (points_per_side - 1) if points_per_side > 1 else 0
    y_interval = height / (points_per_side - 1) if points_per_side > 1 else 0

    # 初始化四个边的点列表
    top_points = []
    right_points = []
    bottom_points = []
    left_points = []

    # 生成上边的点 (从左到右)
    for i in range(points_per_side):
        top_points.append([x1 + i * x_interval, y1])

    # 生成右边的点 (从上到下，跳过第一个和最后一个点以避免重复)
    for i in range(1, points_per_side - 1):
        right_points.append([x2, y1 + i * y_interval])

    # 生成下边的点 (从右到左)
    for i in range(points_per_side - 1, -1, -1):
        bottom_points.append([x1 + i * x_interval, y2])

    # 生成左边的点 (从下到上，跳过第一个和最后一个点以避免重复)
    for i in range(points_per_side - 2, 0, -1):
        left_points.append([x1, y1 + i * y_interval])

    return top_points, right_points, bottom_points, left_points


def is_point_in_rectangle(point, up_left, down_right):
    x, y = point
    x1, y1 = up_left
    x2, y2 = down_right

    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    else:
        return False


def point_on_segment(A, B, P):
    cross = (P[0] - A[0]) * (B[1] - A[1]) - (P[1] - A[1]) * (B[0] - A[0])
    if cross != 0:
        return False
    if min(A[0], B[0]) <= P[0] <= max(A[0], B[0]) and min(A[1], B[1]) <= P[1] <= max(A[1], B[1]):
        return True
    return False


def point_in_contour(contour, point):
    n = len(contour)
    for i in range(n):
        A = contour[i]
        B = contour[(i + 1) % n]
        if point_on_segment(A, B, point):
            return True

    count = 0
    for i in range(n):
        A = contour[i]
        B = contour[(i + 1) % n]
        y1, y2 = A[1], B[1]
        if y1 == y2:
            continue
        if (y1 > point[1] and y2 > point[1]) or (y1 < point[1] and y2 < point[1]):
            continue
        x = (point[1] - A[1]) * (B[0] - A[0]) / (B[1] - A[1]) + A[0]
        if x < point[0]:
            continue
        if max(y1, y2) == point[1]:
            continue
        count += 1

    return count % 2 == 1


def reward_punishment_train_one_epoch(points, trainDatasets, nowDistanceLimit, learningRate, position, distanceFloatWidth, nowTrainLenRatio):

    for i in range(len(points)):
        distanceValue = []
        for j in range(len(trainDatasets)):
            if nowTrainLenRatio >= 0.5:
                if position == 'top':
                    if trainDatasets[j][1] >= points[i][1]:
                        tempDistance = euclidean_distance(points[i], trainDatasets[j])
                        if nowDistanceLimit - distanceFloatWidth <= tempDistance <= nowDistanceLimit + distanceFloatWidth:
                            distanceValue.append(tempDistance)
                        else:
                            tempDistance = float('inf')
                            distanceValue.append(tempDistance)
                    else:
                        tempDistance = float('inf')
                        distanceValue.append(tempDistance)
                elif position == 'bottom':
                    if trainDatasets[j][1] <= points[i][1]:
                        tempDistance = euclidean_distance(points[i], trainDatasets[j])
                        if nowDistanceLimit - distanceFloatWidth <= tempDistance <= nowDistanceLimit + distanceFloatWidth:
                            distanceValue.append(tempDistance)
                        else:
                            tempDistance = float('inf')
                            distanceValue.append(tempDistance)
                    else:
                        tempDistance = float('inf')
                        distanceValue.append(tempDistance)
                elif position == 'left':
                    if trainDatasets[j][0] >= points[i][0]:
                        tempDistance = euclidean_distance(points[i], trainDatasets[j])
                        if nowDistanceLimit - distanceFloatWidth <= tempDistance <= nowDistanceLimit + distanceFloatWidth:
                            distanceValue.append(tempDistance)
                        else:
                            tempDistance = float('inf')
                            distanceValue.append(tempDistance)
                    else:
                        tempDistance = float('inf')
                        distanceValue.append(tempDistance)
                elif position == 'right':
                    if trainDatasets[j][1] <= points[i][0]:
                        tempDistance = euclidean_distance(points[i], trainDatasets[j])
                        if nowDistanceLimit - distanceFloatWidth <= tempDistance <= nowDistanceLimit + distanceFloatWidth:
                            distanceValue.append(tempDistance)
                        else:
                            tempDistance = float('inf')
                            distanceValue.append(tempDistance)
                    else:
                        tempDistance = float('inf')
                        distanceValue.append(tempDistance)


            elif nowTrainLenRatio < 0.5:
                tempDistance = euclidean_distance(points[i], trainDatasets[j])
                if nowDistanceLimit - distanceFloatWidth <= tempDistance <= nowDistanceLimit + distanceFloatWidth:
                    distanceValue.append(tempDistance)
                else:
                    tempDistance = float('inf')
                    distanceValue.append(tempDistance)



        minIndex = get_min_value(distanceValue)


        # 基于距离与方向的奖惩策略
        if minIndex != -1:
            selectPoint = trainDatasets[minIndex]

            if distanceValue[minIndex] != float('inf'):
                if nowTrainLenRatio >= 0.5:
                    if position == 'top':
                        if selectPoint[0] > points[i][0]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] += changeValue
                            points[i][1] += changeValue
                        elif selectPoint[0] < points[i][0]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] -= changeValue
                            points[i][1] += changeValue
                    elif position == 'bottom':
                        if selectPoint[0] > points[i][0]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] += changeValue
                            points[i][1] -= changeValue
                        elif selectPoint[0] < points[i][0]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] -= changeValue
                            points[i][1] -= changeValue
                    elif position == 'left':
                        if selectPoint[1] > points[i][1]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] += changeValue
                            points[i][1] += changeValue
                        elif selectPoint[1] < points[i][1]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] += changeValue
                            points[i][1] -= changeValue
                    elif position == 'right':
                        if selectPoint[1] > points[i][1]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] -= changeValue
                            points[i][1] += changeValue
                        elif selectPoint[1] < points[i][1]:
                            changeValue = learningRate * distanceValue[minIndex]
                            points[i][0] -= changeValue
                            points[i][1] -= changeValue
                elif nowTrainLenRatio < 0.5:
                    if selectPoint[0] > points[i][0] and selectPoint[1] > points[i][1]:
                        changeValue = learningRate * distanceValue[minIndex]
                        points[i][0] += changeValue
                        points[i][1] += changeValue
                    elif selectPoint[0] > points[i][0] and selectPoint[1] < points[i][1]:
                        changeValue = learningRate * distanceValue[minIndex]
                        points[i][0] += changeValue
                        points[i][1] -= changeValue
                    elif selectPoint[0] < points[i][0] and selectPoint[1] > points[i][1]:
                        changeValue = learningRate * distanceValue[minIndex]
                        points[i][0] -= changeValue
                        points[i][1] += changeValue
                    elif selectPoint[0] < points[i][0] and selectPoint[1] < points[i][1]:
                        changeValue = learningRate * distanceValue[minIndex]
                        points[i][0] -= changeValue
                        points[i][1] -= changeValue

    return points







def main():
    readJsonPath = '/home/next_lb/桌面/WYT_S_S/code/tool_program/predicted_boundary/contours_data/contours_data_3.json'

    # 读取数据
    data = load_contour_data(readJsonPath)
    height = int(data['image_info']['height'])
    width = int(data['image_info']['width'])

    # 创建原始画布和处理后的画布
    originalCanvas = np.zeros((height, width, 3), dtype=np.uint8)

    processedCanvas = np.zeros((height, width, 3), dtype=np.uint8)

    clusterCanvas = np.zeros((height, width, 3), dtype=np.uint8)

    obstacleCanvas = np.zeros((height, width, 3), dtype=np.uint8)

    boundaryCanvas = np.zeros((height, width, 3), dtype=np.uint8)


    # ================================================================= #
    allContours = []
    # 处理每个类别 - 添加进度条
    with tqdm(total=len(data['rice_row']), desc="处理轮廓数据") as pbar_main:
        for i in range(len(data['rice_row'])):
            ridge_contours = []
            # 田梗
            for j in range(len(data['ridge'][i])):
                points = np.array(data['ridge'][i][j], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                allContours.append(points)
                ridge_contours.append(points)

            if ridge_contours:
                cv2.fillPoly(originalCanvas, ridge_contours, color=(0, 0, 255))
                cv2.polylines(originalCanvas, ridge_contours, isClosed=True, color=(0, 0, 200), thickness=2)

            grave_contours = []
            # 坟头 (or 障碍)
            for j in range(len(data['grave'][i])):
                points = np.array(data['grave'][i][j], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                allContours.append(points)
                grave_contours.append(points)

            if grave_contours:
                cv2.fillPoly(originalCanvas, grave_contours, color=(255, 0, 0))
                cv2.polylines(originalCanvas, grave_contours, isClosed=True, color=(200, 0, 0), thickness=2)

            pbar_main.update(1)


    # ================================================================= #
    filterContoursOne = []
    representativePoints = []
    with tqdm(total=len(allContours), desc='基于面积的过滤') as pbar_area:
        areaLimit = 3000
        # representativePoints = []
        for i in range(len(allContours)):
            area = cv2.contourArea(allContours[i])
            if area > areaLimit:
                filterContoursOne.append(allContours[i])
                cv2.fillPoly(processedCanvas, [allContours[i]], color=(0, 165, 255))
                cv2.polylines(processedCanvas, [allContours[i]], isClosed=True, color=(0, 140, 230), thickness=2)

                contour = allContours[i]

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    representativePoints.append([cx, cy])
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    representativePoints.append([x + w // 2, y + h // 2])

            pbar_area.update(1)


    # ================================================================= #
    with tqdm(total=1, desc='基于代表点的聚类') as pbar_cluster:
        points = np.array(representativePoints)
        clustering = DBSCAN(eps=500, min_samples=1).fit(points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        pbar_cluster.update(1)

    # 为每个聚类分配一个颜色
    colors = []
    for i in range(n_clusters):
        colors.append((
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255))
        ))

    clusterContours = {}
    for clusterId in range(n_clusters):
        clusterContours[clusterId] = []

    # 将轮廓分配到对应的聚类
    for i, contour in enumerate(filterContoursOne):
        if labels[i] != -1:
            clusterId = labels[i]
            clusterContours[clusterId].append(contour)



    # ================================================================= #
    with tqdm(total=len(clusterContours), desc='可视化聚类结果并保存下来') as pbar_visual:
        for i in range(len(clusterContours)):
            # print(len(clusterContours[i]))
            # 创建空白掩码
            cluster_mask = np.zeros_like(originalCanvas[:, :, 0])

            # 在掩码上绘制所有属于该聚类的轮廓（填充）
            cv2.drawContours(cluster_mask, clusterContours[i], -1, 255, -1)

            # 对掩码进行形态学操作，确保轮廓连接
            kernel = np.ones((5, 5), np.uint8)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)

            # 重新查找该聚类区域的轮廓
            new_contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

            # 将边界绘制到边界画布上
            color = colors[i]

            # 在合并画布上绘制轮廓
            border_color = tuple(max(0, c - 30) for c in color)
            cv2.drawContours(clusterCanvas, new_contours, -1, color, -1)
            cv2.drawContours(clusterCanvas, new_contours, -1, border_color, 2)
            # cv2.imwrite(SAVE_RESULTS_BOUNDARY_PATH.replace('.png', f'_cluster_{i}.png'), clusterCanvas)

            pbar_visual.update(1)



    # ================================================================= #
    obstacleClusterContours = []
    boundaryClusterContours = []
    with tqdm(total=len(clusterContours), desc='对于边界内部障碍物体的初步确认') as pbar_obstacle:
        lenValue = []
        for i in range(len(clusterContours)):
            lenValue.append(len(clusterContours[i]))
        maxValue = max(lenValue)

        for i in range(len(clusterContours)):
            if len(clusterContours[i]) != maxValue:
                obstacleClusterContours.append(clusterContours[i])
                # 创建空白掩码
                cluster_mask = np.zeros_like(originalCanvas[:, :, 0])

                # 在掩码上绘制所有属于该聚类的轮廓（填充）
                cv2.drawContours(cluster_mask, clusterContours[i], -1, 255, -1)

                # 对掩码进行形态学操作，确保轮廓连接
                kernel = np.ones((5, 5), np.uint8)
                cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)

                # 重新查找该聚类区域的轮廓
                new_contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

                # 将边界绘制到边界画布上
                color = colors[i]

                # 在合并画布上绘制轮廓
                border_color = tuple(max(0, c - 30) for c in color)
                cv2.drawContours(obstacleCanvas, new_contours, -1, color, -1)
                cv2.drawContours(obstacleCanvas, new_contours, -1, border_color, 2)
            else:
                boundaryClusterContours.append(clusterContours[i])
            pbar_obstacle.update(1)




    # ================================================================= #
    maxEpochs = 100
    reducePixel = 500
    pointsOnSide = 50
    distanceLimit = 5000
    distanceFloatWidth = 500
    learningRate = 0.01

    with tqdm(total=maxEpochs, desc='基于奖惩策略对边界进行精准的学习') as pbar_boundary_1:
        # 汇总该聚类下所有轮廓的所有点（用于计算整个边界区域的外接矩形）
        allBoundaryPoints = []
        for single_contour in boundaryClusterContours[0]:  # 遍历聚类下的每个轮廓
            # single_contour 是单个轮廓数组，形状 (N, 1, 2)
            for point in single_contour:  # 遍历轮廓的每个点（point 形状 (1, 2)）
                # 提取点的 (x,y) 坐标（转为标量），添加到总列表
                x = int(point[0][0])  # point[0] 是 (2,) 数组，point[0][0] 是 x 标量
                y = int(point[0][1])  # point[0][1] 是 y 标量
                allBoundaryPoints.append([[x, y]])  # 保持 (N,1,2) 的嵌套格式

        allPointsNp = np.array(allBoundaryPoints, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(allPointsNp)
        upLeftPoints = [x+reducePixel, y+reducePixel]
        downRightPoints = [x+w-reducePixel, y+h-reducePixel]


        topEdgePoints, rightEdgePoints, bottomEdgePoints, leftEdgePoints = generate_rectangle_edge_points(upLeftPoints, downRightPoints, pointsOnSide)


        for i in range(len(topEdgePoints)):
            cv2.circle(boundaryCanvas, (int(topEdgePoints[i][0]), int(topEdgePoints[i][1])), 20, (0, 0, 255), -1)
        for i in range(len(rightEdgePoints)):
            cv2.circle(boundaryCanvas, (int(rightEdgePoints[i][0]), int(rightEdgePoints[i][1])), 20, (255, 0, 255), -1)
        for i in range(len(bottomEdgePoints)):
            cv2.circle(boundaryCanvas, (int(bottomEdgePoints[i][0]), int(bottomEdgePoints[i][1])), 20, (255, 255, 0), -1)
        for i in range(len(leftEdgePoints)):
            cv2.circle(boundaryCanvas, (int(leftEdgePoints[i][0]), int(leftEdgePoints[i][1])), 20, (119, 119, 119), -1)


        trainDatasets = []
        # 制作训练数据集
        for i in range(len(allBoundaryPoints)):
            tempFlag = is_point_in_rectangle(allBoundaryPoints[i][0], upLeftPoints, downRightPoints)
            if tempFlag == True:
                trainDatasets.append(allBoundaryPoints[i][0])

        for i in range(len(trainDatasets)):
            cv2.circle(boundaryCanvas, (int(trainDatasets[i][0]), int(trainDatasets[i][1])), 5, (255, 0, 0), -1)


        print(f'train datasets size {len(trainDatasets)}')
        originalTrainLen = len(trainDatasets)
        nowDistanceLimit = distanceLimit
        # 开始训练
        for epoch in range(maxEpochs):
            # 更新距离限制
            nowTrainLenRatio = len(trainDatasets) / originalTrainLen
            nowDistanceLimit = distanceLimit * nowTrainLenRatio
            print('epoch: ', epoch, 'nowTrainLenRatio: ', nowTrainLenRatio, 'nowDistanceLimit: ', nowDistanceLimit)


            position = 'top'
            topEdgePoints = reward_punishment_train_one_epoch(topEdgePoints, trainDatasets, nowDistanceLimit, learningRate, position, distanceFloatWidth, nowTrainLenRatio)
            position = 'bottom'
            bottomEdgePoints = reward_punishment_train_one_epoch(bottomEdgePoints, trainDatasets, nowDistanceLimit, learningRate, position, distanceFloatWidth, nowTrainLenRatio)
            position = 'left'
            leftEdgePoints = reward_punishment_train_one_epoch(leftEdgePoints, trainDatasets, nowDistanceLimit, learningRate, position, distanceFloatWidth, nowTrainLenRatio)
            position = 'right'
            rightEdgePoints = reward_punishment_train_one_epoch(rightEdgePoints, trainDatasets, nowDistanceLimit, learningRate, position, distanceFloatWidth, nowTrainLenRatio)

            edgeContours = []
            for i in range(len(topEdgePoints)):
                edgeContours.append(topEdgePoints[i])
            for i in range(len(rightEdgePoints)):
                edgeContours.append(rightEdgePoints[i])
            for i in range(len(bottomEdgePoints)):
                edgeContours.append(bottomEdgePoints[i])
            for i in range(len(leftEdgePoints)):
                edgeContours.append(leftEdgePoints[i])

            deleteIndex = []
            for i in range(len(trainDatasets)):
                tempFlag = point_in_contour(edgeContours, trainDatasets[i])
                if tempFlag == False:
                    deleteIndex.append(i)


            print('before: ', len(deleteIndex), len(trainDatasets))
            for i in range(len(deleteIndex)):
                del trainDatasets[deleteIndex[i]-i]
            print('after: ', len(deleteIndex), len(trainDatasets))


            pbar_boundary_1.update(1)



        for i in range(len(topEdgePoints)):
            cv2.circle(boundaryCanvas, (int(topEdgePoints[i][0]), int(topEdgePoints[i][1])), 20, (0, 140, 230), -1)
        for i in range(len(bottomEdgePoints)):
            cv2.circle(boundaryCanvas, (int(bottomEdgePoints[i][0]), int(bottomEdgePoints[i][1])), 20, (0, 140, 230), -1)
        for i in range(len(leftEdgePoints)):
            cv2.circle(boundaryCanvas, (int(leftEdgePoints[i][0]), int(leftEdgePoints[i][1])), 20, (0, 140, 230), -1)
        for i in range(len(rightEdgePoints)):
            cv2.circle(boundaryCanvas, (int(rightEdgePoints[i][0]), int(rightEdgePoints[i][1])), 20, (0, 140, 230), -1)





    # ================================================================= #
    # 想一下对于topEdgePoints、bottomEdgePoints、leftEdgePoints、rightEdgePoints 如何使用 allBoundaryPoints  对它做进一步的处理





    # ================================================================= #
    # 保存结果 - 添加进度条
    with tqdm(total=5, desc="保存结果") as pbar_save:
        cv2.imwrite(SAVE_RESULTS_BOUNDARY_PATH.replace('.png', '_original.png'), originalCanvas)
        pbar_save.update(1)
        cv2.imwrite(SAVE_RESULTS_BOUNDARY_PATH.replace('.png', '_processed.png'), processedCanvas)
        pbar_save.update(1)
        cv2.imwrite(SAVE_RESULTS_BOUNDARY_PATH.replace('.png', '_cluster.png'), clusterCanvas)
        pbar_save.update(1)
        cv2.imwrite(SAVE_RESULTS_BOUNDARY_PATH.replace('.png', '_obstacle.png'), obstacleCanvas)
        pbar_save.update(1)
        cv2.imwrite(SAVE_RESULTS_BOUNDARY_PATH.replace('.png', '_boundary.png'), boundaryCanvas)
        pbar_save.update(1)





if __name__ == '__main__':
    main()




