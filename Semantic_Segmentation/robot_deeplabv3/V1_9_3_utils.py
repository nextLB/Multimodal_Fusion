
import math
import numpy as np
import cv2
import copy
import config_parameters


# 根据距离计算置信度的函数
def stable_sigmoid(x):
    a = 0.1
    b = 5
    if x >= 0:
        sig = 1 / (1 + np.exp(a * x - b))
        return sig
    else:
        sig = 1 / (1 + np.exp(a * (-x) - b))
        return sig


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


# The minimum value is selected and its subscript in the original array is returned
def get_min_value(Array):
    length = len(Array)
    if length == 0:
        # print('It is not valid for the array length to be empty.')
        return -1
    elif length == 1:
        return 0
    else:
        min_value = float('inf')
        min_index = -1
        for i in range(length):
            if Array[i] <= min_value:
                min_value = Array[i]
                min_index = i
        return min_index


def get_max_value(Array):
    length = len(Array)
    if length == 0:
        # print('It is not valid for the array length to be empty.')
        return -1
    elif length == 1:
        return 0
    else:
        max_value = float('inf')
        max_index = -1
        for i in range(length):
            if Array[i] >= max_value:
                max_value = Array[i]
                max_index = i
        return max_index


# Calculates the coordinates of the endpoint of the line
def calculate_endpoints_line(k, b):
    pt1X = int(b)
    pt1Y = 0
    pt2X = int(280 * k + b)
    pt2Y = 280

    endPoints = ((pt1X, pt1Y), (pt2X, pt2Y))

    return endPoints


# 判断这条线是否有压着这个识别框
def is_line_crossing_rect(line, rect):
    # 提取矩形的边界坐标
    x_min = min(rect[0][0], rect[1][0])
    x_max = max(rect[0][0], rect[1][0])
    y_min = min(rect[0][1], rect[1][1])
    y_max = max(rect[0][1], rect[1][1])

    # 提取线段的两个端点
    x1, y1 = line[0]
    x2, y2 = line[1]

    # 检查端点是否在矩形内（包括边界）
    if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
        return True

    # 检查线段是否与矩形的四条边相交
    # 上边 (y_min)
    if (y1 < y_min and y2 > y_min) or (y1 > y_min and y2 < y_min):
        t = (y_min - y1) / (y2 - y1)
        x_intersect = x1 + t * (x2 - x1)
        if x_min <= x_intersect <= x_max:
            return True

    # 下边 (y_max)
    if (y1 < y_max and y2 > y_max) or (y1 > y_max and y2 < y_max):
        t = (y_max - y1) / (y2 - y1)
        x_intersect = x1 + t * (x2 - x1)
        if x_min <= x_intersect <= x_max:
            return True

    # 左边 (x_min)
    if (x1 < x_min and x2 > x_min) or (x1 > x_min and x2 < x_min):
        t = (x_min - x1) / (x2 - x1)
        y_intersect = y1 + t * (y2 - y1)
        if y_min <= y_intersect <= y_max:
            return True

    # 右边 (x_max)
    if (x1 < x_max and x2 > x_max) or (x1 > x_max and x2 < x_max):
        t = (x_max - x1) / (x2 - x1)
        y_intersect = y1 + t * (y2 - y1)
        if y_min <= y_intersect <= y_max:
            return True

    return False


def fit(points):
    """
        使用最小二乘法拟合直线，基于三个点坐标计算斜率(k)和截距(b)
        保持原有的斜率计算方式：k = (x2 - x1)/(y2 - y1)

        参数:
        points -- 包含三个点坐标的列表，格式: [[x1, y1], [x2, y2], [x3, y3]]

        返回:
        [k, b] -- 直线的斜率和截距
        None -- 如果无法计算斜率（所有y值相同）
    """
    # 将点转换为numpy数组便于计算
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # 检查y值是否全部相同
    if np.all(y == y[0]):
        return None

    # 使用最小二乘法拟合直线：x = k*y + b
    # 等同于拟合模型：x = k*y + b
    A = np.vstack([y, np.ones(len(y))]).T
    k, b = np.linalg.lstsq(A, x, rcond=None)[0]

    return [float(k), float(b)]


# Calculate the slope of the line with the b-value
def get_k_b(pt1, pt2):
    # Judge the legitimacy
    if len(pt1) != 2:
        # print('The number of coordinates of the first point is invalid.')
        return
    if len(pt2) != 2:
        # print('The number of coordinates of the second point is invalid.')
        return
    if pt1[config_parameters.yIndex] == pt2[config_parameters.yIndex]:
        return None
    else:
        k = (pt2[config_parameters.xIndex] - pt1[config_parameters.xIndex]) / (pt2[config_parameters.yIndex] - pt1[config_parameters.yIndex])
        b = pt1[config_parameters.xIndex] - k * pt1[config_parameters.yIndex]
        KB = [k, b]
        return KB


def predict_middle_line(leftLineK, leftLineB, rightLineK, rightLineB):
    # get index information

    endPoints1 = calculate_endpoints_line(leftLineK, leftLineB)
    endPoints2 = calculate_endpoints_line(rightLineK, rightLineB)
    centerPoints = [[0, 0], [0, 0]]
    centerPoints[0][0] = int((endPoints1[0][0] + endPoints2[0][0]) / 2)
    centerPoints[0][1] = 0
    centerPoints[1][0] = int((endPoints1[1][0] + endPoints2[1][0]) / 2)
    centerPoints[1][1] = 479
    KB = copy.deepcopy(get_k_b(centerPoints[0], centerPoints[1]))

    return KB


def calculate_filter_box_and_line_distance(endPoints, filterBox):
    """
    计算滤波框到直线（由 endPoints 两个端点定义）的垂直距离

    参数:
    endPoints -- 二维列表，包含直线的两个端点坐标，例如 [[x1, y1], [x2, y2]]
    clusterData -- 一维列表，包含点的坐标，例如 [x0, y0]

    返回:
    点到直线的垂直距离（浮点数）
    """
    # 解包端点坐标
    x1, y1 = endPoints[0]
    x2, y2 = endPoints[1]

    # 解包目标点坐标
    x0, y0 = filterBox

    # 1. 检查端点是否重合（视为点）
    if abs(x1 - x2) < 1e-10 and abs(y1 - y2) < 1e-10:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    # 2. 计算直线的 k 和 b
    KB = get_k_b(endPoints[0], endPoints[1])

    # 3. 处理水平线（垂直线）情况
    if KB is None:  # 表示直线是水平线（y恒定）
        return abs(y0 - y1)

    # 4. 提取斜率和截距
    k, b = KB

    # 5. 将直线方程转换为一般式：x - ky - b = 0
    #    即：A = 1, B = -k, C = -b
    A = 1
    B = -k
    C = -b

    # 6. 计算点到直线的距离公式
    numerator = abs(A * x0 + B * y0 + C)
    denominator = math.sqrt(A ** 2 + B ** 2)

    return numerator / denominator


# 计算方框顶点坐标的函数
def calculate_low_middle_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.lowMiddleLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.lowMiddleLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.lowMiddleLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.lowMiddleLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_low_middle(nowBoxes, lowMiddleFilterBoxX, lowMiddleFilterBoxY):
    # get index information
    lowIndex = config_parameters.lowIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[lowIndex])):
        distance.append(euclidean_distance(nowBoxes[lowIndex][i], [lowMiddleFilterBoxX, lowMiddleFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.lowMiddleInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_low_middle(nowBoxes, middleLineK, middleLineB):
    # get index information
    lowIndex = config_parameters.lowIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(middleLineK, middleLineB)
    for i in range(len(nowBoxes[lowIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[lowIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_low_middle(nowBoxes, lowMiddleFilterBoxX, lowMiddleFilterBoxY):
    # get index information
    lowIndex = config_parameters.lowIndex

    distance = []
    for i in range(len(nowBoxes[lowIndex])):
        distance.append(euclidean_distance(nowBoxes[lowIndex][i], [lowMiddleFilterBoxX, lowMiddleFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.lowMiddleFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_low_middle(nowBoxes, middleLineK, middleLineB):
    # get index information
    lowIndex = config_parameters.lowIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(middleLineK, middleLineB)
    for i in range(len(nowBoxes[lowIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[lowIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_middle_middle_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.middleMiddleLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.middleMiddleLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.middleMiddleLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.middleMiddleLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_middle_middle(nowBoxes, middleMiddleFilterBoxX, middleMiddleFilterBoxY):
    # get index information
    middleIndex = config_parameters.middleIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[middleIndex])):
        distance.append(euclidean_distance(nowBoxes[middleIndex][i], [middleMiddleFilterBoxX, middleMiddleFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.middleMiddleInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_middle_middle(nowBoxes, middleLineK, middleLineB):
    # get index information
    middleIndex = config_parameters.middleIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(middleLineK, middleLineB)
    for i in range(len(nowBoxes[middleIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[middleIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_middle_middle(nowBoxes, middleMiddleFilterBoxX, middleMiddleFilterBoxY):
    # get index information
    middleIndex = config_parameters.middleIndex

    distance = []
    for i in range(len(nowBoxes[middleIndex])):
        distance.append(euclidean_distance(nowBoxes[middleIndex][i], [middleMiddleFilterBoxX, middleMiddleFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.middleMiddleFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_middle_middle(nowBoxes, middleLineK, middleLineB):
    # get index information
    middleIndex = config_parameters.middleIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(middleLineK, middleLineB)
    for i in range(len(nowBoxes[middleIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[middleIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_upper_middle_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.upperMiddleLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.upperMiddleLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.upperMiddleLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.upperMiddleLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_upper_middle(nowBoxes, upperMiddleFilterBoxX, upperMiddleFilterBoxY):
    # get index information
    upperIndex = config_parameters.upperIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[upperIndex])):
        distance.append(euclidean_distance(nowBoxes[upperIndex][i], [upperMiddleFilterBoxX, upperMiddleFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.upperMiddleInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_upper_middle(nowBoxes, middleLineK, middleLineB):
    # get index information
    upperIndex = config_parameters.upperIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(middleLineK, middleLineB)
    for i in range(len(nowBoxes[upperIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[upperIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_upper_middle(nowBoxes, upperMiddleFilterBoxX, upperMiddleFilterBoxY):
    # get index information
    upperIndex = config_parameters.upperIndex

    distance = []
    for i in range(len(nowBoxes[upperIndex])):
        distance.append(euclidean_distance(nowBoxes[upperIndex][i], [upperMiddleFilterBoxX, upperMiddleFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.upperMiddleFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_upper_middle(nowBoxes, middleLineK, middleLineB):
    # get index information
    upperIndex = config_parameters.upperIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(middleLineK, middleLineB)
    for i in range(len(nowBoxes[upperIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[upperIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_low_left_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.lowLeftLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.lowLeftLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.lowLeftLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.lowLeftLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_low_left(nowBoxes, lowLeftFilterBoxX, lowLeftFilterBoxY):
    # get index information
    lowIndex = config_parameters.lowIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[lowIndex])):
        distance.append(euclidean_distance(nowBoxes[lowIndex][i], [lowLeftFilterBoxX, lowLeftFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.lowLeftInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_low_left(nowBoxes, leftLineK, leftLineB):
    # get index information
    lowIndex = config_parameters.lowIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(leftLineK, leftLineB)
    for i in range(len(nowBoxes[lowIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[lowIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_low_left(nowBoxes, lowLeftFilterBoxX, lowLeftFilterBoxY):
    # get index information
    lowIndex = config_parameters.lowIndex

    distance = []
    for i in range(len(nowBoxes[lowIndex])):
        distance.append(euclidean_distance(nowBoxes[lowIndex][i], [lowLeftFilterBoxX, lowLeftFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.lowLeftFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_low_left(nowBoxes, leftLineK, leftLineB):
    # get index information
    lowIndex = config_parameters.lowIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(leftLineK, leftLineB)
    for i in range(len(nowBoxes[lowIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[lowIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_middle_left_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.middleLeftLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.middleLeftLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.middleLeftLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.middleLeftLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_middle_left(nowBoxes, middleLeftFilterBoxX, middleLeftFilterBoxY):
    # get index information
    middleIndex = config_parameters.middleIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[middleIndex])):
        distance.append(euclidean_distance(nowBoxes[middleIndex][i], [middleLeftFilterBoxX, middleLeftFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.middleLeftInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_middle_left(nowBoxes, leftLineK, leftLineB):
    # get index information
    middleIndex = config_parameters.middleIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(leftLineK, leftLineB)
    for i in range(len(nowBoxes[middleIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[middleIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_middle_left(nowBoxes, middleLeftFilterBoxX, middleLeftFilterBoxY):
    # get index information
    middleIndex = config_parameters.middleIndex

    distance = []
    for i in range(len(nowBoxes[middleIndex])):
        distance.append(euclidean_distance(nowBoxes[middleIndex][i], [middleLeftFilterBoxX, middleLeftFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.middleLeftFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_middle_left(nowBoxes, leftLineK, leftLineB):
    # get index information
    middleIndex = config_parameters.middleIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(leftLineK, leftLineB)
    for i in range(len(nowBoxes[middleIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[middleIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_upper_left_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.upperLeftLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.upperLeftLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.upperLeftLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.upperLeftLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_upper_left(nowBoxes, upperLeftFilterBoxX, upperLeftFilterBoxY):
    # get index information
    upperIndex = config_parameters.upperIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[upperIndex])):
        distance.append(euclidean_distance(nowBoxes[upperIndex][i], [upperLeftFilterBoxX, upperLeftFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.upperLeftInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_upper_left(nowBoxes, leftLineK, leftLineB):
    # get index information
    upperIndex = config_parameters.upperIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(leftLineK, leftLineB)
    for i in range(len(nowBoxes[upperIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[upperIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_upper_left(nowBoxes, upperLeftFilterBoxX, upperLeftFilterBoxY):
    # get index information
    upperIndex = config_parameters.upperIndex

    distance = []
    for i in range(len(nowBoxes[upperIndex])):
        distance.append(euclidean_distance(nowBoxes[upperIndex][i], [upperLeftFilterBoxX, upperLeftFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.upperLeftFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_upper_left(nowBoxes, leftLineK, leftLineB):
    # get index information
    upperIndex = config_parameters.upperIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(leftLineK, leftLineB)
    for i in range(len(nowBoxes[upperIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[upperIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_low_right_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.lowRightLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.lowRightLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.lowRightLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.lowRightLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_low_right(nowBoxes, lowRightFilterBoxX, lowRightFilterBoxY):
    # get index information
    lowIndex = config_parameters.lowIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[lowIndex])):
        distance.append(euclidean_distance(nowBoxes[lowIndex][i], [lowRightFilterBoxX, lowRightFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.lowRightInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_low_right(nowBoxes, rightLineK, rightLineB):
    # get index information
    lowIndex = config_parameters.lowIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(rightLineK, rightLineB)
    for i in range(len(nowBoxes[lowIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[lowIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_low_right(nowBoxes, lowRightFilterBoxX, lowRightFilterBoxY):
    # get index information
    lowIndex = config_parameters.lowIndex

    distance = []
    for i in range(len(nowBoxes[lowIndex])):
        distance.append(euclidean_distance(nowBoxes[lowIndex][i], [lowRightFilterBoxX, lowRightFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.lowRightFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_low_right(nowBoxes, rightLineK, rightLineB):
    # get index information
    lowIndex = config_parameters.lowIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(rightLineK, rightLineB)
    for i in range(len(nowBoxes[lowIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[lowIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_middle_right_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.middleRightLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.middleRightLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.middleRightLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.middleRightLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_middle_right(nowBoxes, middleRightFilterBoxX, middleRightFilterBoxY):
    # get index information
    middleIndex = config_parameters.middleIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[middleIndex])):
        distance.append(euclidean_distance(nowBoxes[middleIndex][i], [middleRightFilterBoxX, middleRightFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.middleRightInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_middle_right(nowBoxes, rightLineK, rightLineB):
    # get index information
    middleIndex = config_parameters.middleIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(rightLineK, rightLineB)
    for i in range(len(nowBoxes[middleIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[middleIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_middle_right(nowBoxes, middleRightFilterBoxX, middleRightFilterBoxY):
    # get index information
    middleIndex = config_parameters.middleIndex

    distance = []
    for i in range(len(nowBoxes[middleIndex])):
        distance.append(euclidean_distance(nowBoxes[middleIndex][i], [middleRightFilterBoxX, middleRightFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.middleRightFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_middle_right(nowBoxes, rightLineK, rightLineB):
    # get index information
    middleIndex = config_parameters.middleIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(rightLineK, rightLineB)
    for i in range(len(nowBoxes[middleIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[middleIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 计算方框顶点坐标的函数
def calculate_upper_right_vertex_coordinates(centerPoints):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.upperRightLinePressureFrame / 2))
    y1 = int(y_center - int(config_parameters.upperRightLinePressureFrame / 2))
    x2 = int(x_center + int(config_parameters.upperRightLinePressureFrame / 2))
    y2 = int(y_center + int(config_parameters.upperRightLinePressureFrame / 2))

    return [[x1, y1], [x2, y2]]


def initial_distance_upper_right(nowBoxes, upperRightFilterBoxX, upperRightFilterBoxY):
    # get index information
    upperIndex = config_parameters.upperIndex

    distance = []
    # 计算距离
    for i in range(len(nowBoxes[upperIndex])):
        distance.append(euclidean_distance(nowBoxes[upperIndex][i], [upperRightFilterBoxX, upperRightFilterBoxY]))
    # 选择最接近当前聚类的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.upperRightInitialSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def initial_line_upper_right(nowBoxes, rightLineK, rightLineB):
    # get index information
    upperIndex = config_parameters.upperIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(rightLineK, rightLineB)
    for i in range(len(nowBoxes[upperIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[upperIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# 滤波时按照距离查找底部中间识别框
def filter_distance_upper_right(nowBoxes, upperRightFilterBoxX, upperRightFilterBoxY):
    # get index information
    upperIndex = config_parameters.upperIndex

    distance = []
    for i in range(len(nowBoxes[upperIndex])):
        distance.append(euclidean_distance(nowBoxes[upperIndex][i], [upperRightFilterBoxX, upperRightFilterBoxY]))
    # 选择最近的那个目标框
    minIndex = get_min_value(distance)
    if minIndex != -1:
        if distance[minIndex] <= config_parameters.upperRightFilterSize:
            return minIndex
        else:
            return -1
    else:
        return -1


def filter_line_upper_right(nowBoxes, rightLineK, rightLineB):
    # get index information
    upperIndex = config_parameters.upperIndex

    # 选取中间跟踪线压着的底层识别框
    tempIndex = []
    endPoints = calculate_endpoints_line(rightLineK, rightLineB)
    for i in range(len(nowBoxes[upperIndex])):
        boxVertex = calculate_low_middle_vertex_coordinates(nowBoxes[upperIndex][i])
        tempFlag = is_line_crossing_rect(endPoints, boxVertex)
        if tempFlag == True:
            tempIndex.append(i)

    if len(tempIndex) == 1:
        return tempIndex[0]
    else:
        return -1


# A dashed line is drawn by changing the thickness of the line segment according to the number of times it disappears
def cnt_dotted_line(start_point, end_point, img, count, color):
    # Define the start and end points of the dashed line
    thickness = 20 - count * config_parameters.scaleFactor  # Line width

    # Draw a dotted line
    num_segments = 3  # Divide the line into 10 segments
    for i in range(num_segments):
        # Calculate the start and end points of each segment
        segment_start = (
            int(start_point[0] + (end_point[0] - start_point[0]) * (i / num_segments)),
            int(start_point[1] + (end_point[1] - start_point[1]) * (i / num_segments))
        )
        segment_end = (
            int(start_point[0] + (end_point[0] - start_point[0]) * ((i + 0.5) / num_segments)),
            int(start_point[1] + (end_point[1] - start_point[1]) * ((i + 0.5) / num_segments))
        )
        # Draw each segment
        cv2.line(img, segment_start, segment_end, color, thickness)

    return img


# Draw a cluster box with a solid line
def my_rectangle(img, centerPoints, color, thickness):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.longEdgeLength / 2))
    y1 = int(y_center - int(config_parameters.shortEdgeLength / 2))
    x2 = int(x_center + int(config_parameters.longEdgeLength / 2))
    y2 = int(y_center + int(config_parameters.shortEdgeLength / 2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    return img


# Draw a dashed line to get the clustered box
def my_dotted_rectangle(img, centerPoints, color):
    # Gets the coordinates of the center point of the clustered box
    x_center = centerPoints[0]
    y_center = centerPoints[1]

    x1 = int(x_center - int(config_parameters.longEdgeLength / 2))
    y1 = int(y_center - int(config_parameters.shortEdgeLength / 2))
    x2 = int(x_center + int(config_parameters.longEdgeLength / 2))
    y2 = int(y_center - int(config_parameters.shortEdgeLength / 2))
    img = cnt_dotted_line((x1, y1), (x2, y2), img, centerPoints[2], color)

    x1 = x1
    y1 = y1 + config_parameters.shortEdgeLength
    x2 = x2
    y2 = y2 + config_parameters.shortEdgeLength
    img = cnt_dotted_line((x1, y1), (x2, y2), img, centerPoints[2], color)

    x1 = x1
    y1 = y1 - config_parameters.shortEdgeLength
    x2 = x1
    y2 = y2
    img = cnt_dotted_line((x1, y1), (x2, y2), img, centerPoints[2], color)

    x1 = x1 + config_parameters.longEdgeLength
    y1 = y1
    x2 = x1
    y2 = y2
    img = cnt_dotted_line((x1, y1), (x2, y2), img, centerPoints[2], color)

    return img


def line_cnt_dotted_line(start_point, end_point, img, count, color):
    # Define the start and end points of the dashed line
    thickness = 20 - count * config_parameters.scaleFactor  # Line width

    # Draw a dotted line
    num_segments = 10  # Divide the line into 10 segments
    for i in range(num_segments):
        # Calculate the start and end points of each segment
        segment_start = (
            int(start_point[0] + (end_point[0] - start_point[0]) * (i / num_segments)),
            int(start_point[1] + (end_point[1] - start_point[1]) * (i / num_segments))
        )
        segment_end = (
            int(start_point[0] + (end_point[0] - start_point[0]) * ((i + 0.5) / num_segments)),
            int(start_point[1] + (end_point[1] - start_point[1]) * ((i + 0.5) / num_segments))
        )
        # Draw each segment
        cv2.line(img, segment_start, segment_end, color, thickness)

    return img
