
import numpy as np
import copy
import math
import cv2
import os
import csv
import json
import config_parameters

import V1_9_3_utils


class KalmanFilter:
    def __init__(self):
        # 状态向量: [x, y, vx, vy, x, y, z, xa, ] (位置和速度)
        self.state = np.zeros(4)

        # 状态协方差矩阵 (初始不确定性)
        self.P = np.eye(4) * 1000

        # 状态转移矩阵 (假设匀速运动模型)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 观测矩阵 (只能观测到位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 过程噪声协方差 (调整模型不确定性)
        self.Q = np.eye(4) * 0.1

        # 观测噪声协方差 (调整传感器噪声)
        self.R = np.eye(2) * 1

        # 标记是否已初始化
        self.initialized = False

    def predict(self):
        # 预测状态
        self.state = self.F @ self.state
        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # 返回预测的位置(x, y)

    def update(self, measurement):
        # 计算残差 = 测量值 - 预测位置
        y = measurement - self.H @ self.state
        # 残差协方差
        S = self.H @ self.P @ self.H.T + self.R
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # 更新状态估计
        self.state = self.state + K @ y
        # 更新协方差估计
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2]  # 返回更新后的位置(x, y)

    def process(self, state_list, obs_list):
        """
        处理一次数据
        :param state_list: 当前状态值 [x, y]
        :param obs_list: 当前观测值 [x, y]
        :return: 预测位置和更新后的位置（都是[x,y]）
        """
        # 如果是第一次运行，使用传入的状态值初始化位置，速度初始为0
        if not self.initialized:
            self.state = np.array([state_list[0], state_list[1], 0, 0])
            self.initialized = True

        # 预测步骤
        predicted = self.predict()

        # 更新步骤 (使用观测值)
        updated = self.update(np.array(obs_list))

        return predicted, updated



class V193_Mower_Track:
    def __init__(self):
        # get index information
        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        fBF = config_parameters.filterBoxFlag
        fBCNT = config_parameters.filterBoxCnt
        fBCF = config_parameters.filterBoxConfidence
        sIF = config_parameters.successInitialFlag
        sUF = config_parameters.successUpdateFlag
        # ------------------- nowBoxes用于存储三层识别框的信息 ------------------------#
        self.nowBoxes = []
        # ------------------- 九个滤波框的信息设定 ------------------------------------#
        '''底层中间的滤波框'''
        self.lowMiddle = np.zeros(7, dtype=np.float32)
        self.lowMiddle[xI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[xI])
        self.lowMiddle[yI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[yI])
        self.lowMiddleKalmanFilter = KalmanFilter()

        '''中层中间的滤波框'''
        self.middleMiddle = np.zeros(7, dtype=np.float32)
        self.middleMiddle[xI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[xI])
        self.middleMiddle[yI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[yI])
        self.middleMiddleKalmanFilter = KalmanFilter()

        '''顶层中间的滤波框'''
        self.upperMiddle = np.zeros(7, dtype=np.float32)
        self.upperMiddle[xI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[xI])
        self.upperMiddle[yI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[yI])
        self.upperMiddleKalmanFilter = KalmanFilter()

        '''底层左侧的滤波框'''
        self.lowLeft = np.zeros(7, dtype=np.float32)
        self.lowLeft[xI] = copy.deepcopy(config_parameters.lowLeftFilterBox[xI])
        self.lowLeft[yI] = copy.deepcopy(config_parameters.lowLeftFilterBox[yI])
        self.lowLeftKalmanFilter = KalmanFilter()

        '''底层右侧的滤波框'''
        self.lowRight = np.zeros(7, dtype=np.float32)
        self.lowRight[xI] = copy.deepcopy(config_parameters.lowRightFilterBox[xI])
        self.lowRight[yI] = copy.deepcopy(config_parameters.lowRightFilterBox[yI])
        self.lowRightKalmanFilter = KalmanFilter()

        '''中层左侧的滤波框'''
        self.middleLeft = np.zeros(7, dtype=np.float32)
        self.middleLeft[xI] = copy.deepcopy(config_parameters.middleLeftFilterBox[xI])
        self.middleLeft[yI] = copy.deepcopy(config_parameters.middleLeftFilterBox[yI])
        self.middleLeftKalmanFilter = KalmanFilter()

        '''中层右侧的滤波框'''
        self.middleRight = np.zeros(7, dtype=np.float32)
        self.middleRight[xI] = copy.deepcopy(config_parameters.middleRightFilterBox[xI])
        self.middleRight[yI] = copy.deepcopy(config_parameters.middleRightFilterBox[yI])
        self.middleRightKalmanFilter = KalmanFilter()

        '''顶层左侧的滤波框'''
        self.upperLeft = np.zeros(7, dtype=np.float32)
        self.upperLeft[xI] = copy.deepcopy(config_parameters.upperLeftFilterBox[xI])
        self.upperLeft[yI] = copy.deepcopy(config_parameters.upperLeftFilterBox[yI])
        self.upperLeftKalmanFilter = KalmanFilter()

        '''顶层右侧的滤波框'''
        self.upperRight = np.zeros(7, dtype=np.float32)
        self.upperRight[xI] = copy.deepcopy(config_parameters.upperRightFilterBox[xI])
        self.upperRight[yI] = copy.deepcopy(config_parameters.upperRightFilterBox[yI])
        self.upperRightKalmanFilter = KalmanFilter()

        '''左侧跟踪线的信息设定'''
        self.leftLine =np.zeros(4, dtype=np.float32)

        '''中间跟踪线的信息设定'''
        self.middleLine = np.zeros(4, dtype=np.float32)

        '''右侧跟踪线的信息设定'''
        self.rightLine = np.zeros(4, dtype=np.float32)

        '''中间预测线的信息设定'''
        self.predictLine = np.zeros(4, dtype=np.float32)


        # -------------------- decisionPoint用于存储最终的决策点信息 ------------------#
        self.decisionPoint = 0

        # ------------------- COUNTER用于绝对阈值的清零操作 --------------------------#
        self.COUNTER = 0



    # 进行分层
    def layering(self, l_dect_list):
        lowCenterPoints = []
        middleCenterPoints = []
        uppCenterPoints = []

        results = []
        # Legitimacy judgment
        if l_dect_list == []:
            results.append(lowCenterPoints)
            results.append(middleCenterPoints)
            results.append(uppCenterPoints)
        else:
            lowCenterPoints = copy.deepcopy(l_dect_list[2])
            middleCenterPoints = copy.deepcopy(l_dect_list[1])
            uppCenterPoints = copy.deepcopy(l_dect_list[0])
            results.append(lowCenterPoints)
            results.append(middleCenterPoints)
            results.append(uppCenterPoints)
        self.nowBoxes = copy.deepcopy(results)


    # 在处理新一轮的识别框之前，尽可能保证同一层的聚类框和聚类框之间不会具有识别框
    def boxes_in_between_filter_box(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex



        # 左上和中上
        if self.upperLeft[filterBoxFlagI] != 0 and self.upperMiddle[filterBoxFlagI] != 0:
            for i in range(len(self.nowBoxes[upperIndex])):
                if self.upperLeft[xI] + config_parameters.longEdgeLength <= self.nowBoxes[upperIndex][i][xI] <= self.upperMiddle[xI] - config_parameters.longEdgeLength:
                    self.upperLeft[xI] = copy.deepcopy(self.nowBoxes[upperIndex][i][xI])
                    self.upperLeft[yI] = copy.deepcopy(self.nowBoxes[upperIndex][i][yI])
                    break

        # 左中和中中
        if self.middleLeft[filterBoxFlagI] != 0 and self.middleMiddle[filterBoxFlagI] != 0:
            for i in range(len(self.nowBoxes[middleIndex])):
                if self.middleLeft[xI] + config_parameters.longEdgeLength <= self.nowBoxes[middleIndex][i][xI] <= self.middleMiddle[xI] - config_parameters.longEdgeLength:
                    self.middleLeft[xI] = copy.deepcopy(self.nowBoxes[middleIndex][i][xI])
                    self.middleLeft[yI] = copy.deepcopy(self.nowBoxes[middleIndex][i][yI])
                    break

        # 左下和中下
        if self.lowLeft[filterBoxFlagI] != 0 and self.lowMiddle[filterBoxFlagI] != 0:
            for i in range(len(self.nowBoxes[lowIndex])):
                if self.lowLeft[xI] + config_parameters.longEdgeLength <= self.nowBoxes[lowIndex][i][xI] <= self.lowMiddle[xI] - config_parameters.longEdgeLength:
                    self.lowLeft[xI] = copy.deepcopy(self.nowBoxes[lowIndex][i][xI])
                    self.lowLeft[yI] = copy.deepcopy(self.nowBoxes[lowIndex][i][yI])
                    break


        # 右上和中上
        if self.upperRight[filterBoxFlagI] != 0 and self.upperMiddle[filterBoxFlagI] != 0:
            for i in range(len(self.nowBoxes[upperIndex])):
                if self.upperMiddle[xI] + config_parameters.longEdgeLength <= self.nowBoxes[upperIndex][i][xI] <= self.upperRight[xI] - config_parameters.longEdgeLength:
                    self.upperRight[xI] = copy.deepcopy(self.nowBoxes[upperIndex][i][xI])
                    self.upperRight[yI] = copy.deepcopy(self.nowBoxes[upperIndex][i][yI])
                    break

        # 右中和中中
        if self.middleRight[filterBoxFlagI] != 0 and self.middleMiddle[filterBoxFlagI] != 0:
            for i in range(len(self.nowBoxes[middleIndex])):
                if self.middleMiddle[xI] + config_parameters.longEdgeLength <= self.nowBoxes[middleIndex][i][xI] <= self.middleRight[xI] - config_parameters.longEdgeLength:
                    self.middleRight[xI] = copy.deepcopy(self.nowBoxes[middleIndex][i][xI])
                    self.middleRight[yI] = copy.deepcopy(self.nowBoxes[middleIndex][i][yI])
                    break


        # 右下和中下
        if self.lowRight[filterBoxFlagI] != 0 and self.lowMiddle[filterBoxFlagI] != 0:
            for i in range(len(self.nowBoxes[lowIndex])):
                if self.lowMiddle[xI] + config_parameters.longEdgeLength <= self.nowBoxes[lowIndex][i][xI] <= self.lowRight[xI] - config_parameters.longEdgeLength:
                    self.lowRight[xI] = copy.deepcopy(self.nowBoxes[lowIndex][i][xI])
                    self.lowRight[yI] = copy.deepcopy(self.nowBoxes[lowIndex][i][yI])
                    break


    def calculate_low_middle(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看一下底层中间的滤波框是否初始化成功了
        if self.lowMiddle[filterBoxFlagI] == 0:
            self.lowMiddle[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_low_middle(self.nowBoxes, self.lowMiddle[xI], self.lowMiddle[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.middleLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_low_middle(self.nowBoxes, self.middleLine[kI], self.middleLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                self.lowMiddle[xI] = box[xI]
                self.lowMiddle[yI] = box[yI]
                self.lowMiddle[filterBoxFlagI] = 1
                self.lowMiddle[filterBoxCntI] = 0
                self.lowMiddle[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[lowIndex][distanceIndex]
                self.lowMiddle[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                self.lowMiddle[xI] = box[xI]
                self.lowMiddle[yI] = box[yI]
                self.lowMiddle[filterBoxFlagI] = 1
                self.lowMiddle[filterBoxCntI] = 0
                self.lowMiddle[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[lowIndex][lineIndex]
                self.lowMiddle[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    self.lowMiddle[xI] = box[xI]
                    self.lowMiddle[yI] = box[yI]
                    self.lowMiddle[filterBoxFlagI] = 1
                    self.lowMiddle[filterBoxCntI] = 0
                    self.lowMiddle[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[lowIndex][distanceIndex]
                    self.lowMiddle[successInitialFlagI] = 1
                else:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    self.lowMiddle[xI] = box[xI]
                    self.lowMiddle[yI] = box[yI]
                    self.lowMiddle[filterBoxFlagI] = 1
                    self.lowMiddle[filterBoxCntI] = 0
                    self.lowMiddle[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[lowIndex][lineIndex]
                    self.lowMiddle[successInitialFlagI] = 1


        else:
            self.lowMiddle[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_low_middle(self.nowBoxes, self.lowMiddle[xI], self.lowMiddle[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.middleLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_low_middle(self.nowBoxes, self.middleLine[kI], self.middleLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.lowMiddleKalmanFilter.process([self.lowMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[lowIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.lowMiddle[xI] = box[xI]
                self.lowMiddle[yI] = box[yI]
                self.lowMiddle[filterBoxFlagI] = 1
                self.lowMiddle[filterBoxCntI] = 0
                self.lowMiddle[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[lowIndex][distanceIndex]
                self.lowMiddle[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]],
                                                                 self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.lowMiddleKalmanFilter.process([self.lowMiddle[xI], self.lowMiddle[yI]],
                                                                        self.nowBoxes[lowIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.lowMiddle[xI] = box[xI]
                self.lowMiddle[yI] = box[yI]
                self.lowMiddle[filterBoxFlagI] = 1
                self.lowMiddle[filterBoxCntI] = 0
                self.lowMiddle[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[lowIndex][lineIndex]
                self.lowMiddle[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowMiddle[xI], self.lowMiddle[yI]],
                                                                 self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.lowMiddleKalmanFilter.process([self.lowMiddle[xI], self.lowMiddle[yI]],
                                                                            self.nowBoxes[lowIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.lowMiddle[xI] = box[xI]
                    self.lowMiddle[yI] = box[yI]
                    self.lowMiddle[filterBoxFlagI] = 1
                    self.lowMiddle[filterBoxCntI] = 0
                    self.lowMiddle[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[lowIndex][distanceIndex]
                    self.lowMiddle[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.lowMiddleKalmanFilter.process([self.lowMiddle[xI], self.lowMiddle[yI]],
                                                                            self.nowBoxes[lowIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.lowMiddle[xI] = box[xI]
                    self.lowMiddle[yI] = box[yI]
                    self.lowMiddle[filterBoxFlagI] = 1
                    self.lowMiddle[filterBoxCntI] = 0
                    self.lowMiddle[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[lowIndex][lineIndex]
                    self.lowMiddle[successUpdateFlagI] = 1

            else:
                self.lowMiddle[filterBoxCntI] += 1



    def calculate_middle_middle(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看一下中层中间的滤波框是否初始化成功了
        if self.middleMiddle[filterBoxFlagI] == 0:
            self.middleMiddle[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_middle_middle(self.nowBoxes, self.middleMiddle[xI], self.middleMiddle[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.middleLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_middle_middle(self.nowBoxes, self.middleLine[kI], self.middleLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.middleMiddle[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[middleIndex][distanceIndex])
                self.middleMiddle[xI] = box[xI]
                self.middleMiddle[yI] = box[yI]
                self.middleMiddle[filterBoxFlagI] = 1
                self.middleMiddle[filterBoxCntI] = 0
                self.middleMiddle[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[middleIndex][distanceIndex]
                self.middleMiddle[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.middleMiddle[yI]], self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[middleIndex][lineIndex])
                self.middleMiddle[xI] = box[xI]
                self.middleMiddle[yI] = box[yI]
                self.middleMiddle[filterBoxFlagI] = 1
                self.middleMiddle[filterBoxCntI] = 0
                self.middleMiddle[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[middleIndex][lineIndex]
                self.middleMiddle[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.middleMiddle[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.middleMiddle[yI]], self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.middleMiddleFilterBox = copy.deepcopy(self.nowBoxes[middleIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[middleIndex][distanceIndex])
                    self.middleMiddle[xI] = box[xI]
                    self.middleMiddle[yI] = box[yI]
                    self.middleMiddle[filterBoxFlagI] = 1
                    self.middleMiddle[filterBoxCntI] = 0
                    self.middleMiddle[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[middleIndex][distanceIndex]
                    self.middleMiddle[successInitialFlagI] = 1
                else:
                    # self.middleMiddleFilterBox = copy.deepcopy(self.nowBoxes[middleIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[middleIndex][lineIndex])
                    self.middleMiddle[xI] = box[xI]
                    self.middleMiddle[yI] = box[yI]
                    self.middleMiddle[filterBoxFlagI] = 1
                    self.middleMiddle[filterBoxCntI] = 0
                    self.middleMiddle[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[middleIndex][lineIndex]
                    self.middleMiddle[successInitialFlagI] = 1

        else:
            self.middleMiddle[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_middle_middle(self.nowBoxes, self.middleMiddle[xI], self.middleMiddle[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.middleLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_middle_middle(self.nowBoxes, self.middleLine[kI], self.middleLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.lowMiddle[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.middleMiddleKalmanFilter.process([self.middleMiddle[xI], self.middleMiddle[yI]], self.nowBoxes[middleIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.middleMiddle[xI] = box[xI]
                self.middleMiddle[yI] = box[yI]
                self.middleMiddle[filterBoxFlagI] = 1
                self.middleMiddle[filterBoxCntI] = 0
                self.middleMiddle[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[middleIndex][distanceIndex]
                self.middleMiddle[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.middleMiddle[yI]],
                                                                 self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.middleMiddleKalmanFilter.process([self.middleMiddle[xI], self.middleMiddle[yI]],
                                                                        self.nowBoxes[middleIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.middleMiddle[xI] = box[xI]
                self.middleMiddle[yI] = box[yI]
                self.middleMiddle[filterBoxFlagI] = 1
                self.middleMiddle[filterBoxCntI] = 0
                self.middleMiddle[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[middleIndex][lineIndex]
                self.middleMiddle[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.middleMiddle[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleMiddle[xI], self.middleMiddle[yI]],
                                                                 self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.middleMiddleKalmanFilter.process([self.middleMiddle[xI], self.middleMiddle[yI]],
                                                                            self.nowBoxes[middleIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.middleMiddle[xI] = box[xI]
                    self.middleMiddle[yI] = box[yI]
                    self.middleMiddle[filterBoxFlagI] = 1
                    self.middleMiddle[filterBoxCntI] = 0
                    self.middleMiddle[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[middleIndex][distanceIndex]
                    self.middleMiddle[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.middleMiddleKalmanFilter.process([self.middleMiddle[xI], self.middleMiddle[yI]],
                                                                            self.nowBoxes[middleIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.middleMiddle[xI] = box[xI]
                    self.middleMiddle[yI] = box[yI]
                    self.middleMiddle[filterBoxFlagI] = 1
                    self.middleMiddle[filterBoxCntI] = 0
                    self.middleMiddle[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[middleIndex][lineIndex]
                    self.middleMiddle[successUpdateFlagI] = 1
            else:
                self.middleMiddle[filterBoxCntI] += 1


    def calculate_upper_middle(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        if self.upperMiddle[filterBoxFlagI] == 0:
            self.upperMiddle[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_upper_middle(self.nowBoxes, self.upperMiddle[xI], self.upperMiddle[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.middleLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_upper_middle(self.nowBoxes, self.middleLine[kI], self.middleLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                self.upperMiddle[xI] = box[xI]
                self.upperMiddle[yI] = box[yI]
                self.upperMiddle[filterBoxFlagI] = 1
                self.upperMiddle[filterBoxCntI] = 0
                self.upperMiddle[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[upperIndex][distanceIndex]
                self.upperMiddle[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]], self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                self.upperMiddle[xI] = box[xI]
                self.upperMiddle[yI] = box[yI]
                self.upperMiddle[filterBoxFlagI] = 1
                self.upperMiddle[filterBoxCntI] = 0
                self.upperMiddle[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[upperIndex][lineIndex]
                self.upperMiddle[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]], self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.upperMiddleFilterBox = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                    self.upperMiddle[xI] = box[xI]
                    self.upperMiddle[yI] = box[yI]
                    self.upperMiddle[filterBoxFlagI] = 1
                    self.upperMiddle[filterBoxCntI] = 0
                    self.upperMiddle[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[upperIndex][distanceIndex]
                    self.upperMiddle[successInitialFlagI] = 1
                else:
                    # self.upperMiddleFilterBox = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                    self.upperMiddle[xI] = box[xI]
                    self.upperMiddle[yI] = box[yI]
                    self.upperMiddle[filterBoxFlagI] = 1
                    self.upperMiddle[filterBoxCntI] = 0
                    self.upperMiddle[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[upperIndex][lineIndex]
                    self.upperMiddle[successInitialFlagI] = 1

        else:
            self.upperMiddle[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_upper_middle(self.nowBoxes, self.upperMiddle[xI], self.upperMiddle[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.middleLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_upper_middle(self.nowBoxes, self.middleLine[kI], self.middleLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.upperMiddleKalmanFilter.process([self.upperMiddle[xI], self.upperMiddle[yI]], self.nowBoxes[upperIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.upperMiddle[xI] = box[xI]
                self.upperMiddle[yI] = box[yI]
                self.upperMiddle[filterBoxFlagI] = 1
                self.upperMiddle[filterBoxCntI] = 0
                self.upperMiddle[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[upperIndex][distanceIndex]
                self.upperMiddle[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]],
                                                                 self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.upperMiddleKalmanFilter.process([self.upperMiddle[xI], self.upperMiddle[yI]],
                                                                        self.nowBoxes[upperIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.upperMiddle[xI] = box[xI]
                self.upperMiddle[yI] = box[yI]
                self.upperMiddle[filterBoxFlagI] = 1
                self.upperMiddle[filterBoxCntI] = 0
                self.upperMiddle[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[upperIndex][lineIndex]
                self.upperMiddle[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperMiddle[xI], self.upperMiddle[yI]],
                                                                 self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.upperMiddleKalmanFilter.process([self.upperMiddle[xI], self.upperMiddle[yI]],
                                                                            self.nowBoxes[upperIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.upperMiddle[xI] = box[xI]
                    self.upperMiddle[yI] = box[yI]
                    self.upperMiddle[filterBoxFlagI] = 1
                    self.upperMiddle[filterBoxCntI] = 0
                    self.upperMiddle[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[upperIndex][distanceIndex]
                    self.upperMiddle[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.upperMiddleKalmanFilter.process([self.upperMiddle[xI], self.upperMiddle[yI]],
                                                                            self.nowBoxes[upperIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.upperMiddle[xI] = box[xI]
                    self.upperMiddle[yI] = box[yI]
                    self.upperMiddle[filterBoxFlagI] = 1
                    self.upperMiddle[filterBoxCntI] = 0
                    self.upperMiddle[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[upperIndex][lineIndex]
                    self.upperMiddle[successUpdateFlagI] = 1
            else:
                self.upperMiddle[filterBoxCntI] += 1


    def calculate_low_left(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看一下底层中间的滤波框是否初始化成功了
        if self.lowLeft[filterBoxFlagI] == 0:
            self.lowLeft[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_low_left(self.nowBoxes, self.lowLeft[xI], self.lowLeft[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.leftLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_low_left(self.nowBoxes, self.leftLine[kI], self.leftLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                self.lowLeft[xI] = box[xI]
                self.lowLeft[yI] = box[yI]
                self.lowLeft[filterBoxFlagI] = 1
                self.lowLeft[filterBoxCntI] = 0
                self.lowLeft[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[lowIndex][distanceIndex]
                self.lowLeft[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]], self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                self.lowLeft[xI] = box[xI]
                self.lowLeft[yI] = box[yI]
                self.lowLeft[filterBoxFlagI] = 1
                self.lowLeft[filterBoxCntI] = 0
                self.lowLeft[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[lowIndex][lineIndex]
                self.lowLeft[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]], self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    self.lowLeft[xI] = box[xI]
                    self.lowLeft[yI] = box[yI]
                    self.lowLeft[filterBoxFlagI] = 1
                    self.lowLeft[filterBoxCntI] = 0
                    self.lowLeft[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[lowIndex][distanceIndex]
                    self.lowLeft[successInitialFlagI] = 1
                else:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    self.lowLeft[xI] = box[xI]
                    self.lowLeft[yI] = box[yI]
                    self.lowLeft[filterBoxFlagI] = 1
                    self.lowLeft[filterBoxCntI] = 0
                    self.lowLeft[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[lowIndex][lineIndex]
                    self.lowLeft[successInitialFlagI] = 1


        else:
            self.lowLeft[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_low_left(self.nowBoxes, self.lowLeft[xI], self.lowLeft[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.leftLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_low_left(self.nowBoxes, self.leftLine[kI], self.leftLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.lowLeftKalmanFilter.process([self.lowLeft[xI], self.lowLeft[yI]], self.nowBoxes[lowIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.lowLeft[xI] = box[xI]
                self.lowLeft[yI] = box[yI]
                self.lowLeft[filterBoxFlagI] = 1
                self.lowLeft[filterBoxCntI] = 0
                self.lowLeft[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[lowIndex][distanceIndex]
                self.lowLeft[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]],
                                                                 self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.lowLeftKalmanFilter.process([self.lowLeft[xI], self.lowLeft[yI]],
                                                                        self.nowBoxes[lowIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.lowLeft[xI] = box[xI]
                self.lowLeft[yI] = box[yI]
                self.lowLeft[filterBoxFlagI] = 1
                self.lowLeft[filterBoxCntI] = 0
                self.lowLeft[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[lowIndex][lineIndex]
                self.lowLeft[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowLeft[xI], self.lowLeft[yI]],
                                                                 self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.lowLeftKalmanFilter.process([self.lowLeft[xI], self.lowLeft[yI]],
                                                                            self.nowBoxes[lowIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.lowLeft[xI] = box[xI]
                    self.lowLeft[yI] = box[yI]
                    self.lowLeft[filterBoxFlagI] = 1
                    self.lowLeft[filterBoxCntI] = 0
                    self.lowLeft[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[lowIndex][distanceIndex]
                    self.lowLeft[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.lowLeftKalmanFilter.process([self.lowLeft[xI], self.lowLeft[yI]],
                                                                            self.nowBoxes[lowIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.lowLeft[xI] = box[xI]
                    self.lowLeft[yI] = box[yI]
                    self.lowLeft[filterBoxFlagI] = 1
                    self.lowLeft[filterBoxCntI] = 0
                    self.lowLeft[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[lowIndex][lineIndex]
                    self.lowLeft[successUpdateFlagI] = 1

            else:
                self.lowLeft[filterBoxCntI] += 1


    def calculate_middle_left(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看一下底层中间的滤波框是否初始化成功了
        if self.middleLeft[filterBoxFlagI] == 0:
            self.middleLeft[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_middle_left(self.nowBoxes, self.middleLeft[xI], self.middleLeft[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.leftLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_middle_left(self.nowBoxes, self.leftLine[kI], self.leftLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[middleIndex][distanceIndex])
                self.middleLeft[xI] = box[xI]
                self.middleLeft[yI] = box[yI]
                self.middleLeft[filterBoxFlagI] = 1
                self.middleLeft[filterBoxCntI] = 0
                self.middleLeft[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[middleIndex][distanceIndex]
                self.middleLeft[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]], self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[middleIndex][lineIndex])
                self.middleLeft[xI] = box[xI]
                self.middleLeft[yI] = box[yI]
                self.middleLeft[filterBoxFlagI] = 1
                self.middleLeft[filterBoxCntI] = 0
                self.middleLeft[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[middleIndex][lineIndex]
                self.middleLeft[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]], self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[middleIndex][distanceIndex])
                    self.middleLeft[xI] = box[xI]
                    self.middleLeft[yI] = box[yI]
                    self.middleLeft[filterBoxFlagI] = 1
                    self.middleLeft[filterBoxCntI] = 0
                    self.middleLeft[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[middleIndex][distanceIndex]
                    self.middleLeft[successInitialFlagI] = 1
                else:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[middleIndex][lineIndex])
                    self.middleLeft[xI] = box[xI]
                    self.middleLeft[yI] = box[yI]
                    self.middleLeft[filterBoxFlagI] = 1
                    self.middleLeft[filterBoxCntI] = 0
                    self.middleLeft[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[middleIndex][lineIndex]
                    self.middleLeft[successInitialFlagI] = 1


        else:
            self.middleLeft[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_middle_left(self.nowBoxes, self.middleLeft[xI], self.middleLeft[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.leftLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_middle_left(self.nowBoxes, self.leftLine[kI], self.leftLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.middleLeftKalmanFilter.process([self.middleLeft[xI], self.middleLeft[yI]], self.nowBoxes[middleIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.middleLeft[xI] = box[xI]
                self.middleLeft[yI] = box[yI]
                self.middleLeft[filterBoxFlagI] = 1
                self.middleLeft[filterBoxCntI] = 0
                self.middleLeft[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[middleIndex][distanceIndex]
                self.middleLeft[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]],
                                                                 self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.middleLeftKalmanFilter.process([self.middleLeft[xI], self.middleLeft[yI]],
                                                                        self.nowBoxes[middleIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.middleLeft[xI] = box[xI]
                self.middleLeft[yI] = box[yI]
                self.middleLeft[filterBoxFlagI] = 1
                self.middleLeft[filterBoxCntI] = 0
                self.middleLeft[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[middleIndex][lineIndex]
                self.middleLeft[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleLeft[xI], self.middleLeft[yI]],
                                                                 self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.middleLeftKalmanFilter.process([self.middleLeft[xI], self.middleLeft[yI]],
                                                                            self.nowBoxes[middleIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.middleLeft[xI] = box[xI]
                    self.middleLeft[yI] = box[yI]
                    self.middleLeft[filterBoxFlagI] = 1
                    self.middleLeft[filterBoxCntI] = 0
                    self.middleLeft[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[middleIndex][distanceIndex]
                    self.middleLeft[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.middleLeftKalmanFilter.process([self.middleLeft[xI], self.middleLeft[yI]],
                                                                            self.nowBoxes[middleIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.middleLeft[xI] = box[xI]
                    self.middleLeft[yI] = box[yI]
                    self.middleLeft[filterBoxFlagI] = 1
                    self.middleLeft[filterBoxCntI] = 0
                    self.middleLeft[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[middleIndex][lineIndex]
                    self.middleLeft[successUpdateFlagI] = 1

            else:
                self.middleLeft[filterBoxCntI] += 1



    def calculate_upper_left(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        if self.upperLeft[filterBoxFlagI] == 0:
            self.upperLeft[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_upper_left(self.nowBoxes, self.upperLeft[xI], self.upperLeft[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.leftLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_upper_middle(self.nowBoxes, self.leftLine[kI], self.leftLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                self.upperLeft[xI] = box[xI]
                self.upperLeft[yI] = box[yI]
                self.upperLeft[filterBoxFlagI] = 1
                self.upperLeft[filterBoxCntI] = 0
                self.upperLeft[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[upperIndex][distanceIndex]
                self.upperLeft[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]], self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                self.upperLeft[xI] = box[xI]
                self.upperLeft[yI] = box[yI]
                self.upperLeft[filterBoxFlagI] = 1
                self.upperLeft[filterBoxCntI] = 0
                self.upperLeft[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[upperIndex][lineIndex]
                self.upperLeft[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]], self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.upperMiddleFilterBox = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                    self.upperLeft[xI] = box[xI]
                    self.upperLeft[yI] = box[yI]
                    self.upperLeft[filterBoxFlagI] = 1
                    self.upperLeft[filterBoxCntI] = 0
                    self.upperLeft[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[upperIndex][distanceIndex]
                    self.upperLeft[successInitialFlagI] = 1
                else:
                    # self.upperMiddleFilterBox = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                    self.upperLeft[xI] = box[xI]
                    self.upperLeft[yI] = box[yI]
                    self.upperLeft[filterBoxFlagI] = 1
                    self.upperLeft[filterBoxCntI] = 0
                    self.upperLeft[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[upperIndex][lineIndex]
                    self.upperLeft[successInitialFlagI] = 1

        else:
            self.upperLeft[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_upper_left(self.nowBoxes, self.upperLeft[xI], self.upperLeft[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.leftLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_upper_left(self.nowBoxes, self.leftLine[kI], self.leftLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.upperLeftKalmanFilter.process([self.upperLeft[xI], self.upperLeft[yI]], self.nowBoxes[upperIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.upperLeft[xI] = box[xI]
                self.upperLeft[yI] = box[yI]
                self.upperLeft[filterBoxFlagI] = 1
                self.upperLeft[filterBoxCntI] = 0
                self.upperLeft[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[upperIndex][distanceIndex]
                self.upperLeft[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]],
                                                                 self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.upperLeftKalmanFilter.process([self.upperLeft[xI], self.upperLeft[yI]],
                                                                        self.nowBoxes[upperIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.upperLeft[xI] = box[xI]
                self.upperLeft[yI] = box[yI]
                self.upperLeft[filterBoxFlagI] = 1
                self.upperLeft[filterBoxCntI] = 0
                self.upperLeft[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[upperIndex][lineIndex]
                self.upperLeft[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperLeft[xI], self.upperLeft[yI]],
                                                                 self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.upperLeftKalmanFilter.process([self.upperLeft[xI], self.upperLeft[yI]],
                                                                            self.nowBoxes[upperIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.upperLeft[xI] = box[xI]
                    self.upperLeft[yI] = box[yI]
                    self.upperLeft[filterBoxFlagI] = 1
                    self.upperLeft[filterBoxCntI] = 0
                    self.upperLeft[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[upperIndex][distanceIndex]
                    self.upperLeft[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.upperLeftKalmanFilter.process([self.upperLeft[xI], self.upperLeft[yI]],
                                                                            self.nowBoxes[upperIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.upperLeft[xI] = box[xI]
                    self.upperLeft[yI] = box[yI]
                    self.upperLeft[filterBoxFlagI] = 1
                    self.upperLeft[filterBoxCntI] = 0
                    self.upperLeft[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[upperIndex][lineIndex]
                    self.upperLeft[successUpdateFlagI] = 1
            else:
                self.upperLeft[filterBoxCntI] += 1




    def calculate_low_right(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看一下底层中间的滤波框是否初始化成功了
        if self.lowRight[filterBoxFlagI] == 0:
            self.lowRight[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_low_right(self.nowBoxes, self.lowRight[xI], self.lowRight[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.rightLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_low_right(self.nowBoxes, self.rightLine[kI], self.rightLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                self.lowRight[xI] = box[xI]
                self.lowRight[yI] = box[yI]
                self.lowRight[filterBoxFlagI] = 1
                self.lowRight[filterBoxCntI] = 0
                self.lowRight[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[lowIndex][distanceIndex]
                self.lowRight[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]], self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                self.lowRight[xI] = box[xI]
                self.lowRight[yI] = box[yI]
                self.lowRight[filterBoxFlagI] = 1
                self.lowRight[filterBoxCntI] = 0
                self.lowRight[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[lowIndex][lineIndex]
                self.lowRight[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]], self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    self.lowRight[xI] = box[xI]
                    self.lowRight[yI] = box[yI]
                    self.lowRight[filterBoxFlagI] = 1
                    self.lowRight[filterBoxCntI] = 0
                    self.lowRight[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[lowIndex][distanceIndex]
                    self.lowRight[successInitialFlagI] = 1
                else:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    self.lowRight[xI] = box[xI]
                    self.lowRight[yI] = box[yI]
                    self.lowRight[filterBoxFlagI] = 1
                    self.lowRight[filterBoxCntI] = 0
                    self.lowRight[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[lowIndex][lineIndex]
                    self.lowRight[successInitialFlagI] = 1


        else:
            self.lowRight[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_low_right(self.nowBoxes, self.lowRight[xI], self.lowRight[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.rightLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_low_right(self.nowBoxes, self.rightLine[kI], self.rightLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.lowRightKalmanFilter.process([self.lowRight[xI], self.lowRight[yI]], self.nowBoxes[lowIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.lowRight[xI] = box[xI]
                self.lowRight[yI] = box[yI]
                self.lowRight[filterBoxFlagI] = 1
                self.lowRight[filterBoxCntI] = 0
                self.lowRight[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[lowIndex][distanceIndex]
                self.lowRight[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]],
                                                                 self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.lowRightKalmanFilter.process([self.lowRight[xI], self.lowRight[yI]],
                                                                        self.nowBoxes[lowIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.lowRight[xI] = box[xI]
                self.lowRight[yI] = box[yI]
                self.lowRight[filterBoxFlagI] = 1
                self.lowRight[filterBoxCntI] = 0
                self.lowRight[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[lowIndex][lineIndex]
                self.lowRight[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]], self.nowBoxes[lowIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.lowRight[xI], self.lowRight[yI]],
                                                                 self.nowBoxes[lowIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.lowRightKalmanFilter.process([self.lowRight[xI], self.lowRight[yI]],
                                                                            self.nowBoxes[lowIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.lowRight[xI] = box[xI]
                    self.lowRight[yI] = box[yI]
                    self.lowRight[filterBoxFlagI] = 1
                    self.lowRight[filterBoxCntI] = 0
                    self.lowRight[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[lowIndex][distanceIndex]
                    self.lowRight[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.lowRightKalmanFilter.process([self.lowRight[xI], self.lowRight[yI]],
                                                                            self.nowBoxes[lowIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.lowRight[xI] = box[xI]
                    self.lowRight[yI] = box[yI]
                    self.lowRight[filterBoxFlagI] = 1
                    self.lowRight[filterBoxCntI] = 0
                    self.lowRight[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[lowIndex][lineIndex]
                    self.lowRight[successUpdateFlagI] = 1

            else:
                self.lowRight[filterBoxCntI] += 1


    def calculate_middle_right(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看一下底层中间的滤波框是否初始化成功了
        if self.middleRight[filterBoxFlagI] == 0:
            self.middleRight[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_middle_right(self.nowBoxes, self.middleRight[xI], self.middleRight[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.rightLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_middle_right(self.nowBoxes, self.rightLine[kI], self.rightLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[middleIndex][distanceIndex])
                self.middleRight[xI] = box[xI]
                self.middleRight[yI] = box[yI]
                self.middleRight[filterBoxFlagI] = 1
                self.middleRight[filterBoxCntI] = 0
                self.middleRight[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[middleIndex][distanceIndex]
                self.middleRight[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]], self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[middleIndex][lineIndex])
                self.middleRight[xI] = box[xI]
                self.middleRight[yI] = box[yI]
                self.middleRight[filterBoxFlagI] = 1
                self.middleRight[filterBoxCntI] = 0
                self.middleRight[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[middleIndex][lineIndex]
                self.middleRight[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]], self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[middleIndex][distanceIndex])
                    self.middleRight[xI] = box[xI]
                    self.middleRight[yI] = box[yI]
                    self.middleRight[filterBoxFlagI] = 1
                    self.middleRight[filterBoxCntI] = 0
                    self.middleRight[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[middleIndex][distanceIndex]
                    self.middleRight[successInitialFlagI] = 1
                else:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[middleIndex][lineIndex])
                    self.middleRight[xI] = box[xI]
                    self.middleRight[yI] = box[yI]
                    self.middleRight[filterBoxFlagI] = 1
                    self.middleRight[filterBoxCntI] = 0
                    self.middleRight[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[middleIndex][lineIndex]
                    self.middleRight[successInitialFlagI] = 1


        else:
            self.middleRight[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_middle_right(self.nowBoxes, self.middleRight[xI], self.middleRight[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.rightLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_middle_right(self.nowBoxes, self.rightLine[kI], self.rightLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.middleRightKalmanFilter.process([self.middleRight[xI], self.middleRight[yI]], self.nowBoxes[middleIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.middleRight[xI] = box[xI]
                self.middleRight[yI] = box[yI]
                self.middleRight[filterBoxFlagI] = 1
                self.middleRight[filterBoxCntI] = 0
                self.middleRight[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[middleIndex][distanceIndex]
                self.middleRight[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]],
                                                                 self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.middleRightKalmanFilter.process([self.middleRight[xI], self.middleRight[yI]],
                                                                        self.nowBoxes[middleIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.middleRight[xI] = box[xI]
                self.middleRight[yI] = box[yI]
                self.middleRight[filterBoxFlagI] = 1
                self.middleRight[filterBoxCntI] = 0
                self.middleRight[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[middleIndex][lineIndex]
                self.middleRight[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]], self.nowBoxes[middleIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.middleRight[xI], self.middleRight[yI]],
                                                                 self.nowBoxes[middleIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.middleRightKalmanFilter.process([self.middleRight[xI], self.middleRight[yI]],
                                                                            self.nowBoxes[middleIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.middleRight[xI] = box[xI]
                    self.middleRight[yI] = box[yI]
                    self.middleRight[filterBoxFlagI] = 1
                    self.middleRight[filterBoxCntI] = 0
                    self.middleRight[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[middleIndex][distanceIndex]
                    self.middleRight[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.middleRightKalmanFilter.process([self.middleRight[xI], self.middleRight[yI]],
                                                                            self.nowBoxes[middleIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.middleRight[xI] = box[xI]
                    self.middleRight[yI] = box[yI]
                    self.middleRight[filterBoxFlagI] = 1
                    self.middleRight[filterBoxCntI] = 0
                    self.middleRight[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[middleIndex][lineIndex]
                    self.middleRight[successUpdateFlagI] = 1

            else:
                self.middleRight[filterBoxCntI] += 1


    def calculate_upper_right(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看一下底层中间的滤波框是否初始化成功了
        if self.upperRight[filterBoxFlagI] == 0:
            self.upperRight[successInitialFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.initial_distance_upper_right(self.nowBoxes, self.upperRight[xI], self.upperRight[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.rightLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.initial_line_upper_right(self.nowBoxes, self.rightLine[kI], self.rightLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:

                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                self.upperRight[xI] = box[xI]
                self.upperRight[yI] = box[yI]
                self.upperRight[filterBoxFlagI] = 1
                self.upperRight[filterBoxCntI] = 0
                self.upperRight[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[upperIndex][distanceIndex]
                self.upperRight[successInitialFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]], self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                # 初始化底层中间的滤波框
                box = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                self.upperRight[xI] = box[xI]
                self.upperRight[yI] = box[yI]
                self.upperRight[filterBoxFlagI] = 1
                self.upperRight[filterBoxCntI] = 0
                self.upperRight[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[upperIndex][lineIndex]
                self.upperRight[successInitialFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]], self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 初始化底层中间的滤波框
                if tempConfidence_1 >= tempConfidence_2:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][distanceIndex])
                    box = copy.deepcopy(self.nowBoxes[upperIndex][distanceIndex])
                    self.upperRight[xI] = box[xI]
                    self.upperRight[yI] = box[yI]
                    self.upperRight[filterBoxFlagI] = 1
                    self.upperRight[filterBoxCntI] = 0
                    self.upperRight[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[upperIndex][distanceIndex]
                    self.upperRight[successInitialFlagI] = 1
                else:
                    # self.lowMiddleFilterBox = copy.deepcopy(self.nowBoxes[lowIndex][lineIndex])
                    box = copy.deepcopy(self.nowBoxes[upperIndex][lineIndex])
                    self.upperRight[xI] = box[xI]
                    self.upperRight[yI] = box[yI]
                    self.upperRight[filterBoxFlagI] = 1
                    self.upperRight[filterBoxCntI] = 0
                    self.upperRight[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[upperIndex][lineIndex]
                    self.upperRight[successInitialFlagI] = 1


        else:
            self.upperRight[successUpdateFlagI] = 0
            # 首先按距离查找一个底层的识别框试试看
            distanceIndex = V1_9_3_utils.filter_distance_upper_right(self.nowBoxes, self.upperRight[xI], self.upperRight[yI])
            # 如果此时线存在的话，按照跟踪线查找识别框
            if self.rightLine[lineFlagI] != 0:
                lineIndex = V1_9_3_utils.filter_line_upper_right(self.nowBoxes, self.rightLine[kI], self.rightLine[bI])
            else:
                lineIndex = -1

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # 还要加一下判断，上述查找到的识别框，对于该层的左右两侧的采信是否ok
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            # 对查找到的符合要求的识别框，计算
            if distanceIndex != -1 and lineIndex == -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.upperRightKalmanFilter.process([self.upperRight[xI], self.upperRight[yI]], self.nowBoxes[upperIndex][distanceIndex])
                # 更新状态
                box = updated.tolist()
                self.upperRight[xI] = box[xI]
                self.upperRight[yI] = box[yI]
                self.upperRight[filterBoxFlagI] = 1
                self.upperRight[filterBoxCntI] = 0
                self.upperRight[filterBoxConfidenceI] += tempConfidence_1
                del self.nowBoxes[upperIndex][distanceIndex]
                self.upperRight[successUpdateFlagI] = 1
            elif distanceIndex == -1 and lineIndex != -1:
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]],
                                                                 self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)

                # 找到了符合要求的识别框，开始进行卡尔曼滤波
                predicted, updated = self.upperRightKalmanFilter.process([self.upperRight[xI], self.upperRight[yI]],
                                                                        self.nowBoxes[upperIndex][lineIndex])
                # 更新状态
                box = updated.tolist()
                self.upperRight[xI] = box[xI]
                self.upperRight[yI] = box[yI]
                self.upperRight[filterBoxFlagI] = 1
                self.upperRight[filterBoxCntI] = 0
                self.upperRight[filterBoxConfidenceI] += tempConfidence_2
                del self.nowBoxes[upperIndex][lineIndex]
                self.upperRight[successUpdateFlagI] = 1
            elif distanceIndex != -1 and lineIndex != -1:
                tempDistance_1 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]], self.nowBoxes[upperIndex][distanceIndex])
                tempConfidence_1 = V1_9_3_utils.stable_sigmoid(tempDistance_1)
                tempDistance_2 = V1_9_3_utils.euclidean_distance([self.upperRight[xI], self.upperRight[yI]],
                                                                 self.nowBoxes[upperIndex][lineIndex])
                tempConfidence_2 = V1_9_3_utils.stable_sigmoid(tempDistance_2)
                if tempConfidence_1 >= tempConfidence_2:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.upperRightKalmanFilter.process([self.upperRight[xI], self.upperRight[yI]],
                                                                            self.nowBoxes[upperIndex][distanceIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.upperRight[xI] = box[xI]
                    self.upperRight[yI] = box[yI]
                    self.upperRight[filterBoxFlagI] = 1
                    self.upperRight[filterBoxCntI] = 0
                    self.upperRight[filterBoxConfidenceI] += tempConfidence_1
                    del self.nowBoxes[upperIndex][distanceIndex]
                    self.upperRight[successUpdateFlagI] = 1
                else:
                    # 找到了符合要求的识别框，开始进行卡尔曼滤波
                    predicted, updated = self.upperRightKalmanFilter.process([self.upperRight[xI], self.upperRight[yI]],
                                                                            self.nowBoxes[upperIndex][lineIndex])
                    # 更新状态
                    box = updated.tolist()
                    self.upperRight[xI] = box[xI]
                    self.upperRight[yI] = box[yI]
                    self.upperRight[filterBoxFlagI] = 1
                    self.upperRight[filterBoxCntI] = 0
                    self.upperRight[filterBoxConfidenceI] += tempConfidence_2
                    del self.nowBoxes[upperIndex][lineIndex]
                    self.upperRight[successUpdateFlagI] = 1

            else:
                self.upperRight[filterBoxCntI] += 1

    def calculate_filter_boxes_leftLine(self) -> None:
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex


        # 看一下三个聚类框是否都是存在的
        if self.lowLeft[lineFlagI] != 0 and self.middleLeft[lineFlagI] != 0 and self.upperLeft[lineFlagI] != 0:

            # 使用最小二乘法拟合跟踪线
            FITLINE = V1_9_3_utils.fit([[self.lowLeft[xI], self.lowLeft[yI]], [self.middleLeft[xI], self.middleLeft[yI]], [self.upperLeft[xI], self.upperLeft[yI]]])


            if config_parameters.LEFTLINEK <= FITLINE[kI] <= 0.1:
                self.leftLine[kI] = FITLINE[kI]
                self.leftLine[bI] = FITLINE[bI]
                self.leftLine[lineFlagI] = 1
                self.leftLine[lineCntI] = 0
            else:
                self.leftLine[lineCntI] += 1

        # 如果是只有底层和中层存在
        elif self.lowLeft[lineFlagI] != 0 and self.middleLeft[lineFlagI] != 0 and self.upperLeft[lineFlagI] == 0:
            LOWMIDDLELINE = V1_9_3_utils.get_k_b([self.lowLeft[xI], self.lowLeft[yI]],
                                                 [self.middleLeft[xI], self.middleLeft[yI]])
            if config_parameters.LEFTLINEK <= LOWMIDDLELINE[kI] <= 0.1:
                self.leftLine[kI] = LOWMIDDLELINE[kI]
                self.leftLine[bI] = LOWMIDDLELINE[bI]
                self.leftLine[lineFlagI] = 1
                self.leftLine[lineCntI] = 0
            else:
                self.leftLine[lineCntI] += 1
        # 如果是只有底层和顶层存在
        elif self.lowLeft[lineFlagI] != 0 and self.middleLeft[lineFlagI] == 0 and self.upperLeft[lineFlagI] != 0:
            LOWUPPERLINE = V1_9_3_utils.get_k_b([self.lowLeft[xI], self.lowLeft[yI]],
                                                [self.upperLeft[xI], self.upperLeft[yI]])

            if config_parameters.LEFTLINEK <= LOWUPPERLINE[kI] <= 0.1:
                self.leftLine[kI] = LOWUPPERLINE[kI]
                self.leftLine[bI] = LOWUPPERLINE[bI]
                self.leftLine[lineFlagI] = 1
                self.leftLine[lineCntI] = 0
            else:
                self.leftLine[lineCntI] += 1
        # 如果是只有中层和顶层存在
        elif self.lowLeft[lineFlagI] == 0 and self.middleLeft[lineFlagI] != 0 and self.upperLeft[lineFlagI] != 0:
            MIDDLEUPPERLINE = V1_9_3_utils.get_k_b([self.middleLeft[xI], self.middleLeft[yI]],
                                                   [self.upperLeft[xI], self.upperLeft[yI]])

            if config_parameters.LEFTLINEK <= MIDDLEUPPERLINE[kI] <= 0.1:
                self.leftLine[kI] = MIDDLEUPPERLINE[kI]
                self.leftLine[bI] = MIDDLEUPPERLINE[bI]
                self.leftLine[lineFlagI] = 1
                self.leftLine[lineCntI] = 0
            else:
                self.leftLine[lineCntI] += 1
        else:
            self.leftLine[lineCntI] += 1


    def calculate_filter_boxes_rightLine(self) -> None:
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex


        # 看一下三个聚类框是否都是存在的
        if self.lowRight[lineFlagI] != 0 and self.middleRight[lineFlagI] != 0 and self.upperRight[lineFlagI] != 0:

            # 使用最小二乘法拟合跟踪线
            FITLINE = V1_9_3_utils.fit([[self.lowRight[xI], self.lowRight[yI]], [self.middleRight[xI], self.middleRight[yI]], [self.upperRight[xI], self.upperRight[yI]]])


            if -0.1 <= FITLINE[kI] <= config_parameters.RIGHTLINEK:
                self.rightLine[kI] = FITLINE[kI]
                self.rightLine[bI] = FITLINE[bI]
                self.rightLine[lineFlagI] = 1
                self.rightLine[lineCntI] = 0
            else:
                self.rightLine[lineCntI] += 1

        # 如果是只有底层和中层存在
        elif self.lowRight[lineFlagI] != 0 and self.middleRight[lineFlagI] != 0 and self.upperRight[lineFlagI] == 0:
            LOWMIDDLELINE = V1_9_3_utils.get_k_b([self.lowRight[xI], self.lowRight[yI]],
                                                 [self.middleRight[xI], self.middleRight[yI]])
            if -0.1 <= LOWMIDDLELINE[kI] <= config_parameters.RIGHTLINEK:
                self.rightLine[kI] = LOWMIDDLELINE[kI]
                self.rightLine[bI] = LOWMIDDLELINE[bI]
                self.rightLine[lineFlagI] = 1
                self.rightLine[lineCntI] = 0
            else:
                self.rightLine[lineCntI] += 1
        # 如果是只有底层和顶层存在
        elif self.lowRight[lineFlagI] != 0 and self.middleRight[lineFlagI] == 0 and self.upperRight[lineFlagI] != 0:
            LOWUPPERLINE = V1_9_3_utils.get_k_b([self.lowRight[xI], self.lowRight[yI]],
                                                [self.upperRight[xI], self.upperRight[yI]])

            if -0.1 <= LOWUPPERLINE[kI] <= config_parameters.RIGHTLINEK:
                self.rightLine[kI] = LOWUPPERLINE[kI]
                self.rightLine[bI] = LOWUPPERLINE[bI]
                self.rightLine[lineFlagI] = 1
                self.rightLine[lineCntI] = 0
            else:
                self.rightLine[lineCntI] += 1
        # 如果是只有中层和顶层存在
        elif self.lowRight[lineFlagI] == 0 and self.middleRight[lineFlagI] != 0 and self.upperRight[lineFlagI] != 0:
            MIDDLEUPPERLINE = V1_9_3_utils.get_k_b([self.middleRight[xI], self.middleRight[yI]],
                                                   [self.upperRight[xI], self.upperRight[yI]])

            if -0.1 <= MIDDLEUPPERLINE[kI] <= config_parameters.RIGHTLINEK:
                self.rightLine[kI] = MIDDLEUPPERLINE[kI]
                self.rightLine[bI] = MIDDLEUPPERLINE[bI]
                self.rightLine[lineFlagI] = 1
                self.rightLine[lineCntI] = 0
            else:
                self.rightLine[lineCntI] += 1
        else:
            self.rightLine[lineCntI] += 1



    def calculate_filter_boxes_middleLine(self) -> None:
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        if self.lowMiddle[filterBoxFlagI] != 0 and self.middleMiddle[filterBoxFlagI] != 0 and self.upperMiddle[filterBoxFlagI] != 0:
            # 使用最小二乘法拟合跟踪线
            FITLINE = V1_9_3_utils.fit([[self.lowMiddle[xI], self.lowMiddle[yI]], [self.middleMiddle[xI], self.middleMiddle[yI]], [self.upperMiddle[xI], self.upperMiddle[yI]]])

            # 计算两两聚类框组成的各条线
            LOWMIDDLELINE = V1_9_3_utils.get_k_b([self.lowMiddle[xI], self.lowMiddle[yI]], [self.middleMiddle[xI], self.middleMiddle[yI]])
            LOWUPPERLINE = V1_9_3_utils.get_k_b([self.lowMiddle[xI], self.lowMiddle[yI]], [self.upperMiddle[xI], self.upperMiddle[yI]])
            MIDDLEUPPERLINE = V1_9_3_utils.get_k_b([self.middleMiddle[xI], self.middleMiddle[yI]], [self.upperMiddle[xI], self.upperMiddle[yI]])

            tempLINE = [FITLINE, LOWMIDDLELINE, LOWUPPERLINE, MIDDLEUPPERLINE]
            tempK = [abs(FITLINE[kI]), abs(LOWMIDDLELINE[kI]), abs(LOWUPPERLINE[kI]), abs(MIDDLEUPPERLINE[kI])]

            minIndex = V1_9_3_utils.get_min_value(tempK)
            if tempK[minIndex] <= config_parameters.MIDDLELINEK:
                self.middleLine[kI] = tempLINE[minIndex][kI]
                self.middleLine[bI] = tempLINE[minIndex][bI]
                self.middleLine[lineFlagI] = 1
                self.middleLine[lineCntI] = 0
                self.predictLine[lineFlagI] = 0
            else:
                # 当左右两边的线都存在时
                if self.leftLine[lineFlagI] != 0 and self.rightLine[lineFlagI] != 0:
                    # 预测中线
                    KB = V1_9_3_utils.predict_middle_line(self.leftLine[kI], self.leftLine[bI], self.rightLine[kI], self.rightLine[bI])
                    if abs(KB[kI]) <= config_parameters.MIDDLELINEK:
                        self.predictLine[kI] = KB[kI]
                        self.predictLine[bI] = KB[bI]
                        self.predictLine[lineFlagI] = 1
                        self.middleLine[kI] = copy.deepcopy(self.predictLine[kI])
                        self.middleLine[bI] = copy.deepcopy(self.predictLine[bI])
                        self.middleLine[lineFlagI] = 1
                        self.middleLine[lineCntI] = 0
                    else:
                        self.middleLine[lineCntI] += 1
                        self.predictLine[lineFlagI] = 0
                else:
                    self.middleLine[lineCntI] += 1
                    self.predictLine[lineFlagI] = 0


        elif self.lowMiddle[filterBoxFlagI] != 0 and self.middleMiddle[filterBoxFlagI] != 0 and self.upperMiddle[filterBoxFlagI] == 0:
            KB = V1_9_3_utils.get_k_b([self.lowMiddle[xI], self.lowMiddle[yI]], [self.middleMiddle[xI], self.middleMiddle[yI]])
            if abs(KB[kI]) <= config_parameters.MIDDLELINEK:
                self.middleLine[kI] = copy.deepcopy(KB[kI])
                self.middleLine[bI] = copy.deepcopy(KB[bI])
                self.middleLine[lineFlagI] = 1
                self.middleLine[lineCntI] = 0
            else:
                # 当左右两边的线都存在时
                if self.leftLine[lineFlagI] != 0 and self.rightLine[lineFlagI] != 0:
                    # 预测中线
                    KB = V1_9_3_utils.predict_middle_line(self.leftLine[kI], self.leftLine[bI], self.rightLine[kI], self.rightLine[bI])
                    if abs(KB[kI]) <= config_parameters.MIDDLELINEK:
                        self.predictLine[kI] = KB[kI]
                        self.predictLine[bI] = KB[bI]
                        self.predictLine[lineFlagI] = 1
                        self.middleLine[kI] = copy.deepcopy(KB[kI])
                        self.middleLine[bI] = copy.deepcopy(KB[bI])
                        self.middleLine[lineFlagI] = 1
                        self.middleLine[lineCntI] = 0
                    else:
                        self.middleLine[lineCntI] += 1
                        self.predictLine[lineFlagI] = 0
                else:
                    self.middleLine[lineCntI] += 1
                    self.predictLine[lineFlagI] = 0
        elif self.lowMiddle[filterBoxFlagI] != 0 and self.middleMiddle[filterBoxFlagI] == 0 and self.upperMiddle[filterBoxFlagI] != 0:
            KB = V1_9_3_utils.get_k_b([self.lowMiddle[xI], self.lowMiddle[yI]], [self.upperMiddle[xI], self.upperMiddle[yI]])
            if abs(KB[kI]) <= config_parameters.MIDDLELINEK:
                self.middleLine[kI] = copy.deepcopy(KB[kI])
                self.middleLine[bI] = copy.deepcopy(KB[bI])
                self.middleLine[lineFlagI] = 1
                self.middleLine[lineCntI] = 0
            else:
                # 当左右两边的线都存在时
                if self.leftLine[lineFlagI] != 0 and self.rightLine[lineFlagI] != 0:
                    # 预测中线
                    KB = V1_9_3_utils.predict_middle_line(self.leftLine[kI], self.leftLine[bI], self.rightLine[kI], self.rightLine[bI])
                    if abs(KB[kI]) <= config_parameters.MIDDLELINEK:
                        self.predictLine[kI] = KB[kI]
                        self.predictLine[bI] = KB[bI]
                        self.predictLine[lineFlagI] = 1
                        self.middleLine[kI] = copy.deepcopy(self.predictLine[kI])
                        self.middleLine[bI] = copy.deepcopy(self.predictLine[bI])
                        self.middleLine[lineFlagI] = 1
                        self.middleLine[lineCntI] = 0
                    else:
                        self.middleLine[lineCntI] += 1
                        self.predictLine[lineFlagI] = 0
                else:
                    self.middleLine[lineCntI] += 1
                    self.predictLine[lineFlagI] = 0
        elif self.lowMiddle[filterBoxFlagI] == 0 and self.middleMiddle[filterBoxFlagI] != 0 and self.upperMiddle[filterBoxFlagI] != 0:
            KB = V1_9_3_utils.get_k_b([self.middleMiddle[xI], self.middleMiddle[yI]], [self.upperMiddle[xI], self.upperMiddle[yI]])
            if abs(KB[kI]) <= config_parameters.MIDDLELINEK:
                self.middleLine[kI] = copy.deepcopy(KB[kI])
                self.middleLine[bI] = copy.deepcopy(KB[bI])
                self.middleLine[lineFlagI] = 1
                self.middleLine[lineCntI] = 0
            else:
                # 当左右两边的线都存在时
                if self.leftLine[lineFlagI] != 0 and self.rightLine[lineFlagI] != 0:
                    # 预测中线
                    KB = V1_9_3_utils.predict_middle_line(self.leftLine[kI], self.leftLine[bI], self.rightLine[kI], self.rightLine[bI])
                    if abs(KB[kI]) <= config_parameters.MIDDLELINEK:
                        self.predictLine[kI] = KB[kI]
                        self.predictLine[bI] = KB[bI]
                        self.predictLine[lineFlagI] = 1
                        self.middleLine[kI] = copy.deepcopy(self.predictLine[kI])
                        self.middleLine[bI] = copy.deepcopy(self.predictLine[bI])
                        self.middleLine[lineFlagI] = 1
                        self.middleLine[lineCntI] = 0
                    else:
                        self.middleLine[lineCntI] += 1
                        self.predictLine[lineFlagI] = 0
                else:
                    self.middleLine[lineCntI] += 1
                    self.predictLine[lineFlagI] = 0

        else:
            self.middleLine[lineCntI] += 1
            self.predictLine[lineFlagI] = 0



    def reset_absolute_threshold(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        if (config_parameters.LEFTTHRESHOLD > self.lowMiddle[xI]) or (config_parameters.LEFTTHRESHOLD > self.middleMiddle[xI]) or (config_parameters.LEFTTHRESHOLD > self.upperMiddle[xI]) or (config_parameters.RIGHTTHRESHOLD < self.lowMiddle[xI]) or (config_parameters.RIGHTTHRESHOLD < self.middleMiddle[xI]) or (config_parameters.RIGHTTHRESHOLD < self.upperMiddle[xI]):
            self.lowMiddle = np.zeros(7, dtype=np.float32)
            self.lowMiddle[xI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[xI])
            self.lowMiddle[yI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[yI])
            self.lowMiddleKalmanFilter = KalmanFilter()


            self.middleMiddle = np.zeros(7, dtype=np.float32)
            self.middleMiddle[xI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[xI])
            self.middleMiddle[yI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[yI])
            self.middleMiddleKalmanFilter = KalmanFilter()


            self.upperMiddle = np.zeros(7, dtype=np.float32)
            self.upperMiddle[xI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[xI])
            self.upperMiddle[yI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[yI])
            self.upperMiddleKalmanFilter = KalmanFilter()


            self.lowLeft = np.zeros(7, dtype=np.float32)
            self.lowLeft[xI] = copy.deepcopy(config_parameters.lowLeftFilterBox[xI])
            self.lowLeft[yI] = copy.deepcopy(config_parameters.lowLeftFilterBox[yI])
            self.lowLeftKalmanFilter = KalmanFilter()


            self.middleLeft = np.zeros(7, dtype=np.float32)
            self.middleLeft[xI] = copy.deepcopy(config_parameters.middleLeftFilterBox[xI])
            self.middleLeft[yI] = copy.deepcopy(config_parameters.middleLeftFilterBox[yI])
            self.middleLeftKalmanFilter = KalmanFilter()


            self.upperLeft = np.zeros(7, dtype=np.float32)
            self.upperLeft[xI] = copy.deepcopy(config_parameters.upperLeftFilterBox[xI])
            self.upperLeft[yI] = copy.deepcopy(config_parameters.upperLeftFilterBox[yI])
            self.upperLeftKalmanFilter = KalmanFilter()


            self.lowRight = np.zeros(7, dtype=np.float32)
            self.lowRight[xI] = copy.deepcopy(config_parameters.lowRightFilterBox[xI])
            self.lowRight[yI] = copy.deepcopy(config_parameters.lowRightFilterBox[yI])
            self.lowRightKalmanFilter = KalmanFilter()


            self.middleRight = np.zeros(7, dtype=np.float32)
            self.middleRight[xI] = copy.deepcopy(config_parameters.middleRightFilterBox[xI])
            self.middleRight[yI] = copy.deepcopy(config_parameters.middleRightFilterBox[yI])
            self.middleRightKalmanFilter = KalmanFilter()


            self.upperRight = np.zeros(7, dtype=np.float32)
            self.upperRight[xI] = copy.deepcopy(config_parameters.upperRightFilterBox[xI])
            self.upperRight[yI] = copy.deepcopy(config_parameters.upperRightFilterBox[yI])
            self.upperRightKalmanFilter = KalmanFilter()

            self.middleLine = np.zeros(4, dtype=np.float32)

            self.leftLine = np.zeros(4, dtype=np.float32)

            self.rightLine = np.zeros(4, dtype=np.float32)



    def perform_zeroing(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex


        if self.lowMiddle[filterBoxCntI] > config_parameters.zeroingSize:
            self.lowMiddle = np.zeros(7, dtype=np.float32)
            self.lowMiddle[xI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[xI])
            self.lowMiddle[yI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[yI])
            self.lowMiddleKalmanFilter = KalmanFilter()
        if self.middleMiddle[filterBoxCntI] > config_parameters.zeroingSize:
            self.middleMiddle = np.zeros(7, dtype=np.float32)
            self.middleMiddle[xI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[xI])
            self.middleMiddle[yI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[yI])
            self.middleMiddleKalmanFilter = KalmanFilter()
        if self.upperMiddle[filterBoxCntI] > config_parameters.zeroingSize:
            self.upperMiddle = np.zeros(7, dtype=np.float32)
            self.upperMiddle[xI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[xI])
            self.upperMiddle[yI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[yI])
            self.upperMiddleKalmanFilter = KalmanFilter()

        if self.lowLeft[filterBoxCntI] > config_parameters.zeroingSize:
            self.lowLeft = np.zeros(7, dtype=np.float32)
            self.lowLeft[xI] = copy.deepcopy(config_parameters.lowLeftFilterBox[xI])
            self.lowLeft[yI] = copy.deepcopy(config_parameters.lowLeftFilterBox[yI])
            self.lowLeftKalmanFilter = KalmanFilter()
        if self.middleLeft[filterBoxCntI] > config_parameters.zeroingSize:
            self.middleLeft = np.zeros(7, dtype=np.float32)
            self.middleLeft[xI] = copy.deepcopy(config_parameters.middleLeftFilterBox[xI])
            self.middleLeft[yI] = copy.deepcopy(config_parameters.middleLeftFilterBox[yI])
            self.middleLeftKalmanFilter = KalmanFilter()
        if self.upperLeft[filterBoxCntI] > config_parameters.zeroingSize:
            self.upperLeft = np.zeros(7, dtype=np.float32)
            self.upperLeft[xI] = copy.deepcopy(config_parameters.upperLeftFilterBox[xI])
            self.upperLeft[yI] = copy.deepcopy(config_parameters.upperLeftFilterBox[yI])
            self.upperLeftKalmanFilter = KalmanFilter()


        if self.lowRight[filterBoxCntI] > config_parameters.zeroingSize:
            self.lowRight = np.zeros(7, dtype=np.float32)
            self.lowRight[xI] = copy.deepcopy(config_parameters.lowRightFilterBox[xI])
            self.lowRight[yI] = copy.deepcopy(config_parameters.lowRightFilterBox[yI])
            self.lowRightKalmanFilter = KalmanFilter()
        if self.middleRight[filterBoxCntI] > config_parameters.zeroingSize:
            self.middleRight = np.zeros(7, dtype=np.float32)
            self.middleRight[xI] = copy.deepcopy(config_parameters.middleRightFilterBox[xI])
            self.middleRight[yI] = copy.deepcopy(config_parameters.middleRightFilterBox[yI])
            self.middleRightKalmanFilter = KalmanFilter()
        if self.upperRight[filterBoxCntI] > config_parameters.zeroingSize:
            self.upperRight = np.zeros(7, dtype=np.float32)
            self.upperRight[xI] = copy.deepcopy(config_parameters.upperRightFilterBox[xI])
            self.upperRight[yI] = copy.deepcopy(config_parameters.upperRightFilterBox[yI])
            self.upperRightKalmanFilter = KalmanFilter()

        if self.middleLine[lineCntI] > config_parameters.zeroingSize:
            self.middleLine = np.zeros(4, dtype=np.float32)

        if self.leftLine[lineCntI] > config_parameters.zeroingSize:
            self.leftLine = np.zeros(4, dtype=np.float32)

        if self.rightLine[lineCntI] > config_parameters.zeroingSize:
            self.rightLine = np.zeros(4, dtype=np.float32)


    def reset_the_hidden_location(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 先看中间跟踪线
        if self.middleLine[lineFlagI] != 0:
            if self.upperMiddle[filterBoxFlagI] == 0:
                # 重设顶层中间的滤波框
                self.upperMiddle[xI] = int(self.middleLine[kI] * config_parameters.layerUpper + self.middleLine[bI])

            if self.middleMiddle[filterBoxFlagI] == 0:
                # 重设中层中间的滤波框
                self.middleMiddle[xI] = int(self.middleLine[kI] * config_parameters.layerMiddle + self.middleLine[bI])

            if self.lowMiddle[filterBoxFlagI] == 0:
                # 重设底层中间的滤波框
                self.lowMiddle[xI] = int(self.middleLine[kI] * config_parameters.layerLower + self.middleLine[bI])


        if self.leftLine[lineFlagI] != 0:
            if self.upperLeft[filterBoxFlagI] == 0:
                # 重设顶层中间的滤波框
                self.upperLeft[xI] = int(self.leftLine[kI] * config_parameters.layerUpper + self.leftLine[bI])

            if self.middleLeft[filterBoxFlagI] == 0:
                # 重设中层中间的滤波框
                self.middleLeft[xI] = int(self.leftLine[kI] * config_parameters.layerMiddle + self.leftLine[bI])

            if self.lowLeft[filterBoxFlagI] == 0:
                # 重设底层中间的滤波框
                self.lowLeft[xI] = int(self.leftLine[kI] * config_parameters.layerLower + self.leftLine[bI])

        if self.rightLine[lineFlagI] != 0:
            if self.upperRight[filterBoxFlagI] == 0:
                # 重设顶层中间的滤波框
                self.upperRight[xI] = int(self.rightLine[kI] * config_parameters.layerUpper + self.rightLine[bI])

            if self.middleRight[filterBoxFlagI] == 0:
                # 重设中层中间的滤波框
                self.middleRight[xI] = int(self.rightLine[kI] * config_parameters.layerMiddle + self.rightLine[bI])

            if self.lowRight[filterBoxFlagI] == 0:
                # 重设底层中间的滤波框
                self.lowRight[xI] = int(self.rightLine[kI] * config_parameters.layerLower + self.rightLine[bI])


    def line_filter_box_self_check(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # 看中间跟踪线与中间三个滤波框是否符合对应关系，若不符合则对滤波框进行重置
        if self.middleLine[lineFlagI] != 0:
            endPoints = V1_9_3_utils.calculate_endpoints_line(self.middleLine[kI], self.middleLine[bI])
            distance = V1_9_3_utils.calculate_filter_box_and_line_distance(endPoints, [self.lowMiddle[xI], self.lowMiddle[yI]])
            if distance > config_parameters.lowMiddleFilterBoxLineMaxSize:
                self.lowMiddle = np.zeros(7, dtype=np.float32)
                self.lowMiddle[xI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[xI])
                self.lowMiddle[yI] = copy.deepcopy(config_parameters.lowMiddleFilterBox[yI])
                self.lowMiddleKalmanFilter = KalmanFilter()
            endPoints = V1_9_3_utils.calculate_endpoints_line(self.middleLine[kI], self.middleLine[bI])
            distance = V1_9_3_utils.calculate_filter_box_and_line_distance(endPoints, [self.middleMiddle[xI], self.middleMiddle[yI]])
            if distance > config_parameters.middleMiddleFilterBoxLineMaxSize:
                self.middleMiddle = np.zeros(7, dtype=np.float32)
                self.middleMiddle[xI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[xI])
                self.middleMiddle[yI] = copy.deepcopy(config_parameters.middleMiddleFilterBox[yI])
                self.middleMiddleKalmanFilter = KalmanFilter()
            endPoints = V1_9_3_utils.calculate_endpoints_line(self.middleLine[kI], self.middleLine[bI])
            distance = V1_9_3_utils.calculate_filter_box_and_line_distance(endPoints, [self.upperMiddle[xI], self.upperMiddle[yI]])
            if distance > config_parameters.upperMiddleFilterBoxLineMaxSize:
                self.upperMiddle = np.zeros(7, dtype=np.float32)
                self.upperMiddle[xI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[xI])
                self.upperMiddle[yI] = copy.deepcopy(config_parameters.upperMiddleFilterBox[yI])
                self.upperMiddleKalmanFilter = KalmanFilter()


    def position_correction(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        # reset
        if self.lowLeft[xI] >= self.lowMiddle[xI]:
            self.lowLeft = np.zeros(7, dtype=np.float32)
            self.lowLeft[xI] = copy.deepcopy(config_parameters.lowLeftFilterBox[xI])
            self.lowLeft[yI] = copy.deepcopy(config_parameters.lowLeftFilterBox[yI])
            self.lowLeftKalmanFilter = KalmanFilter()

        if self.lowRight[xI] <= self.lowMiddle[xI]:
            self.lowRight = np.zeros(7, dtype=np.float32)
            self.lowRight[xI] = copy.deepcopy(config_parameters.lowRightFilterBox[xI])
            self.lowRight[yI] = copy.deepcopy(config_parameters.lowRightFilterBox[yI])
            self.lowRightKalmanFilter = KalmanFilter()

        if self.middleLeft[xI] >= self.middleMiddle[xI]:
            self.middleLeft = np.zeros(7, dtype=np.float32)
            self.middleLeft[xI] = copy.deepcopy(config_parameters.middleLeftFilterBox[xI])
            self.middleLeft[yI] = copy.deepcopy(config_parameters.middleLeftFilterBox[yI])
            self.middleLeftKalmanFilter = KalmanFilter()

        if self.middleRight[xI] <= self.middleMiddle[xI]:
            self.middleRight = np.zeros(7, dtype=np.float32)
            self.middleRight[xI] = copy.deepcopy(config_parameters.middleRightFilterBox[xI])
            self.middleRight[yI] = copy.deepcopy(config_parameters.middleRightFilterBox[yI])
            self.middleRightKalmanFilter = KalmanFilter()

        if self.upperLeft[xI] >= self.upperMiddle[xI]:
            self.upperLeft = np.zeros(7, dtype=np.float32)
            self.upperLeft[xI] = copy.deepcopy(config_parameters.upperLeftFilterBox[xI])
            self.upperLeft[yI] = copy.deepcopy(config_parameters.upperLeftFilterBox[yI])
            self.upperLeftKalmanFilter = KalmanFilter()

        if self.upperRight[xI] <= self.upperMiddle[xI]:
            self.upperRight = np.zeros(7, dtype=np.float32)
            self.upperRight[xI] = copy.deepcopy(config_parameters.upperRightFilterBox[xI])
            self.upperRight[yI] = copy.deepcopy(config_parameters.upperRightFilterBox[yI])
            self.upperRightKalmanFilter = KalmanFilter()




    def draw(self, image):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        if image is None:
            return

        tempImg = image.copy()
        # 绘制底中的滤波框
        if self.lowMiddle[filterBoxFlagI] != 0 and self.lowMiddle[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.lowMiddle[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.lowMiddle[xI], self.lowMiddle[yI]], (255, 0, 0), 2)
            else:
                centerPoints = [self.lowMiddle[xI], self.lowMiddle[yI], int(self.lowMiddle[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (255, 0, 0))

        # middle middle
        if self.middleMiddle[filterBoxFlagI] != 0 and self.middleMiddle[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.middleMiddle[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.middleMiddle[xI], self.middleMiddle[yI]], (255, 0, 0), 2)
            else:
                centerPoints = [self.middleMiddle[xI], self.middleMiddle[yI], int(self.middleMiddle[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (255, 0, 0))


        # upper middle
        if self.upperMiddle[filterBoxFlagI] != 0 and self.upperMiddle[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.upperMiddle[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.upperMiddle[xI], self.upperMiddle[yI]], (255, 0, 0), 2)
            else:
                centerPoints = [self.upperMiddle[xI], self.upperMiddle[yI], int(self.upperMiddle[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (255, 0, 0))


        # 绘制底left的滤波框
        if self.lowLeft[filterBoxFlagI] != 0 and self.lowLeft[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.lowLeft[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.lowLeft[xI], self.lowLeft[yI]], (0, 255, 255), 2)
            else:
                centerPoints = [self.lowLeft[xI], self.lowLeft[yI], int(self.lowLeft[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (0, 255, 255))

        if self.middleLeft[filterBoxFlagI] != 0 and self.middleLeft[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.middleLeft[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.middleLeft[xI], self.middleLeft[yI]], (0, 255, 255), 2)
            else:
                centerPoints = [self.middleLeft[xI], self.middleLeft[yI], int(self.middleLeft[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (0, 255, 255))


        if self.upperLeft[filterBoxFlagI] != 0 and self.upperLeft[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.upperLeft[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.upperLeft[xI], self.upperLeft[yI]], (0, 255, 255), 2)
            else:
                centerPoints = [self.upperLeft[xI], self.upperLeft[yI], int(self.upperLeft[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (0, 255, 255))



        if self.lowRight[filterBoxFlagI] != 0 and self.lowRight[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.lowRight[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.lowRight[xI], self.lowRight[yI]], (0, 255, 0), 2)
            else:
                centerPoints = [self.lowRight[xI], self.lowRight[yI], int(self.lowRight[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (0, 255, 0))

        if self.middleRight[filterBoxFlagI] != 0 and self.middleRight[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.middleRight[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.middleRight[xI], self.middleRight[yI]], (0, 255, 0), 2)
            else:
                centerPoints = [self.middleRight[xI], self.middleRight[yI], int(self.middleRight[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (0, 255, 0))


        if self.upperRight[filterBoxFlagI] != 0 and self.upperRight[filterBoxCntI] <= config_parameters.zeroingSize:
            if self.upperRight[filterBoxCntI] == 0:
                tempImg = V1_9_3_utils.my_rectangle(tempImg, [self.upperRight[xI], self.upperRight[yI]], (0, 255, 0), 2)
            else:
                centerPoints = [self.upperRight[xI], self.upperRight[yI], int(self.upperRight[filterBoxCntI])]
                tempImg = V1_9_3_utils.my_dotted_rectangle(tempImg, centerPoints, (0, 255, 0))


        # 绘制中间的跟踪线
        if self.predictLine[lineFlagI] != 0 and self.predictLine[kI] != 0 and self.predictLine[bI] != 0:
            endPoints = V1_9_3_utils.calculate_endpoints_line(self.predictLine[kI], self.predictLine[bI])
            cv2.line(tempImg, endPoints[0], endPoints[1], (0, 0, 255), 3)
        elif self.middleLine[lineFlagI] != 0 and self.middleLine[kI] != 0 and self.middleLine[bI] != 0:
            if self.middleLine[lineCntI] == 0:
                endPoints = V1_9_3_utils.calculate_endpoints_line(self.middleLine[kI], self.middleLine[bI])
                cv2.line(tempImg, endPoints[0], endPoints[1], (255, 0, 0), 3)
            else:
                endPoints = V1_9_3_utils.calculate_endpoints_line(self.middleLine[kI], self.middleLine[bI])
                tempImg = V1_9_3_utils.line_cnt_dotted_line(endPoints[0], endPoints[1], tempImg, int(self.middleLine[lineCntI]), (255, 0, 0))


        if self.leftLine[lineFlagI] != 0 and self.leftLine[kI] != 0 and self.leftLine[bI] != 0:
            if self.leftLine[lineCntI] == 0:
                endPoints = V1_9_3_utils.calculate_endpoints_line(self.leftLine[kI], self.leftLine[bI])
                cv2.line(tempImg, endPoints[0], endPoints[1], (0, 255, 255), 3)
            else:
                endPoints = V1_9_3_utils.calculate_endpoints_line(self.leftLine[kI], self.leftLine[bI])
                tempImg = V1_9_3_utils.line_cnt_dotted_line(endPoints[0], endPoints[1], tempImg, int(self.leftLine[lineCntI]), (0, 255, 255))


        if self.rightLine[lineFlagI] != 0 and self.rightLine[kI] != 0 and self.rightLine[bI] != 0:
            if self.rightLine[lineCntI] == 0:
                endPoints = V1_9_3_utils.calculate_endpoints_line(self.rightLine[kI], self.rightLine[bI])
                cv2.line(tempImg, endPoints[0], endPoints[1], (0, 255, 0), 3)
            else:
                endPoints = V1_9_3_utils.calculate_endpoints_line(self.rightLine[kI], self.rightLine[bI])
                tempImg = V1_9_3_utils.line_cnt_dotted_line(endPoints[0], endPoints[1], tempImg, int(self.rightLine[lineCntI]), (0, 255, 0))


        if self.decisionPoint != 0:
            cv2.circle(tempImg, (self.decisionPoint, 200), 10, (0, 0, 255))

        return tempImg


    def calculate_decision_points(self):
        # get index information
        lowIndex = config_parameters.lowIndex
        middleIndex = config_parameters.middleIndex
        upperIndex = config_parameters.upperIndex


        xI = config_parameters.xIndex
        yI = config_parameters.yIndex
        filterBoxFlagI = config_parameters.filterBoxFlag
        filterBoxCntI = config_parameters.filterBoxCnt
        filterBoxConfidenceI = config_parameters.filterBoxConfidence
        successInitialFlagI = config_parameters.successInitialFlag
        successUpdateFlagI = config_parameters.successUpdateFlag

        kI = config_parameters.lineKIndex
        bI = config_parameters.lineBIndex
        lineFlagI = config_parameters.lineFlagIndex
        lineCntI = config_parameters.lineCntIndex

        self.decisionPoint = 0
        if self.middleLine[lineFlagI] != 0:
            self.decisionPoint = int(self.middleLine[kI] * 200 + self.middleLine[bI]) + 100
        elif self.predictLine[lineFlagI] != 0:
            self.decisionPoint = int(self.predictLine[kI] * 200 + self.predictLine[bI]) + 100

