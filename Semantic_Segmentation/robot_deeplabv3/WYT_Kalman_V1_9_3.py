# algo ver  V1.1.0 hongyi track line calculation algo updated
#           V1.2.0 line track algo changed by Willian
#           V1.2.1 - 1.2.2 update based 6.18-01 test result
#           V1.2.5 - hongyi line algo update
# v1.2.6 - Willian select a new line by average K, not last K
# v1.2.7 - Willian, line dist cal para changed
# v1.2.8 - hongyi update with new model
# v1.2.9 - willian fix distance calculation bug, 2024.06.21
# v1.3.0 - add track line calculation, hongyi,   2024.06.21
# v1.5.0 -
# v1.5.1 - the line dist thre change from 50 to 80
#          and select the middle line immediately if it's stable and in middle
# v1.5.2 - enhance the final select mechanism
# v1.5.3 - if turn left but line tilt to right, the wait; same to right
# v1.5.5 - fit to 60degree camera
# v1.6.0 - search from bottom line, and use cross point as lost-function to judge
#        - Organized by next      2025.1.16
#        - Organized and updated by next         2015.1.21
#        - Organized and updated by next         2025.1.24
# v1.7.0 - Fixed some cases where blue lines appeared and disappeared out of compliance, ---next
#        - Fixed a bug where some forecast line non-conformities were occurring, -----next
# v1.7.1 - It mainly solves the problem that the left, middle and right tracking red line cannot be searched, ----next
# v1.7.2 - It mainly solves some related problems when the middle line is maintained,
#          predicted, and the two sides of the hold line are displayed, ----next
# v1.7.3 -
# v1.8.0 - The update algorithm of the clustering box and some restrictions have been adjusted, ----next
# v1.8.1 - Trace lines and prediction lines are visualized, and a framework is added that allows for comparative experiments, ----next
# v1.8.2 - 主要去解决了在跟踪时总是出现串行的问题, ----next
# V1.8.3 - 对于一系列的算法细节部分做了大量的修正，并且增加了图像的畸变矫正以及新的计算决策点的操作,并且还加了很多的规则和调整  ------next
# V1.9.0 - 对于算法流程进行了初步整理, -----next
# V1.9.1 - 对于算法的测试基于新的数据集结合之前的版本增添了许多聚类和聚类、线和聚类、线和线之间加了很多细节操作, --------next
# V1.9.2 - 对于算法的更新数据部分，加上了一个卡尔曼滤波器，实时结合观测数据与先前状态值的信息进行计算滤波框,  ----------next
# V1.9.3 - 对于算法的细节部分，增加了平等的置信度可信机制,  ------------next
# V1.9.4 -



import V1_9_3_implementation

V193_Algorithm = V1_9_3_implementation.V193_Mower_Track()


def ulfd_find_line(image, S_S_Points):

    if V193_Algorithm.COUNTER != 0:
        # TODO: 计算决策点
        V193_Algorithm.calculate_decision_points()

        image = V193_Algorithm.draw(image)
        V193_Algorithm.COUNTER += 1
    else:

        # -----------------------------------------------------------------------#
        # ---------------------------  算法的预处理过程  ---------------------------#
        # -----------------------------------------------------------------------#
        # TODO: 执行识别框的分层
        V193_Algorithm.layering(S_S_Points)

        # TODO: 如果此时跟踪线存在，其对应的滤波框不存在，且此时同一层的其它滤波框符合要求时，根据跟踪线重置一下对应的滤波框的隐藏位置
        V193_Algorithm.reset_the_hidden_location()

        # TODO: 在处理新一轮的识别框之前，尽可能保证同一层的滤波框和滤波框之间不会具有识别框
        V193_Algorithm.boxes_in_between_filter_box()


        # -----------------------------------------------------------------------#
        # ---------------------------  算法的执行过程  ---------------------------#
        # -----------------------------------------------------------------------#

        # TODO: 对底层中间的滤波框执行算法
        V193_Algorithm.calculate_low_middle()

        # TODO: 对中层中间的滤波框执行算法
        V193_Algorithm.calculate_middle_middle()

        # TODO: 对顶层中间的滤波框执行算法
        V193_Algorithm.calculate_upper_middle()

        # TODO: 对底层左侧的滤波框执行算法
        V193_Algorithm.calculate_low_left()
        
        # TODO: 对中层左侧的滤波框执行算法
        V193_Algorithm.calculate_middle_left()
        
        # TODO： 对顶层左侧的滤波框执行算法
        V193_Algorithm.calculate_upper_left()

        # TODO： 对底层右侧的滤波框执行算法
        V193_Algorithm.calculate_low_right()
        
        # TODO： 对中层右侧的滤波框执行算法
        V193_Algorithm.calculate_middle_right()
        
        # TODO： 对顶层右侧的滤波框执行算法
        V193_Algorithm.calculate_upper_right()


        # -----------------------------------------------------------------------#
        # ---------------------------  算法的后处理过程  ---------------------------#
        # -----------------------------------------------------------------------#

        # TODO： 要确保左边的滤波框在中间滤波框的左边，右边的滤波框在中间滤波框的右边
        V193_Algorithm.position_correction()

        # TODO： LSTM


        # TODO: 进行绝对阈值的判断与归零操作
        V193_Algorithm.reset_absolute_threshold()

        # TODO: 按条件执行归零操作
        V193_Algorithm.perform_zeroing()

        # TODO: (直接就选用最小二乘法计算出来的线)根据左侧滤波框计算跟踪线
        V193_Algorithm.calculate_filter_boxes_leftLine()

        # TODO: (直接就选用最小二乘法计算出来的线)根据右侧滤波框计算跟踪线
        V193_Algorithm.calculate_filter_boxes_rightLine()

        # TODO: 根据中间滤波框计算跟踪线
        V193_Algorithm.calculate_filter_boxes_middleLine()

        # TODO: 对本轮计算出的跟踪线与滤波框进行自检
        V193_Algorithm.line_filter_box_self_check()

        # TODO: 计算决策点
        V193_Algorithm.calculate_decision_points()

        # TODO: 存储本次算法数据到历史数据总体集合中


        # TODO: 画图，保存图片和数据等
        image = V193_Algorithm.draw(image)

    if V193_Algorithm.COUNTER > 20:
       V193_Algorithm.COUNTER = 0



    return image, (V193_Algorithm.decisionPoint)


def new_ulfd_find_line(image, S_S_Points):
    if image is None:
        V193_Algorithm.__init__()
        return None, None

    else:
        image, XLine = ulfd_find_line(image, S_S_Points)
        # print(ufldRes)
        return image, XLine



