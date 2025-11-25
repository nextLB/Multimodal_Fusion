
"""
    整个项目的参数配置文件
"""

# 进行训练和推理时图像的尺寸
HEIGHT = 288
WIDTH = 288
NUM_CLASSES = 3



# 划分训练集与验证集的比例
RATIO = 0.3


# 训练过程的参数配置
BATCH_SIZE = 32
NUM_WORKERS = 32
MAX_EPOCHS = 110
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.05





# 跟踪算法的参数配置
layerUpper = 94  # The ordinate of the top-level detection box
layerMiddle = 154  # The ordinate of the middle layer detection frame
layerLower = 214  # The ordinate of the bottom-most detection frame

xIndex = 0
yIndex = 1
filterBoxFlag = 2
filterBoxCnt = 3
filterBoxConfidence = 4
successInitialFlag = 5
successUpdateFlag = 6

lowIndex = 0
middleIndex = 1
upperIndex = 2


lowMiddleFilterBox = [140, layerLower]
lowMiddleInitialSize = 70
lowMiddleFilterSize = 40
lowMiddleLinePressureFrame = 30
lowMiddleFilterBoxLineMaxSize = 40

middleMiddleFilterBox = [140, layerMiddle]
middleMiddleInitialSize = 40
middleMiddleFilterSize = 40
middleMiddleLinePressureFrame = 30
middleMiddleFilterBoxLineMaxSize = 40

upperMiddleFilterBox = [140, layerUpper]
upperMiddleInitialSize = 20
upperMiddleFilterSize = 10
upperMiddleLinePressureFrame = 30
upperMiddleFilterBoxLineMaxSize = 30

lowLeftFilterBox = [60, layerLower]
lowLeftInitialSize = 40
lowLeftFilterSize = 30
lowLeftLinePressureFrame = 30
lowLeftFilterBoxLineMaxSize = 40

middleLeftFilterBox = [70, layerMiddle]
middleLeftInitialSize = 40
middleLeftFilterSize = 30
middleLeftLinePressureFrame = 30
middleLeftFilterBoxLineMaxSize = 40

upperLeftFilterBox = [80, layerUpper]
upperLeftInitialSize = 20
upperLeftFilterSize = 10
upperLeftLinePressureFrame = 30
upperLeftFilterBoxLineMaxSize = 30


lowRightFilterBox = [220, layerLower]
lowRightInitialSize = 40
lowRightFilterSize = 30
lowRightLinePressureFrame = 30
lowRightFilterBoxLineMaxSize = 40

middleRightFilterBox = [210, layerMiddle]
middleRightInitialSize = 40
middleRightFilterSize = 30
middleRightLinePressureFrame = 30
middleRightFilterBoxLineMaxSize = 50

upperRightFilterBox = [190, layerUpper]
upperRightInitialSize = 20
upperRightFilterSize = 10
upperRightLinePressureFrame = 30
upperRightFilterBoxLineMaxSize = 30


lineKIndex = 0
lineBIndex = 1
lineFlagIndex = 2
lineCntIndex = 3


RIGHTLINEK = 0.9
LEFTLINEK = -0.9
MIDDLELINEK = 0.5



zeroingSize = 8


longEdgeLength = 30
shortEdgeLength = 30

scaleFactor = 1


LEFTTHRESHOLD = 70
RIGHTTHRESHOLD = 210


