"""
    深度估计模型的各参数配置文件
"""

import os

# 数据集的根目录路径
DATA_DIR_PATH = '/home/next_lb/桌面/next/nyu_data/data/'

# csv数据文件
TRAIN_CSV_PATH = os.path.join(DATA_DIR_PATH, 'nyu2_train.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR_PATH, 'nyu2_test.csv')

# 可视化数据集存储路径
VISUAL_DATASETS = './visual'


# 训练图像尺寸大小
IMAGE_SIZE = (480, 640)
BATCH_SIZE = 4
NUMBER_WORKERS = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.000001
STEP_SIZE = 20
MAX_EPOCHS = 80



