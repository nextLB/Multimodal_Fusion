
"""
    本项目的路径配置
"""

# 无人机训练数据集的路径
DRONE_DATASETS = '/home/next_lb/桌面/WYT_S_S/data/drone_data/320_datasets'

# 数据增强后可视化的存储路径
SAVE_TRANSFORM_DATASETS = '/home/next_lb/桌面/WYT_S_S/code/drone_deepLabv3/results/transform_datasets'

# 模型保存的路径
MODEL_PATH = '/home/next_lb/桌面/WYT_S_S/code/drone_deepLabv3/results/pytorch_models'

# 训练过程log的保存路径
LOG_FILE_PATH = '/home/next_lb/桌面/WYT_S_S/code/drone_deepLabv3/results/pytorch_models'

# 训练特征图的保存路径
FEATURE_MAPS_PATH = '/home/next_lb/桌面/WYT_S_S/code/drone_deepLabv3/feature_maps'

# 推理图像所用的图像所在路径
INFERENCE_IMAGES_PATH = '../../data/drone_data/inference_datasets/'

# 推理轮廓的存储路径
INFERENCE_CONTOURS_PATH = './results/contours'

# 推理的图像结果存储路径
INFERENCE_RESULT_PATH = './results/inference_images_result/'

# 使用的推理模型的路径
USE_INFERENCE_MODEL_PATH = './results/pytorch_models/maxEpochs_100_learningRate_0.001/deepLabV3_low_loss.pth'


