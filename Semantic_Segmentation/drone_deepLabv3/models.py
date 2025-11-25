
"""
    训练所需要的模型架构文件
"""
import segmentation_models_pytorch as smp
import torch
import config_parameters
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# TODO: 直接调用库进行模型的构建
SimpleDeepLabV3 = smp.DeepLabV3(
    encoder_name='resnet101',    # 预训练架构
    encoder_weights='imagenet',     # 与训练权重
    in_channels=3,      # 输入通道数
    classes=config_parameters.NUM_CLASSES,
    activation=None,  # 不使用激活函数，直接输出logits   (即直接使用最终输出的概率logits，不需要再经过softmax等的函数计算概率，在分类任务中后面直接接着交叉熵损失，效果是不错的)
).to(device)



DeepLabV3Plus = smp.DeepLabV3Plus(
    encoder_name='resnet152',       # 'resnet152'  'resnext50_32x4d'     'resnext101_32x8d'     'resnext101_32x16d'  'efficientnet-b7' 'se_resnext101_32x4d' 'se_resnet152'
    encoder_weights='imagenet',
    in_channels=3,
    classes=config_parameters.NUM_CLASSES,
    activation=None
).to(device)

