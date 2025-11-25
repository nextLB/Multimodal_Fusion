"""
    模型的训练程序
"""

import log
from feature_maps import MyFeatureMapHook
import torch
from torch.utils.tensorboard import SummaryWriter
import utils
import ConvNeXtV2_DeepLabV3Plus


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 创建各训练结果的存储路径
    saveModelPath, saveLogFilePath, saveFeatureMapsPath = utils.create_all_path()
    logger = log.setup_logging(saveLogFilePath)
    # SummaryWriter是使用pytorch的TensorBoard集成功能
    # 在终端运行 tensorboard --logdir=runs 然后浏览器打开 http://localhost:6006
    writer = SummaryWriter()    # 默认创建 runs/当前时间 目录

    ConvNeXtV2_DeepLabV3Plus.main(saveModelPath, saveLogFilePath, saveFeatureMapsPath, logger, writer, MyFeatureMapHook)







if __name__ == '__main__':
    main()
