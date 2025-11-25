"""
    日志文件的构建程序
"""

import logging
import os


# 配置日志
def setup_logging(savePath):
    logFile = os.path.join(savePath, 'training.log')

    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建文件处理器
    fileHandler = logging.FileHandler(logFile, mode='w')
    fileHandler.setLevel(logging.INFO)

    # 创建控制台处理器
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    return logger






