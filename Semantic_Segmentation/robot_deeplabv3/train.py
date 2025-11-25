

"""
    模型训练的程序
"""

import datasets
import log
import utils
import sys
import traceback
import torch.nn as nn
import models
import torch
import torch.optim as optim
import gc
import config_parameters
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from feature_maps import MyFeatureMapHook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 在这里定义一下要可视化的特征层
targetLayers = [
    'identity_1',
    'identity_2',
    'identity_3',
    'identity_4',
    'identity_5'

]


def main():

    # 创建各结果存储路径
    saveModelPath, saveLogFilePath, saveFeatureMapsPath = utils.create_all_path()
    logger = log.setup_logging(saveLogFilePath)
    try:
        logger.info('开始加载本次训练的数据集')
        # 加载数据集
        trainDataLoader, valDataLoader = datasets.main()
        logger.info('加载数据集完毕')



        logger.info('定义本次训练所用的损失函数')
        # 定义损失函数
        crossLossFunction = nn.CrossEntropyLoss()
        logger.info('定义损失函数完毕')



        logger.info('定义本次训练所使用的模型')
        # deepLabV3Model = models.SimpleDeepLabV3
        deepLabV3Model = models.MobileNetV2DeepLabV3().to(device)
        logger.info('模型定义完毕')



        logger.info('定义本次训练所用的优化器')
        optimizer = optim.AdamW(
            deepLabV3Model.parameters(),
            lr=config_parameters.LEARNING_RATE,
            weight_decay=config_parameters.WEIGHT_DECAY,
            betas=(0.9, 0.999)      # 通常保持默认值即可
        )
        logger.info('定义优化器完毕')



        logger.info('定义本次训练所使用的学习率调度器')
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config_parameters.MAX_EPOCHS,  # 半个余弦周期的长度，通常设为总epoch数
            eta_min=1e-5  # 最小学习率，通常是初始学习率的 1/1000 或 1/100
        )
        logger.info('学习率调度器定义完毕')



        logger.info('定义梯度裁剪器')
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        logger.info('定义梯度裁剪器完毕')


        # ======================================================== #
        # 开始训练
        bestValLoss = float('inf')
        bestIou = 0.0
        trainLosses = []
        valLosses = []
        valIous = []
        for epoch in range(config_parameters.MAX_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config_parameters.MAX_EPOCHS}")
            trainLoop = tqdm(trainDataLoader, desc="training")
            deepLabV3Model.train()
            totalLoss = 0

            # TODO: 训练阶段
            for batchIdx, (data, targets) in enumerate(trainLoop):
                data = data.to(device)
                targets = targets.to(device)

                # TODO: 注册特征层
                if epoch % 10 == 0 and batchIdx % 100 == 0:
                    # initial feature hook
                    hookHandler = MyFeatureMapHook(deepLabV3Model, outputDir=f"{saveFeatureMapsPath}/epoch_{epoch}_batchIndex_{batchIdx}", imgIndex=0)
                    hookHandler.register_hooks(targetLayers)

                # 前向传播(使用混合精度)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    predictions = deepLabV3Model(data)
                    loss = crossLossFunction(predictions, targets)

                # TODO: 保存特征层
                if epoch % 10 == 0 and batchIdx % 100 == 0:
                    # save feature maps
                    hookHandler.save_feature_maps()
                    hookHandler.remove_hooks()


                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                totalLoss += loss.item()

                # 更新进度条
                trainLoop.set_postfix(train_loss=loss.item())

            totalLoss /= len(trainDataLoader)
            trainLosses.append(totalLoss)
            print(f"训练损失: {totalLoss:.4f}")



            # TODO: 验证阶段
            deepLabV3Model.eval()
            totalLoss = 0
            totalIou = 0
            with torch.no_grad():
                valLoop = tqdm(valDataLoader, desc='valing')
                for batchIdx, (data, targets) in enumerate(valLoop):
                    data = data.to(device)
                    targets = targets.to(device)
                    # 前向传播
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        predictions = deepLabV3Model(data)
                        loss = crossLossFunction(predictions, targets)

                    totalLoss += loss.item()
                    iou = utils.calculate_iou(predictions, targets, config_parameters.NUM_CLASSES)
                    totalIou += iou.item()
                    valLoop.set_postfix(val_loss=loss.item(), val_iou=iou.item())


            totalLoss /= len(valDataLoader)
            totalIou /= len(valDataLoader)

            # 更新学习率
            scheduler.step()

            valLosses.append(totalLoss)
            valIous.append(totalIou)
            print(f"验证损失: {totalLoss:.4f}, IoU: {totalIou:.4f}")

            # 保存最佳模型
            if totalLoss < bestValLoss:
                bestValLoss = totalLoss
                torch.save(deepLabV3Model.state_dict(), os.path.join(saveModelPath, "deepLabV3_low_loss.pth"))
                print(f"模型已保存至 {saveModelPath}")

            # 保存最佳模型
            if totalIou > bestIou:
                bestIou = totalIou
                torch.save(deepLabV3Model.state_dict(), os.path.join(saveModelPath, "deepLabV3_hight_iou.pth"))
                print(f"模型已保存至 {saveModelPath}")

            # 绘制训练曲线
            utils.draw_train_picture(trainLosses, valLosses, valIous, saveModelPath)


        del deepLabV3Model
        gc.collect()
        torch.cuda.empty_cache()



    except Exception as e:
        logger.error(f"训练过程中发生未预期的错误: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)



if __name__ == '__main__':
    main()



