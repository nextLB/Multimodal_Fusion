
"""
    基于ConvNeXtV2的自监督训练程序
"""

import datasets
import log
import utils
import sys
import traceback
import torch
import torch.optim as optim
import gc
import config_parameters
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import models
from feature_maps import MyFeatureMapHook


# 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 在这里定义一下要可视化的特征层
targetLayers = [
    'encoderBeforeIdentity',
    'encoderAfterIdentity',
    'decoderAfterIdentity'
]


def train_one_epoch(model, optimizer, device, epoch, lrScheduler, scaler, trainLoop, saveFeatureMapsPath):
    count = 0
    totalLoss = 0

    for batchIdx, (originalImage, augmentationImage) in enumerate(trainLoop):
        originalImage = originalImage.to(device)
        augmentationImage = augmentationImage.to(device)


        # TODO: 注册特征层
        if epoch % 10 == 0 and batchIdx % 100 == 0:
            # initial feature hook
            hookHandler = MyFeatureMapHook(model,
                                           outputDir=f"{saveFeatureMapsPath}/epoch_{epoch}_batchIndex_{batchIdx}",
                                           imgIndex=0)
            hookHandler.register_hooks(targetLayers)


        # 使用自动混合精度
        with torch.amp.autocast(device.type if device.type != 'mps' else 'cpu', enabled=(device.type == 'cuda')):
            loss, pred, mask = model(augmentationImage, epoch, batchIdx, saveFeatureMapsPath)


        # TODO: 保存特征层
        if epoch % 10 == 0 and batchIdx % 100 == 0:
            # save feature maps
            hookHandler.save_feature_maps()
            hookHandler.remove_hooks()


        # 使用scaler进行反向传播
        scaler.scale(loss).backward()
        totalLoss += loss.item()

        # 使用scaler更新优化器
        scaler.step(optimizer)
        scaler.update()

        # 清零梯度
        optimizer.zero_grad()

        # 更新进度条
        trainLoop.set_postfix(train_loss=loss.item())
        count += 1

    # 更新学习率
    lrScheduler.step()
    totalLoss /= count

    return totalLoss


def main():

    # 创建各结果存储路径
    saveModelPath, saveLogFilePath, saveFeatureMapsPath = utils.create_all_path()
    logger = log.setup_logging(saveLogFilePath)
    try:
        logger.info('开始加载本次训练的数据集')
        # 加载数据集
        fullDataLoader = datasets.main()
        logger.info('加载数据集完毕')


        logger.info('定义本次训练所使用的模型')
        pretrainModel = models.get_model()
        logger.info('模型定义完毕')
        logger.info(f'本次加载的模型架构如下: {pretrainModel}')


        logger.info('定义本次训练所用的优化器')
        optimizer = optim.AdamW(
            pretrainModel.parameters(),
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
        trainLosses = []
        valLosses = []
        valIous = []
        bestTrainLoss = float('inf')
        for epoch in range(config_parameters.MAX_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config_parameters.MAX_EPOCHS}")
            trainLoop = tqdm(fullDataLoader, desc="training")
            pretrainModel.train()

            trainLoss = train_one_epoch(pretrainModel,
                            optimizer,
                            device,
                            epoch,
                            scheduler,
                            scaler,
                            trainLoop,
                            saveFeatureMapsPath)


            # 美化损失输出
            print("\n" + "=" * 60)
            print(f"📊 Epoch {epoch + 1}/{config_parameters.MAX_EPOCHS} - Training Results:")
            print(f"   ➤ Average Loss: {trainLoss:.6f}")
            print(f"   ➤ Best Loss So Far: {bestTrainLoss:.6f}")



            # 保存最佳模型
            if trainLoss < bestTrainLoss:
                bestTrainLoss = trainLoss
                torch.save(pretrainModel.state_dict(), os.path.join(saveModelPath, "deepLabV3_low_loss.pth"))
                print(f"模型已保存至 {saveModelPath}")


            # 绘制训练曲线
            utils.draw_train_picture(trainLosses, valLosses, valIous, saveModelPath)


        del pretrainModel
        gc.collect()
        torch.cuda.empty_cache()



    except Exception as e:
        logger.error(f"训练过程中发生未预期的错误: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)



if __name__ == '__main__':
    main()







