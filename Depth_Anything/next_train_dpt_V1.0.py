"""
    关于深度估计模型的主程序文件
"""
# V1.0      --- by next, 关于深度估计模型的训练和推理的初步构建

import torch
import NYUV2Datasets
import next_dpt_models
import os
import logging
import dpt_loss
import torch.optim as optim
import CONFIG
from tqdm import tqdm
import matplotlib.pyplot as plt


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def setupLogging():
    """配置日志"""
    logDir = './log'
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logDir, 'dptTrain.log'), mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# 可视化结果函数
def visualize_results(model, val_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        rgb_images, depth_maps = next(iter(val_loader))
        rgb_images = rgb_images.to(device)
        depth_maps = depth_maps.to(device)

        # 预测
        pred_depth = model(rgb_images)

        # 转换为numpy用于可视化
        rgb_np = rgb_images.cpu().numpy()[0].transpose(1, 2, 0)
        true_depth_np = depth_maps.cpu().numpy()[0, 0]
        pred_depth_np = pred_depth.cpu().numpy()[0, 0]

        # 创建可视化图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RGB图像
        axes[0].imshow(rgb_np)
        axes[0].set_title('Input RGB Image')
        axes[0].axis('off')

        # 真实深度图
        im1 = axes[1].imshow(true_depth_np, cmap='plasma')
        axes[1].set_title('Ground Truth Depth')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # 预测深度图
        im2 = axes[2].imshow(pred_depth_np, cmap='plasma')
        axes[2].set_title('Predicted Depth')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.suptitle(f'Epoch {epoch} - Depth Estimation Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'./visual/results_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # 创建日志类
    logger = setupLogging()
    logger.info('本次运行的训练模型的算法版本为: V1.0')

    # 获取数据集数据
    trainLoader, valLoader, testLoader = NYUV2Datasets.main()
    logger.info('数据集加载完毕')

    # 加载与获取其模型架构
    dptModel = next_dpt_models.main('train')
    logger.info('模型加载完毕')

    # 初始化损失函数
    criterion = dpt_loss.DepthLossV1_0()
    # 初始化优化器
    optimizer = optim.Adam(dptModel.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    # 初始化学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG.STEP_SIZE, gamma=0.1)
    logger.info('损失函数、优化器与学习率调度器初始化完毕')

    # 转移到设备上
    os.makedirs("./V1.0_dpt_models/", exist_ok=True)
    os.makedirs("./visual/", exist_ok=True)
    dptModel = dptModel.to(device)
    trainLosses = []
    valLosses = []
    with tqdm(total=CONFIG.MAX_EPOCHS) as trainBar:
        for epoch in range(CONFIG.MAX_EPOCHS):
            dptModel.train()
            runningLoss = 0.0
            batchCount = 0
            for batchIndex, (rgbImages, depthMaps) in enumerate(trainLoader):
                # 转移到设备上
                rgbImages = rgbImages.to(device)
                depthMaps = depthMaps.to(device)

                # 前向传播
                outputs = dptModel(rgbImages)
                loss = criterion(outputs, depthMaps)

                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()
                batchCount += 1

                if batchIndex % 100 == 0:
                    avgLossSoFar = runningLoss / batchCount
                    logger.info(f'Epoch [{epoch+1}/{CONFIG.MAX_EPOCHS}], Step [{batchIndex+1}/{len(trainLoader)}], Avg Loss: {avgLossSoFar:.6f}')

            epochTrainLoss = runningLoss / len(trainLoader)
            trainLosses.append(epochTrainLoss)

            # 验证阶段
            dptModel.eval()
            valLoss = 0.0
            with torch.no_grad():
                for rgbImages, depthMaps in valLoader:
                    rgbImages = rgbImages.to(device)
                    depthMaps = depthMaps.to(device)
                    outputs = dptModel(rgbImages)
                    loss = criterion(outputs, depthMaps)
                    valLoss += loss.item()

            epochValLoss = valLoss / len(valLoader)
            valLosses.append(epochValLoss)
            logger.info(f'Epoch [{epoch + 1}/{CONFIG.MAX_EPOCHS}], Train Loss: {epochTrainLoss:.6f}, Val Loss: {epochValLoss:.6f}')
            # 每个epoch保存一次模型
            torch.save(dptModel.state_dict(), f'./V1.0_dpt_models/depth_model_epoch_{epoch + 1}.pth')

            # 每隔一定的epoch可视化一次结果
            if (epoch + 1) % 3 == 0:
                visualize_results(dptModel, valLoader, device, epoch + 1)

            scheduler.step()
            trainBar.update(1)

    # 训练完毕后绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(trainLosses, label='Training Loss')
    plt.plot(valLosses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./V1.0_dpt_models/training_loss.png', dpi=300, bbox_inches='tight')

    # 保存最终模型
    torch.save(dptModel.state_dict(), './V1.0_dpt_models/final_depth_model.pth')



def main_1():
    # 创建日志类
    logger = setupLogging()
    logger.info('本次运行的训练模型的算法版本为: V1.0')

    # 获取数据集数据
    trainLoader, valLoader, testLoader = NYUV2Datasets.main()
    logger.info('数据集加载完毕')

    # 加载与获取其模型架构
    dptModel = next_dpt_models.main_1('train')
    logger.info('模型加载完毕')

    # 初始化损失函数
    # criterion = dpt_loss.DepthLossV1_0()
    criterion = dpt_loss.ImprovedDepthLoss(alpha=0.5,    # L1损失权重
                                            beta=0.3,     # 梯度损失权重
                                            gamma=0.1,    # SSIM损失权重
                                            delta=0.1     # 尺度不变损失权重
                                            )

    # 初始化优化器
    optimizer = optim.Adam(dptModel.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    # 初始化学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG.STEP_SIZE, gamma=0.1)
    logger.info('损失函数、优化器与学习率调度器初始化完毕')

    # 转移到设备上
    os.makedirs("./V1.0_dpt_models/", exist_ok=True)
    os.makedirs("./visual/", exist_ok=True)
    dptModel = dptModel.to(device)
    trainLosses = []
    valLosses = []
    with tqdm(total=CONFIG.MAX_EPOCHS) as trainBar:
        for epoch in range(CONFIG.MAX_EPOCHS):
            dptModel.train()
            runningLoss = 0.0
            batchCount = 0
            for batchIndex, (rgbImages, depthMaps) in enumerate(trainLoader):
                # 转移到设备上
                rgbImages = rgbImages.to(device)
                depthMaps = depthMaps.to(device)

                # 前向传播
                outputs = dptModel(rgbImages)
                loss = criterion(outputs, depthMaps)

                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()
                batchCount += 1

                if batchIndex % 100 == 0:
                    avgLossSoFar = runningLoss / batchCount
                    logger.info(f'Epoch [{epoch+1}/{CONFIG.MAX_EPOCHS}], Step [{batchIndex+1}/{len(trainLoader)}], Avg Loss: {avgLossSoFar:.6f}')

            epochTrainLoss = runningLoss / len(trainLoader)
            trainLosses.append(epochTrainLoss)

            # 验证阶段
            dptModel.eval()
            valLoss = 0.0
            with torch.no_grad():
                for rgbImages, depthMaps in valLoader:
                    rgbImages = rgbImages.to(device)
                    depthMaps = depthMaps.to(device)
                    outputs = dptModel(rgbImages)
                    loss = criterion(outputs, depthMaps)
                    valLoss += loss.item()

            epochValLoss = valLoss / len(valLoader)
            valLosses.append(epochValLoss)
            logger.info(f'Epoch [{epoch + 1}/{CONFIG.MAX_EPOCHS}], Train Loss: {epochTrainLoss:.6f}, Val Loss: {epochValLoss:.6f}')
            # 每个epoch保存一次模型
            torch.save(dptModel.state_dict(), f'./V1.0_dpt_models/depth_model_epoch_{epoch + 1}.pth')

            # 每隔一定的epoch可视化一次结果
            if (epoch + 1) % 3 == 0:
                visualize_results(dptModel, valLoader, device, epoch + 1)

            scheduler.step()
            trainBar.update(1)

    # 训练完毕后绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(trainLosses, label='Training Loss')
    plt.plot(valLosses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./V1.0_dpt_models/training_loss.png', dpi=300, bbox_inches='tight')

    # 保存最终模型
    torch.save(dptModel.state_dict(), './V1.0_dpt_models/final_depth_model.pth')



if __name__ == '__main__':
    # main()
    main_1()





