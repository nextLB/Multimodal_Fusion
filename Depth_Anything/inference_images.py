"""
    调用训练的模型进行推理图像可视化的程序文件
"""
import os
import torch
import torch.nn as nn
import next_dpt_models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import CONFIG
import NYUV2Datasets



# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 推理和可视化函数
def inference_and_visualize_V1_0(modelPath, outputDir):

    # 加载模型
    model = next_dpt_models.main('inference')

    # 加载模型权重
    model.load_state_dict(torch.load(modelPath, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    print(f"Loaded model from {modelPath}")
    # 获取数据集数据
    trainLoader, valLoader, testLoader = NYUV2Datasets.main()

    # 进行推理和可视化
    with torch.no_grad():
        sample_count = 0  # 总样本计数器
        for batchIndex, (rgbImages, depthMaps) in enumerate(testLoader):
            rgbImages = rgbImages.to(device)
            predDepth = model(rgbImages)

            # 获取批次大小
            batch_size = rgbImages.size(0)

            # 对批次中的每一张图进行可视化
            for i in range(batch_size):
                # 转换为numpy数组
                rgbNp = rgbImages[i].cpu().numpy().transpose(1, 2, 0)
                trueDepthNp = depthMaps[i].cpu().numpy()[0]  # 移除通道维度
                predDepthNp = predDepth[i].cpu().numpy()[0]  # 移除通道维度

                # 创建可视化图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # RGB图像
                axes[0].imshow(rgbNp)
                axes[0].set_title('Input RGB Image', fontsize=14)
                axes[0].axis('off')

                # 真实深度图
                im1 = axes[1].imshow(trueDepthNp, cmap='plasma')
                axes[1].set_title('Ground Truth Depth', fontsize=14)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                # 预测深度图
                im2 = axes[2].imshow(predDepthNp, cmap='plasma')
                axes[2].set_title('Predicted Depth', fontsize=14)
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046)

                plt.suptitle(f'Sample {sample_count + 1} - Depth Estimation Results', fontsize=16)
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)

                # 保存结果
                output_path = os.path.join(outputDir, f'result_{sample_count + 1:03d}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

                sample_count += 1
                print(f"Saved result {sample_count} to {output_path}")

        print(f"All results saved to {outputDir}")
        print(f"Successfully processed {sample_count} samples")







def main():
    inferenceOutputDir = './inference_results'
    os.makedirs(inferenceOutputDir, exist_ok=True)
    modelPath = './V1.0_dpt_models/depth_model_epoch_50.pth'
    inference_and_visualize_V1_0(modelPath, inferenceOutputDir)





if __name__ == '__main__':
    main()







