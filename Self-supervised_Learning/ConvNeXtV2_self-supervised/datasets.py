
"""
    基于ConvNeXtV2自监督数据集的加载程序文件
"""




from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import numpy as np
import os
import config_parameters
import config_path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import utils
from tqdm import tqdm
import cv2


# 训练集的数据增强
trainTransform = A.Compose(
    [
        A.RandomResizedCrop(config_parameters.PRETRAINED_RESIZED_HEIGHT, config_parameters.PRETRAINED_RESIZED_WIDTH,
                                scale=config_parameters.CROPS_SCALE, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5),  # 作用是对图像的颜色属性（亮度、对比度、饱和度、色调）进行随机调整
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=0.1),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()

    ])


# 验证集的数据增强
valTransform = A.Compose(
    [
        A.RandomResizedCrop(config_parameters.PRETRAINED_RESIZED_HEIGHT, config_parameters.PRETRAINED_RESIZED_WIDTH,
                                scale=config_parameters.CROPS_SCALE, interpolation=cv2.INTER_CUBIC),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()

    ])




# 预训练时的数据加载类
class PretrainedDataset(Dataset):
    def __init__(self, rootDir, transform):
        self.root_dir = rootDir
        self.transform = transform
        self.image_paths = []

        self.baseTransform = A.Compose([
        A.Resize(config_parameters.PRETRAINED_RESIZED_HEIGHT,
                 config_parameters.PRETRAINED_RESIZED_WIDTH,
                 interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for subdir, _, files in os.walk(rootDir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    self.image_paths.append(os.path.join(subdir, file))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {rootDir}")

        print(f"Found {len(self.image_paths)} images in dataset")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 使用 OpenCV 读取图像 (Albumentations 推荐方式)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        originalImage = image.copy()

        originalImage = self.baseTransform(image=image)["image"]
        augmentationImage = self.transform(image=image)["image"]


        return originalImage, augmentationImage





def main():
    fullDatasets = PretrainedDataset(config_path.SELF_SUPERVISED_DATA_PATH, trainTransform)

    fullDataLoader = DataLoader(
        fullDatasets,
        batch_size=config_parameters.BATCH_SIZE,
        shuffle=True,
        num_workers=config_parameters.NUM_WORKERS
    )


    # 创建一下可视化存储路径
    os.makedirs(config_path.SAVE_TRANSFORM_DATASETS, exist_ok=True)
    utils.clear_folder(config_path.SAVE_TRANSFORM_DATASETS)
    os.makedirs(config_path.SAVE_TRANSFORM_DATASETS, exist_ok=True)
    # # # 可视化一下加载的数据集
    # with tqdm(total=len(fullDataLoader), desc="数据集可视化中") as pbarDataloader:
    #     for batchIndex, (originalImage, augmentationImage) in enumerate(fullDataLoader):
    #         for i in range(config_parameters.BATCH_SIZE):
    #             utils.save_transform_datasets(originalImage[i], augmentationImage[i], config_path.SAVE_TRANSFORM_DATASETS, 'trainData', batchIndex, i, 0)
    #         pbarDataloader.update(1)


    return fullDataLoader




if __name__ == '__main__':
    main()





