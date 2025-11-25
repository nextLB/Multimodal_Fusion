"""
    针对于无人机数据集的构建
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



trainTransform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        # 添加填充操作，使图像尺寸达到(, )
        A.PadIfNeeded(
            min_height=config_parameters.HEIGHT,  # 最小高度
            min_width=config_parameters.WIDTH,  # 最小宽度
            border_mode=0,  # 填充模式，0表示用0填充
            value=0,  # 填充值，0表示黑色像素
            mask_value=0,  # 掩码填充值为0（背景）
            position="center"  # 居中填充
        ),
        # 对图像按照一定的均值和方差进行归一化处理，便于后续的计算操作
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        # 将图像数据转化为pytorch上的张量
        ToTensorV2(),
    ]
)

valTransform = A.Compose(
    [
        # 添加填充操作，使图像尺寸达到(, )
        A.PadIfNeeded(
            min_height=config_parameters.HEIGHT,  # 最小高度
            min_width=config_parameters.WIDTH,  # 最小宽度
            border_mode=0,  # 填充模式，0表示用0填充
            value=0,  # 填充值，0表示黑色像素
            mask_value=0,  # 掩码填充值为0（背景）
            position="center"  # 居中填充
        ),
        # 对图像按照一定的均值和方差进行归一化处理，便于后续的计算操作
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        # 将图像数据转化为pytorch上的张量
        ToTensorV2(),
    ]
)



# 实现数据集加载类
class DroneDataset(Dataset):
    def __init__(self, imageDir, maskDir, transform):
        self.image_dir = imageDir
        self.mask_dir = maskDir
        self.transform = transform

        # 获取图像和掩码文件列表
        self.images = sorted(
            [f for f in os.listdir(imageDir) if f.startswith("images_") and f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])  # 按数字部分排序
        )
        self.masks = sorted(
            [f for f in os.listdir(maskDir) if f.startswith("images_") and f.endswith("_mask.png")],
            key=lambda x: int(x.split("_")[1])  # 按数字部分排序
        )

        # 验证图像和掩码数量匹配
        if len(self.images) != len(self.masks):
            raise ValueError(f"图像数量({len(self.images)})与掩码数量({len(self.masks)})不匹配")

        # 验证文件名对应关系
        for img, msk in zip(self.images, self.masks):
            img_num = img.split("_")[1].split(".")[0]
            msk_num = msk.split("_")[1]
            if img_num != msk_num:
                raise ValueError(f"图像 {img} 与掩码 {msk} 编号不匹配")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        # 确保图像和掩码都存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # 读取图像和掩码
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)  # 保持原始类别值

        # 应用变换
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]


        return image, mask.long()







def main():
    imagesPath = os.path.join(config_path.DRONE_DATASETS, 'images')
    masksPath = os.path.join(config_path.DRONE_DATASETS, 'masks')
    fullDatasets = DroneDataset(imagesPath, masksPath, trainTransform)

    # 划分索引，整理出训练集与验证集
    indices = list(range(len(fullDatasets)))
    tranIndices, valIndices = train_test_split(
        indices,
        test_size=config_parameters.RATIO,  # 作为验证集的比例
        random_state=42,    # 设定随机数种子 使得每次分配的整体集合是一致的
        shuffle=True        # 设置是否打乱
    )

    tranDatasets = Subset(
        DroneDataset(
            imagesPath,
            masksPath,
            trainTransform
        ),
        tranIndices
    )

    valDatasets = Subset(
        DroneDataset(
            imagesPath,
            masksPath,
            valTransform
        ),
        valIndices
    )

    # 创建数据集加载器
    trainDataLoader = DataLoader(
        tranDatasets,
        batch_size=config_parameters.BATCH_SIZE,
        shuffle=True,
        num_workers=config_parameters.NUM_WORKERS
    )

    valDataLoader = DataLoader(
        valDatasets,
        batch_size=config_parameters.BATCH_SIZE,
        shuffle=False,
        num_workers=config_parameters.NUM_WORKERS
    )

    # 创建一下可视化存储路径
    os.makedirs(config_path.SAVE_TRANSFORM_DATASETS, exist_ok=True)
    utils.clear_folder(config_path.SAVE_TRANSFORM_DATASETS)
    os.makedirs(config_path.SAVE_TRANSFORM_DATASETS, exist_ok=True)
    # # 可视化一下加载的数据集
    # with tqdm(total=len(trainDataLoader)+len(valDataLoader), desc="数据集可视化中") as pbarDataloader:
    #
    #     for batchIndex, (datas, targets) in enumerate(trainDataLoader):
    #         for i in range(config_parameters.BATCH_SIZE):
    #             utils.save_transform_datasets(datas[i], targets[i], config_path.SAVE_TRANSFORM_DATASETS, 'trainData', batchIndex, i, 0)
    #         pbarDataloader.update(1)
    #
    #     for batchIndex, (datas, targets) in enumerate(valDataLoader):
    #         for i in range(config_parameters.BATCH_SIZE):
    #             utils.save_transform_datasets(datas[i], targets[i], config_path.SAVE_TRANSFORM_DATASETS, 'valData', batchIndex, i, 0)
    #         pbarDataloader.update(1)

    return trainDataLoader, valDataLoader




if __name__ == '__main__':
    main()











