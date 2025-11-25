
"""
    数据集处理程序文件
"""


# 导入程序运行所需要的相关库
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import config_parameters
import config_path
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import utils



# 定义训练数据增强的策略
trainTransform = A.Compose(
    [
        # 1. 几何变换类
        A.OneOf([
            A.Rotate(limit=15, p=0.5),  # 随机旋转±15度，保持图像完整性
            A.ShiftScaleRotate(  # 组合变换：平移+缩放+旋转
                shift_limit=0.1,  # 最大平移比例（10%）
                scale_limit=0.1,  # 随机缩放比例±10%
                rotate_limit=15,  # 旋转角度限制
                border_mode=0,  # 边界填充模式（0=常数填充）
                p=0.5
            ),
        ], p=0.5),


        # 2. 色彩空间变换
        A.OneOf([
            A.HueSaturationValue(  # 色相/饱和度/明度调整
                hue_shift_limit=20,  # 色相偏移限制
                sat_shift_limit=30,  # 饱和度偏移限制
                val_shift_limit=20,  # 明度偏移限制
                p=0.7
            ),
            A.CLAHE(clip_limit=3.0, p=0.5),  # 限制对比度自适应直方图均衡化
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # 伽马校正
        ], p=0.5),


        # 3. 空间扭曲变换
        A.OneOf([
            A.ElasticTransform(  # 弹性变换（模拟非刚性形变）
                alpha=1,  # 变形强度
                sigma=50,  # 平滑系数
                alpha_affine=50,  # 仿射变换系数
                border_mode=0,
                p=0.3
            ),
            A.GridDistortion(  # 网格扭曲
                num_steps=5,  # 网格步数
                distort_limit=0.3,  # 扭曲程度
                border_mode=0,
                p=0.3
            ),
            A.OpticalDistortion(  # 光学扭曲（类似透镜效果）
                distort_limit=0.3,  # 扭曲程度
                shift_limit=0.1,  # 偏移程度
                border_mode=0,
                p=0.3
            ),
        ], p=0.5),


        # 4. 遮挡类增强（提升抗遮挡能力）
        A.OneOf([
            A.CoarseDropout(  # 随机矩形遮挡
                max_holes=8,  # 最大遮挡区域数
                max_height=32,  # 遮挡最大高度
                max_width=32,  # 遮挡最大宽度
                min_holes=1,  # 最小遮挡区域数
                fill_value=0,  # 遮挡填充值
                p=0.5
            ),
            A.RandomRain(  # 模拟雨滴（适用于户外场景）
                slant_lower=-10,  # 雨滴倾斜角度
                slant_upper=10,
                drop_length=20,  # 雨滴长度
                drop_width=1,  # 雨滴宽度
                drop_color=(200, 200, 200),  # 雨滴颜色
                blur_value=2,  # 模糊程度
                brightness_coefficient=0.9,  # 亮度系数
                p=0.1
            ),
        ], p=0.4),


        # 5. 天气效果增强（适用于真实场景）
        A.RandomFog(  # 模拟雾效
            fog_coef_lower=0.3,  # 雾浓度下限
            fog_coef_upper=0.5,  # 雾浓度上限
            alpha_coef=0.1,  # 透明度系数
            p=0.1
        ),



        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.PadIfNeeded(
            min_height=config_parameters.HEIGHT,
            min_width=config_parameters.WIDTH,
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


# 定义验证数据增强的策略
valTransform = A.Compose(
    [
        A.PadIfNeeded(
            min_height=config_parameters.HEIGHT,
            min_width=config_parameters.WIDTH,
            border_mode=0,  # 填充模式，0表示用0填充
            value=0,  # 填充值，0表示黑色像素
            mask_value=0,  # 掩码填充值为0（背景）
            position="center"  # 居中填充
        ),
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
class RobotDataset(Dataset):
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
    imagesPath = os.path.join(config_path.ROBOT_DATASETS, 'images')
    masksPath = os.path.join(config_path.ROBOT_DATASETS, 'masks')
    fullDatasets = RobotDataset(imagesPath, masksPath, trainTransform)

    # 划分索引，整理出训练集与验证集
    indices = list(range(len(fullDatasets)))
    tranIndices, valIndices = train_test_split(
        indices,
        test_size=config_parameters.RATIO,  # 作为验证集的比例
        random_state=42,    # 设定随机数种子 使得每次分配的整体集合是一致的
        shuffle=True        # 设置是否打乱
    )

    tranDatasets = Subset(
        RobotDataset(
            imagesPath,
            masksPath,
            trainTransform
        ),
        tranIndices
    )

    valDatasets = Subset(
        RobotDataset(
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





