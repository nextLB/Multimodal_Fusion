
"""
    将images与对应的masks文件重新命名排序为自己想要的数字顺序
"""


import os
import cv2



def rename_images_masks(imagesPath, masksPath, outImagesDir, outMasksDir, number):
    # 获取图像和掩码文件列表
    images = sorted(
        [f for f in os.listdir(imagesPath) if f.startswith("images_") and f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0])  # 按数字部分排序
    )

    masks = sorted(
        [f for f in os.listdir(masksPath) if f.startswith("images_") and f.endswith("_mask.png")],
        key=lambda x: int(x.split("_")[1])  # 按数字部分排序
    )

    # 验证图像和掩码数量匹配
    if len(images) != len(masks):
        raise ValueError(f"图像数量({len(images)})与掩码数量({len(masks)})不匹配")


    # 验证文件名对应关系
    for img, msk in zip(images, masks):
        old_image = cv2.imread(os.path.join(imagesPath, img))
        old_mask = cv2.imread(os.path.join(masksPath, msk), cv2.IMREAD_UNCHANGED)  # 保持原始通道数


        new_img_name = f"images_{number}.png"
        new_img_path = os.path.join(outImagesDir, new_img_name)
        new_mask_name = f"images_{number}_mask.png"
        new_mask_path = os.path.join(outMasksDir, new_mask_name)

        cv2.imwrite(new_img_path, old_image)
        cv2.imwrite(new_mask_path, old_mask)
        number += 1



def main():
    imagesPath = './raw_images'
    masksPath = './raw_masks'
    outImagesDir = './rename_images'
    outMasksDir = './rename_masks'
    number = 1315
    rename_images_masks(imagesPath, masksPath, outImagesDir, outMasksDir, number)



if __name__ == '__main__':
    main()

