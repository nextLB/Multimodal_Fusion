
"""
    切分无人机拍摄的大号图像
"""

import os
import cv2
import math
import re
import shutil
from typing import List

RAW_DRONE_IMAGES_PATH = '/home/next_lb/桌面/WYT_S_S/code/tool_program/split_drone_pictures/raw_big_images'
SAVE_SPLIT_IMAGES_PATH = '/home/next_lb/桌面/WYT_S_S/code/tool_program/split_drone_pictures/split_small_images'

SPLIT_IMAGE_HEIGHT = 960
SPLIT_IMAGE_WIDTH = 960


def clear_folder(folderPath: str):

    # 遍历文件夹内的所有内容
    for item in os.listdir(folderPath):
        itemPath = os.path.join(folderPath, item)
        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(itemPath) or os.path.islink(itemPath):
                os.unlink(itemPath)
                print(f'已删除文件夹: {itemPath}')
            elif os.path.isdir(itemPath):
                # 使用 shutil.rmtree 删除文件夹及其内容
                shutil.rmtree(itemPath)
                print(f"已删除文件夹： {itemPath}")

        except Exception as e:
            print(f"删除 {itemPath} 时出错： {e}")

    print(f"文件夹 {folderPath} 内容已清空")



def main():
    images = sorted(
            [f for f in os.listdir(RAW_DRONE_IMAGES_PATH) if f.startswith("area_") and f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])  # 按数字部分排序
        )

    clear_folder(SAVE_SPLIT_IMAGES_PATH)
    os.makedirs(SAVE_SPLIT_IMAGES_PATH, exist_ok=True)

    NUMBER = 0


    for i in range(len(images)):
        imagePath = os.path.join(RAW_DRONE_IMAGES_PATH, images[i])
        image = cv2.imread(imagePath)
        imageHeight, imageWidth = image.shape[:2]

        HCount = math.floor(imageHeight / SPLIT_IMAGE_HEIGHT)
        WCount = math.floor(imageWidth / SPLIT_IMAGE_WIDTH)
        for j in range(HCount + 1):
            for k in range(WCount + 1):
                if j != HCount and k != WCount:
                    tempImage = image[j * SPLIT_IMAGE_HEIGHT:(j + 1) * SPLIT_IMAGE_HEIGHT,
                                k * SPLIT_IMAGE_WIDTH:(k + 1) * SPLIT_IMAGE_WIDTH]
                    if tempImage is not None:
                        # 进一步检查数组是否非空
                        if tempImage.size > 0:
                            saveImagePath = os.path.join(SAVE_SPLIT_IMAGES_PATH,
                                                         'images_' + str(NUMBER) + '.png')
                            cv2.imwrite(saveImagePath, tempImage)
                            NUMBER += 1
                elif j == HCount and k != WCount:
                    tempImage = image[j * SPLIT_IMAGE_HEIGHT:, k * SPLIT_IMAGE_WIDTH:(k + 1) * SPLIT_IMAGE_WIDTH]
                    if tempImage is not None:
                        # 进一步检查数组是否非空
                        if tempImage.size > 0:
                            saveImagePath = os.path.join(SAVE_SPLIT_IMAGES_PATH,
                                                         'images_' + str(NUMBER) + '.png')
                            cv2.imwrite(saveImagePath, tempImage)
                            NUMBER += 1

                elif j != HCount and k == WCount:
                    tempImage = image[j * SPLIT_IMAGE_HEIGHT:(j + 1) * SPLIT_IMAGE_HEIGHT, k * SPLIT_IMAGE_WIDTH:]
                    if tempImage is not None:
                        # 进一步检查数组是否非空
                        if tempImage.size > 0:
                            saveImagePath = os.path.join(SAVE_SPLIT_IMAGES_PATH,
                                                         'images_' + str(NUMBER) + '.png')
                            cv2.imwrite(saveImagePath, tempImage)
                            NUMBER += 1

                elif j == HCount and k == WCount:
                    tempImage = image[j * SPLIT_IMAGE_HEIGHT:, k * SPLIT_IMAGE_WIDTH:]
                    if tempImage is not None:
                        # 进一步检查数组是否非空
                        if tempImage.size > 0:
                            saveImagePath = os.path.join(SAVE_SPLIT_IMAGES_PATH,
                                                         'images_' + str(NUMBER) + '.png')
                            cv2.imwrite(saveImagePath, tempImage)
                            NUMBER += 1






if __name__ == '__main__':
    main()