"""
    对于无人机图像进行模型推理的程序文件
"""

import config_parameters
import config_path
import utils
import os
import torch
from PIL import Image
import numpy as np
import math
import datasets
import models
from typing import List, Tuple
import cv2
import json

Image.MAX_IMAGE_PIXELS = config_parameters.MAX_IMAGE_PIXELS

InputPixelPoint = List[int]
InputPixelPath = List[InputPixelPoint]
InputPixelPaths = List[InputPixelPath]

PixelPoint = Tuple[int, int]
PixelPath = List[PixelPoint]
PixelPaths = List[PixelPath]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_all_contours_picture(finalHeight: int, finalWidth: int, allRidgePixelResults: list[list[list[int, int]]], allRiceRowPixelResults: list[list[list[int, int]]], allGravePixelResults: list[list[list[int, int]]],allPolePixelResults: list[list[list[int, int]]], nameList: list[str]) -> None:
    allCanvas = np.zeros((finalHeight, finalWidth, 3), dtype=np.uint8)


    # TODO: 水稻



    # for i in range(len(allRiceRowPixelResults)):
    #     # 遍历每个轮廓的点集合
    #     for j in range(len(allRiceRowPixelResults[i])):
    #         # 关键修正：直接使用 (x,y) 二元组构建数组，无需取point[0]
    #         points = np.array(allRiceRowPixelResults[i][j], dtype=np.int32)
    #         # 此时points形状为 (N, 2)（N是该轮廓的点数），reshape为 (-1, 1, 2) 符合OpenCV要求
    #         points = points.reshape((-1, 1, 2))
    #
    #         # 后续逻辑不变
    #         x, y, w, h = cv2.boundingRect(points)
    #         # cv2.rectangle(ridgeCanvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         cv2.polylines(allCanvas, [points], isClosed=False, color=(0, 255, 0), thickness=2)
    #
    #         # writeRiceRowContoursImage = cv2.resize(allCanvas, (config_parameters.RESIZED_IMAGE_WIDTH, config_parameters.RESIZED_IMAGE_HEIGHT))
    #
    #         # savePath = os.path.join(config_path.CONTOURS_SAVE_PATH, f'{nameList[0]}_contours_{i}_{j}_rice_row.png')
    #         # cv2.imwrite(savePath, riceRowCanvas)
    #         # cv2.imwrite(savePath, writeRiceRowContoursImage)


    # TODO: 田埂


    for i in range(len(allRidgePixelResults)):
        # 遍历每个轮廓的点集合
        for j in range(len(allRidgePixelResults[i])):
            # 关键修正：直接使用 (x,y) 二元组构建数组，无需取point[0]
            points = np.array(allRidgePixelResults[i][j], dtype=np.int32)
            # 此时points形状为 (N, 2)（N是该轮廓的点数），reshape为 (-1, 1, 2) 符合OpenCV要求
            points = points.reshape((-1, 1, 2))

            # 后续逻辑不变
            x, y, w, h = cv2.boundingRect(points)
            # cv2.rectangle(ridgeCanvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.polylines(allCanvas, [points], isClosed=False, color=(0, 0, 255), thickness=2)

            # writeRiceRowContoursImage = cv2.resize(allCanvas, (config_parameters.RESIZED_IMAGE_WIDTH, config_parameters.RESIZED_IMAGE_HEIGHT))

            # savePath = os.path.join(config_path.CONTOURS_SAVE_PATH, f'{nameList[0]}_contours_{i}_{j}_rice_row.png')
            # cv2.imwrite(savePath, riceRowCanvas)
            # cv2.imwrite(savePath, writeRiceRowContoursImage)


    # TODO: 坟头

    for i in range(len(allGravePixelResults)):
        # 遍历每个轮廓的点集合
        for j in range(len(allGravePixelResults[i])):
            # 关键修正：直接使用 (x,y) 二元组构建数组，无需取point[0]
            points = np.array(allGravePixelResults[i][j], dtype=np.int32)
            # 此时points形状为 (N, 2)（N是该轮廓的点数），reshape为 (-1, 1, 2) 符合OpenCV要求
            points = points.reshape((-1, 1, 2))

            # 后续逻辑不变
            x, y, w, h = cv2.boundingRect(points)
            # cv2.rectangle(ridgeCanvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.polylines(allCanvas, [points], isClosed=False, color=(255, 0, 0), thickness=2)

            # writeRiceRowContoursImage = cv2.resize(allCanvas, (config_parameters.RESIZED_IMAGE_WIDTH, config_parameters.RESIZED_IMAGE_HEIGHT))

            # savePath = os.path.join(config_path.CONTOURS_SAVE_PATH, f'{nameList[0]}_contours_{i}_{j}_rice_row.png')
            # cv2.imwrite(savePath, riceRowCanvas)
            # cv2.imwrite(savePath, writeRiceRowContoursImage)


    # TODO: 电线杆或其它障碍物

    # for i in range(len(allPolePixelResults)):
    #     # 遍历每个轮廓的点集合
    #     for j in range(len(allPolePixelResults[i])):
    #         # 关键修正：直接使用 (x,y) 二元组构建数组，无需取point[0]
    #         points = np.array(allPolePixelResults[i][j], dtype=np.int32)
    #         # 此时points形状为 (N, 2)（N是该轮廓的点数），reshape为 (-1, 1, 2) 符合OpenCV要求
    #         points = points.reshape((-1, 1, 2))
    #
    #         # 后续逻辑不变
    #         x, y, w, h = cv2.boundingRect(points)
    #         # cv2.rectangle(ridgeCanvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         cv2.polylines(allCanvas, [points], isClosed=False, color=(255, 0, 255), thickness=2)
    #
    #         # writeRiceRowContoursImage = cv2.resize(allCanvas, (config_parameters.RESIZED_IMAGE_WIDTH, config_parameters.RESIZED_IMAGE_HEIGHT))
    #
    #         # savePath = os.path.join(config_path.CONTOURS_SAVE_PATH, f'{nameList[0]}_contours_{i}_{j}_rice_row.png')
    #         # cv2.imwrite(savePath, riceRowCanvas)
    #         # cv2.imwrite(savePath, writeRiceRowContoursImage)


    # writeAllContoursImage = cv2.resize(allCanvas, (config_parameters.RESIZED_IMAGE_WIDTH, config_parameters.RESIZED_IMAGE_HEIGHT))
    savePath = os.path.join(config_path.INFERENCE_CONTOURS_PATH, f'{nameList[0]}_contours_all.png')
    cv2.imwrite(savePath, allCanvas)
    # cv2.imwrite(savePath, writeAllContoursImage)

    return





def main():
    # set paths and clear datas
    saveInferenceImagesPath, saveInferenceContoursPath = utils.inference_create_all_path()

    deepLabV3Model = models.DeepLabV3Plus
    deepLabV3Model.load_state_dict(torch.load(config_path.USE_INFERENCE_MODEL_PATH, map_location=device, weights_only=True))
    deepLabV3Model.eval()

    # sort read images path
    sortedImages = utils.sort_images_by_number(config_path.INFERENCE_IMAGES_PATH)

    NUMBER = 0

    # ======================================================================= #
    for imageName in sortedImages:
        if not imageName.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # get image
        imagePath = os.path.join(config_path.INFERENCE_IMAGES_PATH, imageName)
        print(f"处理图像: {imagePath}")

        # 读取原始图像
        originalImage = np.array(Image.open(imagePath).convert("RGB"))
        originalHeight, originalWidth = originalImage.shape[:2]
        print(f"originalHeight: {originalHeight}, originalWidth: {originalWidth}")

        HCount = math.floor(originalHeight / config_parameters.HEIGHT)
        WCount = math.floor(originalWidth / config_parameters.WIDTH)


        # 初始化一系列需要用的变量
        allRiceRowPixelResults = []
        allRidgePixelResults = []
        allGravePixelResults = []
        allPolePixelResults = []
        allPicture = []


        for i in range(HCount + 1):
            horizontalResult = None
            for j in range(WCount + 1):



                ################################################          1       #######################################################
                if i != HCount and j != WCount:
                    tempImage = originalImage[
                                i * config_parameters.HEIGHT:(i + 1) * config_parameters.HEIGHT,
                                j * config_parameters.WIDTH:(j + 1) * config_parameters.WIDTH]
                    tempHeight, tempWidth = tempImage.shape[:2]

                    # 预处理
                    padded = utils.padTransform(image=tempImage)
                    tempImagePadded = padded["image"]
                    transformed = datasets.valTransform(image=tempImagePadded)
                    inputTensor = transformed["image"].unsqueeze(0).to(device)

                    # 模型推理
                    with torch.no_grad():
                        output = deepLabV3Model(inputTensor)
                        predMask = torch.argmax(output, dim=1)
                        predMask = predMask.squeeze(0).cpu().numpy().astype(np.uint8)

                    contourImage, riceRowContour, ridgeContour, graveContour, poleContour = utils.drone_draw(tempImagePadded, predMask, config_parameters.WIDTH,
                                                               config_parameters.HEIGHT,
                                                               config_path.INFERENCE_RESULT_PATH, NUMBER)
                    NUMBER += 1

                    for m in range(len(riceRowContour)):
                        for n in range(len(riceRowContour[m])):
                            riceRowContour[m][n][0] += j * config_parameters.WIDTH
                            riceRowContour[m][n][1] += i * config_parameters.HEIGHT

                    for m in range(len(ridgeContour)):
                        for n in range(len(ridgeContour[m])):
                            ridgeContour[m][n][0] += j * config_parameters.WIDTH
                            ridgeContour[m][n][1] += i * config_parameters.HEIGHT

                    for m in range(len(graveContour)):
                        for n in range(len(graveContour[m])):
                            graveContour[m][n][0] += j * config_parameters.WIDTH
                            graveContour[m][n][1] += i * config_parameters.HEIGHT

                    for m in range(len(poleContour)):
                        for n in range(len(poleContour[m])):
                            poleContour[m][n][0] += j * config_parameters.WIDTH
                            poleContour[m][n][1] += i * config_parameters.HEIGHT

                    allRiceRowPixelResults.append(riceRowContour)
                    allRidgePixelResults.append(ridgeContour)
                    allGravePixelResults.append(graveContour)
                    allPolePixelResults.append(poleContour)

                    # 第一次迭代直接赋值，后续进行堆叠
                    if horizontalResult is None:
                        horizontalResult = contourImage
                    else:
                        horizontalResult = np.hstack((horizontalResult, contourImage))



                ############################################         2            #############################################################
                elif i == HCount and j != WCount:
                    tempImage = originalImage[i * config_parameters.HEIGHT:,
                                j * config_parameters.WIDTH:(j + 1) * config_parameters.WIDTH]
                    tempHeight, tempWidth = tempImage.shape[:2]


                    # 预处理
                    padded = utils.padTransform(image=tempImage)
                    tempImagePadded = padded["image"]
                    transformed = datasets.valTransform(image=tempImagePadded)
                    inputTensor = transformed["image"].unsqueeze(0).to(device)

                    # 模型推理
                    with torch.no_grad():
                        output = deepLabV3Model(inputTensor)
                        predMask = torch.argmax(output, dim=1)
                        predMask = predMask.squeeze(0).cpu().numpy().astype(np.uint8)

                    contourImage, riceRowContour, ridgeContour, graveContour, poleContour = utils.drone_draw(tempImagePadded, predMask, config_parameters.WIDTH,
                                                               config_parameters.HEIGHT,
                                                               config_path.INFERENCE_RESULT_PATH, NUMBER)
                    NUMBER += 1

                    for m in range(len(riceRowContour)):
                        for n in range(len(riceRowContour[m])):
                            riceRowContour[m][n][0] += j * int(tempWidth)
                            riceRowContour[m][n][1] += i * config_parameters.HEIGHT

                    for m in range(len(ridgeContour)):
                        for n in range(len(ridgeContour[m])):
                            ridgeContour[m][n][0] += j * int(tempWidth)
                            ridgeContour[m][n][1] += i * config_parameters.HEIGHT

                    for m in range(len(graveContour)):
                        for n in range(len(graveContour[m])):
                            graveContour[m][n][0] += j * int(tempWidth)
                            graveContour[m][n][1] += i * config_parameters.HEIGHT

                    for m in range(len(poleContour)):
                        for n in range(len(poleContour[m])):
                            poleContour[m][n][0] += j * int(tempWidth)
                            poleContour[m][n][1] += i * config_parameters.HEIGHT


                    allRiceRowPixelResults.append(riceRowContour)
                    allRidgePixelResults.append(ridgeContour)
                    allGravePixelResults.append(graveContour)
                    allPolePixelResults.append(poleContour)

                    # 第一次迭代直接赋值，后续进行堆叠
                    if horizontalResult is None:
                        horizontalResult = contourImage
                    else:
                        horizontalResult = np.hstack((horizontalResult, contourImage))



                ###########################################         3          ##########################################################
                elif i != HCount and j == WCount:
                    tempImage = originalImage[
                                i * config_parameters.HEIGHT:(i + 1) * config_parameters.HEIGHT,
                                j * config_parameters.WIDTH:]
                    tempHeight, tempWidth = tempImage.shape[:2]


                    # 预处理
                    padded = utils.padTransform(image=tempImage)
                    tempImagePadded = padded["image"]
                    transformed = datasets.valTransform(image=tempImagePadded)
                    inputTensor = transformed["image"].unsqueeze(0).to(device)

                    # 模型推理
                    with torch.no_grad():
                        output = deepLabV3Model(inputTensor)
                        predMask = torch.argmax(output, dim=1)
                        predMask = predMask.squeeze(0).cpu().numpy().astype(np.uint8)

                    contourImage, riceRowContour, ridgeContour, graveContour, poleContour = utils.drone_draw(tempImagePadded, predMask, config_parameters.WIDTH,
                                                               config_parameters.HEIGHT,
                                                               config_path.INFERENCE_RESULT_PATH, NUMBER)
                    NUMBER += 1

                    for m in range(len(riceRowContour)):
                        for n in range(len(riceRowContour[m])):
                            riceRowContour[m][n][0] += j * config_parameters.WIDTH
                            riceRowContour[m][n][1] += i * int(tempHeight)

                    for m in range(len(ridgeContour)):
                        for n in range(len(ridgeContour[m])):
                            ridgeContour[m][n][0] += j * config_parameters.WIDTH
                            ridgeContour[m][n][1] += i * int(tempHeight)


                    for m in range(len(graveContour)):
                        for n in range(len(graveContour[m])):
                            graveContour[m][n][0] += j * config_parameters.WIDTH
                            graveContour[m][n][1] += i * int(tempHeight)

                    for m in range(len(poleContour)):
                        for n in range(len(poleContour[m])):
                            poleContour[m][n][0] += j * config_parameters.WIDTH
                            poleContour[m][n][1] += i * int(tempHeight)

                    allRiceRowPixelResults.append(riceRowContour)
                    allRidgePixelResults.append(ridgeContour)
                    allGravePixelResults.append(graveContour)
                    allPolePixelResults.append(poleContour)



                    # 第一次迭代直接赋值，后续进行堆叠
                    if horizontalResult is None:
                        horizontalResult = contourImage
                    else:
                        horizontalResult = np.hstack((horizontalResult, contourImage))


                ##########################################        4        #################################################################
                elif i == HCount and j == WCount:
                    tempImage = originalImage[i * config_parameters.HEIGHT:, j * config_parameters.WIDTH:]
                    tempHeight, tempWidth = tempImage.shape[:2]



                    # 预处理
                    padded = utils.padTransform(image=tempImage)
                    tempImagePadded = padded["image"]
                    transformed = datasets.valTransform(image=tempImagePadded)
                    inputTensor = transformed["image"].unsqueeze(0).to(device)

                    # 模型推理
                    with torch.no_grad():
                        output = deepLabV3Model(inputTensor)
                        predMask = torch.argmax(output, dim=1)
                        predMask = predMask.squeeze(0).cpu().numpy().astype(np.uint8)

                    contourImage, riceRowContour, ridgeContour, graveContour, poleContour = utils.drone_draw(tempImagePadded, predMask, config_parameters.WIDTH,
                                                               config_parameters.HEIGHT,
                                                               config_path.INFERENCE_RESULT_PATH, NUMBER)
                    NUMBER += 1

                    for m in range(len(riceRowContour)):
                        for n in range(len(riceRowContour[m])):
                            riceRowContour[m][n][0] += j * int(tempWidth)
                            riceRowContour[m][n][1] += i * int(tempHeight)

                    for m in range(len(ridgeContour)):
                        for n in range(len(ridgeContour[m])):
                            ridgeContour[m][n][0] += j * int(tempWidth)
                            ridgeContour[m][n][1] += i * int(tempHeight)


                    for m in range(len(graveContour)):
                        for n in range(len(graveContour[m])):
                            graveContour[m][n][0] += j * int(tempWidth)
                            graveContour[m][n][1] += i * int(tempHeight)

                    for m in range(len(poleContour)):
                        for n in range(len(poleContour[m])):
                            poleContour[m][n][0] += j * int(tempWidth)
                            poleContour[m][n][1] += i * int(tempHeight)

                    allRiceRowPixelResults.append(riceRowContour)
                    allRidgePixelResults.append(ridgeContour)
                    allGravePixelResults.append(graveContour)
                    allPolePixelResults.append(poleContour)

                    # 第一次迭代直接赋值，后续进行堆叠
                    if horizontalResult is None:
                        horizontalResult = contourImage
                    else:
                        horizontalResult = np.hstack((horizontalResult, contourImage))

            if horizontalResult is not None:
                allPicture.append(horizontalResult)

        # =============================================================================#
        # 垂直堆叠同样处理，进行拼接还原成完整的正射图
        finalResult = None
        for result in allPicture:
            if finalResult is None:
                finalResult = result
            else:
                finalResult = np.vstack((finalResult, result))


        # 首先保留下完整的正射图的尺寸信息
        finalHeight, finalWidth = finalResult.shape[:2]



        nameList = imageName.split('.')
        save_all_contours_picture(finalHeight, finalWidth, allRidgePixelResults, allRiceRowPixelResults, allGravePixelResults, allPolePixelResults,  nameList)


        # save draw images
        cv2.imwrite(os.path.join(config_path.INFERENCE_RESULT_PATH, imageName), finalResult)


        # save all contours to json
        # 准备要保存到JSON的数据结构
        contour_data = {
            "ridge": allRidgePixelResults,
            "rice_row": allRiceRowPixelResults,
            "grave": allGravePixelResults,
            "pole": allPolePixelResults,
            "image_info": {
                "height": finalHeight,
                "width": finalWidth,
                "name": nameList[0] if nameList else "unknown"
            }
        }

        # 保存轮廓点集到JSON文件
        json_save_path = os.path.join(config_path.INFERENCE_CONTOURS_PATH, f'{nameList[0]}_contours_data.json')
        with open(json_save_path, 'w', encoding='utf-8') as f:
            # 确保中文正常显示，使用indent参数让JSON格式化输出
            json.dump(contour_data, f, ensure_ascii=False, indent=2)




if __name__ == '__main__':
    main()



