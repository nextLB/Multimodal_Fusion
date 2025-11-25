
"""
    切分视频成指定尺寸的图像的程序
"""

import os
import cv2


PROCESS_VIDEOS_PATH = './raw_videos'
BEGIN_ORDINATE = 0
END_ORDINATE = 280
BEGIN_ABSCISSA = 170
END_ABSCISSA = 450
OUTPUT_PATH = './results'

def main():


    os.makedirs(OUTPUT_PATH, exist_ok=True)
    number = 0
    for nowFile in os.listdir(PROCESS_VIDEOS_PATH):
        if nowFile.endswith('.mp4'):
            cap = cv2.VideoCapture(os.path.join(PROCESS_VIDEOS_PATH, nowFile))
            # 检查视频是否成功打开
            if not cap.isOpened():
                print("无法打开视频文件")
                exit()

            # 循环读取视频帧
            while True:
                # 逐帧读取
                ret, frame = cap.read()

                # ret 为布尔值，表示是否成功读取帧
                # frame 为当前帧的图像数据（numpy 数组，BGR 格式）
                if not ret:
                    # 读取完毕（到达视频末尾）
                    print(f"{os.path.join(PROCESS_VIDEOS_PATH, nowFile)}视频读取完毕")
                    break

                print(f"processing images_{number}.png")
                # 在这里处理帧数据，例如显示、保存或进行图像处理
                frame = frame[int(BEGIN_ORDINATE):int(END_ORDINATE), int(BEGIN_ABSCISSA):int(END_ABSCISSA)]
                cv2.imwrite(os.path.join(OUTPUT_PATH, f"images_{number}.png"), frame)
                number += 1

            # 释放资源
            cap.release()




if __name__ == '__main__':
    main()


