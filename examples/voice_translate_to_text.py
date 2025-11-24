import whisper
import numpy as np
import pyaudio
import wave
import threading
import time

# 全局变量，用于控制录音
is_recording = False
audio_frames = []


def record_audio():
    """在后台线程中录制音频"""
    global is_recording, audio_frames

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    # 尝试列出音频设备，帮助选择正确的设备索引
    print("可用的音频设备:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        print(f"  设备 {i}: {dev_info['name']} - 最大输入通道: {dev_info['maxInputChannels']}")

    # 寻找合适的输入设备
    input_device_index = None
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # 有输入通道的设备
            input_device_index = i
            print(f"选择输入设备: {dev_info['name']}")
            break

    if input_device_index is None:
        print("未找到可用的输入设备")
        return

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)

    print("录音中... (按回车键停止)")
    audio_frames = []

    while is_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


def save_and_transcribe():
    """保存音频文件并进行转录"""
    global audio_frames

    if not audio_frames:
        print("没有录制到音频数据")
        return

    # 保存为临时WAV文件
    filename = "temp_recording.wav"

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(16000)
    wf.writeframes(b''.join(audio_frames))
    wf.close()

    print("音频已保存，开始识别...")

    try:
        # 加载模型 - 你可以根据需要选择不同大小的模型
        # 可选模型: "tiny", "base", "small", "medium", "large"
        model = whisper.load_model("base")

        # 进行转录
        result = model.transcribe(filename, language="zh", task="transcribe")

        print("\n" + "=" * 50)
        print("识别结果:")
        print(result["text"])
        print("=" * 50)

    except Exception as e:
        print(f"识别过程中出错: {e}")


def main():
    global is_recording

    print("=" * 50)
    print("Whisper 语音识别系统")
    print("=" * 50)

    while True:
        print("\n选项:")
        print("1. 开始录音")
        print("2. 退出程序")

        choice = input("请选择 (1/2): ").strip()

        if choice == '1':
            is_recording = True

            # 启动录音线程
            record_thread = threading.Thread(target=record_audio)
            record_thread.daemon = True
            record_thread.start()

            # 等待用户停止录音
            input("")  # 按回车键停止

            is_recording = False
            time.sleep(0.5)  # 等待录音线程结束

            # 进行转录
            save_and_transcribe()

        elif choice == '2':
            print("程序退出，再见！")
            break
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()



# import whisperx
# import numpy as np
# import pyaudio
# import wave
# import threading
# import time
#
# # 全局变量
# is_recording = False
# audio_frames = []
# device = "cpu"  # 如果是NVIDIA GPU，可以设置为 "cuda"
# batch_size = 8
#
#
# def record_audio():
#     """在后台线程中录制音频"""
#     global is_recording, audio_frames
#
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 16000
#
#     p = pyaudio.PyAudio()
#
#     # 自动选择输入设备
#     input_device_index = None
#     for i in range(p.get_device_count()):
#         dev_info = p.get_device_info_by_index(i)
#         if dev_info['maxInputChannels'] > 0:
#             input_device_index = i
#             print(f"使用输入设备: {dev_info['name']}")
#             break
#
#     if input_device_index is None:
#         print("警告: 未找到输入设备，使用默认设备")
#         input_device_index = None
#
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     input_device_index=input_device_index,
#                     frames_per_buffer=CHUNK)
#
#     print("录音中... (按回车键停止)")
#     audio_frames = []
#
#     while is_recording:
#         data = stream.read(CHUNK, exception_on_overflow=False)
#         audio_frames.append(data)
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#
# def save_and_transcribe():
#     """保存音频文件并使用WhisperX进行转录"""
#     global audio_frames
#
#     if not audio_frames:
#         print("没有录制到音频数据")
#         return
#
#     # 保存为临时WAV文件
#     filename = "temp_recording.wav"
#
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(1)
#     wf.setsampwidth(2)
#     wf.setframerate(16000)
#     wf.writeframes(b''.join(audio_frames))
#     wf.close()
#
#     print("音频已保存，开始识别...")
#
#     try:
#         # 1. 加载WhisperX模型
#         model = whisperx.load_model("base", device=device)
#
#         # 2. 转录音频
#         audio = whisperx.load_audio(filename)
#         result = model.transcribe(audio, batch_size=batch_size, language="zh")
#
#         print("\n" + "=" * 60)
#         print("WhisperX 识别结果:")
#         print(result["segments"][0]["text"] if result["segments"] else "未识别到内容")
#         print("=" * 60)
#
#         # 如果需要时间戳信息，可以取消下面的注释
#         # if len(result["segments"]) > 0:
#         #     print("\n详细时间戳:")
#         #     for segment in result["segments"]:
#         #         print(f"  [{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
#
#     except Exception as e:
#         print(f"识别过程中出错: {e}")
#
#
# def main():
#     global is_recording
#
#     print("=" * 50)
#     print("WhisperX 高性能语音识别系统")
#     print("=" * 50)
#
#     # 显示模型信息
#     print("使用的模型: base (平衡速度和精度)")
#     print(f"计算设备: {device}")
#
#     while True:
#         print("\n选项:")
#         print("1. 开始录音")
#         print("2. 退出程序")
#
#         choice = input("请选择 (1/2): ").strip()
#
#         if choice == '1':
#             is_recording = True
#
#             # 启动录音线程
#             record_thread = threading.Thread(target=record_audio)
#             record_thread.daemon = True
#             record_thread.start()
#
#             # 等待用户停止录音
#             input("")  # 按回车键停止
#
#             is_recording = False
#             time.sleep(0.5)
#
#             # 进行转录
#             save_and_transcribe()
#
#         elif choice == '2':
#             print("程序退出，再见！")
#             break
#         else:
#             print("无效选择，请重新输入")
#
#
# if __name__ == "__main__":
#     main()