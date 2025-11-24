


# pip3 install pyaudio speechrecognition -i https://pypi.tuna.tsinghua.edu.cn/simple



import speech_recognition as sr
import threading
import time
import queue
import sys


class RealTimeSpeechRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.stop_event = threading.Event()

        # 调整麦克风对环境噪音的适应
        print("正在调整麦克风对环境噪音的适应...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("麦克风调整完成！")

    def audio_capture(self):
        """后台线程：持续捕获音频数据"""

        def record_audio():
            while not self.stop_event.is_set():
                try:
                    # 使用超时来定期检查停止事件
                    audio = self.recognizer.listen(
                        self.microphone,
                        timeout=1,
                        phrase_time_limit=5
                    )
                    self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    # 超时是正常的，继续循环
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"音频捕获错误: {e}")
                    break

        # 启动音频捕获线程
        audio_thread = threading.Thread(target=record_audio)
        audio_thread.daemon = True
        audio_thread.start()
        return audio_thread

    def speech_recognition(self):
        """后台线程：处理音频队列中的数据进行识别"""

        def process_audio():
            while not self.stop_event.is_set() or not self.audio_queue.empty():
                try:
                    # 从队列获取音频数据，设置超时以便定期检查停止事件
                    audio_data = self.audio_queue.get(timeout=1)

                    try:
                        # 使用Google语音识别（需要网络连接）
                        text = self.recognizer.recognize_google(audio_data, language='zh-CN')
                        print(f"\r识别结果: {text}" + " " * 50, end='\n')

                    except sr.UnknownValueError:
                        # 无法理解音频内容
                        print(f"\r无法识别语音" + " " * 30, end='')
                    except sr.RequestError as e:
                        print(f"\r语音识别服务错误: {e}" + " " * 30, end='')

                    self.audio_queue.task_done()

                except queue.Empty:
                    # 队列为空是正常的，继续循环
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"语音处理错误: {e}")
                    break

        # 启动语音识别线程
        recognition_thread = threading.Thread(target=process_audio)
        recognition_thread.daemon = True
        recognition_thread.start()
        return recognition_thread

    def start_listening(self):
        """开始语音识别"""
        if self.is_listening:
            print("已经在监听中...")
            return

        print("开始实时语音识别...")
        print("请对着麦克风说话（说中文）")
        print("按 Ctrl+C 停止程序")
        print("-" * 50)

        self.is_listening = True
        self.stop_event.clear()

        # 启动两个后台线程
        self.audio_capture()
        self.speech_recognition()

        try:
            # 主线程保持运行
            while self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_listening()

    def stop_listening(self):
        """停止语音识别"""
        print("\n正在停止语音识别...")
        self.is_listening = False
        self.stop_event.set()
        time.sleep(1)  # 给线程一些时间清理
        print("语音识别已停止")


def list_microphones():
    """列出所有可用的麦克风设备"""
    print("可用的音频设备:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{index}: {name}")


def main():
    # 显示可用的麦克风设备
    list_microphones()
    print("-" * 50)

    # 创建语音识别实例
    speech_recog = RealTimeSpeechRecognition()

    try:
        # 开始语音识别
        speech_recog.start_listening()
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        speech_recog.stop_listening()


if __name__ == "__main__":
    main()