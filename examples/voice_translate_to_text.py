import os
import sys
import torch
import whisper
import argparse
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

class SpeechToTextConverter:
    def __init__(self, model_size: str = "base", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化语音转文本转换器
        """
        self.device = device
        print(f"使用设备: {device}")

        # 加载whisper模型
        print("加载Whisper模型...")
        self.model = whisper.load_model(model_size, device=device)
        print(f"已加载 {model_size} 模型")

    def transcribe_audio(self, audio_path: str, language: str = None) -> str:
        """
        将音频文件转换为文本
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        print(f"处理音频文件: {audio_path}")

        # 转录音频
        if language:
            result = self.model.transcribe(audio_path, language=language, fp16=False)
        else:
            result = self.model.transcribe(audio_path, fp16=False)

        text = result["text"].strip()
        print(f"识别结果: {text}")
        return text

    def process_directory(self, directory_path: str,
                         output_file: str = None,
                         language: str = None) -> List[Tuple[str, str]]:
        """
        处理目录中的所有音频文件，仅进行语音转文本
        """
        # 支持的音频格式
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac'}

        # 获取目录中的所有音频文件
        audio_files = []
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(directory_path, file))

        if not audio_files:
            print(f"在目录 {directory_path} 中未找到音频文件")
            return []

        print(f"找到 {len(audio_files)} 个音频文件")

        results = []

        for audio_file in audio_files:
            try:
                print(f"\n处理文件: {os.path.basename(audio_file)}")
                print("-" * 50)

                # 语音转文本
                transcribed_text = self.transcribe_audio(audio_file, language)

                results.append((os.path.basename(audio_file), transcribed_text))

                print(f"✓ 完成: {os.path.basename(audio_file)}")

            except Exception as e:
                print(f"✗ 处理文件 {audio_file} 时出错: {str(e)}")
                continue

        # 保存结果到文件
        if output_file and results:
            self.save_results(results, output_file)

        return results

    def save_results(self, results: List[Tuple[str, str]], output_file: str):
        """保存语音识别结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("语音识别结果\n")
                f.write("=" * 50 + "\n\n")

                for filename, transcribed_text in results:
                    f.write(f"文件: {filename}\n")
                    f.write(f"识别文本: {transcribed_text}\n")
                    f.write("-" * 50 + "\n")

            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="语音转文本工具")
    parser.add_argument("--directory", "-d", type=str, required=True,
                       help="包含音频文件的目录路径")
    parser.add_argument("--output", "-o", type=str, default="speech_to_text_results.txt",
                       help="输出文件路径 (默认: speech_to_text_results.txt)")
    parser.add_argument("--model", "-m", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper模型大小 (默认: base)")
    parser.add_argument("--language", "-l", type=str, default=None,
                       help="源语言（例如: zh, en, ja 等，默认: 自动检测）")

    args = parser.parse_args()

    # 检查目录是否存在
    if not os.path.exists(args.directory):
        print(f"错误: 目录 {args.directory} 不存在")
        sys.exit(1)

    # 创建语音转文本实例
    try:
        converter = SpeechToTextConverter(model_size=args.model)

        # 处理目录中的所有音频文件
        results = converter.process_directory(
            directory_path=args.directory,
            output_file=args.output,
            language=args.language
        )

        # 打印摘要
        print(f"\n{'='*50}")
        print(f"处理完成! 成功处理 {len(results)} 个文件")
        print(f"结果保存在: {args.output}")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
