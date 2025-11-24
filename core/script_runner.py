"""
脚本运行器核心功能
"""

import sys
import os
import subprocess
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class ScriptRunnerThread(QThread):
    """在单独线程中运行外部脚本"""

    # 信号定义
    output_signal = pyqtSignal(str)  # 输出内容
    progress_signal = pyqtSignal(int, str)  # 进度更新 (百分比, 消息)
    finished_signal = pyqtSignal(bool, str)  # 完成信号 (成功, 消息)
    error_signal = pyqtSignal(str)  # 错误信号

    def __init__(self, script_path, working_dir=None):
        super().__init__()
        self.script_path = script_path
        self.working_dir = working_dir or os.path.dirname(script_path)
        self._is_running = True

    def run(self):
        try:
            # 检查脚本是否存在
            if not os.path.exists(self.script_path):
                self.error_signal.emit(f"脚本文件不存在: {self.script_path}")
                self.finished_signal.emit(False, f"脚本文件不存在: {self.script_path}")
                return

            # 获取当前Python解释器路径
            python_executable = sys.executable

            # 更新进度
            self.progress_signal.emit(10, "正在启动脚本...")

            # 运行外部脚本
            process = subprocess.Popen(
                [python_executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=self.working_dir
            )

            self.progress_signal.emit(30, "脚本执行中...")

            # 实时读取输出
            line_count = 0
            for line in iter(process.stdout.readline, ''):
                if not self._is_running:
                    process.terminate()
                    break
                if line:
                    self.output_signal.emit(line.strip())
                    line_count += 1

                    # 模拟进度更新 (实际应用中可以根据输出内容解析真实进度)
                    if line_count % 10 == 0 and line_count < 100:
                        progress = min(30 + (line_count // 10) * 5, 90)
                        self.progress_signal.emit(progress, f"已处理 {line_count} 行输出...")

            process.wait()

            # 根据退出码判断执行结果
            if process.returncode == 0:
                self.progress_signal.emit(100, "脚本执行成功!")
                self.finished_signal.emit(True, f"脚本执行完成 (退出码: {process.returncode})")
            else:
                self.progress_signal.emit(100, "脚本执行完成但有错误")
                self.finished_signal.emit(False, f"脚本执行完成但有错误 (退出码: {process.returncode})")

        except Exception as e:
            error_msg = f"执行错误: {str(e)}"
            self.error_signal.emit(error_msg)
            self.finished_signal.emit(False, error_msg)

    def stop(self):
        """停止脚本执行"""
        self._is_running = False

