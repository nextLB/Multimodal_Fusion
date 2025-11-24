"""
脚本运行器核心功能

运行机理：
1. ScriptRunnerThread继承自QThread，在独立线程中运行外部Python脚本
2. 使用subprocess.Popen启动新的Python进程执行脚本
3. 实时读取脚本的标准输出和错误输出
4. 通过PyQt信号机制与主线程通信，更新UI

关键特性：
- 避免阻塞GUI主线程
- 实时输出捕获
- 支持用户中断执行
- 错误处理和状态报告
"""

import sys
import os
import subprocess
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class ScriptRunnerThread(QThread):
    """在单独线程中运行外部脚本，避免阻塞GUI界面"""

    # 信号定义
    output_signal = pyqtSignal(str)  # 输出内容信号
    progress_signal = pyqtSignal(int, str)  # 进度更新信号 (百分比, 消息)
    finished_signal = pyqtSignal(bool, str)  # 完成信号 (成功状态, 消息)
    error_signal = pyqtSignal(str)  # 错误信号

    def __init__(self, script_path, working_dir=None):
        """
        初始化脚本运行线程

        参数:
            script_path: 要运行的Python脚本路径
            working_dir: 工作目录，默认为脚本所在目录
        """
        super().__init__()
        self.script_path = script_path
        self.working_dir = working_dir or os.path.dirname(script_path)
        self._is_running = True  # 运行状态标志

    def run(self):
        """线程主执行函数 - 在此方法中运行外部脚本"""
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
                stdout=subprocess.PIPE,  # 捕获标准输出
                stderr=subprocess.STDOUT,  # 将标准错误重定向到标准输出
                text=True,  # 以文本模式处理输出
                bufsize=1,  # 行缓冲
                universal_newlines=True,  # 统一换行符
                cwd=self.working_dir  # 设置工作目录
            )

            self.progress_signal.emit(30, "脚本执行中...")

            # 实时读取输出
            line_count = 0
            for line in iter(process.stdout.readline, ''):
                if not self._is_running:  # 检查是否被请求停止
                    process.terminate()  # 终止进程
                    break
                if line:
                    self.output_signal.emit(line.strip())  # 发送输出行
                    line_count += 1

                    # 模拟进度更新 (实际应用中可以根据输出内容解析真实进度)
                    if line_count % 10 == 0 and line_count < 100:
                        progress = min(30 + (line_count // 10) * 5, 90)
                        self.progress_signal.emit(progress, f"已处理 {line_count} 行输出...")

            # 等待进程结束
            process.wait()

            # 根据退出码判断执行结果
            if process.returncode == 0:
                self.progress_signal.emit(100, "脚本执行成功!")
                self.finished_signal.emit(True, f"脚本执行完成 (退出码: {process.returncode})")
            else:
                self.progress_signal.emit(100, "脚本执行完成但有错误")
                self.finished_signal.emit(False, f"脚本执行完成但有错误 (退出码: {process.returncode})")

        except Exception as e:
            # 处理执行过程中的异常
            error_msg = f"执行错误: {str(e)}"
            self.error_signal.emit(error_msg)
            self.finished_signal.emit(False, error_msg)

    def stop(self):
        """停止脚本执行 - 设置标志位以便安全终止线程"""
        self._is_running = False