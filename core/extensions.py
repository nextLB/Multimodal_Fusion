"""
扩展接口定义
为程序功能扩展提供标准接口

运行机理：
1. 定义ExtensionInterface抽象基类，所有扩展必须实现其接口
2. 提供ProgressTracker进度跟踪器，用于统一管理进度信息
3. 实现具体扩展：ResultAnalyzer(结果分析)和ScriptManager(脚本管理)
4. ExtensionManager统一管理所有扩展的注册和初始化

架构特点：
- 插件式架构，易于扩展新功能
- 松耦合设计，扩展与主程序通过接口交互
- 统一的扩展管理机制
"""

from abc import ABC, abstractmethod
from PyQt6.QtCore import QObject, pyqtSignal


class ExtensionInterface(ABC):
    """扩展接口基类 - 所有程序扩展必须实现此接口"""

    @abstractmethod
    def initialize(self, main_window):
        """初始化扩展 - 在程序启动时调用"""
        pass

    @abstractmethod
    def get_name(self):
        """获取扩展名称 - 用于显示和识别"""
        pass


class ProgressTracker(QObject):
    """进度跟踪器 - 统一管理程序执行进度信息"""

    # 信号定义
    progress_updated = pyqtSignal(int)  # 进度百分比更新信号
    progress_message = pyqtSignal(str)  # 进度消息更新信号

    def __init__(self):
        """初始化进度跟踪器"""
        super().__init__()
        self._progress = 0  # 当前进度值 (0-100)
        self._message = ""  # 当前进度消息

    def update_progress(self, value, message=""):
        """更新进度信息"""
        self._progress = max(0, min(100, value))  # 确保进度值在0-100范围内
        self._message = message
        self.progress_updated.emit(self._progress)  # 发射进度更新信号
        if message:
            self.progress_message.emit(message)  # 发射消息更新信号

    def get_progress(self):
        """获取当前进度值"""
        return self._progress

    def get_message(self):
        """获取当前进度消息"""
        return self._message


class ResultAnalyzer(ExtensionInterface):
    """结果分析器扩展 - 分析脚本执行输出结果"""

    def __init__(self):
        """初始化结果分析器"""
        self.main_window = None  # 主窗口引用

    def initialize(self, main_window):
        """初始化扩展，连接必要的信号和槽"""
        self.main_window = main_window
        # 这里可以连接信号，添加菜单项等
        print(f"结果分析器已初始化: {self.get_name()}")

    def get_name(self):
        """返回扩展名称"""
        return "结果分析器"

    def analyze_output(self, output_text):
        """
        分析输出结果

        参数:
            output_text: 要分析的输出文本

        返回:
            dict: 包含分析结果的字典
        """
        lines = output_text.split('\n')
        return {
            "total_lines": len(lines),  # 总行数
            "success_keywords": sum(1 for line in lines if '成功' in line or 'success' in line.lower()),  # 成功关键词计数
            "error_keywords": sum(1 for line in lines if '错误' in line or 'error' in line.lower()),  # 错误关键词计数
            "last_line": lines[-1] if lines else ""  # 最后一行输出
        }


class ScriptManager(ExtensionInterface):
    """脚本管理器扩展 - 管理脚本执行历史记录"""

    def __init__(self):
        """初始化脚本管理器"""
        self.main_window = None  # 主窗口引用
        self.script_history = []  # 脚本执行历史记录

    def initialize(self, main_window):
        """初始化扩展"""
        self.main_window = main_window
        print(f"脚本管理器已初始化: {self.get_name()}")

    def get_name(self):
        """返回扩展名称"""
        return "脚本管理器"

    def add_to_history(self, script_path, timestamp, status):
        """
        添加脚本执行记录到历史

        参数:
            script_path: 脚本路径
            timestamp: 执行时间戳
            status: 执行状态 (success/failed/stopped/running)
        """
        self.script_history.append({
            "path": script_path,
            "timestamp": timestamp,
            "status": status
        })

    def get_history(self):
        """获取脚本执行历史记录"""
        return self.script_history.copy()  # 返回副本以避免外部修改


class ExtensionManager:
    """扩展管理器 - 统一管理所有程序扩展"""

    def __init__(self):
        """初始化扩展管理器"""
        self.extensions = {}  # 存储所有注册的扩展

    def register_extension(self, name, extension):
        """
        注册扩展

        参数:
            name: 扩展名称（唯一标识）
            extension: 扩展实例
        """
        self.extensions[name] = extension

    def initialize_extensions(self, main_window):
        """初始化所有已注册的扩展"""
        for name, extension in self.extensions.items():
            extension.initialize(main_window)  # 调用每个扩展的初始化方法

    def get_extension(self, name):
        """
        获取指定名称的扩展

        参数:
            name: 扩展名称

        返回:
            ExtensionInterface: 扩展实例，如果不存在则返回None
        """
        return self.extensions.get(name)

    def get_all_extensions(self):
        """获取所有已注册的扩展"""
        return self.extensions.copy()  # 返回副本以避免外部修改