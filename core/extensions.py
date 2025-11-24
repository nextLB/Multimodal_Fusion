"""
扩展接口定义
为程序功能扩展提供标准接口
"""

from abc import ABC, abstractmethod
from PyQt6.QtCore import QObject, pyqtSignal


class ExtensionInterface(ABC):
    """扩展接口基类"""

    @abstractmethod
    def initialize(self, main_window):
        """初始化扩展"""
        pass

    @abstractmethod
    def get_name(self):
        """获取扩展名称"""
        pass


class ProgressTracker(QObject):
    """进度跟踪器"""

    progress_updated = pyqtSignal(int)  # 进度百分比
    progress_message = pyqtSignal(str)  # 进度消息

    def __init__(self):
        super().__init__()
        self._progress = 0
        self._message = ""

    def update_progress(self, value, message=""):
        """更新进度"""
        self._progress = max(0, min(100, value))
        self._message = message
        self.progress_updated.emit(self._progress)
        if message:
            self.progress_message.emit(message)

    def get_progress(self):
        """获取当前进度"""
        return self._progress

    def get_message(self):
        """获取当前消息"""
        return self._message


class ResultAnalyzer(ExtensionInterface):
    """结果分析器扩展"""

    def __init__(self):
        self.main_window = None

    def initialize(self, main_window):
        self.main_window = main_window
        # 这里可以连接信号，添加菜单项等
        print(f"结果分析器已初始化: {self.get_name()}")

    def get_name(self):
        return "结果分析器"

    def analyze_output(self, output_text):
        """分析输出结果"""
        # 这里可以实现自定义的结果分析逻辑
        lines = output_text.split('\n')
        return {
            "total_lines": len(lines),
            "success_keywords": sum(1 for line in lines if '成功' in line or 'success' in line.lower()),
            "error_keywords": sum(1 for line in lines if '错误' in line or 'error' in line.lower()),
            "last_line": lines[-1] if lines else ""
        }


class ScriptManager(ExtensionInterface):
    """脚本管理器扩展"""

    def __init__(self):
        self.main_window = None
        self.script_history = []

    def initialize(self, main_window):
        self.main_window = main_window
        print(f"脚本管理器已初始化: {self.get_name()}")

    def get_name(self):
        return "脚本管理器"

    def add_to_history(self, script_path, timestamp, status):
        """添加到历史记录"""
        self.script_history.append({
            "path": script_path,
            "timestamp": timestamp,
            "status": status
        })

    def get_history(self):
        """获取历史记录"""
        return self.script_history.copy()


# 扩展管理器
class ExtensionManager:
    """管理所有扩展"""

    def __init__(self):
        self.extensions = {}

    def register_extension(self, name, extension):
        """注册扩展"""
        self.extensions[name] = extension

    def initialize_extensions(self, main_window):
        """初始化所有扩展"""
        for name, extension in self.extensions.items():
            extension.initialize(main_window)

    def get_extension(self, name):
        """获取扩展"""
        return self.extensions.get(name)

    def get_all_extensions(self):
        """获取所有扩展"""
        return self.extensions.copy()

