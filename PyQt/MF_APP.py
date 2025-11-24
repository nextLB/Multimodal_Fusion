

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton, QDialog,
                             QHBoxLayout)


# 导入自定义模块
sys.path.append('..')
import CSS.main_css as main_css
import CSS.visual_css as visual_css
import CSS.voice_css as voice_css


# 视觉模型窗口类
class VisualModelWindow(QDialog):
    def __init__(self, parent=None):
        """初始化弹出窗口"""
        super().__init__(parent)
        self.setWindowTitle("视觉模型界面")
        self.setGeometry(400, 300, 600, 400)

        # 设置模态，使弹出窗口出现时主窗口不可操作
        self.setModal(True)

        self.init_ui()

    def init_ui(self):
        """初始化弹出窗口UI"""
        layout = QVBoxLayout()

        # 添加内容
        label = QLabel("这是弹出的视觉模型界面")
        label.setStyleSheet("font-size: 16px; margin: 20px;")
        layout.addWidget(label)

        # 添加关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(visual_css.VISUAL_CLOSE)
        layout.addWidget(close_btn)

        self.setLayout(layout)



# 语音模型窗口类
class VoiceModelWindow(QDialog):
    def __init__(self, parent=None):
        """初始化弹出窗口"""
        super().__init__(parent)
        self.setWindowTitle("语音模型界面")
        self.setGeometry(400, 300, 600, 400)

        # 设置模态，使弹出窗口出现时主窗口不可操作
        self.setModal(True)

        self.init_ui()

    def init_ui(self):
        """初始化弹出窗口UI"""
        layout = QVBoxLayout()

        # 添加内容
        label = QLabel("这是弹出的语音模型界面")
        label.setStyleSheet("font-size: 16px; margin: 20px;")
        layout.addWidget(label)

        # 添加关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(voice_css.VOICE_CLOSE)
        layout.addWidget(close_btn)

        self.setLayout(layout)



# 多模态融合主APP类
class MultimodalFusionAPP(QMainWindow):
    def __init__(self):
        """初始化主窗口"""
        super().__init__()

        # 初始化UI界面
        self.init_ui()



    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("多模态融合APP")
        self.setGeometry(300, 200, 1000, 800)  # 设置窗口位置和大小

        # 创建选项卡
        self.tab_widget = QTabWidget()

        # 创建选项卡
        self.main_tab = self.create_main_tab()
        self.settings_tab = self.create_settings_tab()
        # 主控制界面的选项卡
        self.tab_widget.addTab(self.main_tab, "主界面")
        # 设置界面的选项卡
        self.tab_widget.addTab(self.settings_tab, "设置")

        # 设置样式
        self.setStyleSheet(main_css.get_stylesheet())  # 应用样式表

        # 关键：将选项卡设置为主窗口的中央部件
        self.setCentralWidget(self.tab_widget)

    def create_main_tab(self):
        """创建主界面选项卡"""
        tab = QWidget()

        # 创建主垂直布局
        main_layout = QVBoxLayout()

        # 欢迎标签
        label = QLabel("欢迎使用多模态融合APP")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 居中对齐
        main_layout.addWidget(label)

        # 创建水平布局用于放置两个按钮
        button_layout = QHBoxLayout()

        # 视觉模型按钮 - 设置为正方形
        self.visual_model_btn = QPushButton("视觉模型")
        self.visual_model_btn.clicked.connect(self.open_visual_model_window)
        self.visual_model_btn.setFixedSize(120, 120)  # 设置固定大小，形成正方形
        self.visual_model_btn.setStyleSheet(visual_css.VISUAL_BUTTON)

        # 语音模型按钮 - 设置为正方形
        self.voice_model_btn = QPushButton("语音模型")
        self.voice_model_btn.clicked.connect(self.open_voice_model_window)
        self.voice_model_btn.setFixedSize(120, 120)  # 设置固定大小，形成正方形
        self.voice_model_btn.setStyleSheet(voice_css.VOICE_BUTTON)

        # 将按钮添加到水平布局
        button_layout.addWidget(self.visual_model_btn)
        button_layout.addSpacing(20)  # 添加间距
        button_layout.addWidget(self.voice_model_btn)

        # 将水平布局添加到主布局并居中对齐
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(button_layout)

        # 添加弹性空间，使内容在中间显示
        main_layout.addStretch()

        tab.setLayout(main_layout)
        return tab

    def create_settings_tab(self):
        """创建设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("系统设置")
        layout.addWidget(label)
        layout.addStretch()

        tab.setLayout(layout)
        return tab


    def open_visual_model_window(self):
        self.popup_window = VisualModelWindow(self)
        self.popup_window.show()

    def open_voice_model_window(self):
        self.popup_window = VoiceModelWindow(self)
        self.popup_window.show()



def main():
    """主函数 - 应用程序入口点"""
    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("NEXT多模态模型融合APP")
    app.setApplicationVersion("1.0")

    # 创建并显示主窗口
    window = MultimodalFusionAPP()
    window.show()

    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


