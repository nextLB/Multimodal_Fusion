
import os
import sys

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton, QDialog,
                             QHBoxLayout)


# 导入自定义模块
sys.path.append('..')
import CSS.main_css as main_css
import CSS.visual_css as visual_css
import CSS.voice_css as voice_css

from visual_model_window import VisualModelWindow
from voice_model_window import VoiceModelWindow




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

        # 将选项卡设置为主窗口的中央部件
        self.setCentralWidget(self.tab_widget)

    def create_main_tab(self):
        """创建主界面选项卡"""
        tab = QWidget()

        # 创建主垂直布局
        main_layout = QVBoxLayout()

        # 欢迎标签
        label = QLabel("欢迎使用多模态融合APP")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 居中对齐
        label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        main_layout.addWidget(label)

        # 创建水平布局用于放置两个按钮
        button_layout = QHBoxLayout()

        # 视觉模型按钮 - 使用图片
        self.visual_model_btn = QPushButton()
        self.visual_model_btn.clicked.connect(self.open_visual_model_window)

        # 设置视觉按钮图标
        visual_icon_path = "./icons/visual_icon.png"  # 替换为你的图片路径
        if os.path.exists(visual_icon_path):
            # 加载图片并缩放
            pixmap = QPixmap(visual_icon_path)
            scaled_pixmap = pixmap.scaled(140, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.visual_model_btn.setIcon(QIcon(scaled_pixmap))
            self.visual_model_btn.setIconSize(QSize(120, 80))
        else:
            print(f"警告: 图片文件 {visual_icon_path} 不存在")
            self.visual_model_btn.setText("视觉模型")

        self.visual_model_btn.setStyleSheet(visual_css.VISUAL_BUTTON)

        # 语音模型按钮 - 使用图片
        self.voice_model_btn = QPushButton()
        self.voice_model_btn.clicked.connect(self.open_voice_model_window)

        # 设置语音按钮图标
        voice_icon_path = "./icons/voice_icon.png"  # 替换为你的图片路径
        if os.path.exists(voice_icon_path):
            # 加载图片并缩放
            pixmap = QPixmap(voice_icon_path)
            scaled_pixmap = pixmap.scaled(140, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.voice_model_btn.setIcon(QIcon(scaled_pixmap))
            self.voice_model_btn.setIconSize(QSize(120, 80))
        else:
            print(f"警告: 图片文件 {voice_icon_path} 不存在")
            self.voice_model_btn.setText("语音模型")

        self.voice_model_btn.setStyleSheet(voice_css.VOICE_BUTTON)

        # 将按钮添加到水平布局
        button_layout.addWidget(self.visual_model_btn)
        button_layout.addSpacing(100)  # 添加间距
        button_layout.addWidget(self.voice_model_btn)

        # 将水平布局添加到主布局并居中对齐
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(button_layout)

        # 添加标签说明
        label_layout = QHBoxLayout()
        visual_label = QLabel("视觉模型")
        visual_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        visual_label.setStyleSheet("font-size: 16px; margin-top: 10px;")

        voice_label = QLabel("语音模型")
        voice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        voice_label.setStyleSheet("font-size: 16px; margin-top: 10px;")

        label_layout.addWidget(visual_label)
        label_layout.addSpacing(150)  # 与按钮间距对应
        label_layout.addWidget(voice_label)
        label_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(label_layout)
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


