import os
import sys

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton, QDialog,
                             QHBoxLayout, QGridLayout, QGroupBox)

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
        self.setGeometry(300, 200, 1200, 900)

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
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(main_css.MAIN_TITTLE_STYLE)
        main_layout.addWidget(label)

        # 创建视觉模型功能区
        visual_group = self.create_visual_model_group()
        main_layout.addWidget(visual_group)

        # 创建语音模型功能区
        voice_group = self.create_voice_model_group()
        main_layout.addWidget(voice_group)

        # 创建多模态融合模型功能区
        multimodal_group = self.create_multimodal_model_group()
        main_layout.addWidget(multimodal_group)

        main_layout.addStretch()
        tab.setLayout(main_layout)
        return tab

    def create_visual_model_group(self):
        """创建视觉模型功能区"""
        group = QGroupBox("视觉模型")
        group.setStyleSheet(main_css.MAIN_VISUAL_STYLE)

        layout = QHBoxLayout()
        layout.setSpacing(40)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 视觉模型主功能按钮
        self.visual_model_btn = self.create_icon_button(
            "./icons/visual_icon.png",
            "视觉模型",
            self.open_visual_model_window
        )
        layout.addWidget(self.visual_model_btn)

        # 视觉模型训练分析按钮
        self.visual_training_analysis_btn = self.create_icon_button(
            "./icons/training_visual_analysis_icon.png",
            "训练数据分析",
            self.open_visual_training_analysis_window
        )
        layout.addWidget(self.visual_training_analysis_btn)

        # 视觉模型推理分析按钮
        self.visual_inference_analysis_btn = self.create_icon_button(
            "./icons/inference_visual_analysis_icon.png",
            "推理数据分析",
            self.open_visual_inference_analysis_window
        )
        layout.addWidget(self.visual_inference_analysis_btn)

        group.setLayout(layout)
        return group

    def create_voice_model_group(self):
        """创建语音模型功能区"""
        group = QGroupBox("语音模型")
        group.setStyleSheet(main_css.MAIN_VOICE_STYLE)

        layout = QHBoxLayout()
        layout.setSpacing(40)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 语音模型主功能按钮
        self.voice_model_btn = self.create_icon_button(
            "./icons/voice_icon.png",
            "语音模型",
            self.open_voice_model_window
        )
        layout.addWidget(self.voice_model_btn)

        # 语音模型训练分析按钮
        self.voice_training_analysis_btn = self.create_icon_button(
            "./icons/training_voice_analysis_icon.png",
            "训练数据分析",
            self.open_voice_training_analysis_window
        )
        layout.addWidget(self.voice_training_analysis_btn)

        # 语音模型推理分析按钮
        self.voice_inference_analysis_btn = self.create_icon_button(
            "./icons/inference_voice_analysis_icon.png",
            "推理数据分析",
            self.open_voice_inference_analysis_window
        )
        layout.addWidget(self.voice_inference_analysis_btn)

        group.setLayout(layout)
        return group

    def create_multimodal_model_group(self):
        """创建多模态融合模型功能区"""
        group = QGroupBox("多模态融合模型")
        group.setStyleSheet(main_css.MAIN_FUSION_STYLE)

        layout = QHBoxLayout()
        layout.setSpacing(40)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 多模态融合模型主功能按钮
        self.multimodal_fusion_btn = self.create_icon_button(
            "./icons/fusion_icon.png",
            "多模态融合",
            self.open_multimodal_fusion_window
        )
        layout.addWidget(self.multimodal_fusion_btn)

        # 多模态融合模型训练分析按钮
        self.multimodal_training_analysis_btn = self.create_icon_button(
            "./icons/training_fusion_analysis_icon.png",
            "训练数据分析",
            self.open_multimodal_training_analysis_window
        )
        layout.addWidget(self.multimodal_training_analysis_btn)

        # 多模态融合模型推理分析按钮
        self.multimodal_inference_analysis_btn = self.create_icon_button(
            "./icons/inference_fusion_analysis_icon.png",
            "推理数据分析",
            self.open_multimodal_inference_analysis_window
        )
        layout.addWidget(self.multimodal_inference_analysis_btn)

        group.setLayout(layout)
        return group

    def create_icon_button(self, icon_path, button_text, callback):
        """创建带有图标的按钮"""
        button = QPushButton()
        button.clicked.connect(callback)
        button.setToolTip(button_text)  # 鼠标悬停提示

        # 设置按钮图标
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            button.setIcon(QIcon(scaled_pixmap))
            button.setIconSize(QSize(120, 120))
        else:
            print(f"警告: 图片文件 {icon_path} 不存在")
            button.setText(button_text)

        # 设置按钮样式 - 简洁无背景
        button.setStyleSheet(main_css.MAIN_BUTTON_STYLE)

        button.setFixedSize(180, 180)  # 设置固定大小

        # 创建标签显示按钮文本
        label = QLabel(button_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 5px; color: #2c3e50;")

        # 创建垂直布局包含按钮和标签
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(button)
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget.setLayout(layout)

        return widget

    def create_settings_tab(self):
        """创建设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("系统设置")
        label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(label)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    # 以下是所有功能按钮的点击事件处理函数
    def open_visual_model_window(self):
        """打开视觉模型窗口"""
        self.popup_window = VisualModelWindow(self)
        self.popup_window.show()

    def open_voice_model_window(self):
        """打开语音模型窗口"""
        self.popup_window = VoiceModelWindow(self)
        self.popup_window.show()

    def open_multimodal_fusion_window(self):
        """打开多模态融合模型窗口"""
        # 待实现
        pass

    def open_visual_training_analysis_window(self):
        """打开视觉模型训练历史数据分析窗口"""
        # 待实现
        pass

    def open_visual_inference_analysis_window(self):
        """打开视觉模型推理历史数据分析窗口"""
        # 待实现
        pass

    def open_voice_training_analysis_window(self):
        """打开语音模型训练历史数据分析窗口"""
        # 待实现
        pass

    def open_voice_inference_analysis_window(self):
        """打开语音模型推理历史数据分析窗口"""
        # 待实现
        pass

    def open_multimodal_training_analysis_window(self):
        """打开多模态融合模型训练历史数据分析窗口"""
        # 待实现
        pass

    def open_multimodal_inference_analysis_window(self):
        """打开多模态融合模型推理历史数据分析窗口"""
        # 待实现
        pass


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