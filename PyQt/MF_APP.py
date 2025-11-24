

import sys

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton)


# 导入自定义模块
sys.path.append('..')
import CSS.css as css


class MultimodalFusionAPP(QMainWindow):
    def __init__(self):
        """初始化主窗口"""
        super().__init__()

        self.init_ui()



    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("多模态融合APP")
        self.setGeometry(300, 200, 1000, 800)  # 设置窗口位置和大小
        # 设置样式
        self.setStyleSheet(css.COMPLETE_STYLE)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        # 创建选项卡
        self.main_tab = self.create_main_tab()
        self.settings_tab = self.create_settings_tab()
        # 主控制界面的选项卡
        self.tab_widget.addTab(self.main_tab, "主界面")
        # 设置界面的选项卡
        self.tab_widget.addTab(self.settings_tab, "设置")

        # 关键：将选项卡设置为主窗口的中央部件
        self.setCentralWidget(self.tab_widget)


    def create_main_tab(self):
        """创建主界面选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建一个标签
        label = QLabel("欢迎使用多模态融合APP")
        layout.addWidget(label)
        layout.addStretch()

        tab.setLayout(layout)
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


