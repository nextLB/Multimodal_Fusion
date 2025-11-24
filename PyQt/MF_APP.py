

import sys

from PyQt6.QtWidgets import (QApplication, QMainWindow)


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


