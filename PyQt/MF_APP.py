
import os
import sys

from PyQt6.QtCore import Qt, QSize, QProcess, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QTextCursor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton, QDialog,
                             QHBoxLayout, QTextEdit, QProgressBar, QFileDialog, QMessageBox)


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
        self.setGeometry(300, 200, 1000, 800)  # 设置窗口位置和大小

        # 设置模态，使弹出窗口出现时主窗口不可操作
        self.setModal(True)

        # 初始化示例程序运行状态
        self.process = None
        self.is_example_running = False

        # 添加当前运行的脚本路径变量
        self.current_script_path = None

        self.example_script_path = os.path.join("/home/next_lb/桌面/next/Multimodal_Fusion/examples/", "test_script.py")

        self.init_ui()

    def init_ui(self):
        """初始化弹出窗口UI"""
        layout = QVBoxLayout()

        # 添加标题
        label = QLabel("视觉模型示例程序")
        label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 15px;")
        layout.addWidget(label)

        # 添加按钮区域
        button_layout = QHBoxLayout()

        # 添加运行示例程序按钮
        self.run_btn = QPushButton("运行示例程序")
        self.run_btn.clicked.connect(self.run_example_program)
        self.run_btn.setStyleSheet(visual_css.EXAMPLE_BEGIN_BUTTON)
        button_layout.addWidget(self.run_btn)

        # 添加选择并运行任意程序按钮
        self.select_run_btn = QPushButton("选择并运行Python程序")
        self.select_run_btn.clicked.connect(self.select_and_run_program)
        self.select_run_btn.setStyleSheet(visual_css.EXAMPLE_BEGIN_BUTTON)
        button_layout.addWidget(self.select_run_btn)

        # 添加停止按钮
        self.stop_btn = QPushButton("停止程序")
        self.stop_btn.clicked.connect(self.stop_program)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(visual_css.EXAMPLE_END_BUTTON)
        button_layout.addWidget(self.stop_btn)

        # 添加清空按钮
        clear_btn = QPushButton("清空结果")
        clear_btn.clicked.connect(self.clear_results)
        clear_btn.setStyleSheet(visual_css.EXAMPLE_CLEAR_BUTTON)
        button_layout.addWidget(clear_btn)

        layout.addLayout(button_layout)

        # 添加当前运行程序显示标签
        self.current_script_label = QLabel("当前运行: 无")
        self.current_script_label.setStyleSheet("font-size: 12px; color: gray; margin: 5px;")
        layout.addWidget(self.current_script_label)

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 添加结果显示区域
        result_label = QLabel("程序运行输出:")
        result_label.setStyleSheet("font-size: 14px; margin-top: 15px;")
        layout.addWidget(result_label)

        # 创建文本框用于显示结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)  # 设置为只读
        self.result_text.setPlaceholderText("程序输出将显示在这里...")
        self.result_text.setStyleSheet(visual_css.EXAMPLE_SHOW_STYLE)
        layout.addWidget(self.result_text)

        # 添加关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(visual_css.VISUAL_CLOSE)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def select_and_run_program(self):
        """选择并运行任意的Python程序文件"""
        if self.is_example_running:
            self.result_text.append("<span style='color: orange;'>请先停止当前运行的程序</span>")
            return

        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择Python程序文件",
            os.path.expanduser("~"),  # 从用户主目录开始
            "Python Files (*.py);;All Files (*)"
        )

        if file_path:
            self.current_script_path = file_path
            self.run_python_script(file_path)

    def run_python_script(self, script_path):
        """运行指定的Python脚本"""
        # 检查脚本是否存在
        if not os.path.exists(script_path):
            self.result_text.append(f"错误: 找不到程序文件 {script_path}")
            return

        self.is_example_running = True
        self.run_btn.setEnabled(False)
        self.select_run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度

        # 更新当前运行程序显示
        script_name = os.path.basename(script_path)
        self.current_script_label.setText(f"当前运行: {script_name}")
        self.current_script_label.setStyleSheet("font-size: 12px; color: blue; margin: 5px;")

        # 清空之前的结果
        self.result_text.clear()
        self.result_text.append(f"开始运行程序: {script_path}")
        self.result_text.append("=" * 50)

        # 使用QProcess来运行外部Python脚本
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

        # 运行Python脚本
        python_executable = sys.executable  # 使用当前Python解释器
        self.process.start(python_executable, [script_path])

    def run_example_program(self):
        """运行示例程序并显示结果"""
        if self.is_example_running:
            self.result_text.append("<span style='color: orange;'>请先停止当前运行的程序</span>")
            return

        self.current_script_path = self.example_script_path
        self.run_python_script(self.example_script_path)

    def handle_stdout(self):
        """处理标准输出"""
        if self.process:
            data = self.process.readAllStandardOutput()
            if data:
                text = bytes(data).decode('utf-8')
                self.append_output(text)

    def handle_stderr(self):
        """处理标准错误"""
        if self.process:
            data = self.process.readAllStandardError()
            if data:
                text = bytes(data).decode('utf-8')
                self.append_output(f"<span style='color: red;'>{text}</span>")

    def append_output(self, text):
        """添加输出到结果文本框"""
        # 保存当前滚动位置
        scrollbar = self.result_text.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()

        # 添加文本
        cursor = self.result_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(text)  # 使用HTML以支持颜色

        # 如果之前已经在底部，保持滚动到底部
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def process_finished(self, exit_code, exit_status):
        """进程完成时的处理"""
        self.is_example_running = False
        self.run_btn.setEnabled(True)
        self.select_run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        # 更新当前运行程序显示
        self.current_script_label.setText("当前运行: 无")
        self.current_script_label.setStyleSheet("font-size: 12px; color: gray; margin: 5px;")

        self.result_text.append("=" * 50)
        if exit_code == 0:
            self.result_text.append("<span style='color: green;'>程序执行完成！</span>")
        else:
            self.result_text.append(f"<span style='color: red;'>程序异常退出，代码: {exit_code}</span>")

        # 安全地清理进程对象
        if self.process:
            try:
                # 等待进程完全结束
                self.process.waitForFinished(100)
                # 不手动断开连接，让Qt自动管理
                self.process = None
            except:
                self.process = None

    def stop_program(self):
        """停止正在运行的程序"""
        if self.process and self.is_example_running:
            try:
                self.process.terminate()
                # 等待进程终止
                if not self.process.waitForFinished(2000):  # 等待2秒
                    self.process.kill()
                self.result_text.append("<span style='color: orange;'>程序已被用户终止</span>")

                # 更新状态
                self.is_example_running = False
                self.run_btn.setEnabled(True)
                self.select_run_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.progress_bar.setVisible(False)
                self.current_script_label.setText("当前运行: 无")
                self.current_script_label.setStyleSheet("font-size: 12px; color: gray; margin: 5px;")

                # 安全地清理进程对象
                self.process = None
            except Exception as e:
                print(f"停止程序时出错: {e}")
                self.process = None

    def clear_results(self):
        """清空结果文本框"""
        self.result_text.clear()

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        if self.is_example_running and self.process:
            try:
                self.process.terminate()
                if not self.process.waitForFinished(1000):  # 等待1秒
                    self.process.kill()
            except:
                pass
        event.accept()


# 语音模型窗口类
class VoiceModelWindow(QDialog):
    def __init__(self, parent=None):
        """初始化弹出窗口"""
        super().__init__(parent)
        self.setWindowTitle("语音模型界面")
        self.setGeometry(300, 200, 1000, 800)  # 设置窗口位置和大小

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


