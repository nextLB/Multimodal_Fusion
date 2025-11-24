#!/usr/bin/env python3
"""
现代化PyQt脚本运行器 - 主程序
"""

import sys
import os
import datetime


from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QLabel,
                             QFileDialog, QMessageBox, QProgressBar, QFrame,
                             QTabWidget, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon

# 导入自定义模块
sys.path.append('..')
import CSS.css as css
from core.script_runner import ScriptRunnerThread
from core.extensions import ExtensionManager, ResultAnalyzer, ScriptManager, ProgressTracker


class ModernScriptRunner(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化扩展管理器
        self.extension_manager = ExtensionManager()
        self.setup_extensions()

        # 初始化进度跟踪器
        self.progress_tracker = ProgressTracker()

        # 初始化UI
        self.init_ui()

        # 当前状态
        self.script_runner = None
        self.current_script_path = None
        self.output_history = []

    def setup_extensions(self):
        """设置扩展"""
        self.extension_manager.register_extension("result_analyzer", ResultAnalyzer())
        self.extension_manager.register_extension("script_manager", ScriptManager())

    def init_ui(self):
        """初始化现代化UI界面"""
        self.setWindowTitle("Python脚本运行器 - 现代化界面")
        self.setGeometry(300, 200, 1000, 800)

        # 设置样式
        self.setStyleSheet(css.COMPLETE_STYLE)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 添加标题
        self.create_title_section(main_layout)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 主控制选项卡
        self.main_tab = self.create_main_tab()
        self.tab_widget.addTab(self.main_tab, "脚本运行")

        # 扩展选项卡
        self.extensions_tab = self.create_extensions_tab()
        self.tab_widget.addTab(self.extensions_tab, "扩展功能")

        # 添加状态栏
        self.create_status_section(main_layout)

        # 初始化扩展
        self.extension_manager.initialize_extensions(self)

    def create_title_section(self, layout):
        """创建标题区域"""
        title_label = QLabel("Python脚本运行器")
        title_label.setStyleSheet(css.LABEL_STYLES["title"])
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # 添加描述
        desc_label = QLabel("运行外部Python脚本并实时查看输出结果")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #7f8c8d; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(desc_label)

    def create_main_tab(self):
        """创建主控制选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # 脚本选择区域
        script_selection_widget = self.create_script_selection_widget()
        layout.addWidget(script_selection_widget)

        # 进度显示区域
        progress_widget = self.create_progress_widget()
        layout.addWidget(progress_widget)

        # 输出显示区域
        output_widget = self.create_output_widget()
        layout.addWidget(output_widget, 1)  # 给输出区域更多空间

        # 控制按钮区域
        control_widget = self.create_control_widget()
        layout.addWidget(control_widget)

        return tab

    def create_extensions_tab(self):
        """创建扩展功能选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 结果分析扩展
        analysis_group = QGroupBox("结果分析")
        analysis_layout = QVBoxLayout(analysis_group)

        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        self.analysis_result_text.setMaximumHeight(150)
        analysis_layout.addWidget(self.analysis_result_text)

        analyze_btn = QPushButton("分析最近输出")
        analyze_btn.setObjectName("successButton")
        analyze_btn.clicked.connect(self.analyze_output)
        analysis_layout.addWidget(analyze_btn)

        layout.addWidget(analysis_group)

        # 脚本历史扩展
        history_group = QGroupBox("脚本历史")
        history_layout = QVBoxLayout(history_group)

        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        history_layout.addWidget(self.history_text)

        layout.addWidget(history_group)

        # 添加弹性空间
        layout.addStretch(1)

        return tab

    def create_script_selection_widget(self):
        """创建脚本选择部件"""
        frame = QFrame()
        frame.setStyleSheet(css.FRAME_STYLE)
        layout = QHBoxLayout(frame)

        self.script_path_label = QLabel("未选择脚本")
        self.script_path_label.setStyleSheet(css.LABEL_STYLES["path"])
        self.script_path_label.setMinimumHeight(40)

        browse_btn = QPushButton("选择脚本")
        browse_btn.clicked.connect(self.browse_script)

        layout.addWidget(QLabel("目标脚本:"))
        layout.addWidget(self.script_path_label, 1)
        layout.addWidget(browse_btn)

        return frame

    def create_progress_widget(self):
        """创建进度显示部件"""
        frame = QFrame()
        frame.setStyleSheet(css.FRAME_STYLE)
        layout = QVBoxLayout(frame)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 进度消息
        self.progress_label = QLabel("就绪")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #2c3e50; font-size: 14px;")
        layout.addWidget(self.progress_label)

        return frame

    def create_output_widget(self):
        """创建输出显示部件"""
        frame = QFrame()
        frame.setStyleSheet(css.FRAME_STYLE)
        layout = QVBoxLayout(frame)

        # 输出标题和清空按钮
        output_header = QHBoxLayout()
        output_header.addWidget(QLabel("执行输出:"))

        clear_btn = QPushButton("清空输出")
        clear_btn.clicked.connect(self.clear_output)
        output_header.addWidget(clear_btn)
        output_header.addStretch(1)

        layout.addLayout(output_header)

        # 输出文本区域
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.output_text)

        return frame

    def create_control_widget(self):
        """创建控制按钮部件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.run_btn = QPushButton("运行脚本")
        self.run_btn.clicked.connect(self.run_script)
        self.run_btn.setEnabled(False)

        self.stop_btn = QPushButton("停止运行")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self.stop_script)
        self.stop_btn.setEnabled(False)

        layout.addWidget(self.run_btn)
        layout.addWidget(self.stop_btn)

        return widget

    def create_status_section(self, layout):
        """创建状态栏区域"""
        self.status_label = QLabel("就绪 - 请选择要运行的Python脚本")
        self.status_label.setStyleSheet(css.LABEL_STYLES["status"])
        layout.addWidget(self.status_label)

    def browse_script(self):
        """浏览选择Python脚本"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择Python脚本",
            "",
            "Python文件 (*.py);;所有文件 (*)"
        )

        if file_path:
            self.current_script_path = file_path
            self.script_path_label.setText(file_path)
            self.run_btn.setEnabled(True)
            self.status_label.setText(f"已选择脚本: {os.path.basename(file_path)}")
            self.clear_output()
            self.append_output(f"已加载脚本: {file_path}")
            self.append_output(f"加载时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.append_output("-" * 60)

    def run_script(self):
        """运行选定的脚本"""
        if not self.current_script_path or not os.path.exists(self.current_script_path):
            QMessageBox.warning(self, "错误", "请先选择有效的Python脚本文件")
            return

        # 更新UI状态
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 清空输出并显示开始信息
        self.append_output(f"开始执行脚本: {self.current_script_path}")
        self.append_output(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.append_output("=" * 60)

        # 在单独线程中运行脚本
        self.script_runner = ScriptRunnerThread(self.current_script_path)

        # 连接信号
        self.script_runner.output_signal.connect(self.append_output)
        self.script_runner.progress_signal.connect(self.update_progress)
        self.script_runner.finished_signal.connect(self.script_finished)
        self.script_runner.error_signal.connect(self.show_error)

        # 启动线程
        self.script_runner.start()

        # 更新状态
        self.status_label.setText("正在运行脚本...")

        # 记录到脚本历史
        script_manager = self.extension_manager.get_extension("script_manager")
        if script_manager:
            script_manager.add_to_history(
                self.current_script_path,
                datetime.datetime.now(),
                "running"
            )

    def stop_script(self):
        """停止脚本运行"""
        if self.script_runner and self.script_runner.isRunning():
            self.script_runner.stop()
            self.script_runner.wait(2000)  # 等待2秒
            self.append_output("\n*** 用户请求停止执行 ***")
            self.update_progress(0, "执行被用户中断")

            # 记录到脚本历史
            script_manager = self.extension_manager.get_extension("script_manager")
            if script_manager:
                script_manager.add_to_history(
                    self.current_script_path,
                    datetime.datetime.now(),
                    "stopped"
                )

    def script_finished(self, success, message):
        """脚本执行完成回调"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        status = "成功" if success else "失败"
        self.append_output("=" * 60)
        self.append_output(f"执行{status}: {message}")
        self.append_output(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.status_label.setText(f"执行完成 - {message}")

        # 记录到脚本历史
        script_manager = self.extension_manager.get_extension("script_manager")
        if script_manager:
            script_manager.add_to_history(
                self.current_script_path,
                datetime.datetime.now(),
                "success" if success else "failed"
            )
            self.update_history_display()

        # 显示完成消息
        if success:
            QMessageBox.information(self, "完成", "脚本执行成功完成！")
        else:
            QMessageBox.warning(self, "完成", f"脚本执行完成但遇到问题: {message}")

    def update_progress(self, value, message):
        """更新进度显示"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

        # 控制进度条可见性
        if value == 0:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)

            # 如果是100%，3秒后隐藏进度条
            if value == 100:
                QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))

    def show_error(self, error_message):
        """显示错误信息"""
        self.append_output(f"错误: {error_message}")
        self.status_label.setText(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)

    def append_output(self, text):
        """追加输出到文本区域"""
        self.output_text.append(text)
        self.output_history.append(text)

        # 自动滚动到底部
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_output(self):
        """清空输出区域"""
        self.output_text.clear()
        self.output_history.clear()

    def analyze_output(self):
        """分析输出结果"""
        analyzer = self.extension_manager.get_extension("result_analyzer")
        if analyzer and self.output_history:
            output_text = "\n".join(self.output_history)
            result = analyzer.analyze_output(output_text)

            # 显示分析结果
            analysis_text = f"""分析结果:
总输出行数: {result['total_lines']}
成功关键词出现次数: {result['success_keywords']}
错误关键词出现次数: {result['error_keywords']}
最后一行输出: {result['last_line']}
            """
            self.analysis_result_text.setText(analysis_text)

    def update_history_display(self):
        """更新历史记录显示"""
        script_manager = self.extension_manager.get_extension("script_manager")
        if script_manager:
            history = script_manager.get_history()
            history_text = ""
            for item in history[-10:]:  # 显示最近10条记录
                status_icon = "✅" if item["status"] == "success" else "❌" if item["status"] == "failed" else "⏹️"
                history_text += f"{status_icon} {item['timestamp'].strftime('%H:%M:%S')} - {os.path.basename(item['path'])}\n"

            self.history_text.setText(history_text or "暂无历史记录")


def main():
    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("Python脚本运行器")
    app.setApplicationVersion("2.0")

    # 创建并显示主窗口
    window = ModernScriptRunner()
    window.show()

    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()