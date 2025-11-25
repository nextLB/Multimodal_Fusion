
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QTextEdit, QPushButton, QGroupBox)


class VoiceControlWindow(QDialog):
    def __init__(self, parent=None):
        """初始化语音控制窗口"""
        super().__init__(parent)
        self.setWindowTitle("语音控制界面")
        self.setGeometry(300, 200, 1000, 800)  # 设置窗口位置和大小

        # 保存父窗口引用，用于调用主窗口的方法
        self.main_window = parent

        # 设置模态，使弹出窗口出现时主窗口不可操作
        self.setModal(True)

        self.init_ui()

    def init_ui(self):
        """初始化语音控制窗口UI"""
        main_layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("语音控制界面")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px; color: #2c3e50;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # 添加说明文本
        instruction_label = QLabel(
            "请在下方输入框中输入指令编号，然后按回车键执行相应功能：\n"
            "1 - 打开视觉模型窗口\n"
            "2 - 打开语音模型窗口\n"
            "3 - 打开多模态融合窗口\n"
            "4 - 打开语音控制窗口\n"
            "help - 显示帮助信息"
        )
        instruction_label.setStyleSheet("font-size: 14px; margin: 15px; color: #34495e;")
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(instruction_label)

        # 创建输入区域
        input_group = self.create_input_group()
        main_layout.addWidget(input_group)

        # 创建输出显示区域
        output_group = self.create_output_group()
        main_layout.addWidget(output_group)

        # 添加按钮区域
        button_layout = self.create_button_layout()
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def create_input_group(self):
        """创建输入区域"""
        group = QGroupBox("指令输入")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                margin: 10px;
                padding: 15px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout()

        # 输入框
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("请输入指令编号（如：1、2、3...）然后按回车键")
        self.input_line.setStyleSheet("""
            QLineEdit {
                font-size: 14px;
                padding: 10px;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin: 10px;
            }
            QLineEdit:focus {
                border-color: #2980b9;
            }
        """)

        # 连接回车键信号
        self.input_line.returnPressed.connect(self.execute_command)

        layout.addWidget(self.input_line)

        group.setLayout(layout)
        return group

    def create_output_group(self):
        """创建输出显示区域"""
        group = QGroupBox("执行结果")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                margin: 10px;
                padding: 15px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout()

        # 输出文本框
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("""
            QTextEdit {
                font-size: 12px;
                padding: 10px;
                border: 1px solid #7f8c8d;
                border-radius: 5px;
                margin: 5px;
                background-color: #ecf0f1;
            }
        """)

        # 添加欢迎信息
        self.output_text.append("语音控制界面已启动！")
        self.output_text.append("请输入指令编号并按回车执行...")
        self.output_text.append("-" * 50)

        layout.addWidget(self.output_text)

        group.setLayout(layout)
        return group

    def create_button_layout(self):
        """创建按钮布局"""
        layout = QHBoxLayout()

        # 清空按钮
        clear_btn = QPushButton("清空输出")
        clear_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px 15px;
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        clear_btn.clicked.connect(self.clear_output)

        # 帮助按钮
        help_btn = QPushButton("显示帮助")
        help_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px 15px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        help_btn.clicked.connect(self.show_help)

        # 关闭按钮
        close_btn = QPushButton("关闭窗口")
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px 15px;
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        close_btn.clicked.connect(self.close)

        layout.addWidget(clear_btn)
        layout.addWidget(help_btn)
        layout.addStretch()
        layout.addWidget(close_btn)

        return layout

    def execute_command(self):
        """执行输入的命令"""
        command = self.input_line.text().strip()

        if not command:
            self.output_text.append("错误：请输入有效的指令！")
            return

        # 清空输入框
        self.input_line.clear()

        # 记录输入的命令
        self.output_text.append(f">>> 执行命令: {command}")

        # 根据命令执行相应的操作
        if command == "1":
            self.open_visual_model()
        elif command == "2":
            self.open_voice_model()
        elif command == "3":
            self.open_multimodal_fusion()
        elif command == "4":
            self.output_text.append("提示：语音控制窗口已经打开！")
        elif command.lower() == "help":
            self.show_help()
        else:
            self.output_text.append(f"错误：未知指令 '{command}'")
            self.output_text.append("请输入 'help' 查看可用指令")

        # 滚动到最新内容
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )

    def open_visual_model(self):
        """打开视觉模型窗口"""
        try:
            if self.main_window:
                self.main_window.open_visual_model_window()
                self.output_text.append("✓ 成功打开视觉模型窗口")
            else:
                self.output_text.append("错误：无法访问主窗口功能")
        except Exception as e:
            self.output_text.append(f"错误：打开视觉模型窗口失败 - {str(e)}")

    def open_voice_model(self):
        """打开语音模型窗口"""
        try:
            if self.main_window:
                self.main_window.open_voice_model_window()
                self.output_text.append("✓ 成功打开语音模型窗口")
            else:
                self.output_text.append("错误：无法访问主窗口功能")
        except Exception as e:
            self.output_text.append(f"错误：打开语音模型窗口失败 - {str(e)}")

    def open_multimodal_fusion(self):
        """打开多模态融合窗口"""
        try:
            if self.main_window:
                self.main_window.open_multimodal_fusion_window()
                self.output_text.append("✓ 成功打开多模态融合窗口")
            else:
                self.output_text.append("错误：无法访问主窗口功能")
        except Exception as e:
            self.output_text.append(f"错误：打开多模态融合窗口失败 - {str(e)}")

    def show_help(self):
        """显示帮助信息"""
        help_text = """
可用指令列表：
1 - 打开视觉模型窗口
2 - 打开语音模型窗口  
3 - 打开多模态融合窗口
4 - 语音控制窗口（当前窗口）
help - 显示此帮助信息

使用方法：
在输入框中输入指令编号，然后按回车键执行。
        """
        self.output_text.append(help_text)

    def clear_output(self):
        """清空输出区域"""
        self.output_text.clear()
        self.output_text.append("输出已清空")
        self.output_text.append("-" * 50)