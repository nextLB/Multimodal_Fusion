
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QTextEdit, QPushButton, QGroupBox)

# 导入自定义模块
sys.path.append('..')
import CSS.voice_control_css as voice_control_css


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

        # 添加按钮区域
        button_layout = self.create_button_layout()
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def create_input_group(self):
        """创建输入区域"""
        group = QGroupBox("指令输入")
        group.setStyleSheet(voice_control_css.COMMMAND_INPUT)

        layout = QVBoxLayout()

        # 输入框
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("请输入指令编号（如：1、2、3...）然后按回车键")
        self.input_line.setStyleSheet(voice_control_css.INSTRUCTION_NUMBER)

        # 连接回车键信号
        self.input_line.returnPressed.connect(self.execute_command)

        layout.addWidget(self.input_line)

        group.setLayout(layout)
        return group

    def create_button_layout(self):
        """创建按钮布局"""
        layout = QHBoxLayout()

        # 帮助按钮
        help_btn = QPushButton("显示帮助")
        help_btn.setStyleSheet(voice_control_css.SHOW_HELP)
        help_btn.clicked.connect(self.show_help)

        # 关闭按钮
        close_btn = QPushButton("关闭窗口")
        close_btn.setStyleSheet(voice_control_css.CLOSE_BUTTON)
        close_btn.clicked.connect(self.close)


        layout.addWidget(help_btn)
        layout.addStretch()
        layout.addWidget(close_btn)

        return layout

    def execute_command(self):
        """执行输入的命令"""
        command = self.input_line.text().strip()

        if not command:
            return

        # 清空输入框
        self.input_line.clear()


        # 根据命令执行相应的操作
        if command == "1":
            self.open_visual_model()
        elif command == "2":
            self.open_voice_model()
        elif command == "3":
            self.open_multimodal_fusion()
        elif command == "4":
            pass
        elif command.lower() == "help":
            self.show_help()
        else:
            print(f"错误：未知指令 '{command}'")
            print("请输入 'help' 查看可用指令")


    def open_visual_model(self):
        """打开视觉模型窗口"""
        try:
            if self.main_window:
                self.main_window.open_visual_model_window()
                print("✓ 成功打开视觉模型窗口")
            else:
                print("错误：无法访问主窗口功能")
        except Exception as e:
            print(f"错误：打开视觉模型窗口失败 - {str(e)}")

    def open_voice_model(self):
        """打开语音模型窗口"""
        try:
            if self.main_window:
                self.main_window.open_voice_model_window()
                print("✓ 成功打开语音模型窗口")
            else:
                print("错误：无法访问主窗口功能")
        except Exception as e:
            print(f"错误：打开语音模型窗口失败 - {str(e)}")

    def open_multimodal_fusion(self):
        """打开多模态融合窗口"""
        try:
            if self.main_window:
                self.main_window.open_multimodal_fusion_window()
                print("✓ 成功打开多模态融合窗口")
            else:
                print("错误：无法访问主窗口功能")
        except Exception as e:
            print(f"错误：打开多模态融合窗口失败 - {str(e)}")

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

        print(help_text)

    # TODO: 获取外部语音的输入，将其转换为文本，传递给self.input_line之类的实现控制
    def listen_audio_convert_text(self):
        pass



