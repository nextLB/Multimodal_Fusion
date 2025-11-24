"""
样式定义文件
"""

# 主窗口样式
MAIN_WINDOW_STYLE = """
QMainWindow {
    background-color: #f5f7fa;
}
"""

# 按钮样式
BUTTON_STYLES = {
    "primary": """
        QPushButton {
            background-color: #4a90e2;
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            min-width: 120px;
        }
        QPushButton:hover {
            background-color: #357abd;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    """,

    "danger": """
        QPushButton {
            background-color: #e74c3c;
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            min-width: 120px;
        }
        QPushButton:hover {
            background-color: #c0392b;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    """,

    "success": """
        QPushButton {
            background-color: #27ae60;
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            min-width: 120px;
        }
        QPushButton:hover {
            background-color: #219653;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    """
}

# 文本编辑框样式
TEXT_EDIT_STYLE = """
QTextEdit {
    background-color: #2c3e50;
    color: #ecf0f1;
    border: 1px solid #34495e;
    border-radius: 5px;
    padding: 10px;
    font-family: 'Courier New', monospace;
}
"""

# 标签样式
LABEL_STYLES = {
    "default": """
        QLabel {
            color: #2c3e50;
            font-weight: bold;
        }
    """,

    "title": """
        QLabel {
            font-size: 24px; 
            color: #2c3e50; 
            margin-bottom: 10px;
        }
    """,

    "status": """
        QLabel {
            color: #7f8c8d; 
            font-size: 12px;
        }
    """,

    "path": """
        QLabel {
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 3px;
            padding: 8px;
            color: #2c3e50;
        }
    """
}

# 进度条样式
PROGRESS_BAR_STYLE = """
QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 5px;
    text-align: center;
    background-color: #ecf0f1;
}
QProgressBar::chunk {
    background-color: #27ae60;
    border-radius: 4px;
}
"""

# 框架样式
FRAME_STYLE = """
QFrame {
    background-color: white;
    border: 1px solid #e1e8ed;
    border-radius: 8px;
    padding: 15px;
}
"""

# 组合样式
COMPLETE_STYLE = f"""
{MAIN_WINDOW_STYLE}

QPushButton {{
    {BUTTON_STYLES['primary'].split('QPushButton {')[1].split('}')[0]}
}}

QPushButton#stopButton {{
    {BUTTON_STYLES['danger'].split('QPushButton {')[1].split('}')[0]}
}}

QPushButton#successButton {{
    {BUTTON_STYLES['success'].split('QPushButton {')[1].split('}')[0]}
}}

{TEXT_EDIT_STYLE}

{LABEL_STYLES['default']}

{PROGRESS_BAR_STYLE}
"""