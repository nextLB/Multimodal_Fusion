# CSS/css.py
# 多模态融合APP的样式定义 - Qt兼容版本

# 主样式表
STYLESHEET = """
/* ===== 全局样式 ===== */
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                               stop:0 #2c3e50, stop:1 #34495e);
    color: #ecf0f1;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
}

/* ===== 应用程序全局样式 ===== */
QWidget {
    background: transparent;
    color: #ecf0f1;
}

/* ===== 选项卡控件样式 ===== */
QTabWidget::pane {
    border: 2px solid #34495e;
    border-radius: 8px;
    background: rgba(52, 73, 94, 0.9);
    margin: 2px;
}

QTabWidget::tab-bar {
    alignment: left;
}

/* 左侧选项卡标签 */
QTabBar::tab {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #34495e, stop:1 #2c3e50);
    border: 1px solid #1a252f;
    border-radius: 6px;
    padding: 12px 20px;
    margin: 4px 2px;
    color: #bdc3c7;
    font-weight: bold;
    min-width: 80px;
}

/* 选项卡悬停效果 */
QTabBar::tab:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #3498db, stop:1 #2980b9);
    color: white;
    border: 1px solid #2980b9;
}

/* 选项卡选中状态 */
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #e74c3c, stop:1 #c0392b);
    color: white;
    border: 1px solid #c0392b;
    font-weight: bold;
}

/* 选项卡禁用状态 */
QTabBar::tab:disabled {
    background: #7f8c8d;
    color: #95a5a6;
}

/* ===== 按钮样式 ===== */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #3498db, stop:1 #2980b9);
    border: 2px solid #2980b9;
    border-radius: 8px;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    margin: 5px;
    min-width: 100px;
}

/* 按钮悬停效果 */
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #5dade2, stop:1 #3498db);
    border: 2px solid #5dade2;
}

/* 按钮按下效果 */
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #2471a3, stop:1 #1b4f72);
    border: 2px solid #1b4f72;
}

/* 按钮禁用状态 */
QPushButton:disabled {
    background: #7f8c8d;
    border: 2px solid #95a5a6;
    color: #bdc3c7;
}

/* ===== 标签样式 ===== */
QLabel {
    background: transparent;
    color: #ecf0f1;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 14px;
}

/* 重要标签样式 */
QLabel[important="true"] {
    background: rgba(231, 76, 60, 0.2);
    border: 1px solid #e74c3c;
    color: #e74c3c;
    font-weight: bold;
}

/* 标签悬停效果 */
QLabel:hover {
    background: rgba(52, 152, 219, 0.2);
    color: #5dade2;
}

/* ===== 输入框样式 ===== */
QLineEdit, QTextEdit, QPlainTextEdit {
    background: rgba(236, 240, 241, 0.1);
    border: 2px solid #34495e;
    border-radius: 6px;
    padding: 8px 12px;
    color: #ecf0f1;
    font-size: 14px;
    selection-background-color: #3498db;
}

/* 输入框悬停效果 */
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {
    border: 2px solid #5dade2;
    background: rgba(236, 240, 241, 0.15);
}

/* 输入框焦点效果 */
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 2px solid #3498db;
    background: rgba(236, 240, 241, 0.2);
}

/* ===== 组合框样式 ===== */
QComboBox {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #34495e, stop:1 #2c3e50);
    border: 2px solid #34495e;
    border-radius: 6px;
    padding: 8px 12px;
    color: #ecf0f1;
    min-width: 120px;
}

QComboBox:hover {
    border: 2px solid #5dade2;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #3d566e, stop:1 #34495e);
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #34495e;
    border-left-style: solid;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}

QComboBox QAbstractItemView {
    background: #2c3e50;
    border: 1px solid #34495e;
    selection-background-color: #3498db;
    color: #ecf0f1;
}

/* ===== 滚动条样式 ===== */
QScrollBar:vertical {
    background: #34495e;
    width: 15px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #3498db;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #5dade2;
}

QScrollBar::handle:vertical:pressed {
    background: #2471a3;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ===== 进度条样式 ===== */
QProgressBar {
    border: 2px solid #34495e;
    border-radius: 8px;
    text-align: center;
    color: white;
    background: #2c3e50;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                               stop:0 #e74c3c, stop:0.5 #3498db, stop:1 #2ecc71);
    border-radius: 6px;
}

/* ===== 分组框样式 ===== */
QGroupBox {
    border: 2px solid #34495e;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    background: rgba(52, 73, 94, 0.5);
    font-weight: bold;
    color: #ecf0f1;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 8px;
    background: #e74c3c;
    border-radius: 4px;
    color: white;
}

/* ===== 菜单栏样式 ===== */
QMenuBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #34495e, stop:1 #2c3e50);
    color: #ecf0f1;
    border-bottom: 1px solid #34495e;
}

QMenuBar::item {
    background: transparent;
    padding: 4px 12px;
}

QMenuBar::item:selected {
    background: #3498db;
    border-radius: 4px;
}

QMenu {
    background: #2c3e50;
    border: 1px solid #34495e;
    color: #ecf0f1;
}

QMenu::item {
    padding: 4px 20px;
}

QMenu::item:selected {
    background: #3498db;
}

/* ===== 状态栏样式 ===== */
QStatusBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #2c3e50, stop:1 #34495e);
    color: #bdc3c7;
    border-top: 1px solid #34495e;
}

/* ===== 工具提示样式 ===== */
QToolTip {
    background: #e74c3c;
    color: white;
    border: 1px solid #c0392b;
    border-radius: 4px;
    padding: 4px 8px;
}

/* ===== 自定义按钮类 ===== */
.animated-button {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #2ecc71, stop:1 #27ae60);
    border: 2px solid #27ae60;
}

.animated-button:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #58d68d, stop:1 #2ecc71);
    border: 2px solid #2ecc71;
}

.warning-button {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #e67e22, stop:1 #d35400);
    border: 2px solid #d35400;
}

.warning-button:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #f39c12, stop:1 #e67e22);
    border: 2px solid #e67e22;
}

/* 特殊标题标签 */
.title-label {
    font-size: 18px;
    font-weight: bold;
    color: #3498db;
    background: rgba(52, 152, 219, 0.1);
    border: 1px solid #3498db;
    border-radius: 8px;
    padding: 12px;
    margin: 5px;
}

.subtitle-label {
    font-size: 14px;
    font-weight: bold;
    color: #2ecc71;
    background: rgba(46, 204, 113, 0.1);
    border: 1px solid #2ecc71;
    border-radius: 6px;
    padding: 8px;
    margin: 3px;
}
"""

# 极简版样式表（如果还有问题使用这个）
MINIMAL_STYLESHEET = """
QMainWindow {
    background: #2c3e50;
    color: #ecf0f1;
}

QTabWidget::pane {
    border: 1px solid #34495e;
    background: #34495e;
}

QTabBar::tab {
    background: #34495e;
    border: 1px solid #1a252f;
    padding: 8px 16px;
    color: #bdc3c7;
}

QTabBar::tab:hover {
    background: #3498db;
    color: white;
}

QTabBar::tab:selected {
    background: #e74c3c;
    color: white;
}

QPushButton {
    background: #3498db;
    border: 1px solid #2980b9;
    color: white;
    padding: 8px 16px;
}

QPushButton:hover {
    background: #5dade2;
}

QPushButton:pressed {
    background: #2471a3;
}

QLabel {
    color: #ecf0f1;
}
"""


def apply_stylesheet(app):
    """
    应用样式表到应用程序

    Args:
        app: QApplication实例
    """
    try:
        app.setStyleSheet(STYLESHEET)
        print("样式表应用成功！")
    except Exception as e:
        print(f"样式表应用失败: {e}")
        # 如果完整样式表有问题，使用极简版
        app.setStyleSheet(MINIMAL_STYLESHEET)
        print("已应用极简版样式表")


def get_stylesheet():
    """
    获取样式表字符串

    Returns:
        str: 样式表字符串
    """
    return STYLESHEET


def get_minimal_stylesheet():
    """
    获取极简版样式表字符串

    Returns:
        str: 极简版样式表字符串
    """
    return MINIMAL_STYLESHEET