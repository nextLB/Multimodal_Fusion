import sys
import json
import numpy as np
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel,
                             QSlider, QComboBox, QFileDialog, QCheckBox,
                             QMessageBox, QProgressBar, QGroupBox,
                             QSplitter, QFrame, QScrollArea, QProgressDialog)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette, QLinearGradient, QPainter
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import os
from typing import Dict, List, Optional
import time


class ModernGLWidget(QOpenGLWidget):
    """
    现代化OpenGL点云可视化组件 - 简化版本，避免VAO问题
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pointClouds = {}  # 存储点云数据
        self.visibleClouds = {}  # 可见的点云
        self.colors = {}  # 点云颜色
        self.pointSizes = {}  # 各个点云的点大小

        # 相机参数
        self.cameraDistance = 800.0
        self.cameraRotationX = -45.0
        self.cameraRotationY = 0.0
        self.cameraTarget = [0.0, 0.0, 0.0]

        # 鼠标控制
        self.lastMousePos = None
        self.isRotating = False
        self.isPanning = False

        # 点大小
        self.basePointSize = 2.0

        # GPU数据 - 使用简单的VBO方法
        self.gpuBuffers = {}  # 存储GPU上的VBO

        # 渲染优化
        self.frameCount = 0
        self.lastFpsTime = time.time()
        self.fps = 0

        # 可视化设置
        self.showAxes = True
        self.showGrid = True
        self.backgroundColor = [0.08, 0.08, 0.12, 1.0]

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(1000, 700)

        # 设置定时器用于FPS计算
        self.fpsTimer = QTimer(self)
        self.fpsTimer.timeout.connect(self.updateFPS)
        self.fpsTimer.start(1000)  # 每秒更新一次FPS

    def initializeGL(self):
        """初始化OpenGL"""
        try:
            # 检查OpenGL版本
            version = glGetString(GL_VERSION).decode()
            vendor = glGetString(GL_VENDOR).decode()
            renderer = glGetString(GL_RENDERER).decode()

            print(f"OpenGL版本: {version}")
            print(f"GPU厂商: {vendor}")
            print(f"渲染器: {renderer}")

            # 初始化OpenGL状态
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_PROGRAM_POINT_SIZE)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glClearColor(*self.backgroundColor)

            print("OpenGL初始化成功")

        except Exception as e:
            print(f"OpenGL初始化错误: {e}")
            QMessageBox.critical(self, "OpenGL错误", f"OpenGL初始化失败: {e}")

    def resizeGL(self, w, h):
        """调整视口大小"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h > 0 else 1.0
        gluPerspective(45.0, aspect, 1.0, 20000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """渲染场景"""
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # 设置相机
            glTranslatef(0.0, 0.0, -self.cameraDistance)
            glRotatef(self.cameraRotationX, 1.0, 0.0, 0.0)
            glRotatef(self.cameraRotationY, 0.0, 1.0, 0.0)
            glTranslatef(-self.cameraTarget[0], -self.cameraTarget[1], -self.cameraTarget[2])

            # 渲染坐标轴和网格
            if self.showAxes:
                self._drawCoordinateAxes()
            if self.showGrid:
                self._drawGrid()

            # 渲染点云 - 使用VBO或立即模式
            for cloud_name in list(self.visibleClouds.keys()):
                if cloud_name in self.pointClouds:
                    self._renderPointCloud(cloud_name)

            self.frameCount += 1

        except Exception as e:
            print(f"渲染错误: {e}")

    def _drawCoordinateAxes(self):
        """绘制坐标轴"""
        glLineWidth(2.0)
        glBegin(GL_LINES)

        # X轴 - 红色
        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(200.0, 0.0, 0.0)

        # Y轴 - 绿色
        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 200.0, 0.0)

        # Z轴 - 蓝色
        glColor3f(0.2, 0.4, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 200.0)

        glEnd()

        # 坐标轴标签
        self._drawText3D(210, 0, 0, "X", (1.0, 0.2, 0.2))
        self._drawText3D(0, 210, 0, "Y", (0.2, 1.0, 0.2))
        self._drawText3D(0, 0, 210, "Z", (0.2, 0.4, 1.0))

    def _drawGrid(self):
        """绘制地面网格"""
        grid_size = 1000
        grid_step = 50
        half_size = grid_size // 2

        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(0.3, 0.3, 0.3, 0.6)

        for i in range(-half_size, half_size + 1, grid_step):
            # 水平线
            glVertex3f(-half_size, i, 0)
            glVertex3f(half_size, i, 0)
            # 垂直线
            glVertex3f(i, -half_size, 0)
            glVertex3f(i, half_size, 0)

        glEnd()

    def _drawText3D(self, x, y, z, text, color):
        """在3D空间中绘制文本（简单实现）"""
        # 在实际应用中，这里应该使用纹理字体渲染
        # 这里使用点来简单表示标签位置
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glColor3f(*color)
        glVertex3f(x, y, z)
        glEnd()

    def _renderPointCloud(self, cloud_name: str):
        """渲染单个点云"""
        if cloud_name not in self.colors:
            return

        color = self.colors[cloud_name]
        glColor3f(color[0], color[1], color[2])

        points = self.pointClouds[cloud_name]
        point_size = self.pointSizes.get(cloud_name, self.basePointSize)
        glPointSize(point_size)

        # 使用VBO渲染（如果可用）
        if cloud_name in self.gpuBuffers:
            self._renderWithVBO(cloud_name, points)
        else:
            # 使用优化的立即模式渲染
            self._renderOptimizedImmediate(points)

    def _renderWithVBO(self, cloud_name: str, points: np.ndarray):
        """使用VBO渲染（避免VAO问题）"""
        try:
            vbo = self.gpuBuffers[cloud_name]
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glEnableClientState(GL_VERTEX_ARRAY)

            glDrawArrays(GL_POINTS, 0, len(points))

            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        except Exception as e:
            print(f"VBO渲染失败: {e}")
            # 回退到立即模式
            self._renderOptimizedImmediate(points)

    def _renderOptimizedImmediate(self, points: np.ndarray):
        """优化的立即模式渲染"""
        try:
            # 使用glDrawArrays替代glBegin/glEnd
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, points)
            glDrawArrays(GL_POINTS, 0, len(points))
            glDisableClientState(GL_VERTEX_ARRAY)
        except Exception as e:
            print(f"顶点数组渲染失败，使用传统模式: {e}")
            # 最终回退到传统立即模式
            self._renderTraditionalImmediate(points)

    def _renderTraditionalImmediate(self, points: np.ndarray):
        """传统立即模式渲染（备用方案）"""
        glBegin(GL_POINTS)
        for i in range(len(points)):
            glVertex3f(points[i][0], points[i][1], points[i][2])
        glEnd()

    def updateFPS(self):
        """更新FPS显示"""
        current_time = time.time()
        if current_time - self.lastFpsTime > 0:
            self.fps = self.frameCount / (current_time - self.lastFpsTime)
        self.frameCount = 0
        self.lastFpsTime = current_time

    def loadPointCloud(self, filename: str, cloud_name: str = None) -> bool:
        """
        加载点云数据

        Args:
            filename: JSON文件名
            cloud_name: 点云名称，如果为None则使用文件名

        Returns:
            bool: 是否加载成功
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            points = np.array(data['points'], dtype=np.float32)

            if cloud_name is None:
                cloud_name = os.path.splitext(os.path.basename(filename))[0]

            self.pointClouds[cloud_name] = points
            self.visibleClouds[cloud_name] = True

            # 为不同类型点云分配特定颜色
            if 'ground' in cloud_name.lower():
                self.colors[cloud_name] = (0.4, 0.3, 0.2)  # 大地色
                self.pointSizes[cloud_name] = 1.5
            elif 'road' in cloud_name.lower():
                self.colors[cloud_name] = (0.3, 0.3, 0.3)  # 灰色
                self.pointSizes[cloud_name] = 1.8
            elif 'building' in cloud_name.lower():
                self.colors[cloud_name] = (0.7, 0.5, 0.3)  # 建筑色
                self.pointSizes[cloud_name] = 2.0
            elif 'tree' in cloud_name.lower():
                self.colors[cloud_name] = (0.2, 0.6, 0.2)  # 绿色
                self.pointSizes[cloud_name] = 2.2
            elif 'water' in cloud_name.lower() or 'lake' in cloud_name.lower() or 'stream' in cloud_name.lower():
                self.colors[cloud_name] = (0.2, 0.4, 0.8)  # 蓝色
                self.pointSizes[cloud_name] = 2.0
            elif 'cloud' in cloud_name.lower():
                self.colors[cloud_name] = (1.0, 1.0, 1.0)  # 白色
                self.pointSizes[cloud_name] = 3.0
            elif 'farm' in cloud_name.lower():
                self.colors[cloud_name] = (0.3, 0.5, 0.2)  # 农田绿
                self.pointSizes[cloud_name] = 1.8
            else:
                # 生成随机但美观的颜色
                hue = np.random.random()
                saturation = 0.7 + np.random.random() * 0.3
                value = 0.6 + np.random.random() * 0.4
                self.colors[cloud_name] = self.hsv_to_rgb(hue, saturation, value)
                self.pointSizes[cloud_name] = self.basePointSize

            # 尝试创建GPU缓冲（仅VBO，无VAO）
            self._createVBOBuffer(cloud_name, points)

            print(f"✓ 加载点云 '{cloud_name}': {len(points):,} 个点")
            self.update()
            return True

        except Exception as e:
            print(f"✗ 加载点云失败 {filename}: {e}")
            return False

    def hsv_to_rgb(self, h, s, v):
        """HSV转RGB"""
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        if i == 5:
            return (v, p, q)

    def _createVBOBuffer(self, cloud_name: str, points: np.ndarray):
        """在GPU上创建顶点缓冲对象（仅VBO，无VAO）"""
        try:
            # 创建VBO
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            self.gpuBuffers[cloud_name] = vbo
            print(f"为 '{cloud_name}' 创建VBO缓冲")

        except Exception as e:
            print(f"创建VBO缓冲失败: {e}")
            # 如果VBO缓冲创建失败，我们仍然可以使用立即模式

    def setCloudVisibility(self, cloud_name: str, visible: bool):
        """设置点云可见性"""
        if cloud_name in self.pointClouds:
            if visible:
                self.visibleClouds[cloud_name] = True
            else:
                self.visibleClouds.pop(cloud_name, None)
            self.update()

    def setPointSize(self, size: float):
        """设置基础点大小"""
        self.basePointSize = max(1.0, min(10.0, size))
        self.update()

    def setCloudPointSize(self, cloud_name: str, size: float):
        """设置特定点云的点大小"""
        self.pointSizes[cloud_name] = max(0.5, min(15.0, size))
        self.update()

    def resetCamera(self):
        """重置相机"""
        self.cameraDistance = 800.0
        self.cameraRotationX = -45.0
        self.cameraRotationY = 0.0
        self.cameraTarget = [0.0, 0.0, 0.0]
        self.update()

    def setBackgroundColor(self, color):
        """设置背景颜色"""
        self.backgroundColor = color
        self.makeCurrent()
        glClearColor(*color)
        self.doneCurrent()
        self.update()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.isRotating = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.isPanning = True

        self.lastMousePos = event.position()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.isRotating = False
        self.isPanning = False
        self.lastMousePos = None

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.lastMousePos is None:
            return

        delta = event.position() - self.lastMousePos

        if self.isRotating:
            self.cameraRotationY += delta.x() * 0.5
            self.cameraRotationX += delta.y() * 0.5
            self.cameraRotationX = max(-90.0, min(90.0, self.cameraRotationX))

        elif self.isPanning:
            pan_sensitivity = self.cameraDistance * 0.0015
            self.cameraTarget[0] -= delta.x() * pan_sensitivity
            self.cameraTarget[1] += delta.y() * pan_sensitivity

        self.lastMousePos = event.position()
        self.update()

    def wheelEvent(self, event):
        """鼠标滚轮事件"""
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            self.cameraDistance /= zoom_factor
        else:
            self.cameraDistance *= zoom_factor

        self.cameraDistance = max(10.0, min(10000.0, self.cameraDistance))
        self.update()


class ModernControlPanel(QWidget):
    """
    现代化控制面板
    """

    visibilityChanged = pyqtSignal(str, bool)
    pointSizeChanged = pyqtSignal(float)
    cloudPointSizeChanged = pyqtSignal(str, float)
    resetCameraRequested = pyqtSignal()
    loadDataRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cloudCheckboxes = {}
        self.cloudSliders = {}
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title = QLabel("点云可视化控制系统")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 数据加载部分
        load_group = self._createLoadGroup()
        layout.addWidget(load_group)

        # 显示控制部分
        display_group = self._createDisplayGroup()
        layout.addWidget(display_group)

        # 点云控制部分
        clouds_group = self._createCloudsGroup()
        layout.addWidget(clouds_group)

        # 状态信息
        status_group = self._createStatusGroup()
        layout.addWidget(status_group)

        # 操作说明
        info_group = self._createInfoGroup()
        layout.addWidget(info_group)

        layout.addStretch()

    def _createLoadGroup(self):
        """创建数据加载组"""
        group = QGroupBox("数据管理")
        group.setStyleSheet(self._getGroupBoxStyle())
        layout = QVBoxLayout(group)

        load_btn = QPushButton("📁 加载点云数据目录")
        load_btn.setStyleSheet(self._getButtonStyle("primary"))
        load_btn.clicked.connect(self.loadDataRequested.emit)
        layout.addWidget(load_btn)

        return group

    def _createDisplayGroup(self):
        """创建显示控制组"""
        group = QGroupBox("显示设置")
        group.setStyleSheet(self._getGroupBoxStyle())
        layout = QVBoxLayout(group)

        # 点大小控制
        size_layout = QHBoxLayout()
        size_label = QLabel("基础点大小:")
        size_label.setStyleSheet("color: #ffffff;")
        size_layout.addWidget(size_label)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 10)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(self.pointSizeChanged.emit)
        size_layout.addWidget(self.size_slider)
        layout.addLayout(size_layout)

        # 相机控制
        camera_btn = QPushButton("🔄 重置相机视角")
        camera_btn.setStyleSheet(self._getButtonStyle("secondary"))
        camera_btn.clicked.connect(self.resetCameraRequested.emit)
        layout.addWidget(camera_btn)

        return group

    def _createCloudsGroup(self):
        """创建点云控制组"""
        group = QGroupBox("点云控制")
        group.setStyleSheet(self._getGroupBoxStyle())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.clouds_container = QWidget()
        self.clouds_layout = QVBoxLayout(self.clouds_container)
        self.clouds_layout.setSpacing(8)

        scroll.setWidget(self.clouds_container)

        layout = QVBoxLayout(group)
        layout.addWidget(scroll)

        # 全选/全不选按钮
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.setStyleSheet(self._getButtonStyle("success"))
        select_all_btn.clicked.connect(self.selectAllClouds)
        select_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.setStyleSheet(self._getButtonStyle("danger"))
        deselect_all_btn.clicked.connect(self.deselectAllClouds)
        select_layout.addWidget(deselect_all_btn)

        layout.addLayout(select_layout)

        return group

    def _createStatusGroup(self):
        """创建状态信息组"""
        group = QGroupBox("系统状态")
        group.setStyleSheet(self._getGroupBoxStyle())
        layout = QVBoxLayout(group)

        self.status_label = QLabel("就绪 - 请加载点云数据")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #cccccc; 
                background-color: #2a2a2a; 
                border-radius: 6px; 
                padding: 10px;
                font-size: 12px;
            }
        """)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # 性能指标
        perf_layout = QHBoxLayout()
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet("color: #ffffff;")
        perf_layout.addWidget(fps_label)

        self.fps_value = QLabel("0")
        self.fps_value.setStyleSheet("color: #00ff00; font-weight: bold;")
        perf_layout.addWidget(self.fps_value)
        perf_layout.addStretch()

        layout.addLayout(perf_layout)

        return group

    def _createInfoGroup(self):
        """创建操作说明组"""
        group = QGroupBox("操作说明")
        group.setStyleSheet(self._getGroupBoxStyle())
        layout = QVBoxLayout(group)

        info_text = """
        • 左键拖动: 旋转视角
        • 右键拖动: 平移视角  
        • 鼠标滚轮: 缩放
        • 可单独控制每个点云可见性
        • 支持GPU加速渲染
        • 渲染所有数据点
        """

        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #aaaaaa; font-size: 11px; line-height: 1.4;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        return group

    def _getGroupBoxStyle(self):
        """获取GroupBox样式"""
        return """
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """

    def _getButtonStyle(self, button_type):
        """获取按钮样式"""
        styles = {
            "primary": """
                QPushButton {
                    background-color: #007acc;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 6px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #005a9e;
                }
                QPushButton:pressed {
                    background-color: #004578;
                }
            """,
            "secondary": """
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #545b62;
                }
            """,
            "success": """
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """,
            "danger": """
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """
        }
        return styles.get(button_type, styles["secondary"])

    def addCloudControl(self, cloud_name: str, points_count: int):
        """添加点云控制项"""
        if cloud_name in self.cloudCheckboxes:
            return

        # 创建点云控制项容器
        cloud_widget = QWidget()
        cloud_layout = QHBoxLayout(cloud_widget)
        cloud_layout.setContentsMargins(5, 2, 5, 2)

        # 复选框
        checkbox = QCheckBox(f"{cloud_name}")
        checkbox.setChecked(True)
        checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #666666;
                background-color: #333333;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #007acc;
                background-color: #007acc;
                border-radius: 3px;
            }
        """)
        checkbox.toggled.connect(
            lambda checked, name=cloud_name: self.visibilityChanged.emit(name, checked)
        )
        cloud_layout.addWidget(checkbox)

        # 点数量标签
        count_label = QLabel(f"({points_count:,})")
        count_label.setStyleSheet("color: #888888; font-size: 10px;")
        count_label.setFixedWidth(80)
        cloud_layout.addWidget(count_label)

        # 点大小滑块
        size_slider = QSlider(Qt.Orientation.Horizontal)
        size_slider.setRange(1, 15)
        size_slider.setValue(2)
        size_slider.setFixedWidth(60)
        size_slider.valueChanged.connect(
            lambda size, name=cloud_name: self.cloudPointSizeChanged.emit(name, float(size))
        )
        cloud_layout.addWidget(size_slider)

        cloud_layout.addStretch()

        self.cloudCheckboxes[cloud_name] = checkbox
        self.cloudSliders[cloud_name] = size_slider
        self.clouds_layout.addWidget(cloud_widget)

    def selectAllClouds(self):
        """选择所有点云"""
        for cloud_name, checkbox in self.cloudCheckboxes.items():
            checkbox.setChecked(True)

    def deselectAllClouds(self):
        """取消选择所有点云"""
        for cloud_name, checkbox in self.cloudCheckboxes.items():
            checkbox.setChecked(False)

    def updateStatus(self, message: str):
        """更新状态信息"""
        self.status_label.setText(message)

    def updateFPS(self, fps: float):
        """更新FPS显示"""
        self.fps_value.setText(f"{fps:.1f}")


class ModernPointCloudVisualizer(QMainWindow):
    """
    现代化点云数据可视化主窗口
    """

    def __init__(self):
        super().__init__()
        self.glWidget = None
        self.controlPanel = None
        self.loadedClouds = 0
        self.totalPoints = 0

        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("高级点云数据可视化系统 - GPU加速")
        self.setGeometry(100, 100, 1600, 1000)

        # 设置应用程序样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
        """)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #444444;
            }
            QSplitter::handle:hover {
                background-color: #666666;
            }
        """)

        # 左侧控制面板
        self.controlPanel = ModernControlPanel()
        self.controlPanel.setMinimumWidth(350)
        self.controlPanel.setMaximumWidth(450)

        # 右侧OpenGL部件
        self.glWidget = ModernGLWidget()

        # 添加到分割器
        splitter.addWidget(self.controlPanel)
        splitter.addWidget(self.glWidget)

        # 设置分割比例
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # 连接信号
        self._connectSignals()

        # 状态栏
        self.statusBar().showMessage("就绪 - 欢迎使用高级点云可视化系统")

    def _connectSignals(self):
        """连接信号和槽"""
        self.controlPanel.visibilityChanged.connect(self.toggleCloudVisibility)
        self.controlPanel.pointSizeChanged.connect(self.changePointSize)
        self.controlPanel.cloudPointSizeChanged.connect(self.changeCloudPointSize)
        self.controlPanel.resetCameraRequested.connect(self.resetCamera)
        self.controlPanel.loadDataRequested.connect(self.loadPointClouds)

        # 连接FPS更新
        timer = QTimer(self)
        timer.timeout.connect(self.updateFPSDisplay)
        timer.start(500)  # 每500ms更新一次

    def loadPointClouds(self):
        """加载点云数据"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择点云数据目录", "",
            QFileDialog.Option.ShowDirsOnly
        )

        if not dir_path:
            return

        self.controlPanel.updateStatus("正在扫描和加载点云数据，请稍候...")
        QApplication.processEvents()  # 更新UI

        # 扫描JSON文件
        json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]

        if not json_files:
            self.controlPanel.updateStatus("在选定目录中未找到.json文件")
            QMessageBox.warning(self, "无数据", "在选定目录中未找到.json文件")
            return

        self.controlPanel.updateStatus(f"找到 {len(json_files)} 个点云文件，开始加载...")

        # 创建进度对话框
        progress = QProgressDialog("加载点云数据...", "取消", 0, len(json_files), self)
        progress.setWindowTitle("加载进度")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        loaded_count = 0
        total_points = 0

        for i, json_file in enumerate(json_files):
            if progress.wasCanceled():
                break

            file_path = os.path.join(dir_path, json_file)
            cloud_name = os.path.splitext(json_file)[0]

            progress.setLabelText(f"正在加载: {json_file}")
            progress.setValue(i)

            if self.glWidget.loadPointCloud(file_path, cloud_name):
                loaded_count += 1
                points_count = len(self.glWidget.pointClouds[cloud_name])
                total_points += points_count
                self.controlPanel.addCloudControl(cloud_name, points_count)

            QApplication.processEvents()

        progress.setValue(len(json_files))

        self.loadedClouds = loaded_count
        self.totalPoints = total_points

        status_message = f"加载完成! 已加载 {loaded_count} 个点云文件，总计 {total_points:,} 个点"
        self.controlPanel.updateStatus(status_message)
        self.statusBar().showMessage(status_message)

        if loaded_count == 0:
            QMessageBox.warning(self, "加载失败", "未能成功加载任何点云文件")
        else:
            QMessageBox.information(self, "加载成功",
                                    f"成功加载 {loaded_count} 个点云文件\n总计 {total_points:,} 个点")

    def toggleCloudVisibility(self, cloud_name: str, visible: bool):
        """切换点云可见性"""
        self.glWidget.setCloudVisibility(cloud_name, visible)

    def changePointSize(self, size: int):
        """改变基础点大小"""
        self.glWidget.setPointSize(float(size))

    def changeCloudPointSize(self, cloud_name: str, size: float):
        """改变特定点云的点大小"""
        self.glWidget.setCloudPointSize(cloud_name, size)

    def resetCamera(self):
        """重置相机"""
        self.glWidget.resetCamera()
        self.statusBar().showMessage("相机视角已重置")

    def updateFPSDisplay(self):
        """更新FPS显示"""
        if self.glWidget:
            self.controlPanel.updateFPS(self.glWidget.fps)


def main():
    """主函数"""
    # 设置高DPI支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 设置应用程序调色板
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 122, 204))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    visualizer = ModernPointCloudVisualizer()
    visualizer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()