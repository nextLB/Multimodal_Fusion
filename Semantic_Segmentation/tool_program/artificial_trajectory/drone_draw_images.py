import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import math
import time
import threading
import logging
import sv_ttk  # 用于现代主题

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

Image.MAX_IMAGE_PIXELS = 5000000000


class TrajectoryVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("轨迹可视化工具")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2c3e50")

        # 设置应用图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")  # 如果有图标文件的话
        except:
            pass

        # 初始化变量
        self.original_image = None  # 存储原始图像
        self.display_image = None  # 存储显示图像（带轨迹）
        self.tk_image = None
        self.canvas = None
        self.json_data = None
        self.original_image_size = (0, 0)  # 原始图像尺寸
        self.display_image_size = (0, 0)  # 当前显示图像尺寸
        self.scale_factor = 1.0  # 初始缩放因子
        self.user_scale_factor = 1.0  # 用户缩放因子
        self.max_display_size = 800  # 最大显示尺寸
        self.trajectory_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f",
                                  "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
                                  "#e84393", "#00cec9"]  # 更现代的颜色

        # 视口相关变量
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = 800
        self.viewport_height = 600
        self.canvas_width = 800
        self.canvas_height = 600
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        # 轨迹控制变量
        self.current_trajectory_idx = 0
        self.current_point_idx = 0
        self.is_playing = False
        self.play_speed = 50  # 毫秒
        self.trajectory_lines = []  # 存储轨迹线的ID
        self.trajectory_points = []  # 存储轨迹点的ID
        self.all_trajectory_items = []  # 存储所有轨迹项（用于清除）
        self.completed_trajectories = set()  # 存储已完成的轨迹索引
        self.trajectory_draw = None  # 用于绘制轨迹的ImageDraw对象

        # 性能优化变量
        self.redraw_timer = None
        self.last_zoom_time = 0
        self.zoom_debounce_time = 100  # 毫秒
        self.scaled_points_cache = {}  # 缓存缩放后的点坐标
        self.current_scale_key = None  # 当前缩放键

        # 创建UI
        self.create_ui()

        # 应用现代主题
        sv_ttk.set_theme("dark")

        logging.info("应用程序初始化完成")

    def create_ui(self):
        # 创建主框架
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建工具栏
        toolbar = tk.Frame(main_frame, bg="#34495e", height=60, relief=tk.RAISED, bd=1)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        # 文件选择区域
        file_frame = tk.Frame(toolbar, bg="#34495e")
        file_frame.pack(side=tk.LEFT, padx=10)

        load_image_btn = ttk.Button(file_frame, text="选择图像", command=self.load_image, style="Accent.TButton")
        load_image_btn.pack(side=tk.LEFT, padx=5)

        load_json_btn = ttk.Button(file_frame, text="选择JSON", command=self.load_json, style="Accent.TButton")
        load_json_btn.pack(side=tk.LEFT, padx=5)

        # 添加分隔线
        separator = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # 轨迹控制区域
        control_frame = tk.Frame(toolbar, bg="#34495e")
        control_frame.pack(side=tk.LEFT, padx=10)

        self.play_btn = ttk.Button(control_frame, text="播放", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(control_frame, text="重置当前", command=self.reset_current_trajectory,
                                    state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(control_frame, text="清除所有轨迹", command=self.clear_all_trajectories,
                                    state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # 添加分隔线
        separator2 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator2.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # 速度控制
        speed_frame = tk.Frame(toolbar, bg="#34495e")
        speed_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(speed_frame, text="速度:", bg="#34495e", fg="white", font=("Arial", 10)).pack(side=tk.LEFT)
        self.speed_scale = ttk.Scale(speed_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                     length=100, command=self.set_speed)
        self.speed_scale.set(self.play_speed)
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # 速度值显示
        self.speed_var = tk.StringVar(value=f"{self.play_speed}ms")
        speed_label = tk.Label(speed_frame, textvariable=self.speed_var, bg="#34495e", fg="white", font=("Arial", 9))
        speed_label.pack(side=tk.LEFT, padx=5)

        # 添加分隔线
        separator3 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator3.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # 轨迹选择
        traj_frame = tk.Frame(toolbar, bg="#34495e")
        traj_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(traj_frame, text="轨迹:", bg="#34495e", fg="white", font=("Arial", 10)).pack(side=tk.LEFT)
        self.traj_var = tk.StringVar()
        self.traj_combo = ttk.Combobox(traj_frame, textvariable=self.traj_var, state="readonly", width=15)
        self.traj_combo.pack(side=tk.LEFT, padx=5)
        self.traj_combo.bind("<<ComboboxSelected>>", self.on_trajectory_select)

        # 添加分隔线
        separator4 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator4.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # 缩放控制
        zoom_frame = tk.Frame(toolbar, bg="#34495e")
        zoom_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(zoom_frame, text="缩放:", bg="#34495e", fg="white", font=("Arial", 10)).pack(side=tk.LEFT)
        self.zoom_out_btn = ttk.Button(zoom_frame, text="−", width=3, command=lambda: self.zoom_with_debounce(0.8))
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_in_btn = ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.zoom_with_debounce(1.2))
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_reset_btn = ttk.Button(zoom_frame, text="↺", width=3, command=self.zoom_reset)
        self.zoom_reset_btn.pack(side=tk.LEFT, padx=2)

        # 缩放比例显示
        self.zoom_var = tk.StringVar(value="100%")
        zoom_label = tk.Label(zoom_frame, textvariable=self.zoom_var, bg="#34495e", fg="white", font=("Arial", 9))
        zoom_label.pack(side=tk.LEFT, padx=5)

        # 状态显示
        status_frame = tk.Frame(toolbar, bg="#34495e")
        status_frame.pack(side=tk.RIGHT, padx=10)

        tk.Label(status_frame, text="状态:", bg="#34495e", fg="white", font=("Arial", 10)).pack(side=tk.LEFT)
        self.status_label = tk.Label(status_frame, text="请选择图像和JSON文件", bg="#34495e", fg="#ecf0f1", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 创建画布和滚动条
        canvas_frame = tk.Frame(main_frame, bg="#2c3e50")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 添加水平和垂直滚动条
        self.h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.h_scrollbar.config(command=self.on_scroll_x)

        self.v_scrollbar = ttk.Scrollbar(canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.v_scrollbar.config(command=self.on_scroll_y)

        # 使用双缓冲画布减少闪烁
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#34495e",  # 深色背景
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set,
            highlightthickness=0,
            width=800,
            height=600
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_pan_end)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # 绑定鼠标滚轮事件（用于缩放）
        self.canvas.bind("<MouseWheel>", self.on_mousewheel_zoom)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel_zoom)  # Linux
        self.canvas.bind("<Button-5>", self.on_mousewheel_zoom)  # Linux

    def on_canvas_configure(self, event):
        """画布大小改变时调用"""
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.viewport_width = self.canvas_width
        self.viewport_height = self.canvas_height
        self.update_viewport()

    def on_scroll_x(self, *args):
        """水平滚动条回调"""
        if not self.original_image:
            return

        if args[0] == "moveto":
            self.viewport_x = float(args[1]) * (
                        self.original_image_size[0] * self.user_scale_factor - self.viewport_width)
        elif args[0] == "scroll":
            delta = int(args[1])
            if args[2] == "units":
                delta *= 10
            self.viewport_x = max(0, min(self.original_image_size[0] * self.user_scale_factor - self.viewport_width, self.viewport_x + delta))

        self.update_viewport()

    def on_scroll_y(self, *args):
        """垂直滚动条回调"""
        if not self.original_image:
            return

        if args[0] == "moveto":
            self.viewport_y = float(args[1]) * (
                        self.original_image_size[1] * self.user_scale_factor - self.viewport_height)
        elif args[0] == "scroll":
            delta = int(args[1])
            if args[2] == "units":
                delta *= 10
            self.viewport_y = max(0, min(self.original_image_size[1] * self.user_scale_factor - self.viewport_height,
                                         self.viewport_y + delta))

        self.update_viewport()

    def update_scrollbars(self):
        """更新滚动条位置和大小"""
        if not self.original_image:
            return

        # 计算滚动条参数
        total_width = self.original_image_size[0] * self.user_scale_factor
        total_height = self.original_image_size[1] * self.user_scale_factor

        # 设置滚动条范围
        if total_width > self.viewport_width:
            self.h_scrollbar.set(self.viewport_x / (total_width - self.viewport_width),
                                 (self.viewport_x + self.viewport_width) / total_width)
            self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self.h_scrollbar.pack_forget()

        if total_height > self.viewport_height:
            self.v_scrollbar.set(self.viewport_y / (total_height - self.viewport_height),
                                 (self.viewport_y + self.viewport_height) / total_height)
            self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.v_scrollbar.pack_forget()

    def on_pan_start(self, event):
        """开始拖拽视图"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")

    def on_pan_move(self, event):
        """拖拽视图移动"""
        if self.is_panning and self.original_image:
            dx = self.pan_start_x - event.x
            dy = self.pan_start_y - event.y

            self.viewport_x += dx
            self.viewport_y += dy

            # 限制视口不超出图像范围
            max_x = max(0, self.original_image_size[0] * self.user_scale_factor - self.viewport_width)
            max_y = max(0, self.original_image_size[1] * self.user_scale_factor - self.viewport_height)

            self.viewport_x = max(0, min(max_x, self.viewport_x))
            self.viewport_y = max(0, min(max_y, self.viewport_y))

            self.pan_start_x = event.x
            self.pan_start_y = event.y

            self.update_viewport()

    def on_pan_end(self, event):
        """结束拖拽视图"""
        self.is_panning = False
        self.canvas.config(cursor="")

    def on_canvas_motion(self, event):
        """在画布上移动鼠标时显示坐标"""
        if self.original_image and self.original_image_size[0] > 0:
            # 计算实际坐标（考虑缩放和视口）
            actual_x = (self.viewport_x + event.x) / self.user_scale_factor
            actual_y = (self.viewport_y + event.y) / self.user_scale_factor

            # 更新状态栏
            self.status_label.config(
                text=f"坐标: ({actual_x:.1f}, {actual_y:.1f}) | 缩放: {self.user_scale_factor * 100:.1f}% | 视口: ({self.viewport_x:.0f}, {self.viewport_y:.0f})")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("所有图片格式", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.PNG *.JPG *.JPEG *.BMP *.GIF *.TIFF"),
                ("PNG 图片", "*.png *.PNG"),
                ("JPEG 图片", "*.jpg *.jpeg *.JPG *.JPEG"),
                ("BMP 图片", "*.bmp *.BMP"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                logging.info(f"开始加载图像: {file_path}")
                self.original_image = Image.open(file_path).convert("RGB")
                self.display_image = self.original_image.copy()
                self.trajectory_draw = ImageDraw.Draw(self.display_image)
                self.original_image_size = self.original_image.size
                self.user_scale_factor = 1.0  # 重置用户缩放因子
                self.viewport_x = 0
                self.viewport_y = 0
                self.scaled_points_cache = {}  # 清除点缓存
                self.update_viewport()
                self.status_label.config(text=f"已加载图像: {os.path.basename(file_path)}")
                self.zoom_var.set(f"{self.user_scale_factor * 100:.0f}%")
                logging.info(f"图像加载成功: {self.original_image_size}")

                # 如果JSON也已加载，启用控制按钮
                if self.json_data is not None:
                    self.play_btn.config(state=tk.NORMAL)
                    self.reset_btn.config(state=tk.NORMAL)
                    self.clear_btn.config(state=tk.NORMAL)
                    logging.info("图像和JSON都已加载，启用控制按钮")
            except Exception as e:
                error_msg = f"无法加载图像: {str(e)}"
                logging.error(error_msg)
                messagebox.showerror("错误", error_msg)

    def load_json(self):
        file_path = filedialog.askopenfilename(
            title="选择JSON文件",
            filetypes=[
                ("JSON 文件", "*.json"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                logging.info(f"开始加载JSON: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 检查JSON结构并转换为程序期望的格式
                self.json_data = self.parse_json_structure(data)

                if not self.json_data:
                    error_msg = "JSON文件格式不支持：应为轨迹列表或包含轨迹列表的对象"
                    logging.error(error_msg)
                    messagebox.showerror("错误", error_msg)
                    return

                self.status_label.config(text=f"已加载JSON: {os.path.basename(file_path)}")
                logging.info(f"JSON加载成功，包含 {len(self.json_data)} 条轨迹")

                # 更新轨迹选择下拉框
                trajectories = []
                for i, traj in enumerate(self.json_data):
                    if isinstance(traj, dict):
                        label = traj.get('label', f'轨迹_{i + 1}')
                    else:
                        label = f'轨迹_{i + 1}'
                    trajectories.append(label)

                self.traj_combo['values'] = trajectories
                if trajectories:
                    self.traj_var.set(trajectories[0])
                    logging.info(f"轨迹下拉框更新，当前选择: {trajectories[0]}")

                # 如果图像也已加载，启用控制按钮
                if self.original_image is not None:
                    self.play_btn.config(state=tk.NORMAL)
                    self.reset_btn.config(state=tk.NORMAL)
                    self.clear_btn.config(state=tk.NORMAL)
                    logging.info("图像和JSON都已加载，启用控制按钮")

            except Exception as e:
                error_msg = f"无法加载JSON文件: {str(e)}"
                logging.error(error_msg)
                messagebox.showerror("错误", error_msg)

    def parse_json_structure(self, data):
        """
        解析JSON结构，转换为程序期望的轨迹列表格式
        期望格式: [{"label": "轨迹1", "points": [[x1,y1], [x2,y2], ...]}, ...]
        """
        # 如果已经是列表，直接返回
        if isinstance(data, list):
            return data

        # 如果是字典，尝试提取轨迹数据
        if isinstance(data, dict):
            # 尝试常见的键名
            for key in ['trajectories', 'tracks', 'paths', 'data']:
                if key in data and isinstance(data[key], list):
                    return data[key]

            # 如果字典中有points键，将其包装为列表
            if 'points' in data:
                return [data]

        # 无法识别的格式
        return None

    def update_viewport(self):
        """更新视口显示"""
        if not self.original_image:
            return

        # 清除画布
        self.canvas.delete("all")
        self.trajectory_lines = []  # 清空当前轨迹线
        self.trajectory_points = []  # 清空当前轨迹点

        # 计算视口在原始图像中的位置和大小
        src_x = self.viewport_x / self.user_scale_factor
        src_y = self.viewport_y / self.user_scale_factor
        src_width = self.viewport_width / self.user_scale_factor
        src_height = self.viewport_height / self.user_scale_factor

        # 确保不超出图像边界
        src_x = max(0, min(self.original_image_size[0] - src_width, src_x))
        src_y = max(0, min(self.original_image_size[1] - src_height, src_y))

        # 裁剪图像
        cropped_img = self.display_image.crop((src_x, src_y, src_x + src_width, src_y + src_height))

        # 缩放到视口大小
        resized_img = cropped_img.resize((self.viewport_width, self.viewport_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)

        # 在画布上显示图像
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # 更新滚动条
        self.update_scrollbars()
        logging.info(
            f"视口更新: ({self.viewport_x:.0f}, {self.viewport_y:.0f}) {self.viewport_width}x{self.viewport_height}")

    def draw_trajectory_on_image(self, traj_idx):
        """将轨迹绘制到图像上"""
        if traj_idx >= len(self.json_data):
            return

        trajectory = self.json_data[traj_idx]
        points = trajectory.get('points', [])

        if not points:
            return

        # 获取轨迹颜色
        color = self.trajectory_colors[traj_idx % len(self.trajectory_colors)]

        # 绘制所有点和线
        for i, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue

            x, y = point[0], point[1]

            # 绘制点
            self.trajectory_draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline="white", width=1)

            # 如果不是第一个点，绘制连线
            if i > 0:
                prev_point = points[i - 1]
                if isinstance(prev_point, (list, tuple)) and len(prev_point) >= 2:
                    prev_x, prev_y = prev_point[0], prev_point[1]
                    self.trajectory_draw.line([prev_x, prev_y, x, y], fill=color, width=3)

    def on_trajectory_select(self, event):
        selected_idx = self.traj_combo.current()
        if selected_idx >= 0 and selected_idx < len(self.json_data):
            self.current_trajectory_idx = selected_idx
            logging.info(f"选择了轨迹: {self.traj_var.get()}, 索引: {selected_idx}")
            self.reset_current_trajectory()

    def toggle_play(self):
        if not self.json_data or not self.original_image:
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_btn.config(text="暂停")
            logging.info(f"开始播放轨迹 {self.current_trajectory_idx + 1}, 速度: {self.play_speed}ms")
            self.visualize_trajectory()
        else:
            self.play_btn.config(text="播放")
            logging.info("暂停播放轨迹")

    def set_speed(self, value):
        self.play_speed = int(float(value))
        self.speed_var.set(f"{self.play_speed}ms")
        logging.info(f"设置播放速度: {self.play_speed}ms")

    def reset_current_trajectory(self):
        # 停止播放
        self.is_playing = False
        self.play_btn.config(text="播放")

        # 重置当前轨迹的播放进度
        self.current_point_idx = 0

        # 重新显示视口
        logging.info(f"重置当前轨迹: {self.current_trajectory_idx}")
        self.update_viewport()

    def clear_all_trajectories(self):
        # 停止播放
        self.is_playing = False
        self.play_btn.config(text="播放")

        # 重置所有轨迹的播放进度
        self.current_point_idx = 0

        # 清空已完成的轨迹记录
        self.completed_trajectories.clear()

        # 重新创建显示图像（清除所有轨迹）
        self.display_image = self.original_image.copy()
        self.trajectory_draw = ImageDraw.Draw(self.display_image)

        # 更新视口
        self.update_viewport()

        # 更新状态
        self.status_label.config(text="已清除所有轨迹")
        logging.info("已清除所有轨迹")

    def visualize_trajectory(self):
        if not self.is_playing or not self.json_data or self.current_trajectory_idx >= len(self.json_data):
            return

        trajectory = self.json_data[self.current_trajectory_idx]

        # 确保轨迹有points字段
        if not isinstance(trajectory, dict) or 'points' not in trajectory:
            self.is_playing = False
            self.play_btn.config(text="播放")
            error_msg = "轨迹数据格式不正确，缺少points字段"
            logging.error(error_msg)
            messagebox.showerror("错误", error_msg)
            return

        points = trajectory.get('points', [])

        if self.current_point_idx >= len(points):
            # 轨迹播放完成
            self.is_playing = False
            self.play_btn.config(text="播放")
            self.completed_trajectories.add(self.current_trajectory_idx)
            logging.info(f"轨迹 {self.current_trajectory_idx + 1} 播放完成")
            return

        # 获取当前点
        point = points[self.current_point_idx]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            # 跳过无效的点
            self.current_point_idx += 1
            if self.current_point_idx < len(points) and self.is_playing:
                self.root.after(self.play_speed, self.visualize_trajectory)
            return

        # 获取轨迹颜色
        color = self.trajectory_colors[self.current_trajectory_idx % len(self.trajectory_colors)]

        # 绘制当前点
        x, y = point[0], point[1]
        self.trajectory_draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline="white", width=1)

        # 如果不是第一个点，绘制连线
        if self.current_point_idx > 0:
            prev_point = points[self.current_point_idx - 1]
            if isinstance(prev_point, (list, tuple)) and len(prev_point) >= 2:
                prev_x, prev_y = prev_point[0], prev_point[1]
                self.trajectory_draw.line([prev_x, prev_y, x, y], fill=color, width=3)

        # 更新视口
        self.update_viewport()

        # 更新状态
        status_text = f"轨迹 {self.current_trajectory_idx + 1}/{len(self.json_data)}, 点 {self.current_point_idx + 1}/{len(points)}"
        self.status_label.config(text=status_text)

        # 移动到下一个点
        self.current_point_idx += 1

        # 如果不是最后一个点，安排下一次绘制
        if self.current_point_idx < len(points) and self.is_playing:
            self.root.after(self.play_speed, self.visualize_trajectory)
        else:
            # 轨迹播放完成
            self.is_playing = False
            self.play_btn.config(text="播放")
            self.completed_trajectories.add(self.current_trajectory_idx)
            logging.info(f"轨迹 {self.current_trajectory_idx + 1} 播放完成")

    def on_mousewheel_zoom(self, event):
        if not self.original_image:
            return

        # 计算缩放因子
        scale = 1.1
        if event.num == 5 or event.delta < 0:  # 向下滚动或Linux上的向下滚动
            scale = 0.9

        # 使用防抖机制
        current_time = time.time() * 1000
        if current_time - self.last_zoom_time < self.zoom_debounce_time:
            return

        self.last_zoom_time = current_time

        # 计算缩放中心
        if hasattr(event, 'x') and hasattr(event, 'y'):
            zoom_center_x = self.viewport_x + event.x
            zoom_center_y = self.viewport_y + event.y
        else:
            # 如果没有鼠标位置信息，使用视口中心
            zoom_center_x = self.viewport_x + self.viewport_width / 2
            zoom_center_y = self.viewport_y + self.viewport_height / 2

        # 保存缩放前的视口参数
        old_scale = self.user_scale_factor
        old_viewport_x = self.viewport_x
        old_viewport_y = self.viewport_y

        # 更新用户缩放因子
        self.user_scale_factor *= scale

        # 限制缩放范围
        self.user_scale_factor = max(0.1, min(10.0, self.user_scale_factor))

        # 计算新的视口位置，使缩放中心保持不变
        self.viewport_x = zoom_center_x - (zoom_center_x - old_viewport_x) * (self.user_scale_factor / old_scale)
        self.viewport_y = zoom_center_y - (zoom_center_y - old_viewport_y) * (self.user_scale_factor / old_scale)

        logging.info(f"鼠标滚轮缩放: {old_scale:.2f} -> {self.user_scale_factor:.2f}")

        # 重绘图像和轨迹
        self.schedule_redisplay()

    def zoom_with_debounce(self, scale_factor):
        """带防抖的缩放函数"""
        if not self.original_image:
            return

        current_time = time.time() * 1000
        if current_time - self.last_zoom_time < self.zoom_debounce_time:
            return

        self.last_zoom_time = current_time

        # 使用视口中心作为缩放中心
        zoom_center_x = self.viewport_x + self.viewport_width / 2
        zoom_center_y = self.viewport_y + self.viewport_height / 2

        # 保存缩放前的视口参数
        old_scale = self.user_scale_factor
        old_viewport_x = self.viewport_x
        old_viewport_y = self.viewport_y

        # 更新用户缩放因子
        self.user_scale_factor *= scale_factor

        # 限制缩放范围
        self.user_scale_factor = max(0.1, min(10.0, self.user_scale_factor))

        # 计算新的视口位置，使缩放中心保持不变
        self.viewport_x = zoom_center_x - (zoom_center_x - old_viewport_x) * (self.user_scale_factor / old_scale)
        self.viewport_y = zoom_center_y - (zoom_center_y - old_viewport_y) * (self.user_scale_factor / old_scale)

        action = "放大" if scale_factor > 1 else "缩小"
        logging.info(f"{action}: {old_scale:.2f} -> {self.user_scale_factor:.2f}")

        # 重绘图像和轨迹
        self.schedule_redisplay()

    def zoom_reset(self):
        """重置缩放"""
        if not self.original_image:
            return

        # 保存当前视口中心
        viewport_center_x = self.viewport_x + self.viewport_width / 2
        viewport_center_y = self.viewport_y + self.viewport_height / 2

        logging.info(f"重置缩放: {self.user_scale_factor:.2f} -> 1.0")
        self.user_scale_factor = 1.0

        # 调整视口位置，使中心保持不变
        self.viewport_x = viewport_center_x - self.viewport_width / 2
        self.viewport_y = viewport_center_y - self.viewport_height / 2

        self.schedule_redisplay()

    def schedule_redisplay(self):
        """安排重绘任务，避免频繁重绘"""
        if self.redraw_timer:
            self.root.after_cancel(self.redraw_timer)

        self.redraw_timer = self.root.after(50, self.update_viewport)


if __name__ == '__main__':
    root = tk.Tk()
    app = TrajectoryVisualizer(root)
    root.mainloop()