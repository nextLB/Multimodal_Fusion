import torch
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random
import gc
import math


class PointCloudGenerator:
    """
    修复版点云数据生成器类，用于生成模拟现实世界的三维点云数据
    利用NVIDIA GPU加速生成过程，并优化内存管理
    """

    def __init__(self, device: str = "auto"):
        """
        初始化点云生成器

        Args:
            device: 使用的设备，"auto"自动选择，可选"cuda"或"cpu"
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU型号: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 基础参数
        self.ground_size = 2000.0  # 地面尺寸
        self.max_height = 300.0  # 最大高度

    def cleanup_memory(self):
        """清理内存和GPU缓存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("内存清理完成")

    def generateGroundPlane(self, num_points: int = 1500000,
                            size: float = 2000.0,
                            height_variation: float = 8.0) -> torch.Tensor:
        """
        生成大地平面点云

        Args:
            num_points: 点的数量
            size: 平面尺寸
            height_variation: 高度变化范围

        Returns:
            torch.Tensor: 形状为 [num_points, 3] 的点云张量
        """
        print(f"生成大地平面点云，点数: {num_points}")

        # 在GPU上生成随机点
        x = (torch.rand(num_points, device=self.device) - 0.5) * size
        y = (torch.rand(num_points, device=self.device) - 0.5) * size

        # 使用柏林噪声生成更自然的地形
        noise_scale = 0.01
        z_noise = torch.sin(x * noise_scale) * torch.cos(y * noise_scale) * height_variation
        z = torch.randn(num_points, device=self.device) * 2.0 + z_noise

        points = torch.stack([x, y, z], dim=1)

        return points.cpu()

    def generateRoadNetwork(self, num_roads: int = 50) -> torch.Tensor:
        """
        生成道路网络点云

        Args:
            num_roads: 道路数量

        Returns:
            torch.Tensor: 道路点云数据
        """
        print(f"生成 {num_roads} 条道路点云")

        all_points = []
        points_per_road_segment = 5000  # 每个道路段的点数

        # 生成主要干道
        main_roads = num_roads // 3
        for i in range(main_roads):
            # 主干道 - 更宽更直
            start_x = (torch.rand(1).item() - 0.5) * self.ground_size
            start_y = (torch.rand(1).item() - 0.5) * self.ground_size

            # 随机选择水平或垂直方向
            if random.choice([True, False]):
                end_x = (torch.rand(1).item() - 0.5) * self.ground_size
                end_y = start_y + (torch.randn(1).item() * 100)
            else:
                end_x = start_x + (torch.randn(1).item() * 100)
                end_y = (torch.rand(1).item() - 0.5) * self.ground_size

            # 生成道路点
            t = torch.linspace(0, 1, points_per_road_segment, device=self.device)
            road_width = 12.0  # 主干道宽度

            # 道路中心线
            center_x = start_x + (end_x - start_x) * t
            center_y = start_y + (end_y - start_y) * t

            # 添加道路宽度
            offset = (torch.rand(points_per_road_segment, device=self.device) - 0.5) * road_width

            # 计算垂直向量（修正张量操作）
            dx = end_x - start_x
            dy = end_y - start_y
            length_val = math.sqrt(dx ** 2 + dy ** 2)
            if length_val < 1e-6:  # 避免除零
                perpendicular_x, perpendicular_y = 1.0, 0.0
            else:
                perpendicular_x = -dy / length_val
                perpendicular_y = dx / length_val

            x = center_x + offset * perpendicular_x
            y = center_y + offset * perpendicular_y
            z = torch.ones(points_per_road_segment, device=self.device) * 0.05  # 道路略高于地面

            road_points = torch.stack([x, y, z], dim=1)
            all_points.append(road_points.cpu())

        # 生成次要道路
        for i in range(num_roads - main_roads):
            # 弯曲的次要道路
            control_points = []
            num_segments = random.randint(3, 8)
            for j in range(num_segments):
                control_points.append((
                    (torch.rand(1).item() - 0.5) * self.ground_size * 0.8,
                    (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
                ))

            # 使用贝塞尔曲线生成道路
            road_width = 6.0  # 次要道路宽度

            segment_points = []
            for j in range(len(control_points) - 1):
                start_p = control_points[j]
                end_p = control_points[j + 1]

                t_segment = torch.linspace(0, 1, points_per_road_segment // num_segments, device=self.device)

                center_x = start_p[0] + (end_p[0] - start_p[0]) * t_segment
                center_y = start_p[1] + (end_p[1] - start_p[1]) * t_segment

                offset = (torch.rand(len(t_segment), device=self.device) - 0.5) * road_width

                # 计算垂直向量
                dx_seg = end_p[0] - start_p[0]
                dy_seg = end_p[1] - start_p[1]
                length_seg = math.sqrt(dx_seg ** 2 + dy_seg ** 2)
                if length_seg < 1e-6:
                    perp_x, perp_y = 1.0, 0.0
                else:
                    perp_x = -dy_seg / length_seg
                    perp_y = dx_seg / length_seg

                x = center_x + offset * perp_x
                y = center_y + offset * perp_y
                z = torch.ones(len(t_segment), device=self.device) * 0.03

                segment_points.append(torch.stack([x, y, z], dim=1).cpu())

            all_points.extend(segment_points)

        return torch.cat(all_points, dim=0)

    def generateClouds(self, num_clouds: int = 30) -> torch.Tensor:
        """
        生成云朵点云

        Args:
            num_clouds: 云朵数量

        Returns:
            torch.Tensor: 云朵点云数据
        """
        print(f"生成 {num_clouds} 朵云点云")

        all_points = []
        points_per_cloud = 15000

        for i in range(num_clouds):
            # 随机云朵位置（在空中）
            center_x = (torch.rand(1).item() - 0.5) * self.ground_size
            center_y = (torch.rand(1).item() - 0.5) * self.ground_size
            center_z = 150 + torch.rand(1).item() * 100  # 云朵高度

            # 云朵大小和形状
            cloud_size = 20 + torch.rand(1).item() * 50
            num_spheres = random.randint(3, 8)  # 每个云朵由多个球体组成

            cloud_points = []
            points_per_sphere = points_per_cloud // num_spheres
            for j in range(num_spheres):
                # 每个球体的偏移
                offset_x = (torch.rand(1).item() - 0.5) * cloud_size * 0.8
                offset_y = (torch.rand(1).item() - 0.5) * cloud_size * 0.8
                offset_z = (torch.rand(1).item() - 0.5) * cloud_size * 0.3

                sphere_radius = 5 + torch.rand(1).item() * 15
                sphere_points = self._generateSphere(
                    points_per_sphere,
                    radius=sphere_radius,
                    center_x=center_x + offset_x,
                    center_y=center_y + offset_y,
                    center_z=center_z + offset_z
                )
                cloud_points.append(sphere_points)

            all_points.extend(cloud_points)

        return torch.cat(all_points, dim=0)

    def generateWindows(self, num_buildings_with_windows: int = 80) -> torch.Tensor:
        """
        生成建筑物窗户点云

        Args:
            num_buildings_with_windows: 带窗户的建筑物数量

        Returns:
            torch.Tensor: 窗户点云数据
        """
        print(f"生成 {num_buildings_with_windows} 栋建筑物的窗户点云")

        all_points = []
        windows_per_building = 50

        for i in range(num_buildings_with_windows):
            # 建筑物位置和尺寸
            building_x = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
            building_y = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
            building_width = 15 + torch.rand(1).item() * 35
            building_depth = 15 + torch.rand(1).item() * 35
            building_height = 20 + torch.rand(1).item() * 60

            # 生成窗户
            for j in range(windows_per_building):
                # 窗户在建筑物表面的位置
                face = random.choice(['front', 'back', 'left', 'right'])

                if face in ['front', 'back']:
                    window_x = building_x + (torch.rand(1).item() - 0.5) * building_width * 0.8
                    window_z = 2 + torch.rand(1).item() * (building_height - 4)
                    if face == 'front':
                        window_y = building_y + building_depth / 2
                    else:
                        window_y = building_y - building_depth / 2
                else:  # left or right
                    window_y = building_y + (torch.rand(1).item() - 0.5) * building_depth * 0.8
                    window_z = 2 + torch.rand(1).item() * (building_height - 4)
                    if face == 'right':
                        window_x = building_x + building_width / 2
                    else:
                        window_x = building_x - building_width / 2

                # 窗户尺寸
                window_width = 1.5 + torch.rand(1).item() * 1.0
                window_height = 2.0 + torch.rand(1).item() * 1.0

                # 生成窗户矩形 - 修正张量形状问题
                num_window_points = 200
                u = torch.rand(num_window_points, device=self.device)
                v = torch.rand(num_window_points, device=self.device)

                x = window_x + (u - 0.5) * window_width
                y = torch.full((num_window_points,), window_y, device=self.device)  # 修正：确保y是张量
                z = window_z + (v - 0.5) * window_height

                window_points = torch.stack([x, y, z], dim=1)
                all_points.append(window_points.cpu())

        return torch.cat(all_points, dim=0)

    def generateBridge(self, num_bridges: int = 5) -> torch.Tensor:
        """
        生成桥梁点云

        Args:
            num_bridges: 桥梁数量

        Returns:
            torch.Tensor: 桥梁点云数据
        """
        print(f"生成 {num_bridges} 座桥梁点云")

        all_points = []
        points_per_bridge = 50000

        for i in range(num_bridges):
            # 桥梁跨越河流或山谷
            bridge_length = 80 + torch.rand(1).item() * 120
            bridge_width = 8 + torch.rand(1).item() * 6
            bridge_height = 15 + torch.rand(1).item() * 20

            # 桥梁位置
            center_x = (torch.rand(1).item() - 0.5) * self.ground_size * 0.6
            center_y = (torch.rand(1).item() - 0.5) * self.ground_size * 0.6

            # 桥面
            deck_points = self._generateBox(
                points_per_bridge // 3,
                width=bridge_width,
                height=1.0,
                depth=bridge_length,
                center_x=center_x,
                center_y=center_y,
                center_z=bridge_height
            )

            # 桥墩
            num_piers = max(3, int(bridge_length / 20))
            for j in range(num_piers):
                pier_x = center_x
                pier_y = center_y - bridge_length / 2 + (j + 0.5) * (bridge_length / num_piers)
                pier_points = self._generateCylinder(
                    points_per_bridge // (num_piers * 3),
                    radius=1.5,
                    height=bridge_height,
                    center_x=pier_x,
                    center_y=pier_y
                )
                all_points.append(pier_points)

            # 桥栏杆
            railing_points1 = self._generateBox(
                points_per_bridge // 10,
                width=0.3,
                height=1.2,
                depth=bridge_length,
                center_x=center_x + bridge_width / 2,
                center_y=center_y,
                center_z=bridge_height + 0.6
            )
            railing_points2 = self._generateBox(
                points_per_bridge // 10,
                width=0.3,
                height=1.2,
                depth=bridge_length,
                center_x=center_x - bridge_width / 2,
                center_y=center_y,
                center_z=bridge_height + 0.6
            )

            all_points.extend([deck_points, railing_points1, railing_points2])

        return torch.cat(all_points, dim=0)

    def generateStream(self, num_streams: int = 10) -> torch.Tensor:
        """
        生成溪流点云

        Args:
            num_streams: 溪流数量

        Returns:
            torch.Tensor: 溪流点云数据
        """
        print(f"生成 {num_streams} 条溪流点云")

        all_points = []
        points_per_stream_segment = 3000  # 每个溪流段的点数

        for i in range(num_streams):
            # 溪流起点
            start_x = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
            start_y = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8

            # 生成蜿蜒的溪流路径
            stream_length = 200 + torch.rand(1).item() * 300
            num_segments = 20
            segment_length = stream_length / num_segments

            current_x = start_x
            current_y = start_y
            stream_width = 3 + torch.rand(1).item() * 4

            for j in range(num_segments):
                # 随机角度变化模拟蜿蜒
                angle = torch.randn(1).item() * 0.5  # 弧度
                end_x = current_x + segment_length * math.cos(angle)
                end_y = current_y + segment_length * math.sin(angle)

                # 生成溪流段
                t = torch.linspace(0, 1, points_per_stream_segment, device=self.device)
                center_x = current_x + (end_x - current_x) * t
                center_y = current_y + (end_y - current_y) * t

                # 溪流宽度变化
                width_variation = 0.5 + 0.5 * torch.sin(t * 2 * math.pi * 3)  # 周期性宽度变化
                current_width = stream_width * width_variation

                offset = (torch.rand(points_per_stream_segment, device=self.device) - 0.5) * current_width

                # 计算垂直向量
                dx_seg = end_x - current_x
                dy_seg = end_y - current_y
                length_seg = math.sqrt(dx_seg ** 2 + dy_seg ** 2)
                if length_seg < 1e-6:
                    perp_x, perp_y = 1.0, 0.0
                else:
                    perp_x = -dy_seg / length_seg
                    perp_y = dx_seg / length_seg

                x = center_x + offset * perp_x
                y = center_y + offset * perp_y
                z = -1.0 - torch.rand(points_per_stream_segment, device=self.device) * 2.0  # 溪流低于地面

                stream_points = torch.stack([x, y, z], dim=1)
                all_points.append(stream_points.cpu())

                current_x, current_y = end_x, end_y

        return torch.cat(all_points, dim=0)

    def generateLake(self, num_lakes: int = 8) -> torch.Tensor:
        """
        生成湖泊点云

        Args:
            num_lakes: 湖泊数量

        Returns:
            torch.Tensor: 湖泊点云数据
        """
        print(f"生成 {num_lakes} 个湖泊点云")

        all_points = []
        points_per_lake = 80000

        for i in range(num_lakes):
            # 湖泊位置
            center_x = (torch.rand(1).item() - 0.5) * self.ground_size * 0.7
            center_y = (torch.rand(1).item() - 0.5) * self.ground_size * 0.7

            # 湖泊尺寸和形状
            lake_radius = 20 + torch.rand(1).item() * 50

            # 生成椭圆形湖泊（更自然）
            radius_x = lake_radius * (0.8 + torch.rand(1).item() * 0.4)
            radius_y = lake_radius * (0.8 + torch.rand(1).item() * 0.4)

            # 使用极坐标生成湖泊点
            theta = torch.rand(points_per_lake, device=self.device) * 2 * math.pi
            r = torch.sqrt(torch.rand(points_per_lake, device=self.device))  # 均匀分布

            x = center_x + r * radius_x * torch.cos(theta)
            y = center_y + r * radius_y * torch.sin(theta)

            # 湖泊深度变化
            depth = 3 + torch.rand(points_per_lake, device=self.device) * 5
            # 中心更深，边缘更浅
            depth_variation = 1.0 - r * 0.3
            z = -depth * depth_variation

            # 添加湖底地形
            bottom_variation = torch.sin(x * 0.02) * torch.cos(y * 0.02) * 1.0
            z += bottom_variation

            lake_points = torch.stack([x, y, z], dim=1)
            all_points.append(lake_points.cpu())

        return torch.cat(all_points, dim=0)

    def generateFarmland(self, num_fields: int = 40,
                         points_per_field: int = 40000) -> torch.Tensor:
        """
        生成农田地块点云

        Args:
            num_fields: 农田数量
            points_per_field: 每个农田的点数

        Returns:
            torch.Tensor: 农田点云数据
        """
        print(f"生成 {num_fields} 块农田点云")

        all_points = []

        for i in range(num_fields):
            # 随机农田位置和大小
            center_x = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
            center_y = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
            field_width = 25 + torch.rand(1).item() * 60
            field_length = 25 + torch.rand(1).item() * 60

            # 在GPU上生成农田点
            u = torch.rand(points_per_field, device=self.device)
            v = torch.rand(points_per_field, device=self.device)

            x = center_x + (u - 0.5) * field_width
            y = center_y + (v - 0.5) * field_length
            z = torch.ones(points_per_field, device=self.device) * 0.08  # 农田略高于地面

            # 添加行状结构模拟农作物
            row_frequency = 5 + torch.rand(1).item() * 10
            row_pattern = torch.sin(v * row_frequency * math.pi) * 0.3
            z += row_pattern

            # 添加随机噪声模拟作物高度变化
            crop_variation = torch.randn(points_per_field, device=self.device) * 0.1
            z += crop_variation

            field_points = torch.stack([x, y, z], dim=1)
            all_points.append(field_points.cpu())

        return torch.cat(all_points, dim=0)

    def generateTrafficLights(self, num_lights: int = 150) -> torch.Tensor:
        """
        生成红绿灯点云

        Args:
            num_lights: 红绿灯数量

        Returns:
            torch.Tensor: 红绿灯点云数据
        """
        print(f"生成 {num_lights} 个红绿灯点云")

        all_points = []
        points_per_light = 1200

        for i in range(num_lights):
            # 随机位置（沿道路分布）
            road_axis = random.choice(['x', 'y'])
            if road_axis == 'x':
                x = (torch.rand(1).item() - 0.5) * self.ground_size
                y = (round(torch.rand(1).item() * 20) - 10) * 15  # 在网格上
            else:
                x = (round(torch.rand(1).item() * 20) - 10) * 15
                y = (torch.rand(1).item() - 0.5) * self.ground_size

            height = 5 + torch.rand(1).item() * 4

            # 生成红绿灯柱
            pillar_points = self._generateCylinder(
                points_per_light // 3,
                radius=0.15,
                height=height,
                center_x=x,
                center_y=y
            )

            # 生成灯头
            light_head_points = self._generateBox(
                points_per_light // 3,
                width=1.0,
                height=0.4,
                depth=0.4,
                center_x=x,
                center_y=y,
                center_z=height
            )

            # 生成支撑臂
            arm_length = 1.5 + torch.rand(1).item() * 1.0
            arm_points = self._generateCylinder(
                points_per_light // 3,
                radius=0.08,
                height=arm_length,
                center_x=x + arm_length / 2,
                center_y=y,
                center_z=height,
                rotation='y'
            )

            all_points.extend([pillar_points, light_head_points, arm_points])

        return torch.cat(all_points, dim=0)

    def generateBuildings(self, num_buildings: int = 100) -> torch.Tensor:
        """
        生成建筑物点云

        Args:
            num_buildings: 建筑物数量

        Returns:
            torch.Tensor: 建筑物点云数据
        """
        print(f"生成 {num_buildings} 个建筑物点云")

        all_points = []

        for i in range(num_buildings):
            # 随机建筑物参数
            x = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
            y = (torch.rand(1).item() - 0.5) * self.ground_size * 0.8
            width = 12 + torch.rand(1).item() * 45
            length = 12 + torch.rand(1).item() * 45
            height = 15 + torch.rand(1).item() * 75

            points_per_building = 6000

            # 生成建筑物立方体
            building_points = self._generateBox(
                points_per_building,
                width=width,
                height=height,
                depth=length,
                center_x=x,
                center_y=y,
                center_z=height / 2
            )

            # 添加屋顶（金字塔形或平顶）
            if torch.rand(1).item() > 0.3:  # 70%的建筑有斜屋顶
                roof_height = height * 0.15
                roof_points = self._generatePyramidRoof(
                    points_per_building // 4,
                    width=width * 0.9,
                    depth=length * 0.9,
                    height=roof_height,
                    center_x=x,
                    center_y=y,
                    center_z=height + roof_height / 2
                )
                all_points.append(roof_points)
            else:  # 平顶建筑
                roof_points = self._generateBox(
                    points_per_building // 8,
                    width=width * 1.02,
                    height=0.5,
                    depth=length * 1.02,
                    center_x=x,
                    center_y=y,
                    center_z=height + 0.25
                )
                all_points.append(roof_points)

            all_points.append(building_points)

        return torch.cat(all_points, dim=0)

    def generateTrees(self, num_trees: int = 600) -> torch.Tensor:
        """
        生成树木点云

        Args:
            num_trees: 树木数量

        Returns:
            torch.Tensor: 树木点云数据
        """
        print(f"生成 {num_trees} 棵树木点云")

        all_points = []
        points_per_tree = 1000

        for i in range(num_trees):
            # 随机位置（避免与建筑物重叠）
            max_attempts = 10
            for attempt in range(max_attempts):
                x = (torch.rand(1).item() - 0.5) * self.ground_size * 0.9
                y = (torch.rand(1).item() - 0.5) * self.ground_size * 0.9
                # 简单的位置验证（在实际应用中需要更复杂的碰撞检测）
                if torch.rand(1).item() > 0.1:  # 90%的通过率
                    break

            trunk_height = 4 + torch.rand(1).item() * 8
            crown_radius = 2.5 + torch.rand(1).item() * 5

            # 生成树干
            trunk_points = self._generateCylinder(
                points_per_tree // 4,
                radius=0.4,
                height=trunk_height,
                center_x=x,
                center_y=y
            )

            # 生成树冠（使用多个球体创造更自然的形状）
            crown_shape = random.choice(['sphere', 'cone', 'multi_sphere'])

            if crown_shape == 'sphere':
                crown_points = self._generateSphere(
                    points_per_tree * 3 // 4,
                    radius=crown_radius,
                    center_x=x,
                    center_y=y,
                    center_z=trunk_height + crown_radius * 0.8
                )
            elif crown_shape == 'cone':
                crown_points = self._generateCone(
                    points_per_tree * 3 // 4,
                    radius=crown_radius,
                    height=crown_radius * 1.5,
                    center_x=x,
                    center_y=y,
                    center_z=trunk_height
                )
            else:  # multi_sphere
                crown_points = []
                num_spheres = random.randint(2, 4)
                points_per_sphere = points_per_tree * 3 // (4 * num_spheres)
                for j in range(num_spheres):
                    offset_x = (torch.rand(1).item() - 0.5) * crown_radius * 0.6
                    offset_y = (torch.rand(1).item() - 0.5) * crown_radius * 0.6
                    offset_z = j * crown_radius * 0.6
                    sphere_radius = crown_radius * (0.7 - j * 0.15)
                    sphere_points = self._generateSphere(
                        points_per_sphere,
                        radius=sphere_radius,
                        center_x=x + offset_x,
                        center_y=y + offset_y,
                        center_z=trunk_height + offset_z
                    )
                    crown_points.append(sphere_points)
                crown_points = torch.cat(crown_points, dim=0)

            all_points.extend([trunk_points, crown_points])

        return torch.cat(all_points, dim=0)

    def _generateCylinder(self, num_points: int, radius: float, height: float,
                          center_x: float = 0, center_y: float = 0, center_z: float = 0,
                          rotation: str = 'z') -> torch.Tensor:
        """生成圆柱体点云"""
        theta = torch.rand(num_points, device=self.device) * 2 * math.pi
        r = torch.sqrt(torch.rand(num_points, device=self.device)) * radius
        h = torch.rand(num_points, device=self.device) * height

        if rotation == 'z':
            x = center_x + r * torch.cos(theta)
            y = center_y + r * torch.sin(theta)
            z = center_z + h
        elif rotation == 'x':
            x = center_x + h
            y = center_y + r * torch.cos(theta)
            z = center_z + r * torch.sin(theta)
        else:  # 'y'
            x = center_x + r * torch.cos(theta)
            y = center_y + h
            z = center_z + r * torch.sin(theta)

        return torch.stack([x, y, z], dim=1).cpu()

    def _generateBox(self, num_points: int, width: float, height: float, depth: float,
                     center_x: float = 0, center_y: float = 0, center_z: float = 0) -> torch.Tensor:
        """生成立方体点云"""
        # 在GPU上生成随机点
        u = torch.rand(num_points, device=self.device)
        v = torch.rand(num_points, device=self.device)
        w_face = torch.rand(num_points, device=self.device)

        # 选择面
        face = torch.randint(0, 6, (num_points,), device=self.device)

        x = torch.zeros(num_points, device=self.device)
        y = torch.zeros(num_points, device=self.device)
        z = torch.zeros(num_points, device=self.device)

        # 前后面
        mask = face == 0
        x[mask] = (u[mask] - 0.5) * width
        y[mask] = (v[mask] - 0.5) * depth
        z[mask] = -height / 2

        mask = face == 1
        x[mask] = (u[mask] - 0.5) * width
        y[mask] = (v[mask] - 0.5) * depth
        z[mask] = height / 2

        # 左右面
        mask = face == 2
        x[mask] = -width / 2
        y[mask] = (u[mask] - 0.5) * depth
        z[mask] = (v[mask] - 0.5) * height

        mask = face == 3
        x[mask] = width / 2
        y[mask] = (u[mask] - 0.5) * depth
        z[mask] = (v[mask] - 0.5) * height

        # 上下面
        mask = face == 4
        x[mask] = (u[mask] - 0.5) * width
        y[mask] = -depth / 2
        z[mask] = (v[mask] - 0.5) * height

        mask = face == 5
        x[mask] = (u[mask] - 0.5) * width
        y[mask] = depth / 2
        z[mask] = (v[mask] - 0.5) * height

        # 应用中心偏移
        x += center_x
        y += center_y
        z += center_z

        return torch.stack([x, y, z], dim=1).cpu()

    def _generateSphere(self, num_points: int, radius: float,
                        center_x: float = 0, center_y: float = 0, center_z: float = 0) -> torch.Tensor:
        """生成球体点云"""
        # 在GPU上生成均匀分布的球面点
        u = torch.rand(num_points, device=self.device)
        v = torch.rand(num_points, device=self.device)

        theta = u * 2 * math.pi
        phi = torch.acos(2 * v - 1)

        r = radius * torch.pow(torch.rand(num_points, device=self.device), 1 / 3)

        x = center_x + r * torch.sin(phi) * torch.cos(theta)
        y = center_y + r * torch.sin(phi) * torch.sin(theta)
        z = center_z + r * torch.cos(phi)

        return torch.stack([x, y, z], dim=1).cpu()

    def _generatePyramidRoof(self, num_points: int, width: float, depth: float, height: float,
                             center_x: float = 0, center_y: float = 0, center_z: float = 0) -> torch.Tensor:
        """生成金字塔形屋顶点云"""
        u = torch.rand(num_points, device=self.device)
        v = torch.rand(num_points, device=self.device)
        w = torch.rand(num_points, device=self.device)

        # 在基础矩形上生成点
        base_x = (u - 0.5) * width
        base_y = (v - 0.5) * depth

        # 计算金字塔高度（从中心到边缘线性减少）
        dist_x = torch.abs(base_x) / (width / 2)
        dist_y = torch.abs(base_y) / (depth / 2)
        max_dist = torch.max(dist_x, dist_y)

        # 高度从中心到边缘线性减少
        roof_z = height * (1 - max_dist)

        x = center_x + base_x
        y = center_y + base_y
        z = center_z + roof_z * w  # 在屋顶表面随机分布

        return torch.stack([x, y, z], dim=1).cpu()

    def _generateCone(self, num_points: int, radius: float, height: float,
                      center_x: float = 0, center_y: float = 0, center_z: float = 0) -> torch.Tensor:
        """生成圆锥体点云"""
        u = torch.rand(num_points, device=self.device)
        v = torch.rand(num_points, device=self.device)

        # 底面半径随高度线性减少
        r = radius * (1 - v) * torch.sqrt(u)
        theta = torch.rand(num_points, device=self.device) * 2 * math.pi

        x = center_x + r * torch.cos(theta)
        y = center_y + r * torch.sin(theta)
        z = center_z + v * height

        return torch.stack([x, y, z], dim=1).cpu()

    def pointsToDict(self, points: torch.Tensor) -> Dict:
        """
        将点云张量转换为字典格式用于JSON保存

        Args:
            points: 点云张量

        Returns:
            Dict: 可序列化的字典
        """
        points_np = points.numpy()
        return {
            "points": points_np.tolist(),
            "num_points": len(points_np),
            "bounds": {
                "min_x": float(points_np[:, 0].min()),
                "max_x": float(points_np[:, 0].max()),
                "min_y": float(points_np[:, 1].min()),
                "max_y": float(points_np[:, 1].max()),
                "min_z": float(points_np[:, 2].min()),
                "max_z": float(points_np[:, 2].max())
            },
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "total_points": len(points_np)
            }
        }

    def generateCompleteScene(self, save_dir: str = None) -> str:
        """
        生成完整的场景点云数据并保存

        Args:
            save_dir: 保存目录，如果为None则使用时间戳创建

        Returns:
            str: 保存目录路径
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"enhanced_point_cloud_data_{timestamp}"

        os.makedirs(save_dir, exist_ok=True)
        print(f"数据将保存到: {save_dir}")

        # 生成各种点云数据（按从大到小的顺序，优化内存使用）
        objects_to_generate = [
            ("ground_plane", lambda: self.generateGroundPlane(1800000)),
            ("roads", lambda: self.generateRoadNetwork(100)),  # 减少道路数量避免内存问题
            ("lakes", lambda: self.generateLake(10)),
            ("streams", lambda: self.generateStream(10)),  # 减少溪流数量
            ("farmland", lambda: self.generateFarmland(50, 35000)),
            ("buildings", lambda: self.generateBuildings(100)),
            ("windows", lambda: self.generateWindows(80)),  # 减少窗户数量
            ("bridges", lambda: self.generateBridge(6)),
            ("trees", lambda: self.generateTrees(800)),
            ("traffic_lights", lambda: self.generateTrafficLights(300)),
            ("clouds", lambda: self.generateClouds(20))
        ]

        total_points = 0
        file_sizes = {}
        successful_objects = []

        for obj_name, generator_func in objects_to_generate:
            print(f"\n正在生成 {obj_name}...")
            try:
                # 生成前清理内存
                self.cleanup_memory()

                points = generator_func()
                data_dict = self.pointsToDict(points)

                filename = os.path.join(save_dir, f"{obj_name}.json")
                with open(filename, 'w') as f:
                    json.dump(data_dict, f, indent=2)

                file_size_mb = os.path.getsize(filename) / (1024 * 1024)
                file_sizes[obj_name] = file_size_mb

                print(f"✓ 已保存 {obj_name}: {len(points):,} 个点 ({file_size_mb:.2f} MB)")
                total_points += len(points)
                successful_objects.append(obj_name)

                # 生成后清理内存
                del points
                self.cleanup_memory()

            except Exception as e:
                print(f"✗ 生成 {obj_name} 时出错: {e}")
                import traceback
                traceback.print_exc()

        # 生成场景摘要
        summary = {
            "total_points": total_points,
            "successful_objects": successful_objects,
            "failed_objects": [obj[0] for obj in objects_to_generate if obj[0] not in successful_objects],
            "file_sizes": file_sizes,
            "generation_time": datetime.now().isoformat(),
            "scene_bounds": {
                "ground_size": self.ground_size,
                "max_height": self.max_height
            }
        }

        summary_file = os.path.join(save_dir, "scene_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"生成完成! 成功生成 {len(successful_objects)}/{len(objects_to_generate)} 个对象")
        print(f"总点数: {total_points:,}")
        print(f"数据保存在: {save_dir}")
        print(f"{'=' * 60}")

        # 打印文件大小统计
        print("\n文件大小统计:")
        for obj_name, size in file_sizes.items():
            print(f"  {obj_name}: {size:.2f} MB")

        return save_dir


def main():
    """主函数示例"""
    print("修复版点云数据生成器启动...")
    generator = PointCloudGenerator()

    try:
        save_path = generator.generateCompleteScene()

        # 显示生成的文件
        print("\n生成的文件:")
        for file in sorted(os.listdir(save_path)):
            file_path = os.path.join(save_path, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file}: {size_mb:.2f} MB")

    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()