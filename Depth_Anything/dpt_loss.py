"""
    关于深度估计模型的loss函数的构建程序文件
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLossV1_0(nn.Module):
    def __init__(self):
        super(DepthLossV1_0, self).__init__()

    # 进行梯度差异
    def gradient_loss(self, pred, target):
        grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        grad_loss_x = torch.abs(grad_pred_x - grad_target_x).mean()
        grad_loss_y = torch.abs(grad_pred_y - grad_target_y).mean()

        return grad_loss_x + grad_loss_y

    def forward(self, pred, target):
        # 确保预测和目标尺寸相同
        if pred.size() != target.size():
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

        # 添加诊断信息
        if torch.rand(1) < 0.001:  # 随机采样0.1%的批次进行诊断
            print(f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")

        l1_loss = F.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)

        # 组合损失 - 调整权重
        total_loss = l1_loss + 0.1 * grad_loss  # 降低梯度损失的权重

        # 添加诊断信息
        if torch.rand(1) < 0.001:
            print(f"L1 Loss: {l1_loss.item():.6f}, Grad Loss: {grad_loss.item():.6f}, Total: {total_loss.item():.6f}")

        return total_loss


class ImprovedDepthLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.1, delta=0.1, epsilon=1e-6):
        """
        改进的深度估计损失函数

        Args:
            alpha: L1损失的权重
            beta: 梯度损失的权重
            gamma: SSIM损失的权重
            delta: 尺度不变损失的权重
            epsilon: 数值稳定性参数
        """
        super(ImprovedDepthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        # 修复：在forward中动态创建高斯核，避免设备不匹配
        self.window_size = 5
        self.sigma = 1.5

    def _create_gaussian_kernel(self, device):
        """动态创建高斯核，确保在正确的设备上"""
        size = self.window_size
        sigma = self.sigma

        coords = torch.arange(size, device=device).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
        return kernel

    def multi_scale_gradient_loss(self, pred, target, scales=4):
        """多尺度梯度损失"""
        total_loss = 0
        current_weight = 1.0

        for scale in range(scales):
            if scale > 0:
                # 下采样
                pred_scaled = F.avg_pool2d(pred, kernel_size=2 ** scale, stride=2 ** scale)
                target_scaled = F.avg_pool2d(target, kernel_size=2 ** scale, stride=2 ** scale)
            else:
                pred_scaled, target_scaled = pred, target

            # Sobel算子计算梯度 - 确保在正确的设备上
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

            grad_pred_x = F.conv2d(pred_scaled, sobel_x, padding=1)
            grad_pred_y = F.conv2d(pred_scaled, sobel_y, padding=1)
            grad_target_x = F.conv2d(target_scaled, sobel_x, padding=1)
            grad_target_y = F.conv2d(target_scaled, sobel_y, padding=1)

            # 梯度幅度
            grad_pred_mag = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + self.epsilon)
            grad_target_mag = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2 + self.epsilon)

            # 梯度方向
            grad_pred_dir = torch.atan2(grad_pred_y, grad_pred_x + self.epsilon)
            grad_target_dir = torch.atan2(grad_target_y, grad_target_x + self.epsilon)

            # 幅度损失 + 方向损失
            mag_loss = F.l1_loss(grad_pred_mag, grad_target_mag)
            dir_loss = 1 - torch.cos(grad_pred_dir - grad_target_dir).mean()

            scale_loss = mag_loss + 0.5 * dir_loss
            total_loss += current_weight * scale_loss
            current_weight *= 0.5  # 降低权重

        return total_loss / scales

    def ssim_loss(self, pred, target, window_size=11, size_average=True):
        """结构相似性损失 - 修复设备不匹配问题"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 动态创建高斯核，确保在正确的设备上
        gaussian_kernel = self._create_gaussian_kernel(pred.device)

        # 使用高斯滤波
        mu_pred = F.conv2d(pred, gaussian_kernel, padding=window_size // 2, groups=1)
        mu_target = F.conv2d(target, gaussian_kernel, padding=window_size // 2, groups=1)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, gaussian_kernel, padding=window_size // 2, groups=1) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, gaussian_kernel, padding=window_size // 2, groups=1) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, gaussian_kernel, padding=window_size // 2,
                                     groups=1) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        if size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

    def scale_invariant_loss(self, pred, target):
        """尺度不变对数损失 - 对光照变化更鲁棒"""
        # 添加小值避免log(0)
        pred_log = torch.log(torch.clamp(pred, min=self.epsilon))
        target_log = torch.log(torch.clamp(target, min=self.epsilon))

        diff_log = pred_log - target_log
        return torch.sqrt(torch.mean(diff_log ** 2) - 0.5 * torch.mean(diff_log) ** 2 + self.epsilon)

    def berhu_loss(self, pred, target):
        """反向Huber损失 - 对异常值更鲁棒"""
        diff = torch.abs(pred - target)
        c = 0.2 * torch.max(diff).item()

        mask = diff <= c
        loss = torch.where(mask, diff, (diff ** 2 + c ** 2) / (2 * c + self.epsilon))
        return loss.mean()

    def edge_aware_smoothness_loss(self, pred, images):
        """边缘感知平滑度损失"""
        grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        # 使用图像梯度作为权重
        grad_image_x = torch.mean(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]), 1, keepdim=True)
        grad_image_y = torch.mean(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]), 1, keepdim=True)

        weight_x = torch.exp(-grad_image_x)
        weight_y = torch.exp(-grad_image_y)

        smoothness_x = grad_pred_x * weight_x
        smoothness_y = grad_pred_y * weight_y

        return (smoothness_x.mean() + smoothness_y.mean())

    def forward(self, pred, target, images=None):
        # 确保预测和目标尺寸相同
        if pred.size() != target.size():
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

        # 基础L1损失
        l1_loss = F.l1_loss(pred, target)

        # 改进的梯度损失
        grad_loss = self.multi_scale_gradient_loss(pred, target)

        # 结构相似性损失
        ssim_loss = self.ssim_loss(pred, target)

        # 尺度不变损失
        scale_inv_loss = self.scale_invariant_loss(pred, target)

        # BerHu损失作为L1的补充
        berhu_loss = self.berhu_loss(pred, target)

        # 组合损失
        total_loss = (self.alpha * l1_loss +
                      self.beta * grad_loss +
                      self.gamma * ssim_loss +
                      self.delta * scale_inv_loss +
                      0.1 * berhu_loss)  # BerHu损失权重较小

        # 如果提供了输入图像，添加边缘感知平滑度损失
        if images is not None:
            smooth_loss = self.edge_aware_smoothness_loss(pred, images)
            total_loss += 0.05 * smooth_loss

        # 诊断信息
        if torch.rand(1) < 0.001:
            print(f"L1: {l1_loss.item():.4f}, Grad: {grad_loss.item():.4f}, "
                  f"SSIM: {ssim_loss.item():.4f}, ScaleInv: {scale_inv_loss.item():.4f}, "
                  f"Total: {total_loss.item():.4f}")

        return total_loss


