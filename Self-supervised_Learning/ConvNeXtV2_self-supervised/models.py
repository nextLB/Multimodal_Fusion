"""
    ConvNeXtV2模型构建程序文件
"""
import os

import torch
import torch.nn as nn
import base_models
import config_parameters
from torchvision.utils import save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# TODO: 构建FCAME模型
class FCMAE(nn.Module):
    """
        Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(self, imageSize, inChannels, depths, dims, decoderDepth, decoderEmbedDim, patchSize, maskRatio, normPixLoss):
        super().__init__()

        # configs
        self.imageSize = imageSize
        self.depths = depths
        self.imds = dims
        self.patchSize = patchSize
        self.maskRatio = maskRatio
        self.numPatches = (imageSize // patchSize) ** 2
        self.decoderEmbedDim = decoderEmbedDim
        self.decoderDepth = decoderDepth
        self.normPixLoss = normPixLoss

        # 编码层
        self.encoder = base_models.SparseConvNeXtV2(inChannels, config_parameters.NUM_CLASSES, depths, dims, config_parameters.DROP_PATH_RATE, config_parameters.D)

        # decoder
        self.proj = nn.Conv2d(in_channels=dims[-1], out_channels=decoderEmbedDim, kernel_size=1)
        # mask tokens
        self.maskToken = nn.Parameter(torch.zeros(1, decoderEmbedDim, 1, 1))
        decoder = [base_models.Block(dim=decoderEmbedDim, dropPath=0.) for i in range(decoderDepth)]

        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(in_channels=decoderEmbedDim, out_channels=patchSize ** 2 * inChannels, kernel_size=1)


        self.encoderBeforeIdentity = nn.Identity()
        self.encoderAfterIdentity = nn.Identity()
        self.decoderAfterIdentity = nn.Identity()




    # 随机产生遮盖的掩码的方法函数
    def gen_random_mask(self, x, maskRatio):
        """生成随机掩码
        Args:
            x: 输入张量，形状为 [N, C, H, W]
            mask_ratio: 掩码比例，0-1之间的浮点数，表示要掩码的patch比例
        Returns:
            mask: 二进制掩码，0表示保留，1表示掩码/移除
        """

        # N: batch size
        N = x.shape[0]
        # L: 总的patch数量，假设输入是正方形，patchSize是每个patch的边长
        L = (x.shape[2] // self.patchSize) ** 2
        # lenKeep: 需要保留的patch数量
        lenKeep = int(L * (1 - maskRatio))

        # 生成随机噪声，用于决定patch的随机排序
        # 形状: [N, L] - 每个样本有L个随机值
        noise = torch.randn(N, L, device=x.device)

        # 对每个样本的噪声进行排序，得到随机排列的索引
        # idsShuffle: 按噪声值排序后的索引，形状 [N, L]
        idsShuffle = torch.argsort(noise, dim=1)
        # idsRestore: 恢复原始顺序的索引，用于后续还原掩码顺序
        idsRestore = torch.argsort(idsShuffle, dim=1)

        # 生成初始二进制掩码：全1表示全部掩码
        mask = torch.ones([N, L], device=x.device)
        # 将前len_keep个位置设为0（保留）
        mask[:, :lenKeep] = 0
        # 使用恢复索引将掩码还原到原始patch顺序
        # 这样掩码就对应到原始的patch布局
        mask = torch.gather(mask, dim=1, index=idsRestore)

        return mask


    # 计算损失
    def forward_loss(self, images, pred, mask, epoch, batchIdx, savePath):
        """
        计算前向传播的损失函数
        Args:
            imgs: 原始图像 [N, 3, H, W]
            pred: 模型预测结果 [N, L, p*p*3] 或 [N, C, H, W]
            mask: 二进制掩码 [N, L], 0表示保留, 1表示移除/掩码
        Returns:
            loss: 计算得到的损失值
        """
        # 如果pred是4维张量（通常是卷积层的输出），需要reshape为3维
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            # 将特征图展平为patch序列: [N, C, H, W] -> [N, C, L]
            pred = pred.reshape(n, c, -1)
            # 调整维度顺序: [N, C, L] -> [N, L, C] 以匹配目标格式
            pred = torch.einsum('ncl->nlc', pred)


        # 将原始图像分割成patch，与pred的形状对齐
        target = self.patchify(images)  # [N, L, p*p*3]


        # TODO: 如果符合轮数要求，就将其保存下来
        if epoch % 5 == 0 and batchIdx % 100 == 0:
            completeSavePath = os.path.join(savePath, f'forward_loss_epoch_{epoch}_batchIdx_{batchIdx}')
            os.makedirs(completeSavePath, exist_ok=True)

            # 保存原始图像
            save_image(images, f'{completeSavePath}/original_images.png', nrow=4, normalize=True)

            # 还原预测图像
            pred_restored = self.unpatchify(pred)  # [N, C, H, W]
            save_image(pred_restored, f'{completeSavePath}/predicted_images.png', nrow=4, normalize=True)

            # 还原掩码图像
            mask_vis = self.mask_to_image(mask)  # [N, 1, H, W]
            save_image(mask_vis, f'{completeSavePath}/mask_images.png', nrow=4, normalize=True)

            # 创建带掩码的原始图像可视化
            masked_images = images * (1 - mask_vis)  # 将掩码区域置为黑色
            save_image(masked_images, f'{completeSavePath}/masked_original_images.png', nrow=4, normalize=True)

            # 创建带掩码的预测图像可视化
            masked_pred = pred_restored * mask_vis  # 只显示被掩码区域的预测
            save_image(masked_pred, f'{completeSavePath}/masked_predicted_images.png', nrow=4, normalize=True)




        # 如果启用像素值归一化
        if self.normPixLoss:
            # 计算每个patch的均值，保持维度用于广播
            mean = target.mean(dim=-1, keepdim=True)  # [N, L, 1]
            # 计算每个patch的方差，保持维度
            var = target.var(dim=-1, keepdim=True)  # [N, L, 1]
            # 对target进行标准化：(x - μ) / σ
            target = (target - mean) / (var + 1.e-6) ** .5

        # 计算预测值与目标值之间的均方误差
        loss = (pred - target) ** 2  # [N, L, p*p*3]

        # 在每个patch内求平均，得到每个patch的损失值
        loss = loss.mean(dim=-1)  # [N, L] - 每个patch的平均损失

        # 只计算被掩码patch的损失（mask=1的位置）
        # (loss * mask): 保留被掩码patch的损失，其他位置为0
        # mask.sum(): 被掩码patch的总数
        loss = (loss * mask).sum() / mask.sum()  # 被掩码patch的平均损失

        return loss


    def mask_to_image(self, mask):
        """
        将掩码转换为图像格式用于可视化
        Args:
            mask: [N, L] 二进制掩码
        Returns:
            [N, 1, H, W] 掩码图像，其中掩码区域为1，非掩码区域为0
        """
        N, L = mask.shape
        p = self.patchSize
        H = W = self.imageSize
        h = w = H // p

        # 将掩码reshape为图像格式
        mask_img = mask.reshape(N, h, w, 1, 1, 1)
        mask_img = mask_img.repeat(1, 1, 1, p, p, 1)
        mask_img = mask_img.permute(0, 5, 1, 3, 2, 4).contiguous()
        mask_img = mask_img.reshape(N, 1, H, W)

        return mask_img

    def patchify(self, imgs):
        """
        将图像分割成多个块(patch)
        Args:
            imgs: 输入图像张量，形状为 (N, 3, H, W)
        Returns:
            x: 分块后的张量，形状为 (N, L, patch_size**2 * 3)
            其中 L 是块的数量，L = (H // patch_size) * (W // patch_size)
        """
        p = self.patchSize  # 每个patch的边长
        # 确保图像是正方形且能被patch大小整除
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # 计算在高度和宽度方向上的patch数量
        h = w = imgs.shape[2] // p

        # 第一步reshape: 将图像分解为patch网格
        # [N, 3, H, W] -> [N, 3, h, p, w, p]
        # 其中 h, w 是patch网格的维度，p 是patch大小
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))

        # 使用爱因斯坦求和记号调整维度顺序
        # 从 [N, 3, h, p, w, p] -> [N, h, w, p, p, 3]
        # 这样每个patch的像素和通道信息都在最后三个维度
        x = torch.einsum('nchpwq->nhwpqc', x)

        # 最后reshape: 将空间维度合并，展平每个patch
        # [N, h, w, p, p, 3] -> [N, h*w, p*p*3]
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x


    def unpatchify(self, x):
        """
        将分块后的张量重组回原始图像格式
        Args:
            x: 分块后的张量，形状为 (N, L, patch_size**2 * 3)
        Returns:
            imgs: 重组后的图像，形状为 (N, 3, H, W)
        """
        p = self.patchSize
        # 计算原始图像中patch网格的维度
        # 假设原始图像是正方形，所以 h = w = sqrt(L)
        h = w = int(x.shape[1] ** .5)
        # 验证L确实是完全平方数
        assert h * w == x.shape[1]

        # 第一步reshape: 恢复patch网格结构
        # [N, L, p*p*3] -> [N, h, w, p, p, 3]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))

        # 使用爱因斯坦求和记号调整维度顺序，与patchify相反
        # 从 [N, h, w, p, p, 3] -> [N, 3, h, p, w, p]
        x = torch.einsum('nhwpqc->nchpwq', x)

        # 最后reshape: 合并patch维度，恢复原始图像形状
        # [N, 3, h, p, w, p] -> [N, 3, h*p, w*p]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs



    def forward_encoder(self, imgs, maskRatio):
        # generate random masks
        mask = self.gen_random_mask(imgs, maskRatio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.maskToken.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred


    def forward(self, imgs, epoch, batchIdx, savePath):



        x = self.encoderBeforeIdentity(imgs)
        x, mask = self.forward_encoder(x, self.maskRatio)
        x = self.encoderAfterIdentity(x)



        pred = self.forward_decoder(x, mask)
        pred = self.decoderAfterIdentity(pred)



        loss = self.forward_loss(imgs, pred, mask, epoch, batchIdx, savePath)



        return loss, pred, mask











def get_model():
    model = FCMAE(config_parameters.PRETRAINED_IMAGE_SIZE,
                  config_parameters.IN_CHANNELS,
                  config_parameters.DEPTHS,
                  config_parameters.DIMS,
                  config_parameters.DECODER_DEPTH,
                  config_parameters.DECODER_EMBED_DIM,
                  config_parameters.PATCH_SIZE,
                  config_parameters.MASK_RATIO,
                  config_parameters.NORM_PIX_LOSS).to(device)

    return model



