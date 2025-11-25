

"""
    转换端训练的pytorch模型为嵌入式端的RK模型的程序文件
"""

import torch
import segmentation_models_pytorch as smp
from rknn.api import RKNN
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2



PYTORCH_MODEL_PATH = '/home/next_lb/桌面/WYT_S_S/code/robot_deeplabv3/results/pytorch_models/maxEpochs_110_learningRate_0.001/deepLabV3_low_loss.pth'
ONNX_MODEL_PATH = './onnx_models'
RKNN_MODEL_PATH = 'rknn_models'

HEIGHT = 288
WIDTH = 288
NUM_CLASSES = 3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SimpleDeepLabV3 = smp.DeepLabV3(
    encoder_name='mobilenet_v2',    # 预训练架构
    encoder_weights='imagenet',     # 与训练权重
    in_channels=3,      # 输入通道数
    classes=NUM_CLASSES,
    activation=None, # 不使用激活函数，直接输出logits   (即直接使用最终输出的概率logits，不需要再经过softmax等的函数计算概率，在分类任务中后面直接接着交叉熵损失，效果是不错的)
).to(device)


# TODO: 自主手动构建的模型架构
# 自主构建基于mobilenetv2的deeplabv3模型
class MobileNetV2DeepLabV3(nn.Module):
    def __init__(self):
        super(MobileNetV2DeepLabV3, self).__init__()

        # 定义特征提取层
        self.identity_1 = nn.Identity()
        self.identity_2 = nn.Identity()
        self.identity_3 = nn.Identity()
        self.identity_4 = nn.Identity()
        self.identity_5 = nn.Identity()



        asppInputChannels = 1280

        self.num_classes = NUM_CLASSES
        self.backbone = MobileNetV2Backbone()



        # 低级特征处理层（处理来自backbone的低层特征）
        self.low_level_conv_1 = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),  # 24是backbone中第二个block的输出通道数
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.low_level_conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.low_level_conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.low_level_conv_4 = nn.Sequential(
            nn.Conv2d(96, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.low_level_conv_5 = nn.Sequential(
            nn.Conv2d(160, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )


        # ASPP 模块
        self.aspp = ASPPModule(asppInputChannels, 256)


        # 特征融合后的分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 48 + 64 + 128 + 192 + 320, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, self.num_classes, 1)
        )


        # 初始化权重
        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 获取输入尺寸
        input_size = x.size()[2:]

        # TODO: 特征提取
        x = self.identity_1(x)

        # 骨干网络前向传播
        x, features = self.backbone(x)


        # TODO: 低维度的特征处理
        # 提取低级特征（来自backbone的第二个block）
        low_level_feat_1 = features[1]  # 假设features[1]是24通道的低级特征

        # 处理低级特征
        low_level_feat_1 = self.low_level_conv_1(low_level_feat_1)

        low_level_feat_2 = features[2]

        low_level_feat_2 = self.low_level_conv_2(low_level_feat_2)

        low_level_feat_3 = features[3]

        low_level_feat_3 = self.low_level_conv_3(low_level_feat_3)

        low_level_feat_4 = features[4]

        low_level_feat_4 = self.low_level_conv_4(low_level_feat_4)

        low_level_feat_5 = features[5]

        low_level_feat_5 = self.low_level_conv_5(low_level_feat_5)



        # TODO: 特征提取
        x = self.identity_2(x)

        # ASPP 模块
        high_level_feat = self.aspp(x)

        # 将高级特征上采样到低级特征的分辨率
        high_level_feat = F.interpolate(high_level_feat, size=low_level_feat_1.shape[2:],
                                        mode='bilinear', align_corners=True)

        low_level_feat_2 = F.interpolate(low_level_feat_2, size=low_level_feat_1.shape[2:],
                                         mode='bilinear', align_corners=True)

        low_level_feat_3 = F.interpolate(low_level_feat_3, size=low_level_feat_1.shape[2:],
                                         mode='bilinear', align_corners=True)

        low_level_feat_4 = F.interpolate(low_level_feat_4, size=low_level_feat_1.shape[2:],
                                         mode='bilinear', align_corners=True)

        low_level_feat_5 = F.interpolate(low_level_feat_5, size=low_level_feat_1.shape[2:],
                                         mode='bilinear', align_corners=True)


        # 特征融合：拼接低级和高级特征
        fused_feat = torch.cat([high_level_feat, low_level_feat_1, low_level_feat_2, low_level_feat_3, low_level_feat_4, low_level_feat_5], dim=1)

        # TODO: 特征提取
        x = self.identity_3(x)

        # 分类
        x = self.classifier(fused_feat)

        # TODO: 特征提取
        x = self.identity_4(x)

        # 上采样到原始输入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        # TODO: 特征提取
        x = self.identity_5(x)

        return x




class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super(MobileNetV2Backbone, self).__init__()

        # 输入通道配置
        widthMult = 1.0
        inputChannel = 32
        lastChannel = 1280

        # 倒残差模块的配置：[t, c, n, s]
        # t: 扩展因子， c: 输出通道， n: 重复次数， s: 步长
        invertedResidualSetting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 1
            [6, 24, 2, 2],  # 3
            [6, 32, 3, 2],  # 6
            [6, 64, 4, 2],  # 10
            [6, 96, 3, 1],  # 13
            [6, 160, 3, 2],  # 16
            [6, 320, 1, 1]  # 17
        ]

        # 构建第一个卷积层
        inputChannel = int(inputChannel * widthMult)
        self.lastChannel = int(lastChannel * max(1.0, widthMult))
        features = [ConvBNReLU(3, inputChannel, 3, 2, 1)]


        # 构建倒残差模块
        for t, c, n, s in invertedResidualSetting:
            outputChannel = int(c * widthMult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(inputChannel, outputChannel, stride, t))
                inputChannel = outputChannel


        # 构建最后的1x1卷积层
        features.append(ConvBNReLU(inputChannel, self.lastChannel, 1, 1, 1))

        self.features = nn.Sequential(*features)

        # 加载预训练参数
        self._load_pretrained_weights()


    def _load_pretrained_weights(self):
        """加载torchvision的预训练权重"""
        try:
            pretrained_model = mobilenet_v2(pretrained=True)

            # 手动对齐参数
            state_dict = self.state_dict()
            pretrained_dict = pretrained_model.state_dict()

            # 参数名称映射（因为我们的实现可能与torchvision略有不同）
            mapping = {}
            for i, (name, param) in enumerate(pretrained_dict.items()):
                if 'features' in name:
                    mapping[name] = name

            # 加载匹配的参数
            for name, param in pretrained_dict.items():
                if name in mapping and mapping[name] in state_dict:
                    if state_dict[mapping[name]].shape == param.shape:
                        state_dict[mapping[name]] = param

            self.load_state_dict(state_dict)
            print("成功加载预训练权重")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化的权重")

    def forward(self, x):
        # 存储中间特征 (用于可能的特征融合)
        features = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            # 记录不同的分辨率的特征图 (可选，用于多尺度特征)
            if i in [1, 3, 6, 10, 13, 16]:      # 这些索引对应不同阶段的输出
                features.append(x)

        return x, features






class ConvBNReLU(nn.Sequential):
    """卷积 + BatchNorm + ReLU6 的基本模块"""
    def __init__(self, inChannels, outChannels, kernelSize, stride, groups):
        padding = (kernelSize - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU6(inplace=True)
        )




class InvertedResidual(nn.Module):
    """MobileNetV2的倒残差模块"""
    def __init__(self, inChannels, outChannels, stride, expandRatio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        assert stride in [1, 2]

        hiddenDim = int(round(inChannels * expandRatio))
        self.use_res_connect = self.stride == 1 and inChannels == outChannels

        layers = []
        if expandRatio != 1:
            # 扩展层   (逐点卷积)
            layers.append(ConvBNReLU(inChannels, hiddenDim, 1, 1, 1))

        layers.extend([
            # 深度卷积
            ConvBNReLU(hiddenDim, hiddenDim, 3, stride, hiddenDim),
            # 逐点卷积(线性激活，无ReLU)
            nn.Conv2d(hiddenDim, outChannels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outChannels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




class SeparableConv2d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding, dilation, bias):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(inChannels, inChannels, kernelSize,
                                  stride=stride, padding=padding,
                                  dilation=dilation, groups=inChannels, bias=bias)
        self.pointwise = nn.Conv2d(inChannels, outChannels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class ASPPModule(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling) 模块"""

    def __init__(self, inChannels, outChannels):
        super(ASPPModule, self).__init__()

        # 1x1 卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

        # 3x3 空洞卷积, rate=6
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

        # 3x3 空洞卷积, rate=12
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

        # 3x3 空洞卷积, rate=18
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inChannels, outChannels, 1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

        # 输出卷积
        self.conv_out = nn.Sequential(
            nn.Conv2d(outChannels * 5, outChannels, 1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # 各个分支的前向传播
        feat1x1 = self.conv1x1(x)
        feat3x3_1 = self.conv3x3_1(x)
        feat3x3_2 = self.conv3x3_2(x)
        feat3x3_3 = self.conv3x3_3(x)

        # 全局平均池化分支
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=True)

        # 拼接所有特征
        out = torch.cat([feat1x1, feat3x3_1, feat3x3_2, feat3x3_3, global_feat], dim=1)
        out = self.conv_out(out)

        return out






def clear_folder(folderPath: str):

    # 遍历文件夹内的所有内容
    for item in os.listdir(folderPath):
        itemPath = os.path.join(folderPath, item)
        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(itemPath) or os.path.islink(itemPath):
                os.unlink(itemPath)
                print(f'已删除文件夹: {itemPath}')
            elif os.path.isdir(itemPath):
                # 使用 shutil.rmtree 删除文件夹及其内容
                shutil.rmtree(itemPath)
                print(f"已删除文件夹： {itemPath}")

        except Exception as e:
            print(f"删除 {itemPath} 时出错： {e}")

    print(f"文件夹 {folderPath} 内容已清空")




def main():

    pytorchModelPathList = PYTORCH_MODEL_PATH.split('/')


    # 首先构建一下转换模型后的存储路径
    saveONNXModelPath = os.path.join(ONNX_MODEL_PATH, pytorchModelPathList[-2])
    saveRKNNModelPath = os.path.join(RKNN_MODEL_PATH, pytorchModelPathList[-2])
    os.makedirs(saveONNXModelPath, exist_ok=True)
    clear_folder(saveONNXModelPath)
    os.makedirs(saveONNXModelPath, exist_ok=True)
    os.makedirs(saveRKNNModelPath, exist_ok=True)
    clear_folder(saveRKNNModelPath)
    os.makedirs(saveRKNNModelPath, exist_ok=True)

    saveONNXModelPath = os.path.join(saveONNXModelPath, 'deepLabV3_onnx_model.onnx')
    saveRKNNModelPath = os.path.join(saveRKNNModelPath, 'deepLabV3_rknn_model.rknn')


    # 导入模型架构
    # model = SimpleDeepLabV3
    model = MobileNetV2DeepLabV3().to(device)


    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
    model.eval()

    # 创建固定batch size的输入张量
    dummyInput = torch.randn(1, 3, HEIGHT, WIDTH).to(device)


    torch.onnx.export(
        model,
        dummyInput,
        saveONNXModelPath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None       # 不设置动态轴，使用固定的batch size
    )
    print(f"ONNX model saved to {saveONNXModelPath}")


    # 创建 RKNN 对象
    rknn = RKNN()

    # 设置模型配置
    rknn.config(
        target_platform='rk3568',
        optimization_level=3
    )

    print('--> Loading model')
    ret = rknn.load_onnx(
        model=saveONNXModelPath,
        input_size_list=[[1, 3, HEIGHT, WIDTH]]     # 明确指定输入尺寸
    )

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')


    # 构建RKNN模型
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')


    # 到处RKNN模型
    print('--> Export RKNN model')
    ret = rknn.export_rknn(saveRKNNModelPath)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 释放 RKNN 对象
    rknn.release()






if __name__ == '__main__':
    main()


