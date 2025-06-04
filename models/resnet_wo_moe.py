import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.detr_seg import SegmentationTransformer
# MoEAdapter：包含多个专家和门控网络，并在输出后添加激活函数
class MoEAdapter(nn.Module):
    def __init__(self, input_channels, num_experts=3, output_channels=128, negative_slope=0.2, top_k=2):
        super(MoEAdapter, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # 激活的专家数量

        # 创建专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, int(input_channels // 2), kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(int(input_channels // 2), input_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            ) for _ in range(num_experts)
        ])

        # 创建门控网络
        self.gating_network = nn.Conv2d(input_channels, num_experts, kernel_size=1)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: (B, num_experts, C, H, W)

        # 获取门控网络的权重
        gate_weights = self.gating_network(x)  # Shape: (B, num_experts, H, W)
        gate_weights = F.softmax(gate_weights, dim=1)  # Softmax归一化

        # Top-K gating 稀疏化
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=1)  # 获取Top-K权重及其索引
        sparse_gate_weights = torch.zeros_like(gate_weights)
        sparse_gate_weights.scatter_(1, topk_indices, topk_values)  # 构造稀疏权重
        sparse_gate_weights = sparse_gate_weights / sparse_gate_weights.sum(dim=1, keepdim=True)  # 重新归一化

        # 稀疏加权求和
        sparse_gate_weights = sparse_gate_weights.unsqueeze(2)  # Shape: (B, num_experts, 1, H, W)
        weighted_sum = (sparse_gate_weights * expert_outputs).sum(dim=1)  # 加权求和，Shape: (B, C, H, W)

        # 激活函数
        return self.leaky_relu(weighted_sum)

#------------用于resnet浅层的专家------------#
from models.deal.dehaze import DehazeNet, dehaze_image
from models.deal.low_enhance import Low_enhance_net, low_enhance_image

class MoEAdapter_shallow(nn.Module):
    def __init__(self, top_k=2):
        super(MoEAdapter_shallow, self).__init__()
        self.dehazenet = DehazeNet()
        self.lownet = Low_enhance_net()
        self.experts = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # 门控网络
        self.gating_network = nn.Conv2d(16, 3, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU()
        self.top_k = top_k  # 激活的专家数量

    def forward(self, x):
        # 专家输出
        tx, a = self.dehazenet(x)
        r = self.lownet(x)
        x1 = dehaze_image(x, tx, a) + x  # zero dce
        x2 = low_enhance_image(x, r) + x
        x3 = self.experts(x)
        expert_outputs = torch.stack([x1, x2, x3], dim=1)  # Shape: (B, 3, C, H, W)

        # 门控网络生成权重
        gate_weights = self.gating_network(x)  # Shape: (B, 3, H, W)
        gate_weights = F.softmax(gate_weights, dim=1)

        # 稀疏化权重：Top-K gating
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=1)  # 选择前 K 个专家
        sparse_gate_weights = torch.zeros_like(gate_weights)
        sparse_gate_weights.scatter_(1, topk_indices, topk_values)  # 构造稀疏权重
        sparse_gate_weights = sparse_gate_weights / sparse_gate_weights.sum(dim=1, keepdim=True)  # 重新归一化

        # 稀疏加权求和
        sparse_gate_weights = sparse_gate_weights.unsqueeze(2)  # Shape: (B, 3, 1, H, W)
        weighted_sum = (sparse_gate_weights * expert_outputs).sum(dim=1)  # 加权求和，Shape: (B, C, H, W)

        return self.leaky_relu(weighted_sum)

#----------写一个融合网络，不使用ResNet------------------#
class feature_exbase(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(feature_exbase, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=1, padding=1, kernel_size=3)

    def forward(self, x):
        act = nn.LeakyReLU()
        x = self.conv(x)
        x = act(x)
        return x



class ResNetWithMOE(nn.Module):
    def __init__(self):
        super(ResNetWithMOE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.moe1 = MoEAdapter_shallow()
        # self.moe2 = MoEAdapter(input_channels=32, output_channels=32, num_experts=2)
        # self.moe3 = MoEAdapter(input_channels=64, output_channels=64, num_experts=2)
        # self.moe4 = MoEAdapter(input_channels=128, output_channels=128, num_experts=2)

    def forward(self, x, vi=False):
        x = self.conv1(x)
        # if vi:
        #     x = self.moe1(x) + x
        y1 = x
        x = self.conv2(x)

        y2 = x
        x = self.conv3(x)

        y3 = x
        x = self.conv4(x)

        return x, y3, y2, y1



class ResNetSegmentationModelWithMoE(nn.Module):
    def __init__(self):
        super(ResNetSegmentationModelWithMoE, self).__init__()

        # 加载预训练的ResNet50模型，并去掉最后的全连接层
        self.resnet = ResNetWithMOE()

        # 定义卷积块（包含卷积和激活函数）
        self.conv_block4 = self._conv_block(256, 128)
        self.conv_block5 = self._conv_block(128, 64)

        # 最终的分割输出层
        self.seg_head = SegmentationTransformer(9, num_queries=100, hidden_dim=64, feature_dim=64)

        # 最终的融合输出层
        self.fusion_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
        )

        # 重建图像的分支
        self.decode_vi = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, stride=1, padding=1, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, stride=1, padding=1, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, stride=1, padding=1, kernel_size=3)
        )

        self.decode_ir = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, stride=1, padding=1, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, stride=1, padding=1, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, stride=1, padding=1, kernel_size=3)
        )

    def _conv_block(self, in_channels, out_channels):
        """卷积块，包含卷积、激活函数、MoE-Adapter"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU()
        )

    def forward(self, vi, ir):
        vi_img, ir_img = vi, ir
        vi, vi3, vi2, vi1 = self.resnet(vi, vi=True)
        ir, ir3, ir2, ir1 = self.resnet(ir)

        #print(vi1.shape, vi2.shape, vi3.shape, vi4.shape, vi_last.shape)
        x = torch.cat([vi,ir],dim=1)
        # x = self.fusion_module(x)
        #x = torch.cat([self.cross(x, feature), x], dim=1)

        # 经过每个卷积块并添加MoE-Adapter
        x = self.conv_block4(x)
        x = self.conv_block5(x) + ir3 + vi3

        # 最后卷积输出分割掩码
        seg = self.seg_head(x)
        fusion = self.fusion_head(x)
        # 重建出可见光和红外,这段推理阶段给他注释掉
        vi = self.decode_vi(x)
        ir = self.decode_ir(x)

        return fusion[:,0:1, :, :] * vi_img + fusion[:,1:2, :, :] * ir_img, fusion[:,0:1, :, :] * vi_img + fusion[:,1:2, :, :] * ir_img, vi, ir, seg

# # # 示例输入（假设输入是3通道图像，尺寸为256x256）
# if __name__ == '__main__':
#
#     input_image = torch.randn(1, 3, 640, 480).cuda()
#     feature = torch.rand(1,512).cuda()
#
#     # 创建带MoE优化的ResNet分割模型并进行前向传播
#     model = ResNetSegmentationModelWithMoE(num_classes=9).cuda()
#     output = model(input_image, input_image, feature)

