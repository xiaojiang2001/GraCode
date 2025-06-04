import torch
import torch.nn as nn
import torch.nn.functional as F
## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x
from .nets_segformer.segformer import SegFormer
class SegmentationTransformer(nn.Module):
    def __init__(self, num_classes, num_queries, hidden_dim, feature_dim):
        super(SegmentationTransformer, self).__init__()
        self.head = SegFormer(num_classes=num_classes)

    def forward(self, features):
        return self.head(features)





# class SegmentationTransformer(nn.Module):
#     def __init__(self, num_classes, num_queries, hidden_dim, feature_dim):
#         """
#         Args:
#             num_classes (int): 分类类别数（包含背景类）。
#             num_queries (int): 查询的数量。
#             hidden_dim (int): Transformer 的隐藏维度。
#             feature_dim (int): 输入特征的通道数。
#         """
#         super(SegmentationTransformer, self).__init__()
#
#         # self.decoder = nn.Sequential(
#         #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(inplace=True),
#         #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1,padding=1),
#         #     nn.BatchNorm2d(32),
#         #     nn.ReLU(inplace=True),
#         #     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1,padding=1),
#         #     nn.BatchNorm2d(16),
#         #     nn.ReLU(inplace=True),
#         #     nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=1,padding=1)
#         # )
#
#         # 查询向量
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)  # (num_queries, hidden_dim)
#
#         # 输入特征降维
#         self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)  # (b, hidden_dim, h, w)
#
#         # Transformer 编码器-解码器
#         self.transformer = nn.Transformer(hidden_dim, nhead=1, num_encoder_layers=1, num_decoder_layers=1)
#
#         self.text_emb = FeatureWiseAffine(in_channels=512, out_channels=64)
#
#         # 分类头
#         self.class_embed = nn.Linear(hidden_dim, num_classes)  # 每个查询输出类别分布
#
#         # 掩码头
#         self.mask_embed = nn.Linear(hidden_dim, hidden_dim)  # 每个查询生成掩码
#
#
#
#     def forward(self, features, text_feature):
#         act = nn.LeakyReLU()
#         """
#         Args:
#             features (Tensor): 输入特征，形状为 (b, c, h, w)。
#
#         Returns:
#             logits (Tensor): 分割结果，形状为 (b, num_classes, h, w)。
#         """
#         b, c, h, w = features.shape
#
#         features = self.text_emb(features, text_feature) + features
#
#         # 1. 特征图降维
#         features_proj = self.input_proj(features)  # (b, hidden_dim, h, w)
#         features_proj = act(features_proj)
#         features_flat = features_proj.flatten(2).permute(2, 0, 1)  # (h*w, b, hidden_dim)
#
#         # 2. 查询向量初始化
#         queries = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)  # (num_queries, b, hidden_dim)
#
#         # 3. Transformer 编码器-解码器
#         transformer_output = self.transformer(features_flat, queries)  # (num_queries, b, hidden_dim)
#         transformer_output = act(transformer_output)
#
#
#         # 4. 分类分支
#         class_logits = self.class_embed(transformer_output)  # (num_queries, b, num_classes)
#         class_logits = act(class_logits)
#         class_logits = class_logits.permute(1, 0, 2)  # (b, num_queries, num_classes)
#
#         # 5. 掩码分支
#         mask_features = self.mask_embed(transformer_output)  # (num_queries, b, hidden_dim)
#         mask_features = act(mask_features)
#         mask_features = mask_features.permute(1, 0, 2).contiguous()  # (b, num_queries, hidden_dim)
#
#         # 将 features_proj 转置以匹配维度
#         features_proj = features_proj.permute(0, 2, 3, 1).contiguous()  # (b, h, w, hidden_dim)
#
#         # 点乘生成掩码
#         masks = torch.einsum("bhwc,bqk->bqhw", features_proj, mask_features)  # (b, num_queries, h, w)
#         masks = torch.sigmoid(masks)  # 概率化
#
#         # 6. 合并类别和掩码
#         class_probs = F.softmax(class_logits, dim=-1)  # (b, num_queries, num_classes)
#         final_masks = torch.einsum("bqhw,bqc->bchw", masks, class_probs)  # (b, num_classes, h, w)
#
#
#
#
#         return act(final_masks)


# # 测试代码
# if __name__ == "__main__":
#     # 参数设置
#     batch_size = 2
#     feature_dim = 256  # 输入特征的通道数
#     hidden_dim = 256  # Transformer 的隐藏维度
#     num_classes = 5  # 分割类别数（例如 VOC 数据集）
#     num_queries = 100  # 查询数量
#     height, width = 512, 512  # 输入特征图的高度和宽度
#
#     # 输入特征
#     features = torch.randn(batch_size, feature_dim, height, width)
#
#     # 初始化模型
#     model = SegmentationTransformer(num_classes, num_queries, hidden_dim, feature_dim)
#
#     # 前向传播
#     outputs = model(features)  # 输出 (b, num_classes, h, w)
#     print("输出形状:", outputs.shape)  # 应输出 (2, 21, 32, 32)
