import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadInvertedAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8):
        super(MultiHeadInvertedAttention, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Conv2d(4, out_dim, kernel_size=1)  # 处理未解耦特征
        self.value = nn.Conv2d(4, out_dim, kernel_size=1)
        self.scale_factor = torch.sqrt(torch.FloatTensor([out_dim // num_heads]))

    def forward(self, F_undecoupled, F_similar):
        # 将相似特征映射为查询矩阵 [N, 1024] -> [N, num_heads, out_dim // num_heads] [N, 8, 32]
        Q = self.query(F_similar).view(F_similar.shape[0], self.num_heads, -1).permute(0, 2, 1) 

        # 将未解耦特征映射为键和值矩阵
        K = self.key(F_undecoupled).view(F_undecoupled.shape[0], self.num_heads, -1).permute(0, 2, 1)
        V = self.value(F_undecoupled).view(F_undecoupled.shape[0], self.num_heads, -1).permute(0, 2, 1)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 反转权重以突出非相似性特征
        inverted_attention_weights = 1 - attention_weights

        # 应用反转后的注意力权重
        F_output = torch.matmul(inverted_attention_weights, V).permute(0, 2, 1).contiguous()
        F_output = F_output.view(F_undecoupled.shape[0], -1, F_undecoupled.shape[2], F_undecoupled.shape[3])

        return F_output, inverted_attention_weights

