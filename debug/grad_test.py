import torch
import torch.nn as nn
import torch.optim as optim

# 假设 CrossAttention 和 FeedForward 已定义
class CrossAttention(nn.Module):
    def __init__(self, query_dim, heads, dim_head, dropout=0.):
        super().__init__()
        # ...

    def forward(self, x, context=None):
        # ...
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., glu=False):
        super().__init__()
        # ...

    def forward(self, x):
        # ...
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        # 新加入的并行注意力权重
        self.attn_cpsd = CrossAttention(query_dim=dim, context_dim=context_dim,
                                        heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # 新加入的norm层
        self.norm_cpsd = nn.LayerNorm(dim)

        self.checkpoint = checkpoint

    def forward(self, x, context=None, context_cpsd=None):
        return self._forward(x, context, context_cpsd)

    def _forward(self, x, context=None, context_cpsd=None):
        # 确保输入张量的 requires_grad 状态与期望一致
        x = self.norm1(x)
        x = self.attn1(x) + x

        x = self.norm2(x)
        x = self.attn2(x, context=context) + x

        # 新加入并行的交叉注意力权重
        x = self.norm_cpsd(x)
        x = self.attn_cpsd(x, context=context_cpsd) + x

        x = self.norm3(x)
        x = self.ff(x) + x
        return x

# 初始化模型
dim = 512
n_heads = 8
d_head = 64
model = BasicTransformerBlock(dim=dim, n_heads=n_heads, d_head=d_head)

# 冻结不包含 "cpsd" 的层的参数
for name, param in model.named_parameters():
    if 'cpsd' not in name:
        param.requires_grad = False

# 打印所有参数的 requires_grad 状态，用于验证
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# 定义优化器，只优化包含 "cpsd" 的层
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# 测试代码，输入随机数据
x = torch.randn(2, 16, dim, requires_grad=True)  # 假设输入维度为 [batch_size, seq_length, dim]
context = torch.randn(2, 16, dim, requires_grad=True)
context_cpsd = torch.randn(2, 16, dim, requires_grad=True)

# 前向传播
output = model(x, context=context, context_cpsd=context_cpsd)
print("Output shape:", output.shape)

# 定义损失函数
criterion = nn.MSELoss()
target = torch.randn_like(output)

# 反向传播
loss = criterion(output, target)
loss.backward()

# 更新参数
optimizer.step()
