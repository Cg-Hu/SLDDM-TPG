import torch
import torch.nn as nn
import torch.nn.init as init

'''
TODO 实验
在64x64维度上, 768维度上进行仿射
'''

# Is SATs
# 处理亮度和对比度
# 亮度和对比度调整
class EnhancedAffineTransformModule(nn.Module):
    def __init__(self, feature_dim):
        super(EnhancedAffineTransformModule, self).__init__()
        # 初始化仿射变换的参数 A 和 b
        self.A = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.b = nn.Parameter(torch.randn(feature_dim))
        # 亮度和对比度的可学习参数
        self.scale = nn.Parameter(torch.ones(1) * 1.2)  # 初始比例稍微增强为1.2
        self.shift = nn.Parameter(torch.zeros(1) + 0.1)  # 初始偏移稍微增强为0.1
        # 添加可学习的概率权重
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5)  # 初始概率为0.5

    def forward(self, x):
        # 仿射变换：y = A * x + b
        y = torch.matmul(x, self.A) + self.b

        # 调整亮度和对比度：scale * y + shift
        y = self.scale * y + self.shift
        # 根据可学习的概率权重进行调整
        y = self.prob_weight * y + (1 - self.prob_weight) * x
        return y

# 模块改进：处理清晰度
# 高斯滤波和反卷积
class ClarityAdjustmentModule(nn.Module):
    def __init__(self, feature_dim):
        super(ClarityAdjustmentModule, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        # 使用简单的卷积和反卷积操作来调整清晰度
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose1d(1, 1, kernel_size=3, padding=1)
        # 额外增加一个去模糊的卷积层来增强清晰度
        self.sharpen_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # 添加可学习的概率权重
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5)  # 初始概率为0.5

    def forward(self, x):
        x_original = x
        x = self.fc(x).unsqueeze(1)  # 添加通道维度 [batch_size, 1, feature_dim]
        x = self.conv(x)  # 卷积操作
        x = self.deconv(x)  # 反卷积操作
        x = self.sharpen_conv(x)  # 增加清晰度
        x = x.squeeze(1)  # 去掉通道维度
        # 根据可学习的概率权重进行调整
        x = self.prob_weight * x + (1 - self.prob_weight) * x_original
        return x

# 模块改进：处理褶皱和扭曲
# 非线性激活：加入非线性激活层（如 ReLU、Tanh）来模拟衣物的褶皱和扭曲效果。
# 自注意力机制
class WrinkleAdjustmentModule(nn.Module):
    def __init__(self, feature_dim):
        super(WrinkleAdjustmentModule, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
        # 添加可学习的概率权重
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5)  # 初始概率为0.5

    def forward(self, x):
        x_original = x
        x = self.fc(x)
        x = self.activation(x)
        # 注意力机制
        x = x.unsqueeze(0)  # 添加时间维度 [1, batch_size, feature_dim]
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)  # 去掉时间维度
        # 根据可学习的概率权重进行调整
        x = self.prob_weight * x + (1 - self.prob_weight) * x_original
        return x
    


# 模块改进：处理背景噪声
# 噪声过滤：噪声过滤模块，通过卷积层来对特征进行过滤，去除背景噪声。
# 门控机制：引入门控机制（如 Sigmoid 激活）来对噪声进行选择性过滤。
class NoiseFilterModule(nn.Module):
    def __init__(self, feature_dim):
        super(NoiseFilterModule, self).__init__()
        # 使用1x1卷积对背景噪声进行去除
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
        # 增强去噪能力的卷积操作
        self.noise_reduction_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # 添加可学习的概率权重
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5)  # 初始概率为0.5

    def forward(self, x):
        x_original = x
        x = self.fc(x).unsqueeze(1)  # 添加通道维度 [batch_size, 1, feature_dim]
        x = self.sigmoid(x) * x  # 使用Sigmoid门控来过滤掉背景噪声
        x = self.noise_reduction_conv(x)  # 增强去噪
        x = x.squeeze(1)  # 去掉通道维度
        # 根据可学习的概率权重进行调整
        x = self.prob_weight * x + (1 - self.prob_weight) * x_original
        return x

class ComprehensiveAffineTransformModule(nn.Module):
    def __init__(self, feature_dim=1024, first_train=False):
        super(ComprehensiveAffineTransformModule, self).__init__()
        self.affine_transform = EnhancedAffineTransformModule(feature_dim)
        self.clarity_adjustment = ClarityAdjustmentModule(feature_dim)
        self.wrinkle_adjustment = WrinkleAdjustmentModule(feature_dim)
        self.noise_filter = NoiseFilterModule(feature_dim)
        if first_train:
            self._initialize_weights()
    
    def _initialize_weights(self):
        # 初始化 EnhancedAffineTransformModule 的权重
        nn.init.xavier_uniform_(self.affine_transform.A)
        nn.init.zeros_(self.affine_transform.b)
        nn.init.constant_(self.affine_transform.prob_weight, 0.5)

        # 初始化 ClarityAdjustmentModule 的权重
        nn.init.xavier_uniform_(self.clarity_adjustment.fc.weight)
        nn.init.zeros_(self.clarity_adjustment.fc.bias)
        nn.init.kaiming_normal_(self.clarity_adjustment.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.clarity_adjustment.conv.bias)
        nn.init.kaiming_normal_(self.clarity_adjustment.deconv.weight, nonlinearity='relu')
        nn.init.zeros_(self.clarity_adjustment.deconv.bias)
        nn.init.constant_(self.clarity_adjustment.sharpen_conv.weight, -1)
        nn.init.constant_(self.clarity_adjustment.sharpen_conv.weight[:, :, 1], 2)
        nn.init.constant_(self.clarity_adjustment.prob_weight, 0.5)

        # 初始化 WrinkleAdjustmentModule 的权重
        nn.init.xavier_uniform_(self.wrinkle_adjustment.fc.weight)
        nn.init.zeros_(self.wrinkle_adjustment.fc.bias)
        for param in self.wrinkle_adjustment.attention.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        nn.init.constant_(self.wrinkle_adjustment.prob_weight, 0.5)

        # 初始化 NoiseFilterModule 的权重
        nn.init.xavier_uniform_(self.noise_filter.fc.weight)
        nn.init.zeros_(self.noise_filter.fc.bias)
        nn.init.constant_(self.noise_filter.noise_reduction_conv.weight, -0.5)
        nn.init.constant_(self.noise_filter.noise_reduction_conv.weight[:, :, 1], 1.5)
        nn.init.constant_(self.noise_filter.prob_weight, 0.5)

    def forward(self, x):
        x = self.affine_transform(x)
        x = self.clarity_adjustment(x)
        x = self.wrinkle_adjustment(x)
        x = self.noise_filter(x)
        return x


class DimensionReover(nn.Module):
    def __init__(self, feature_dim=2048, recover_dim=4096) -> None:
        super().__init__()
        if recover_dim % 64 != 0:
            raise ValueError("recover_dim must be a multiple of 64")
        self.feature_dim = feature_dim
        self.recover_dim = recover_dim

        self.recover_diff = nn.Sequential(
            nn.Linear(self.feature_dim, self.recover_dim, bias=False),
            nn.BatchNorm1d(self.recover_dim, affine=False),  # affine=True 允许 BatchNorm 学习缩放和偏移参数
        )
        # first train use that
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 初始化（Glorot 初始化）
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                # 将 BatchNorm 的 weight 初始化为 1，bias 初始化为 0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # 检查输入维度
        assert x1.shape[1] == self.feature_dim, f"Expected input feature dim {self.feature_dim}, got {x1.shape[1]}"
        assert x2.shape[1] == self.feature_dim, f"Expected input feature dim {self.feature_dim}, got {x2.shape[1]}"
        
        # 通过 recover 层进行线性变换和标准化
        x1 = self.recover(x1)
        x2 = self.recover(x2)

        # 打印中间输出形状
        # print(f"x1 shape after recover: {x1.shape}")

        # 重塑张量的形状为 (-1, 64, recover_dim // 64)
        x1 = x1.view(-1, -1, 64, x1.shape[1] // 64)
        x2 = x2.view(-1, -1, 64, x2.shape[1] // 64)

        return x1, x2


    
if __name__ == '__main__':
    feature_dim = 1024
    x = torch.randn(2, feature_dim)
    comprehensive_module = ComprehensiveAffineTransformModule(feature_dim)
    y = comprehensive_module(x)
    print(y.shape)
