import torch
import torch.nn as nn
import torch.nn.init as init


class EnhancedAffineTransformModule(nn.Module):
    def __init__(self, feature_dim):
        super(EnhancedAffineTransformModule, self).__init__()
        self.A = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.b = nn.Parameter(torch.randn(feature_dim))
        self.scale = nn.Parameter(torch.ones(1) * 1.2)
        self.shift = nn.Parameter(torch.zeros(1) + 0.1)
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        y = torch.matmul(x, self.A) + self.b
        y = self.scale * y + self.shift
        y = self.prob_weight * y + (1 - self.prob_weight) * x
        return y

class ClarityAdjustmentModule(nn.Module):
    def __init__(self, feature_dim):
        super(ClarityAdjustmentModule, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose1d(1, 1, kernel_size=3, padding=1)
        self.sharpen_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5) 

    def forward(self, x):
        x_original = x
        x = self.fc(x).unsqueeze(1)  
        x = self.conv(x)  
        x = self.deconv(x)  
        x = self.sharpen_conv(x)  
        x = x.squeeze(1)  
        x = self.prob_weight * x + (1 - self.prob_weight) * x_original
        return x

class WrinkleAdjustmentModule(nn.Module):
    def __init__(self, feature_dim):
        super(WrinkleAdjustmentModule, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5) 

    def forward(self, x):
        x_original = x
        x = self.fc(x)
        x = self.activation(x)
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0) 
        x = self.prob_weight * x + (1 - self.prob_weight) * x_original
        return x

class NoiseFilterModule(nn.Module):
    def __init__(self, feature_dim):
        super(NoiseFilterModule, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.noise_reduction_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.prob_weight = nn.Parameter(torch.ones(1) * 0.5)  # 初始概率为0.5

    def forward(self, x):
        x_original = x
        x = self.fc(x).unsqueeze(1)  
        x = self.sigmoid(x) * x  
        x = self.noise_reduction_conv(x)
        x = x.squeeze(1)
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
        nn.init.xavier_uniform_(self.affine_transform.A)
        nn.init.zeros_(self.affine_transform.b)
        nn.init.constant_(self.affine_transform.prob_weight, 0.5)
        nn.init.xavier_uniform_(self.clarity_adjustment.fc.weight)
        nn.init.zeros_(self.clarity_adjustment.fc.bias)
        nn.init.kaiming_normal_(self.clarity_adjustment.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.clarity_adjustment.conv.bias)
        nn.init.kaiming_normal_(self.clarity_adjustment.deconv.weight, nonlinearity='relu')
        nn.init.zeros_(self.clarity_adjustment.deconv.bias)
        nn.init.constant_(self.clarity_adjustment.sharpen_conv.weight, -1)
        nn.init.constant_(self.clarity_adjustment.sharpen_conv.weight[:, :, 1], 2)
        nn.init.constant_(self.clarity_adjustment.prob_weight, 0.5)
        nn.init.xavier_uniform_(self.wrinkle_adjustment.fc.weight)
        nn.init.zeros_(self.wrinkle_adjustment.fc.bias)
        for param in self.wrinkle_adjustment.attention.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        nn.init.constant_(self.wrinkle_adjustment.prob_weight, 0.5)
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
            nn.BatchNorm1d(self.recover_dim, affine=False), 
        )
        # first train use that
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        assert x1.shape[1] == self.feature_dim, f"Expected input feature dim {self.feature_dim}, got {x1.shape[1]}"
        assert x2.shape[1] == self.feature_dim, f"Expected input feature dim {self.feature_dim}, got {x2.shape[1]}"
        x1 = self.recover(x1)
        x2 = self.recover(x2)
        x1 = x1.view(-1, -1, 64, x1.shape[1] // 64)
        x2 = x2.view(-1, -1, 64, x2.shape[1] // 64)
        return x1, x2


    
if __name__ == '__main__':
    feature_dim = 1024
    x = torch.randn(2, feature_dim)
    comprehensive_module = ComprehensiveAffineTransformModule(feature_dim)
    y = comprehensive_module(x)
    print(y.shape)
