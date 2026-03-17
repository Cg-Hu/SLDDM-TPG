import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
from omegaconf import OmegaConf
import numpy as np
import torch.nn.init as init
from collections import OrderedDict
import os
from PIL import Image
import matplotlib.pyplot as plt
from ldn.affine_blocks import ComprehensiveAffineTransformModule
from sldm.modules.encoders.modules import FrozenCLIPImageEmbedder
from sldm.util import instantiate_from_config

HOME_DIR = os.path.expanduser("~")

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.view(t.size(0), -1, h, t.size(-1) // h).permute(2, 0, 1, 3).reshape(-1, t.size(1), t.size(-1) // h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = mask.view(mask.size(0), -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = mask.unsqueeze(1).repeat(1, h, 1).reshape(-1, mask.size(-1))
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = out.view(h, x.size(0), x.size(1), -1).permute(1, 2, 0, 3).reshape(x.size(0), x.size(1), -1) # 这边的内容和Unet一致对应
        return self.to_out(out)
    

class ResidualAttentionModule(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.5):
        super().__init__()
        inner_dim = dim_head * heads # 512
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, query_dim)
        )

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)    

    def forward(self, x, context=None, mask=None, last=False):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.view(t.size(0), -1, h, t.size(-1) // h).permute(2, 0, 1, 3).reshape(-1, t.size(1), t.size(-1) // h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = mask.view(mask.size(0), -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = mask.unsqueeze(1).repeat(1, h, 1).reshape(-1, mask.size(-1))
            sim.masked_fill_(~mask, max_neg_value)
        if last:
            attn = 1 - sim.softmax(dim=-1) 
        else:
            attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v) # out的shape is: torch.Size([16, 1, 64]) v的shape is: torch.Size([16, 4, 64])
        # idx = len(os.listdir("/home/hcg/cloth_pattern/sim_diff_diffusion_106/vae_sim_diff"))
        # save_attention_heatmap(out, heatmap_save_path=f'out_ap_{idx}.png')

        out = out.view(h, x.size(0), x.size(1), -1).permute(1, 2, 0, 3).reshape(x.size(0), x.size(1), -1)
        out = self.to_out(out)
        out = self.norm1(out + x)

        F_ffn_output = self.ffn(out)
        F_output = self.norm2(F_ffn_output + out)

        return F_output, attn


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

class ResNetWithCA(nn.Module):

    def __init__(self, layers=[3, 4, 6, 3], feature_dim=2048, first_train=False):
        super(ResNetWithCA, self).__init__()
        self.inplanes = 64
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=False)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(Bottleneck, 64, layers[0]) # torch.Size([2, 256, 16, 16])
        # q,k,v [2*8, 16*16, 32]
        sa1 = SpatialTransformer(in_channels=256, n_heads=8, d_head=40) # [2, 320, 16, 16] -> [2, 256, 16, 16]
        layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        sa2 = SpatialTransformer(in_channels=512, n_heads=8, d_head=80) # [2, 640, 8, 8]
        layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        sa3 = SpatialTransformer(in_channels=1024, n_heads=8, d_head=160) # [2, 1280, 4, 4]
        layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2) # [2, 2048, 1, 1]
        # conv2 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0)
        # bn2 = nn.BatchNorm2d(1024)
        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_extract = nn.Sequential(OrderedDict([
            ('conv1', conv1),
            ('bn1', bn1),
            ('relu1', relu),
            ('maxpool', maxpool),
            ('layer1', layer1),
            # ('sa1', sa1),
            ('layer2', layer2),
            # ('sa2', sa2),
            ('layer3', layer3),
            # ('sa3', sa3),
            ('layer4', layer4),
            # ('conv2', conv2),
            # ('n2', bn2),
            # ('relu2', relu),
            ('avgpool', avgpool),
        ]))
        
        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(feature_dim, feature_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(feature_dim)),
            ('relu1', relu),  # first layer
            ('linear2', nn.Linear(feature_dim, feature_dim, bias=False)),
            ('bn2', nn.BatchNorm1d(feature_dim)),
            ('relu2', relu),  # second layer
        ]))
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extract(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) + x
        return x

def get_latent_autoencoder_kl(config):
    config = OmegaConf.load(config)
    auntoencoder_kl = instantiate_from_config(config.autoencoder_kl)
    auntoencoder_kl = no_grad_set(auntoencoder_kl)
    return auntoencoder_kl

    
def no_grad_set(model):
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False
    return model



class SimBranch(nn.Module):
    def __init__(self, pred_dim=1024, feature_dim=2048, hid_dim=512, first_train=False) -> None:
        super(SimBranch, self).__init__()
    
        self.fc_sim = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(feature_dim, pred_dim, bias=False)),  # [2, 1024]
            ('bn', nn.BatchNorm1d(pred_dim, affine=False))
        ]))
        self.predictor = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(pred_dim, hid_dim, bias=False)),  # [2, 512]
            ('bn1', nn.BatchNorm1d(hid_dim)),
            ('relu', nn.ReLU(inplace=False)),  # hidden layer
            ('linear2', nn.Linear(hid_dim, pred_dim, bias=True))  # output layer [2, 1024]
        ]))
                                
    def forward(self, x):
        z = self.fc_sim(x)
        p = self.predictor(z)
        return z, p

class SimBranchFeature1024(nn.Module):
    def __init__(self, pred_dim=1024, feature_dim=2048) -> None:
        super(SimBranchFeature1024, self).__init__()
        
        self.fc_sim = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(feature_dim, pred_dim, bias=False)),  # [2, 1024]
            ('bn', nn.BatchNorm1d(pred_dim, affine=False))
        ]))
                   
    def forward(self, x):
        z = self.fc_sim(x)
        # p = self.predictor(z)
        return z

class DiffBranchAttn(nn.Module):
    def __init__(self, feature_dim=2048, recover_dim=4096, pred_dim=1024, heads=8, layers=4, fisrt_train=False) -> None:
        super(DiffBranchAttn, self).__init__()
        self.feature_dim = feature_dim
        self.recover_dim = recover_dim
        self.pred_dim = pred_dim
        self.W_x = nn.Parameter(torch.randn(recover_dim, pred_dim))  # 图像投影矩阵 [d_i, d_e]
        self.W_xvae = nn.Parameter(torch.randn(feature_dim, pred_dim))  # 文本投影矩阵 [d_t, d_e]
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionModule(
                    pred_dim,
                    pred_dim,
                    heads=heads,
                )
                for _ in range(layers)
            ]
        )
        self.final_ln = LayerNorm(1024)


    def forward(self, x: torch.Tensor, x_vae: torch.Tensor):
        x_vae = x_vae.view(-1, 4, self.recover_dim) # [N,4,4096]
        x = x.unsqueeze(1) # [N,1,2048]
        x_vae = F.normalize(torch.matmul(x_vae, self.W_x), p=2, dim=1)  # 图像嵌入 [N,4,1024]
        x = F.normalize(torch.matmul(x, self.W_xvae), p=2, dim=1)  # 图像嵌入 [N,1,1024]
        for idx, block in enumerate(self.resblocks):
            if idx == len(self.resblocks) - 1:
                x, _ = block(x, x_vae, last=True)
            else:
                x, _ = block(x, x_vae, last=False)
        # print(f"x 的shape is: {x.shape}")
        x = self.final_ln(x)
        x = x.squeeze(1)
        return x


class DiffBranchSubtract(nn.Module):
    def __init__(self, feature_dim=2048, recover_dim=4096, pred_dim=1024) -> None:
        super(DiffBranchSubtract, self).__init__()
        self.feature_dim = feature_dim
        self.recover_dim = recover_dim
        self.pred_dim = pred_dim

        self.recover2vae_diff = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(feature_dim, recover_dim, bias=False)),
            ('bn', nn.BatchNorm1d(recover_dim, affine=False))  # affine=True 允许 BatchNorm 学习缩放和偏移参数
        ]))
        self.down_channels = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.fc_diff = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(recover_dim, feature_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(feature_dim, affine=False)),  # affine=True 允许 BatchNorm 学习缩放和偏移参数
            ('relu1', nn.ReLU(inplace=False)),
            ('linear2', nn.Linear(feature_dim, feature_dim, bias=False)),
            ('bn2', nn.BatchNorm1d(feature_dim)),
            ('relu2', nn.ReLU(inplace=False)),  # second layer
            ('linear3', nn.Linear(feature_dim, pred_dim, bias=False))
        ]))

    def project_orthogonal(F_undecoupled, F_similar):
        norm_squared = torch.sum(F_similar ** 2, dim=1, keepdim=True)  # [N, 1]
        dot_product = torch.sum(F_undecoupled * F_similar.unsqueeze(-1), dim=1, keepdim=True)  # [N, 1, 64, 64]
        projection = (dot_product / norm_squared) * F_similar.unsqueeze(-1)  # [N, 2048, 64, 64]
        F_projected = F_undecoupled - projection  # [N, 4, 64, 64]

        return F_projected
    
    def forward(self, x, x_vae):
        x = self.recover2vae_diff(x) # Linear [2048->4096]
        x = x.view(-1, 1, 64, 64) # [N, 1, 64, 64]
        x_vae = self.down_channels(x_vae) # [N, 1, 64, 64]
        x_diff = self.project_orthogonal(x_vae, x)
        x_diff = torch.flatten(nn.ReLU(x_diff), 1)
        x = self.fc_diff(x) # [N, 1024] 
        return x

class ClothPatternSimDiffFeature(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], feature_dim=2048, pred_dim=1024):
        super(ClothPatternSimDiffFeature, self).__init__()
        self.feature_encoder = ResNetWithCA(layers=layers, feature_dim=feature_dim)
        self.sim = SimBranchFeature1024(pred_dim=pred_dim, feature_dim=feature_dim)
    
    def forward(self, x):
        return self.sim(self.feature_encoder(x))

class ClothPatternSimDiff(nn.Module):
    def __init__(self, config, layers=[3, 4, 6, 3], feature_dim=2048, pred_dim=1024, recover_dim=4096, feature_mode='res', first_train=False, scm_epochs=2):
        super(ClothPatternSimDiff, self).__init__()
        self.scm_epochs = scm_epochs
        self.auntoencoder_kl = get_latent_autoencoder_kl(config)
        assert feature_mode in ['res', 'vit'], print(f"feature_mode only support 'res' and 'vit'")
        if feature_mode == 'res':
            self.feature_encoder = ResNetWithCA(layers=layers, feature_dim=feature_dim, first_train=first_train)
        elif feature_mode == 'vit':
            self.feature_encoder = FrozenCLIPImageEmbedder()
            feature_dim = 1024
        self.sim = SimBranch(pred_dim=pred_dim, feature_dim=feature_dim, first_train=first_train)

        self.diff = DiffBranchAttn(feature_dim=feature_dim, recover_dim=recover_dim, pred_dim=pred_dim,fisrt_train=first_train)
        self.affine_block = ComprehensiveAffineTransformModule(pred_dim, first_train=first_train)
    

    def forward(self, x1, x2, xaug, epoch):
        p1, p2, p_aug, z1, z2, z_aug, x1_vae, x2_vae, xaug_vae, x1_hid, x2_hid, xaug_hid = self.sim_branch(x1, x2, xaug)
        if epoch > self.scm_epochs:
            x1_diff, x2_diff, xaug_diff = self.diff_branch(x1_vae, x2_vae, xaug_vae, x1_hid, x2_hid, xaug_hid)
            x2_affine = self.affine_block(x2_diff)
            xaug_affine = self.affine_block(xaug_diff)
            return p1, p2, p_aug, z1, z2, z_aug, x1_diff, x2_diff, xaug_diff, x2_affine, xaug_affine
        return p1, p2, p_aug, z1, z2, z_aug

            
    def sim_branch(self, x1, x2, xaug):
   
        combined_x = torch.cat((x1, x2, xaug), dim=0)
        output, sample_x = self.auntoencoder_kl(combined_x)
        x1_vae, x2_vae, xaug_vae = torch.chunk(sample_x, chunks=3, dim=0) 
        xaug_hid = self.feature_encoder(xaug_vae)
        z_aug, p_aug = self.sim(xaug_hid) # NxC
    
        x1_hid = self.feature_encoder(x1_vae)
        z1, p1 = self.sim(x1_hid) # NxC
        x2_hid = self.feature_encoder(x2_vae)
        z2, p2 = self.sim(x2_hid) # NxC
        return p1, p2, p_aug, z1.detach(), z2.detach(), z_aug.detach(), x1_vae, x2_vae, xaug_vae, x1_hid, x2_hid, xaug_hid,

    def diff_branch(self, x1_vae, x2_vae, xaug_vae, x1_hid, x2_hid, xaug_hid):
        x1_diff = self.diff(x1_hid, x1_vae)
        x2_diff = self.diff(x2_hid, x2_vae)
        xaug_diff = self.diff(xaug_hid, xaug_vae)
        return x1_diff, x2_diff, xaug_diff
          
class ClothPatternSimDiffFeatureNew(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], feature_dim=2048, pred_dim=1024, recover_dim=4096, first_train=False):
        super(ClothPatternSimDiffFeatureNew, self).__init__()
        # self.auntoencoder_kl = get_latent_autoencoder_kl("/home/hcg/cloth_pattern/sim_diff_diffusion_106/vae_sim_diff/first_autoencoder/encoder.yaml")
        self.feature_encoder = ResNetWithCA(layers=layers, feature_dim=feature_dim, first_train=first_train)
        # Sim Branch
        self.sim = SimBranch(pred_dim=pred_dim, feature_dim=feature_dim, first_train=first_train)
        # Diff Branch
        self.diff = DiffBranchAttn(feature_dim=feature_dim, recover_dim=recover_dim, pred_dim=pred_dim,fisrt_train=first_train)
        # Residual Branch
        self.affine_block = ComprehensiveAffineTransformModule(pred_dim, first_train=first_train)

    def forward(self, x):
        # _, x = self.auntoencoder_kl(x)
        x_simz, x_simp, x_hid = self.sim_branch(x)
        x_diff = self.diff_branch(x_hid, x)
        x_affine = self.affine_block(x_diff)
        return x_simz, x_simp, x_diff, x_affine, x_hid # [N, 1024], [N, 1024], [N, 1024], [N, 2048]

    def sim_branch(self, x):
        x_hid = self.feature_encoder(x)
        z, p = self.sim(x_hid) # NxC
        return z, p, x_hid

    def diff_branch(self, x_hid, x_vae):
        x_diff = self.diff(x_hid, x_vae)
        return x_diff
