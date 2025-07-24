import torch
import torch.nn as nn
from functools import partial
from transformers import CLIPVisionModel, CLIPModel
from xf import LayerNorm, Transformer

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c




class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to requirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text




class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.transformer.vision_model.embeddings.patch_embedding = nn.Conv2d(4, 1024, kernel_size=(4, 4), stride=(4, 4), bias=False)
        
        self.mapper = Transformer(
            1,
            1024,
            5,
            1,
        )
        self.final_ln = LayerNorm(1024)
        self.freeze()
        # self._initialize_weights() # 第一次训练，后面就是加载权重

    def freeze(self):
        for name, param in self.named_parameters():
            if "patch_embedding" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.transformer.vision_model.embeddings.patch_embedding.weight, mode='fan_out', nonlinearity='relu')
        # Initialize the weights of the Transformer mapper
        for name, param in self.mapper.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        # pooler_output：表示从 last_hidden_state 中进行池化后的输出，它可以看作是整个输入图像的全局表示，
        # 通常形状为 [batch_size, hidden_size]。[bs, 1024]
        z = outputs.pooler_output
        z = z.unsqueeze(1)
        z = self.mapper(z) # [N, 1024]
        z = self.final_ln(z)
        z = z.squeeze(1)
        return z

    def encode(self, image):
        return self(image)

class SimCLIPImageEmbedder(AbstractEncoder):
    pass

if __name__ == "__main__":
    # 初始化模型
    model = FrozenCLIPImageEmbedder()

    # 创建测试输入数据，假设输入是 [batch_size, channels, height, width]
    test_image = torch.randn(2, 4, 64, 64)

    # 前向传播测试
    with torch.no_grad():
        output = model.encode(test_image)
        print("Output shape:", output.shape)

    # # 断言输出形状是否符合预期
    # assert output.shape == (2, 1024), f"Unexpected output shape: {output.shape}"
    torch.save(model.mapper.state_dict(), "frozen_clip_image_embedder_mapper_weights.pth")