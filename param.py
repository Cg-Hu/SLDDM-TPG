import torch
import os

# 权重文件路径
weight_path = "/nfs5/hcg/repository/params/stable-diffusion/diffuers_version/unet/diffusion_pytorch_model.bin"

# ...existing code...
def calculate_param_size(weight_path):
    # 加载权重文件
    weights = torch.load(weight_path, map_location="cpu")
    
    # 统计所有参数的总大小（以字节为单位）
    total_size = sum(param.numel() * param.element_size() for param in weights.values())
    
    # 统计参数的数量（参数量）
    total_params = sum(param.numel() for param in weights.values())
    
    # 转换为 MB 和 M (百万)
    total_size_mb = total_size / (1024 * 1024)
    total_params_m = total_params / 1_000_000
    
    print(f"权重文件 {os.path.basename(weight_path)} 的存储大小为：{total_size_mb:.2f} MB")
    print(f"模型的总参数数量为：{total_params_m:.2f} M (百万个参数)")

# 执行统计操作
calculate_param_size(weight_path)