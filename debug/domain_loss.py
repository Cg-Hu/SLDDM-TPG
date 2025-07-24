import torch
import torch.nn.functional as F

# 假设你的类是这样定义的
class MyModel:
    def __init__(self):
        pass

    def compute_in_dom_loss(self, c_hid, p_hid):
        c_hid = c_hid.squeeze(1)  # 变为 [N, 4]
        p_hid = p_hid.squeeze(1)  # 变为 [N, 4]
        # c_hid /= c_hid.clone().norm(dim=-1, keepdim=True)
        # p_hid /= p_hid.clone().norm(dim=-1, keepdim=True)

        result = []
        for i in range(c_hid.shape[0]):
            # 获取当前行的元素
            current_row = c_hid[i]
            # 计算当前行减去后面所有行的差值
            for j in range(i+1, c_hid.shape[0]):
                # 当前行减去下面的行，得到差值
                diff = current_row - c_hid[j]
                result.append(diff)

        # 将所有结果合并为一个tensor
        c_diff = torch.stack(result)

        result = []
        for i in range(p_hid.shape[0]):
            # 获取当前行的元素
            current_row = p_hid[i]
            # 计算当前行减去后面所有行的差值
            for j in range(i+1, p_hid.shape[0]):
                # 当前行减去下面的行，得到差值
                diff = current_row - p_hid[j]
                result.append(diff)

        # 将所有结果合并为一个tensor
        p_diff = torch.stack(result)

        in_dom_loss = (1 - F.cosine_similarity(c_diff, p_diff, dim=1)).mean()
        return in_dom_loss            

    def compute_cross_dom_loss(self, c_hid, p_hid):
        cross_dom =  p_hid - c_hid
        # cross_dom /= cross_dom.clone().norm(dim=-1, keepdim=True)
        
        # 1. 重塑 domain_sub 的形状为 (N, 1024)
        cross_dom = cross_dom.squeeze(1)  # 变为 (N, 4)
        cosine_sim = F.cosine_similarity(cross_dom.unsqueeze(0), cross_dom.unsqueeze(1), dim=2)
        cross_sim = torch.triu(cosine_sim, diagonal=0)  # `diagonal=0` 保留对角线
        cross_sim = cross_sim.triu(diagonal=1)  # `diagonal=1` 将对角线以上部分（包括对角线）置为0
        cross_sim = cross_sim[cross_sim != 0]

        cross_dom_loss = (1 - cross_sim).mean()
        return cross_dom_loss

# 测试代码
def test_loss_functions():
    model = MyModel()
    
    # 创建一些示例数据
    N = 2  # 现在有 4 个样本


    c_hid = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], 
                        [[1.0, 3.0, 2.0, 4.0]],
                        [[1.0, 2.0, 4.0, 3.0]],])  # 随机生成 [2, 4] 的 c_hid

    p_hid = torch.tensor([[[3.0, 2.0, 1.0, 4.0]],
                        [[4.0, 3.0, 2.0, 1.0]],
                        [[3.0, 1.0, 4.0, 2.0]]])  # 随机生成 [2, 4] 的 p_hid
    
    # 调用 compute_in_dom_loss
    in_dom_loss = model.compute_in_dom_loss(c_hid, p_hid)
    print(f"In-domain loss: {in_dom_loss.item()}")

    # 调用 compute_cross_dom_loss
    cross_dom_loss = model.compute_cross_dom_loss(c_hid, p_hid)
    print(f"Cross-domain loss: {cross_dom_loss.item()}")

# 运行测试
test_loss_functions()
