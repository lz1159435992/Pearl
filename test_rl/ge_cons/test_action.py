import torch

# 假设我们有一个二维张量
two_dim_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

# 定义一个函数来递归地生成更高维度的张量
def generate_higher_dim_tensor(tensor, target_dim):
    if tensor.dim() == target_dim:
        return tensor
    else:
        # 将行作为整体进行排列组合
        new_tensor = torch.cartesian_prod(*tensor.t())
        # 递归调用直到达到目标维度
        return generate_higher_dim_tensor(new_tensor, target_dim)

# 生成一个10维张量
ten_dim_tensor = generate_higher_dim_tensor(two_dim_tensor, 10)
print(ten_dim_tensor)
print(f"Generated tensor shape: {ten_dim_tensor.shape}")