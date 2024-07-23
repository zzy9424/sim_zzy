import torch
from torch.distributions import Normal

mean = 0.0
std = 1.0

# 设置随机数生成器的种子
torch.manual_seed(42)

seed = torch.initial_seed()
print(seed)

# 创建 Normal 分布对象
dist = Normal(mean, std)

# 采样
sample1 = dist.sample()
sample2 = dist.sample()

print(sample1)  # 输出第一次采样结果
print(sample2)  # 输出第二次采样结果

seed = torch.initial_seed()
print(seed)