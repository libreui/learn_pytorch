import torch
from torch.nn import L1Loss
from torch import nn

# L1Loss函数的使用部分
input = torch.tensor([1, 2, 3], dtype=torch.float)
output = torch.tensor([1, 2, 5], dtype=torch.float)

input = torch.reshape(input, [1, 1, 1, 3])
output = torch.reshape(output, [1, 1, 1, 3])

loss = L1Loss(reduction='sum')
result = loss(input, output)

loss_mse = nn.MSELoss()
result_mse = loss_mse(input, output)

print(result)
print(result_mse)

# 对分类问题 写损失函数
# 模拟分类结果
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
# 变换形状符合损失函数
x = torch.reshape(x, (1, 3))
# 创建损失函数
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
