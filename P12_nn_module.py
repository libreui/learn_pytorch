import torch
import torch.nn as nn


class TuDui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x + 1
        return output


# 1. 定义一个神经网络
tudui = TuDui()
# 2. 定义一个输入
input = torch.tensor(1.0)
# 3. 传入神经网络
output = tudui(input)
print(output)
