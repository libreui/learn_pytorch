import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

        self. model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


tudui = Tudui()


# 检查网络模型结构是否正确
input = torch.ones([64, 3, 32, 32])
output = tudui(input)
print(output.shape)

writer = SummaryWriter('./logs/sequential')
writer.add_graph(tudui, input)
writer.close()
