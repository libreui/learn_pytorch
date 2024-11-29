import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        return self.linear1(x)


tudui = Tudui()

for data in dataloader:
    img, target = data
    print(img.shape)
    # 摊平操作
    output = torch.flatten(img)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
