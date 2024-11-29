import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, [-1, 1, 2, 2])

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        return self.sigmoid1(x)


tudui = Tudui()

writer = SummaryWriter('./logs/sigmoid')
step = 0
for data in dataloader:
    img, target = data
    output = tudui(img)
    writer.add_images("input", img, step)
    writer.add_images("Sigmoid", output, step)
    step += 1

writer.close()
