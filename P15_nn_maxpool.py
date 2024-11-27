import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

input_matrix = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
], dtype=torch.float32)

input_matrix = torch.reshape(input_matrix, (-1, 1, 5, 5))
print(input_matrix.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.max_pool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        return self.max_pool(x)


tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("./logs/max-pool")
step = 0
for data in dataloader:
    img, target = data
    output = tudui(img)
    writer.add_images("input", img, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()
