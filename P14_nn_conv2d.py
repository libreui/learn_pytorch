import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 设置DataSet
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

# 设置DataLoader
dataloader = DataLoader(dataset, batch_size=64)


# 设置模型类
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
print(tudui)

# 设置日志
writer = SummaryWriter("./logs/conv2d")
step = 1
for data in dataloader:
    imgs, target = data
    output = tudui(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # torch.Size([3, 32, 32])
    # torch.Size([6, 30, 30])
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step += 1
