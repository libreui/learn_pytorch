import torchvision.datasets
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, Module
from torch.utils.data import DataLoader
from torch import nn


# 损失函数相关学习

class Tudui(Module):
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


dataset = torchvision.datasets.CIFAR10('./dataset', False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)
tudui = Tudui()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    img, target = data
    output = tudui(img)
    result_loss = loss(output, target)
    # 在损失函数计算除了结果以后，使用backward来进行反向传播
    # 在这里如果执行backward，神经网络模型的grad(梯度)是不会自行计算的
    result_loss.backward()
    print(result_loss)
