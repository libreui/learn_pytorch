import torch.optim
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
# 创建损失函数
loss = nn.CrossEntropyLoss()
# 创建优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

# 进行20轮训练，在真实的训练学习中，都是几万几十万次训练
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        # 首先要梯度清零
        optim.zero_grad()

        img, target = data
        output = tudui(img)

        # 执行损失函数进行计算
        result_loss = loss(output, target)
        # 在损失函数计算除了结果以后，使用backward来进行反向传播
        # 在这里如果不执行backward，神经网络模型的grad(梯度)是不会自行计算的
        result_loss.backward()
        # 执行优化器
        optim.step()
        running_loss += result_loss
    print(running_loss)

