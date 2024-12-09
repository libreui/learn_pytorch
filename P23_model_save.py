import torch
import torchvision
import torch.nn as nn

vgg16 = torchvision.models.vgg16()

# 保存方式1 模型结构+模型参数
torch.save(vgg16, "./models/vgg16_method1.pth")

# 保存方式2 模型参数 (官方推荐)
torch.save(vgg16.state_dict(), "./models/vgg16_method2.pth")


# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        return self.conv1(x)


tudui = Tudui()
torch.save(tudui, "./models/tudui_method1.pth")
