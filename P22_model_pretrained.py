import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet('./dataset', split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16 = torchvision.models.vgg16()

train_data = torchvision.datasets.CIFAR10('./dataset', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

# 在vgg16模型中添加一层网络
vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))

# 在vgg16模型中修改一层网络
# vgg16.classifier[6] = nn.Linear(4096, 10)

print(vgg16)
