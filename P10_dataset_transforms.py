import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)

# print(train_set[0])
# img, target = train_set[0]
# print(train_set.classes)
# print(img)
# print(target)

# print(test_set[0])

writer = SummaryWriter('./logs/p10')
for i in range(10):
    img_tensor, target = test_set[i]
    writer.add_image('test_set', img_tensor, i)
writer.close()
