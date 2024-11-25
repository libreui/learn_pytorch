import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 测试数据集
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0)

# 测试数据集中第一章图片及target
img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter('logs/dataloader')
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f'Epoch: {epoch}', imgs, step)
        step += 1
writer.close()
