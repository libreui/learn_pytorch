import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度是：{train_data_size}")
print(f"测试数据集的长度是：{test_data_size}")

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 创建神经网络模型
tudui = Tudui()

# 创建损失函数，注意： 分类问题可以用“交叉熵”损失算法
loss_fn = nn.CrossEntropyLoss()

# 创建优化器: 使用 SGD(随机梯度下降) 学习速率(lr): 0.01
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# tensorboard记录日志
writer = SummaryWriter('./logs/train')

# 遍历数据并训练
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10
for i in range(epoch):
    print(f"------第 {i+1} 轮训练开始------")

    # 训练步骤开始
    tudui.train()
    for image, target in train_dataloader:
        # 梯度清零
        optimizer.zero_grad()
        # 训练数据
        output = tudui(image)
        # 计算损失
        loss_result = loss_fn(output, target)
        # 梯度反向传播
        loss_result.backward()
        # 优化器迭代
        optimizer.step()

        total_train_step += 1
        # 为了日志便于查看不用每次都输出
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {loss_result.item()}")
            writer.add_scalar("train_loss", loss_result.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    # 在测试集上进行测试，累计每轮的总损失
    total_test_loss = 0
    # 记录整体识别正确的个数
    total_accuracy = 0
    with torch.no_grad():
        for img, target in test_dataloader:
            output = tudui(img)
            loss = loss_fn(output, target)
            total_test_loss += loss.item()

            # 一部分数据的正确个数
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy

        print(f"整体测试集上的Loss: {total_test_loss}")
        print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, f"./models/tudui_model_{i}.pth")
    # 官方推荐的方式，只保存状态数据
    # torch.save(tudui.state_dict(), f"./models/tudui_model_{i}.pth")
    print(f"tudui_model_{i}.pth 模型已保存")

writer.close()



