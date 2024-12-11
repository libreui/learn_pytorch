import torch
from PIL import Image
import torchvision
from torch import nn

# image_path = "./images/dog.png"
image_path = "./images/airplane.png"

image = Image.open(image_path)

# 转换大小，适合模型的输入
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# 加载网络模型
model_dict = torch.load("./models/tudui_model_29.pth", weights_only=True)
model = Tudui()
model.load_state_dict(model_dict)
image = torch.reshape(image, [1, 3, 32, 32])
print(image.shape)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

classics = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

print(output.argmax(1))
