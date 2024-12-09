import torch
import torchvision.models

# 加载方式1 对应 torch.save()
model = torch.load("./models/vgg16_method1.pth")
# print(model.py)

# 加载方式2 对应 torch.state_dict()
model_dict = torch.load("./models/vgg16_method2.pth")
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(model_dict)
# print(vgg16)


# 陷阱1 报错，因为没有模型对象的定义，需要引入进来
# 引入模型定义 "from P23_model_save import Tudui"
from P23_model_save import Tudui
model_tudui = torch.load("./models/tudui_method1.pth")
print(model_tudui)
