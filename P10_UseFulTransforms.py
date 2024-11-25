from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img = Image.open('images/lxq.jpg')

# ToTensor
trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img:PIL -> resize -> img_resize:PIL
img_resize = trans_resize(img)
print(img_resize)
# img_resize:PIL -> ToTensor -> img_resize_tensor:Tensor
img_resize = trans_to_tensor(img_resize)
writer.add_image('Resize', img_resize, 0)

# Compose - Resize - 2
trans_resize_2 = transforms.Resize([512])
trans_compose = transforms.Compose([
    trans_resize_2,
    trans_to_tensor
])
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2, 1)

# RandomCrop
trans_random_crop = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random_crop, trans_to_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)

writer.close()

