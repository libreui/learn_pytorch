from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "dataset/hymenoptera_data/train/ants/28847243_e79fe052cd.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor = transforms.ToTensor()
tensor_img = tensor(img)
writer.add_image("Tensor Image", tensor_img)
writer.close()
