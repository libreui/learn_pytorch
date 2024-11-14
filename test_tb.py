from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# Create a SummaryWriter object
writer = SummaryWriter('logs')

# Add a Image to the SummaryWriter object
image_path = 'dataset/hymenoptera_data/train/bees/16838648_415acd9e3f.jpg'
img = Image.open(image_path)
img_array = np.array(img)
writer.add_image('image', img_array, 2, dataformats='HWC')


# y = x
for x in range(-100, 100):
    writer.add_scalar('y=x**2', x**2, x)

# Close the SummaryWriter object
writer.close()
