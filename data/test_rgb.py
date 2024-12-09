from PIL import Image
import numpy as np

mask = Image.open('/home/disk1/lyf/seg/data/test/video_41/segmentation/000013140.png')
mask_array = np.array(mask)
print(mask_array.shape)

print("mask mode: ", mask.mode)

unique_values = np.unique(mask_array)
print(unique_values)