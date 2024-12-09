# data_loader.py
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
class SurgicalToolDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

        # # 遍历每个视频文件夹，读取frames_segmentation.csv
        # for video_folder in os.listdir(root_folder):
        #     # print("video_folder: ", root_folder)
        #     video_path = os.path.join(root_folder, video_folder)
        #     # print(video_path)
        #     if os.path.isdir(video_path):
        #         csv_file = os.path.join(video_path, 'frames_segmentation_10HZ.csv')
        #         frames_folder = os.path.join(video_path, 'frames_10HZ')
        #         masks_folder = os.path.join(video_path, 'segmentation')

        #         if os.path.exists(csv_file):
        #             df = pd.read_csv(csv_file)
        #             # 只保留每10帧一个掩膜的记录
        #             for _, row in df.iterrows():  # 每10帧一组
        #                 frame_paths = os.path.join(frames_folder, row['frame_file'])
        #                 mask_path = os.path.join(masks_folder, row['mask_file'])  # 每10帧对应一个掩膜
                        
        #                 # 确保帧和掩膜文件存在
        #                 if os.path.exists(frame_paths) and os.path.exists(mask_path):
        #                     self.pairs.append((frame_paths, mask_path))
        #                 else:
        #                     print(f"Warning: Frame or mask not found - {frame_paths}, {mask_path}")
 
    
    # if len(self.pairs) == 0:
    #     print("Warning: No valid frame-mask pairs found in the dataset.")
    # else:
    #     print(f"Total valid frame-mask pairs found: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame_paths, mask_path = self.pairs[idx]

        frame = cv2.imread(frame_paths)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 数据增强
        if self.transform:
            # 将帧和掩膜从torch.Tensor转换为numpy数组
            # frames = [frame.transpose(2, 0, 1) for frame in frames]  # Convert HWC to CHW
            # frames = [frame.astype('float32') / 255.0 for frame in frames]  # Normalize to [0, 1]
            # mask = mask.astype('float32') / 255.0

            augmented = self.transform(image=frame, mask=mask)
            frame = augmented['image']
            mask = augmented['mask']

        frame = frame.float() / 255.0
        mask  = mask.long()
        return frame, mask

# load for train 
root_folder_train = "/home/disk1/lyf/seg/data/train/"  # 数据集根目录
root_folder_test = "/home/disk1/lyf/seg/data/test/"  # 数据集根目录

# transform = A.Compose([
#     A.Resize(256, 256),  # 确保帧和掩膜的尺寸一致
#     ToTensorV2()         # 转换为张量
# ])

# dataset_train = SurgicalToolDataset(root_folder=root_folder_train, transform=transform)
# dataset_test = SurgicalToolDataset(root_folder=root_folder_test, transform=transform)
    
# def visualize_data_sample(dataset, num_samples=5):
#     for i in range(num_samples):
#         frame, mask = dataset[i]
#         frame = frame.permute(1, 2, 0).numpy()  # 将Tensor转换为numpy格式用于显示
#         mask = mask.numpy()

#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.title("Frame")
#         plt.imshow(frame)

#         plt.subplot(1, 2, 2)
#         plt.title("Mask")
#         plt.imshow(mask, cmap="gray")

#         plt.show()



# 可视化样本
# visualize_data_sample(dataset, num_samples=5)