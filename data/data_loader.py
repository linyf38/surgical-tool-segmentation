import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


class SurgicalToolDataset11(Dataset):
    def __init__(self, root_folder, transform):
        self.pairs = []
        self.transform = transform

        # 遍历每个视频文件夹，读取frames_segmentation.csv
        for video_folder in os.listdir(root_folder):
            # print(video_folder)
            video_path = os.path.join(root_folder, video_folder)
            # print(video_path)
            if os.path.isdir(video_path):
                # print("Yes")
                csv_file = os.path.join(video_path, 'frames_segmentation.csv')
                # print(csv_file)
                frames_folder = os.path.join(video_path, 'frames')
                masks_folder = os.path.join(video_path, 'segmentation')

                # 读取CSV文件并保存帧和掩膜的路径
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    # print(df[:5])
                    for _, row in df.iterrows():
                        frame_path = os.path.join(frames_folder, row['frame_file'])
                        mask_path = os.path.join(masks_folder, row['mask_file'])

                        # 检查帧和掩膜文件是否存在
                        if os.path.exists(frame_path) and os.path.exists(mask_path):
                            self.pairs.append((frame_path, mask_path))
                        else:
                            print(f"Warning: Frame or mask not found - {frame_path}, {mask_path}")

        # 打印数据集的大小以帮助调试
        if len(self.pairs) == 0:
            print("Warning: No valid frame-mask pairs found in the dataset.")
        else:
            print(f"Total valid frame-mask pairs found: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame_path, mask_path = self.pairs[idx]

        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=frame, mask=mask)
            frame = augmented['image']
            mask = augmented['mask']
        # frame = torch.from_numpy(frame).permute(2,0,1) # (C,H,W)
        # mask = torch.from_numpy(mask) # (H,W)

        # transform unit8 to float32 and long
        frame = frame.float() / 255.0
        mask  = mask.long()
        return frame, mask

# transform = A.Compose([
#     A.Resize(256, 256),  # 调整大小
#     # A.HorizontalFlip(p=0.5),  # 随机水平翻转
#     # A.RandomBrightnessContrast(p=0.2),  # 随机亮度对比度调整
#     # ToTensorV2()  # 转换为Tensor
# ])

# root_folder = '/home/disk1/lyf/seg/data/test'
# dataset = SurgicalToolDataset(root_folder=root_folder, transform=transform)
# data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# data type after loading
# frames, masks = next(iter(data_loader))
# print(f"Frames shape: {frames.shape}")  
# print(f"Masks shape: {masks.shape}")    
# print(f"Frames dtype: {frames.dtype}")  
# print(f"Masks dtype: {masks.dtype}")  


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

# # 使用示例：可视化数据集中的样本
# visualize_data_sample(dataset)

# test the last segmentation
# def visualize_last_50_frames(root_folder):
#     for video_folder in os.listdir(root_folder):
#         video_path = os.path.join(root_folder, video_folder)
#         if os.path.isdir(video_path):
#             csv_file = os.path.join(video_path, 'frames_segmentation.csv')
#             frames_folder = os.path.join(video_path, 'frames')
#             masks_folder = os.path.join(video_path, 'segmentation')

#             if os.path.exists(csv_file):
#                 df = pd.read_csv(csv_file)
#                 last_50_pairs = df.tail(10)
#                 print(f"Video folder: {video_folder}")
                
#                 # 可视化最后50个frame及其对应的mask
#                 for _, row in last_50_pairs.iterrows():
#                     frame_path = os.path.join(frames_folder, row['frame_file'])
#                     mask_path = os.path.join(masks_folder, row['mask_file'])

#                     if os.path.exists(frame_path) and os.path.exists(mask_path):
#                         frame = cv2.imread(frame_path)
#                         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#                         plt.figure(figsize=(10, 5))
#                         plt.subplot(1, 2, 1)
#                         plt.title("Frame")
#                         plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                         plt.subplot(1, 2, 2)
#                         plt.title("Mask")
#                         plt.imshow(mask, cmap="gray")

#                         plt.show()

# # 使用示例：可视化每个视频文件夹中的倒数50个frame的对应情况
# visualize_last_50_frames(root_folder)

