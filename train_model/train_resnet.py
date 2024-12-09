import os
import random
import sys
sys.path.append('/home/disk1/lyf/seg')
import segmentation_models_pytorch as smp
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.data_loader_10HZ import SurgicalToolDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler  
import kornia.augmentation as K
import kornia
import pandas as pd

def get_unet_model():
    model = smp.Unet(
        encoder_name="resnet34",        # 使用ResNet34作为编码器
        encoder_weights="imagenet",     # 使用在ImageNet上预训练的权重
        in_channels=3,                  # 输入通道数
        classes=10                       # 输出通道数，对应分割的类别数（背景和手术工具）
    )
    return model


# 随机拆分训练集和验证集
def split_train_val(root_folder, train_ratio=0.8, seed=42):
    
    random.seed(seed)
    all_pairs = []

    # 遍历每个 video_xx 文件夹
    for video_folder in os.listdir(root_folder):
        video_path = os.path.join(root_folder, video_folder)
        if os.path.isdir(video_path):
            frames_folder = os.path.join(video_path, 'frames_10HZ')
            masks_folder = os.path.join(video_path, 'segmentation')
            csv_file = os.path.join(video_path, 'frames_segmentation_10HZ.csv')

            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    frame_path = os.path.join(frames_folder, row['frame_file'])
                    mask_path = os.path.join(masks_folder, row['mask_file'])

                    # 检查帧和掩膜是否存在
                    if os.path.exists(frame_path) and os.path.exists(mask_path):
                        all_pairs.append((frame_path, mask_path))
                    else:
                        print(f"Warning: Missing frame or mask: {frame_path}, {mask_path}")

    # 打乱数据对并拆分为训练集和验证集
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"Total pairs: {len(all_pairs)}, Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    return train_pairs, val_pairs



# 计算准确率
def calculate_accuracy(outputs, masks):
    preds = torch.argmax(outputs, dim=1)  # 获取每个像素的预测类别
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / total


# 定义设备
device = torch.device("cuda")

# 定义数据增强
transform = A.Compose([
    A.Resize(256, 256),  # 调整大小
    ToTensorV2()         # 转换为Tensor
])

# 数据集拆分
root_folder = '/home/disk1/lyf/seg/data/train'
root_folder_test = '/home/disk1/lyf/seg/data/test'
train_videos, val_videos = split_train_val(root_folder)

# 定义训练集和验证集
train_dataset = SurgicalToolDataset(root_folder=train_videos, transform=transform)
val_dataset = SurgicalToolDataset(root_folder=val_videos, transform=transform)
test_dataset = SurgicalToolDataset(root_folder=root_folder_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, persistent_workers=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, persistent_workers=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, persistent_workers=True, pin_memory=True)

# 定义模型、损失函数和优化器
model = get_unet_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scaler = GradScaler()

# GPU数据增强
gpu_augment = torch.nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.2, contrast=0.2, p=0.2),
).to(device)

# 训练模型
epochs = 5
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    train_correct = 0
    train_total = 0

    for frames, masks in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        frames = frames.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            frames = gpu_augment(frames)

        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        # 累计训练集准确率
        train_correct += (torch.argmax(outputs, dim=1) == masks).sum().item()
        train_total += masks.numel()

    avg_train_loss = epoch_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    end_time = time.time() - start_time
    print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Time: {end_time:.4f}s")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for vframes, vmasks in val_loader:
            vframes = vframes.to(device)
            vmasks = vmasks.to(device)

            voutputs = model(vframes)
            vloss = criterion(voutputs, vmasks)
            val_loss += vloss.item()
            # 累计验证集准确率
            val_correct += (torch.argmax(voutputs, dim=1) == vmasks).sum().item()
            val_total += vmasks.numel()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    print(f"Epoch [{epoch + 1}/{epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 测试阶段
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for tframes, tmasks in test_loader:
            tframes = tframes.to(device)
            tmasks = tmasks.to(device)

            toutputs = model(tframes)
            tloss = criterion(toutputs, tmasks)
            test_loss += tloss.item()
            # 累计测试集准确率
            test_correct += (torch.argmax(toutputs, dim=1) == tmasks).sum().item()
            test_total += tmasks.numel()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    print(f"Epoch [{epoch + 1}/{epochs}], Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), f"/home/disk1/lyf/seg/models_results/surgical_tool_segmentation_unet_resnet34_epoch{epoch}.pth")
