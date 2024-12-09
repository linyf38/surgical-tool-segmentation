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
import matplotlib.pyplot as plt
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
train_dataset = SurgicalToolDataset(pairs=train_videos, transform=transform)
val_dataset = SurgicalToolDataset(pairs=val_videos, transform=transform)
test_dataset = SurgicalToolDataset(pairs=split_train_val(root_folder_test)[0], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, persistent_workers=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, persistent_workers=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, persistent_workers=True, pin_memory=True)

# 模型和优化器
model = get_unet_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scaler = GradScaler()

gpu_augment = torch.nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.2, contrast=0.2, p=0.2),
).to(device)

# 存储训练结果
epoch_results = {'epoch': [], 'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

# 训练模型
epochs = 60
for epoch in range(1, epochs + 1):
    start_time = time.time()
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for frames, masks in tqdm(train_loader, desc=f"Training Epoch {epoch}/{epochs}"):
        frames, masks = frames.to(device), masks.to(device)
        with torch.no_grad():
            frames = gpu_augment(frames)

        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_correct += (torch.argmax(outputs, dim=1) == masks).sum().item()
        train_total += masks.numel()

    train_accuracy = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # 验证阶段
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for vframes, vmasks in val_loader:
            vframes, vmasks = vframes.to(device), vmasks.to(device)
            voutputs = model(vframes)
            vloss = criterion(voutputs, vmasks)
            val_loss += vloss.item()
            val_correct += (torch.argmax(voutputs, dim=1) == vmasks).sum().item()
            val_total += vmasks.numel()

    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    # 记录结果
    epoch_results['epoch'].append(epoch)
    epoch_results['train_acc'].append(train_accuracy)
    epoch_results['val_acc'].append(val_accuracy)
    epoch_results['train_loss'].append(avg_train_loss)
    epoch_results['val_loss'].append(avg_val_loss)

    print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Train Acc {train_accuracy:.4f}, "
          f"Val Loss {avg_val_loss:.4f}, Val Acc {val_accuracy:.4f}")

    # 保存模型（每5个epoch保存一次）
    if epoch % 5 == 0 and epoch > 20:
        torch.save(model.state_dict(), f"/home/disk1/lyf/seg/models_results/unet_epoch{epoch}.pth")

# 可视化结果
def plot_training_results(epoch_results):
    epochs = epoch_results['epoch']
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_results['train_loss'], label="Train Loss")
    plt.plot(epochs, epoch_results['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_results['train_acc'], label="Train Accuracy")
    plt.plot(epochs, epoch_results['val_acc'], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig("/home/disk1/lyf/seg/models_results/training_results.png")
    plt.show()

# 绘制训练结果
plot_training_results(epoch_results)
