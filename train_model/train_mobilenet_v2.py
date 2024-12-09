import sys
sys.path.append('/home/disk1/lyf/seg')
import segmentation_models_pytorch as smp
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# import data
from data.data_loader import SurgicalToolDataset  
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler  
import kornia.augmentation as K
import kornia
# DALI imports

def get_unet_model():
    # 定义U-Net模型
    model = smp.Unet(
        encoder_name="mobilenet_v2",        # shufflenet_v2_x0_5
        encoder_weights="imagenet",     # 使用在ImageNet上预训练的权重
        in_channels=3,                  # 输入通道数
        classes=10                       # 输出通道数，对应分割的类别数（背景和手术工具）
    )
    return model


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

transform = A.Compose([
    A.Resize(224, 224), # 调整大小
#     A.HorizontalFlip(p=0.5),  # 随机水平翻转
#     # A.RandomBrightnessContrast(p=0.2),  # 随机亮度对比度调整
    ToTensorV2()  # 转换为Tensor
])
# 定义数据集和数据加载器
train_dataset = SurgicalToolDataset(root_folder='/home/disk1/lyf/seg/data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, persistent_workers=True, pin_memory=True )


val_dataset = SurgicalToolDataset(root_folder='/home/disk1/lyf/seg/data/test', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, persistent_workers=True, pin_memory=True)
# 定义模型、损失函数和优化器
model = get_unet_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scaler = GradScaler()
# 训练模型
epochs = 5

gpu_augment = torch.nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.2, contrast=0.2, p=0.2), 
).to(device)

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    for frames, masks in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        frames = frames.to(device)
        masks = masks.to(device)

        # # 前向传播
        # outputs = model(frames)
        # loss = criterion(outputs, masks)
        # 在GPU上进行数据增强
        with torch.no_grad():
            frames = gpu_augment(frames)
        # # 反向传播
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        # 使用scaler进行loss缩放和反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    avg_train_loss = epoch_loss / len(train_loader)
    end_time = time.time() - start_time
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}, Time: {end_time:.4f})s")
     # 验证阶段
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for vframes, vmasks in val_loader:
            vframes = vframes.to(device)
            vmasks = vmasks.to(device)
            
            voutputs = model(vframes)
            vloss = criterion(voutputs, vmasks)
            total_val_loss += vloss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
# 保存训练后的模型
torch.save(model.state_dict(), f"/home/disk1/lyf/seg/models_results/surgical_tool_segmentation_unet_epoch{epoch}.pth")