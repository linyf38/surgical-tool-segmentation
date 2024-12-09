import torch
import numpy as np
import sys
sys.path.append('/home/disk1/lyf/seg')
import os
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.data_loader_10HZ import SurgicalToolDataset
import random
from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer
import pandas as pd

# 数据集划分函数
def split_train_val(root_folder, train_ratio=0.8, seed=42):
    
    random.seed(seed)
    all_pairs = []

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
                    if os.path.exists(frame_path) and os.path.exists(mask_path):
                        all_pairs.append((frame_path, mask_path))

    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"Total pairs: {len(all_pairs)}, Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    return train_pairs, val_pairs


# 加载SegFormer模型
def load_segformer_model(num_classes):
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model


# 数据增强
transform = A.Compose([
    A.Resize(512, 512),  # Resize to match SegFormer input size
    ToTensorV2()
])

# 划分数据集
root_folder = '/home/disk1/lyf/seg/data/train'
train_pairs, val_pairs = split_train_val(root_folder)

# 创建训练集和验证集
train_dataset = SurgicalToolDataset(pairs=train_pairs, transform=transform)
val_dataset = SurgicalToolDataset(pairs=val_pairs, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# 加载模型
num_classes = 10  
model = load_segformer_model(num_classes)
model.to("cuda")


training_args = TrainingArguments(
    output_dir="./results",  
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,  
    load_best_model_at_end=True
)

# 自定义数据收集器
def data_collator(features):
    pixel_values = torch.stack([f[0] for f in features])
    labels = torch.stack([f[1] for f in features])
    return {"pixel_values": pixel_values, "labels": labels}

#训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("/home/disk1/lyf/seg/models_results/segformer")
