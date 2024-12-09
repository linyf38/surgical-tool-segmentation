import matplotlib.pyplot as plt
from data_loader import SurgicalToolDataset
import torch

def visualize_mask_conversion(dataset, idx):
    frame, mask_class = dataset[idx]

    # 检查 mask_class 是否是 PyTorch Tensor
    if isinstance(mask_class, torch.Tensor):
        mask_class = mask_class.numpy()  # 转换为 numpy 格式

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Frame")
    plt.imshow(frame.permute(1, 2, 0).numpy())  # 转换为 (H, W, C) 以用于显示

    plt.subplot(1, 2, 2)
    plt.title("Converted Mask (Class Index)")
    plt.imshow(mask_class, cmap="tab10")  # 使用类别颜色显示掩膜

    plt.show()

# 示例使用
visualize_mask_conversion(SurgicalToolDataset(root_folder='/home/disk1/lyf/seg/data/test'), idx=0)
