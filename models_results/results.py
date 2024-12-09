import torch
import segmentation_models_pytorch as smp
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
def load_model(weight_path, device="cuda"):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # 使用训练好的权重，不加载ImageNet权重
        in_channels=3,
        classes=10
    )
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model
# 自定义颜色映射 (根据类别数定义)
# Define custom color map for different classes (e.g., 10 classes for different surgical tools)
COLORS = np.array([
    [0, 0, 0],        # Class 0: Background (Black)
    [255, 0, 0],      # Class 1: Red
    [0, 255, 0],      # Class 2: Green
    [0, 0, 255],      # Class 3: Blue
    [255, 255, 0],    # Class 4: Yellow
    [255, 0, 255],    # Class 5: Magenta
    [0, 255, 255],    # Class 6: Cyan
    [128, 0, 0],      # Class 7: Dark Red
    [0, 128, 0],      # Class 8: Dark Green
    [0, 0, 128],      # Class 9: Dark Blue
], dtype=np.uint8)

# Define color map application function
def apply_color_map(mask, colors=COLORS):
    
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        color_mask[mask == class_id] = color
    return color_mask

# Overlay mask on image
def overlay_mask_on_image(image, mask, alpha=0.5):
    
    return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

# Visualize prediction results
def visualize_prediction(model, frame_path, mask_path, transform, device="cuda"):
    
    # 读取图片和真实掩膜
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Error: Mask file not found or could not be read: {mask_path}")
        return

    # 数据预处理
    augmented = transform(image=frame, mask=mask)
    frame_tensor = augmented['image'].unsqueeze(0).to(device)
    mask_tensor = augmented['mask']

    # 确保图像是 float32 类型并归一化
    frame_tensor = frame_tensor.float() / 255.0

    # 模型预测
    with torch.no_grad():
        output = model(frame_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 检查预测值是否有意义
    unique_values = np.unique(pred_mask)
    print(f"Unique values in predicted mask: {unique_values}")

    # 将预测掩膜映射为彩色图像
    pred_colored_mask = apply_color_map(pred_mask)

    # 将真实掩膜映射为彩色图像
    true_colored_mask = apply_color_map(mask)

    # 确保输入图像和掩膜是相同的大小
    pred_colored_mask = cv2.resize(pred_colored_mask, (frame.shape[1], frame.shape[0]))
    true_colored_mask = cv2.resize(true_colored_mask, (frame.shape[1], frame.shape[0]))

    # 叠加预测掩膜到输入图片
    overlay_image = overlay_mask_on_image(frame, pred_colored_mask)

    # 可视化
    plt.figure(figsize=(16, 8))

    # 输入图片
    plt.subplot(1, 4, 1)
    plt.imshow(frame)
    plt.title("Input Image")
    plt.axis("off")

    # 真实掩膜
    plt.subplot(1, 4, 2)
    plt.imshow(true_colored_mask)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # 预测掩膜
    plt.subplot(1, 4, 3)
    plt.imshow(pred_colored_mask)
    plt.title("Predicted Mask")
    plt.axis("off")

    # 叠加结果
    plt.subplot(1, 4, 4)
    plt.imshow(overlay_image)
    plt.title("Overlay Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("/home/disk1/lyf/seg/models_results/training_results_unet_epoch15_60.png")
    plt.show()



# 定义数据增强，与训练时一致
transform = Compose([
    Resize(256, 256),  # 确保尺寸一致
    ToTensorV2()
])

# 加载模型
weight_path = "/home/disk1/lyf/seg/models_results/unet_epoch15.pth"
model = load_model(weight_path)

# 设置测试图片路径
frame_path = "/home/disk1/lyf/seg/data/train/video_06/frames/frame_00060.jpg"
mask_path = "/home/disk1/lyf/seg/data/train/video_06/segmentation/000000360.png"

# 可视化预测结果
visualize_prediction(model, frame_path, mask_path, transform)

