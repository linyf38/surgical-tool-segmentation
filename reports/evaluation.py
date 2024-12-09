import torch
import numpy as np
import sys
sys.path.append('/home/disk1/lyf/seg')
import os
import pandas as pd
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.data_loader_10HZ import SurgicalToolDataset

# Define model loading function
def load_model(weight_path, device="cuda"):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Don't load ImageNet weights, use the trained weights
        in_channels=3,
        classes=10
    )
    model.load_state_dict(torch.load(weight_path))  # Load the pre-trained weights
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Compute IoU and Dice score for evaluation
def compute_iou_dice_combined(y_true, y_pred, num_classes):
    
    combined_scores = []

    for cls in range(num_classes):
        true_class = (y_true == cls).astype(np.uint8)
        pred_class = (y_pred == cls).astype(np.uint8)

        intersection = np.sum(np.logical_and(true_class, pred_class))
        union = np.sum(np.logical_or(true_class, pred_class))

        dice_score = 2 * intersection / (np.sum(true_class) + np.sum(pred_class)) if (np.sum(true_class) + np.sum(pred_class)) else 0
        iou_score = intersection / union if union else 0

        combined_scores.append((iou_score + dice_score) / 2)

    mean_combined_score = np.mean(combined_scores)
    return mean_combined_score

# Get DataLoader for the test dataset
def get_test_loader(dataset_root, batch_size=16):
    transform = A.Compose([
        A.Resize(256, 256),  # Resize to match the model's input size
        ToTensorV2()         # Convert images and masks to tensors
    ])

    all_pairs = []

    # Collect all frame-mask pairs from the test dataset (no split needed)
    for video_folder in os.listdir(dataset_root):
        video_folder_path = os.path.join(dataset_root, video_folder)

        frames_folder = os.path.join(video_folder_path, 'frames_10HZ')
        masks_folder = os.path.join(video_folder_path, 'segmentation')
        csv_file = os.path.join(video_folder_path, 'frames_segmentation_10HZ.csv')

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                frame_path = os.path.join(frames_folder, row['frame_file'])
                mask_path = os.path.join(masks_folder, row['mask_file'])
                all_pairs.append((frame_path, mask_path))

    # Create dataset and DataLoader
    test_dataset = SurgicalToolDataset(pairs=all_pairs, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return test_loader

# Model evaluation function
def evaluate_model(model, dataloader, device, num_classes):
    
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for frames, masks in dataloader:
            inputs = frames.to(device, non_blocking=True)
            outputs = model(inputs)

            # Convert ground truth and predicted masks to numpy arrays
            masks_true = masks.cpu().numpy()
            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            # Store true and predicted values
            y_true.append(masks_true)
            y_pred.append(pred)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Compute combined IoU and Dice score
    mean_combined_score = compute_iou_dice_combined(y_true, y_pred, num_classes)

    print(f"Mean Combined IoU-Dice Score: {mean_combined_score:.4f}")
    return mean_combined_score

# Main evaluation script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10  # Adjust based on your dataset
    dataset_root = '/home/disk1/lyf/seg/data/test'  # Path to your test dataset
    weight_path = "/home/disk1/lyf/seg/models_results/unet_epoch25.pth"  # Path to your trained model weights

    # Load pre-trained model
    trained_model = load_model(weight_path, device)

    # Get DataLoader for test dataset
    test_loader = get_test_loader(dataset_root)

    # Evaluate the model
    evaluate_model(trained_model, test_loader, device, num_classes)
