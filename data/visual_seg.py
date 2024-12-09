import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(frame_path, mask_path):
    frame = cv2.imread(frame_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 使用不同的颜色显示分割掩码
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    unique_mask_value = np.unique(mask)
    # 合并视频帧和分割掩码
    combined = cv2.addWeighted(frame, 0.3, color_mask, 0.7, 0)
    print(f"Unique values in mask: {unique_mask_value}")
    # 使用matplotlib显示图像
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # 不显示坐标轴
    plt.show()

# 示例：处理一个视频帧和对应的分割掩码
frame_path = "/home/disk1/lyf/seg/data/test/video_42/frames/frame_0060.jpg"
mask_path = "/home/disk1/lyf/seg/data/test/video_42/segmentation/000003600.png"
visualize_segmentation(frame_path, mask_path)
# import numpy as np

# # 定义颜色到类别的映射
# COLOR_TO_CLASS = {
#     (0, 0, 0): 0,   # Background
#     (1, 1, 1): 1,   # Tool clasper
#     (2, 2, 2): 2,   # Tool wrist
#     (3, 3, 3): 3,   # Tool shaft
#     (4, 4, 4): 4,   # Suturing needle
#     (5, 5, 5): 5,   # Thread
#     (6, 6, 6): 6,   # Suction tool
#     (7, 7, 7): 7,   # Needle Holder
#     (8, 8, 8): 8,   # Clamps
#     (9, 9, 9): 9    # Catheter
# }

# # 处理mask，将RGB值转换为类别标签
# def convert_rgb_to_class(mask):
#     height, width, _ = mask.shape
#     label_mask = np.zeros((height, width), dtype=np.int64)

#     for rgb, cls in COLOR_TO_CLASS.items():
#         label_mask[(mask == rgb).all(axis=2)] = cls

#     return label_mask
