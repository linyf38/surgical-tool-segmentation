import os
import csv
import tqdm
def save_frame_mask_pairs(frames_folder, masks_folder, output_csv):
    frames = sorted(os.listdir(frames_folder))
    masks = sorted(os.listdir(masks_folder))

    # 每个mask对应60帧
    frames_per_mask = 60
    pairs = []

    # 获取帧文件的数量
    num_frames = len(frames)
    num_masks = len(masks)

    for i, frame_file in enumerate(frames):
        # 获取当前视频帧的编号 (处理五位数命名)
        frame_number = int(frame_file.split('_')[1].split('.')[0])  # frame_00001.jpg -> 1
        print(f"Processing {frame_file}...")

        # 计算对应的 segmentation mask 编号
        mask_index = frame_number // frames_per_mask  # 每60帧对应一个mask
        if mask_index >= num_masks:
            mask_index = num_masks - 1  # 超过最后一个mask时，使用最后一个mask
        
        mask_file = masks[mask_index]  # 使用实际的mask文件名

        # 保存帧文件和mask文件的配对
        pairs.append([frame_file, mask_file])

    # 保存配对到CSV文件
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_file', 'mask_file'])
        writer.writerows(pairs)

    print(f"Saved frame-mask pairs to {output_csv}")

# 使用示例
# test
# for i in tqdm.trange(42, 51):
#     frames_folder = f'/home/disk1/lyf/seg/data/test/video_{i}/frames'
#     masks_folder = f'/home/disk1/lyf/seg/data/test/video_{i}/segmentation'
#     output_csv = f'/home/disk1/lyf/seg/data/test/video_{i}/frames_segmentation.csv'

#     save_frame_mask_pairs(frames_folder, masks_folder, output_csv)

# train
for i in tqdm.trange(30, 41):
    frames_folder = f'/home/disk1/lyf/seg/data/train/video_{i}/frames'
    masks_folder = f'/home/disk1/lyf/seg/data/train/video_{i}/segmentation'
    output_csv = f'/home/disk1/lyf/seg/data/train/video_{i}/frames_segmentation.csv'

    save_frame_mask_pairs(frames_folder, masks_folder, output_csv)

# for i in tqdm.trange(1,3):
#     frames_folder = f"/home/disk1/lyf/seg/data/train/video_29_{i}/frames"  # 具体的视频路径
#     masks_folder = f"/home/disk1/lyf/seg/data/train/video_29_{i}/segmentation"  # 提取后的帧保存的文件夹
#     output_csv = f'/home/disk1/lyf/seg/data/train/video_29_{i}/frames_segmentation.csv'
#     print(f"video_29_{i} finished")
#     save_frame_mask_pairs(frames_folder, masks_folder, output_csv)

# video_path = f"/home/disk1/lyf/seg/data/train/video_16/video_left.avi"  # 具体的视频路径
# output_folder = f"/home/disk1/lyf/seg/data/train/video_16/frames"  # 提取后的帧保存的文件夹
# extract_frames(video_path, output_folder)
# print(f"video_16 finished")