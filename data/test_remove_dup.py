import cv2
import os
import hashlib
import tqdm

def remove_duplicate_frames(frames_folder, deduped_folder):
    """
    移除重复的帧。
    
    :param frames_folder: 原始帧文件夹
    :param deduped_folder: 去重后的帧保存文件夹
    """
    os.makedirs(deduped_folder, exist_ok=True)
    frame_files = sorted(os.listdir(frames_folder))
    previous_hash = None
    saved_count = 0

    for frame_file in tqdm.tqdm(frame_files, desc="Removing duplicates"):
        frame_path = os.path.join(frames_folder, frame_file)
        with open(frame_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        if file_hash != previous_hash:
            # 复制帧到去重文件夹
            deduped_path = os.path.join(deduped_folder, frame_file)
            cv2.imwrite(deduped_path, cv2.imread(frame_path))
            saved_count += 1
            previous_hash = file_hash

    print(f"Removed duplicates. Saved {saved_count} unique frames.")

# 示例使用：
# for i in tqdm.trange(41, 51):
#     frames_folder = f"/home/disk1/lyf/seg/data/test/video_{i}/frames"
#     deduped_folder = f"/home/disk1/lyf/seg/data/test/video_{i}/frames_deduped"
#     remove_duplicate_frames(frames_folder, deduped_folder)
#     print(f"video_{i} duplicate removal finished")
i = 41
frames_folder = f"/home/disk1/lyf/seg/data/test/video_{i}/frames"
deduped_folder = f"/home/disk1/lyf/seg/data/test/video_{i}/frames_deduped"
remove_duplicate_frames(frames_folder, deduped_folder)
print(f"video_{i} duplicate removal finished")