import cv2
import os
import tqdm

def extract_frames(video_path, output_folder):
    """
    从视频中提取帧并保存为图像文件。
    
    :param video_path: 视频文件路径
    :param output_folder: 提取帧保存的文件夹
    :param frame_rate: 提取帧的频率
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps/60)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1

    cap.release()
    print(f"video {video_path} had processed {saved_frame_count} frames.")

# 示例使用：提取视频帧
# test
# for i in tqdm.trange(41, 51):
#     video_path = f"/home/disk1/lyf/seg/data/test/video_{i}/video_left.avi"  # 具体的视频路径
#     output_folder = f"/home/disk1/lyf/seg/data/test/video_{i}/frames"  # 提取后的帧保存的文件夹
#     extract_frames(video_path, output_folder)
#     print(f"video_{i} finished")

# train
for i in tqdm.trange(30, 41):
    video_path = f"/home/disk1/lyf/seg/data/train/video_{i}/video_left.avi"  # 具体的视频路径
    output_folder = f"/home/disk1/lyf/seg/data/train/video_{i}/frames"  # 提取后的帧保存的文件夹
    extract_frames(video_path, output_folder)
    print(f"video_{i} finished")

# for i in tqdm.trange(1,3):
#     video_path = f"/home/disk1/lyf/seg/data/train/video_29_{i}/video_left.avi"  # 具体的视频路径
#     output_folder = f"/home/disk1/lyf/seg/data/train/video_29_{i}/frames"  # 提取后的帧保存的文件夹
#     extract_frames(video_path, output_folder)
#     print(f"video_29_{i} finished")

# video_path = f"/home/disk1/lyf/seg/data/train/video_16/video_left.avi"  # 具体的视频路径
# output_folder = f"/home/disk1/lyf/seg/data/train/video_16/frames"  # 提取后的帧保存的文件夹
# extract_frames(video_path, output_folder)
# print(f"video_16 finished")