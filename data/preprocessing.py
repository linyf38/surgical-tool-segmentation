
import av
import os
import tqdm

def extract_frames_with_av(video_path, output_folder, target_fps=10):
    
    os.makedirs(output_folder, exist_ok=True)

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    source_fps = video_stream.average_rate
    frame_interval = int(source_fps / target_fps)

    frame_count = 0
    saved_frame_count = 0

    for packet in container.demux(video_stream):
        for frame in packet.decode():
            if frame_count % frame_interval == 0:
                # 转换为 RGB 格式
                frame_rgb = frame.to_image()
                frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
                frame_rgb.save(frame_filename)
                saved_frame_count += 1
            frame_count += 1

    print(f"Video {video_path} processed {saved_frame_count} frames.")

# test
# for i in tqdm.trange(41, 51):
#     video_path = f"/home/disk1/lyf/seg/data/test/video_{i}/video_left.avi"  # 具体的视频路径
#     output_folder = f"/home/disk1/lyf/seg/data/test/video_{i}/frames_10HZ"  # 提取后的帧保存的文件夹
#     extract_frames_with_av(video_path, output_folder, target_fps=10)
#     print(f"video_{i} finished")
# train
# for i in tqdm.trange(1, 10):
#     video_path = f"/home/disk1/lyf/seg/data/train/video_0{i}/video_left.avi"  # 具体的视频路径
#     output_folder = f"/home/disk1/lyf/seg/data/train/video_0{i}/frames_10HZ"  # 提取后的帧保存的文件夹
#     extract_frames_with_av(video_path, output_folder, target_fps=10)
#     print(f"video_{i} finished")

# for i in tqdm.trange(1,3):
#     video_path = f"/home/disk1/lyf/seg/data/train/video_29_{i}/video_left.avi"  # 具体的视频路径
#     output_folder = f"/home/disk1/lyf/seg/data/train/video_29_{i}/frames_10HZ"  # 提取后的帧保存的文件夹
#     extract_frames(video_path, output_folder)
#     print(f"video_29_{i} finished")

# video_path = f"/home/disk1/lyf/seg/data/train/video_16/video_left.avi"  # 具体的视频路径
# output_folder = f"/home/disk1/lyf/seg/data/train/video_16/frames_10HZ"  # 提取后的帧保存的文件夹
# extract_frames(video_path, output_folder)
# print(f"video_16 finished")