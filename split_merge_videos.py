from pathlib import Path
import json
import random
import pandas as pd


def split_videos(
        root_dir, 
        file_list, 
        split_id = ["A", "B"],
        ratio=0.5,
        ):
    if file_list:
        mp4_files = [Path(p) for p in file_list]
    else:
        mp4_files = sorted(list(root_dir.rglob("*.mp4")))

    print(f"Total videos: {len(mp4_files)}")

    random.seed(42)
    random.shuffle(mp4_files)

    split_index = int(len(mp4_files) * ratio)

    split_1 = mp4_files[:split_index]
    split_2 = mp4_files[split_index:]

    print(f"{split_id[0]}: {len(split_1)} videos")
    print(f"{split_id[1]}: {len(split_2)} videos")

    with open(f"split_{split_id[0]}.json", "w") as f:
        json.dump([str(p) for p in split_1], f, indent=2)

    with open(f"split_{split_id[1]}.json", "w") as f:
        json.dump([str(p) for p in split_2], f, indent=2)

def merge_split_csvs(root_dir, csv_list):
    # 假设 split_A.csv 和 split_B.csv 都在当前目录下
    all_lines = []
    for csv_file in csv_list:
        csv_path = Path(root_dir) / csv_file
        if not csv_path.exists():
            print(f"请确保 {csv_file} 存在于 {root_dir} 目录下")
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        all_lines.append(lines)
    # 使用 split_A 的表头
    header = all_lines[0][0]
    combined_lines = [header]
    for lines in all_lines:
        combined_lines.extend(lines[1:])  # 跳过表头
        # 写入新的 CSV 文件
    combined_csv = Path(root_dir) / "metadata.csv"
    with open(combined_csv, "w", encoding="utf-8") as f:
        f.writelines(combined_lines)
    print(f"已合并 CSV 文件，保存为 {combined_csv}")


def resume_unfinished_video(json_path, metadata_csv_path):
    with open(json_path, "r") as f:
        video_list = set(p.strip() for p in json.load(f))

    print(f"Total videos in {json_path}: {len(video_list)}")

    df = pd.read_csv(metadata_csv_path)
    completed_paths = set(p.strip() for p in df.iloc[:, 0].astype(str))
    print(f"Completed paths: {len(completed_paths)}")

    unfinished_paths = list(video_list - completed_paths)
    print(f"Unfinished videos: {len(unfinished_paths)}")

    if len(unfinished_paths) == 0:
        print("✅ All videos are completed!")
        exit()

    split_videos(root_dir, file_list=unfinished_paths, 
                 split_id=["C", "D"], ratio=0.5)


if __name__ == "__main__":
    root_dir = Path("/mnt/hdd/dataset/MultiCamVideo-Dataset")
    metadata_csv_path = root_dir / "metadata_split_B.csv"
    # Resume the unfinished video of split_B
    resume_unfinished_video(json_path="./split_B.json", metadata_csv_path=metadata_csv_path)