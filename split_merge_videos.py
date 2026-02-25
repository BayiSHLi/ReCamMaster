from pathlib import Path
import json
import random


def split_videos(root_dir):
    mp4_files = sorted(list(root_dir.rglob("*.mp4")))

    print(f"Total videos: {len(mp4_files)}")

    # 如果你希望打乱顺序（推荐）
    random.seed(42)
    random.shuffle(mp4_files)

    ratio = 0.55  # 60% 给机器A

    split_index = int(len(mp4_files) * ratio)

    split_A = mp4_files[:split_index]
    split_B = mp4_files[split_index:]

    print(f"A: {len(split_A)} videos")
    print(f"B: {len(split_B)} videos")

    with open("split_A.json", "w") as f:
        json.dump([str(p) for p in split_A], f, indent=2)

    with open("split_B.json", "w") as f:
        json.dump([str(p) for p in split_B], f, indent=2)

def merge_split_csvs(root_dir):
    # 假设 split_A.csv 和 split_B.csv 都在当前目录下
    split_A_csv = Path(root_dir) / "metadata_split_A.csv"
    split_B_csv = Path(root_dir) / "metadata_split_B.csv"

    if not split_A_csv.exists() or not split_B_csv.exists():
        print("请确保 split_A.csv 和 split_B.csv 都存在于当前目录下")
        return

    # 读取两个 CSV 文件的内容
    with open(split_A_csv, "r", encoding="utf-8") as f:
        lines_A = f.readlines()

    with open(split_B_csv, "r", encoding="utf-8") as f:
        lines_B = f.readlines()

    # 合并内容（假设第一行是表头）
    header = lines_A[0]  # 使用 split_A 的表头
    combined_lines = [header] + lines_A[1:] + lines_B[1:]

    # 写入新的 CSV 文件
    combined_csv = Path(root_dir) / "metadata.csv"
    with open(combined_csv, "w", encoding="utf-8") as f:
        f.writelines(combined_lines)

    print(f"已合并 CSV 文件，保存为 {combined_csv}")