from pathlib import Path
import json
import random
import pandas as pd


def split_metadata(root_dir, metadata_csv_path, split_ratio=0.5, split_id=["ws1", "ws2"]):
    df = pd.read_csv(root_dir / metadata_csv_path)
    total_videos = len(df)
    print(f"Total videos in metadata: {total_videos}")

    # 打乱数据
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(total_videos * split_ratio)

    split_1 = df_shuffled.iloc[:split_index]
    split_2 = df_shuffled.iloc[split_index:]

    print(f"{split_id[0]}: {len(split_1)} videos")
    print(f"{split_id[1]}: {len(split_2)} videos")

    split_1.to_csv(root_dir / f"metadata_split_{split_id[0]}.csv", index=False)
    split_2.to_csv(root_dir / f"metadata_split_{split_id[1]}.csv", index=False)



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
    return combined_csv


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


def verify_all_video_processed(root_dir, csv_path):
    root_dir = Path(root_dir)
    mp4_files = {
        str(p.resolve()) for p in root_dir.rglob("*.mp4")
    }

    print(f"Total videos in {root_dir}: {len(mp4_files)}")

    df = pd.read_csv(csv_path)
    path_col = df.iloc[:, 0].astype(str).str.strip()
    caption_col = df.iloc[:, 1]

    invalid_values = {"", "none", "nan", "error", "failed"}

    valid_mask = (
        caption_col.notna()
        & (~caption_col.astype(str).str.strip().str.lower().isin(invalid_values))
    )

    completed_paths = {
        str(Path(p).resolve()) for p in path_col[valid_mask]
    }
    print(f"Completed paths: {len(completed_paths)}")

    unfinished_paths = list(mp4_files - completed_paths)
    print(f"Unfinished videos: {len(unfinished_paths)}")

    if len(unfinished_paths) == 0:
        print("✅ All videos are completed!")
        return None
    else:
        print("❌ Some videos are still unfinished: saved in new split")
        json_path = Path("./split_unfinished.json")
        with open(json_path, "w") as f:
            json.dump([str(p) for p in unfinished_paths], f, indent=2)
        print(f"Unfinished video paths saved to {json_path}")
        return json_path

def post_process(metadata_path, root_dir):
    # Split the metadata.csv file containing absolute paths into:
    # - metadata_train.csv
    # - metadata_val.csv
    # And change the absolute paths to paths relative to train/ or val/.

    metadata_path = Path(metadata_path)
    root_dir = Path(root_dir)

    train_dir = root_dir / "train"
    val_dir = root_dir / "val"

    df = pd.read_csv(metadata_path)

    if df.shape[1] == 0:
        raise ValueError("metadata.csv 为空")

    # 默认第一列是 file_name
    file_col = df.columns[0]

    # 用 mask 方式分离
    train_mask = []
    val_mask = []
    new_paths = []

    for abs_path_str in df[file_col]:
        abs_path = Path(abs_path_str).resolve()

        if train_dir in abs_path.parents:
            rel_path = abs_path.relative_to(train_dir)
            train_mask.append(True)
            val_mask.append(False)
            new_paths.append(str(rel_path))

        elif val_dir in abs_path.parents:
            rel_path = abs_path.relative_to(val_dir)
            train_mask.append(False)
            val_mask.append(True)
            new_paths.append(str(rel_path))

        else:
            train_mask.append(False)
            val_mask.append(False)
            new_paths.append(None)
            print(f"⚠️ 跳过非法路径: {abs_path}")

    # 更新路径列
    df[file_col] = new_paths

    # 过滤
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    # 保存（明确写 header=True）
    train_csv_path = metadata_path.parent / "metadata_train.csv"
    val_csv_path = metadata_path.parent / "metadata_val.csv"

    df_train.to_csv(train_csv_path, index=False, header=True)
    df_val.to_csv(val_csv_path, index=False, header=True)

    print(f"✅ Train: {len(df_train)}")
    print(f"✅ Val: {len(df_val)}")
    return train_csv_path, val_csv_path

if __name__ == "__main__":
    root_dir = Path("/mnt/hdd/dataset/MultiCamVideo-Dataset")

    # metadata_csv_path = root_dir / "metadata_split_B.csv"
    # # Resume the unfinished video of split_B
    # resume_unfinished_video(json_path="./split_B.json", metadata_csv_path=metadata_csv_path)

    # csv_list = ["metadata_split_A.csv", "metadata_split_B.csv", "metadata_split_C.csv", "metadata_split_D.csv"]
    # combined_csv = merge_split_csvs(root_dir, csv_list)
    # # Verify all videos are processed
    # verify_all_video_processed(root_dir, combined_csv)

    # train_csv_path, val_csv_path = post_process(combined_csv, root_dir)

    metadata_csv_path = "metadata.csv"
    split_metadata(root_dir, metadata_csv_path, split_ratio=0.5, split_id=["ws1", "ws2"])
