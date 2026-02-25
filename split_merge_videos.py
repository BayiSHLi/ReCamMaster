from pathlib import Path
import json
import random

root_dir = Path("/mnt/hdd/dataset/MultiCamVideo-Dataset")
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