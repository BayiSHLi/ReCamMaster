from __future__ import annotations

import argparse
import hashlib
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan all files named original.mp4 under a root directory and print exact or similar videos."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/hdd/dataset/webvid10m",
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="original.mp4",
        help="Filename to scan (default: original.mp4).",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=8,
        help="Number of frames sampled per video for perceptual hashing.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=10,
        help="Maximum Hamming distance to consider two videos similar.",
    )
    parser.add_argument(
        "--prefix-bits",
        type=int,
        default=16,
        help="Bucket videos by hash prefix bits to reduce pair comparisons.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Optional cap for debug runs; -1 means no limit.",
    )
    return parser.parse_args()


def find_target_files(root: Path, target_name: str, max_files: int) -> List[Path]:
    files = sorted(path for path in root.rglob(target_name) if path.is_file())
    if max_files > 0:
        files = files[:max_files]
    return files


def _sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def find_exact_duplicates(paths: Sequence[Path]) -> List[List[Path]]:
    size_groups: Dict[int, List[Path]] = defaultdict(list)
    for path in paths:
        size_groups[path.stat().st_size].append(path)

    duplicates: List[List[Path]] = []
    for _, same_size_paths in size_groups.items():
        if len(same_size_paths) < 2:
            continue

        hash_groups: Dict[str, List[Path]] = defaultdict(list)
        for path in same_size_paths:
            hash_groups[_sha1(path)].append(path)
        duplicates.extend(group for group in hash_groups.values() if len(group) > 1)
    return duplicates


def _phash_frame(gray_frame: np.ndarray) -> int:
    resized = cv2.resize(gray_frame, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8]

    # Ignore the DC component when computing threshold.
    values = low.flatten()
    threshold = np.median(values[1:]) if values.shape[0] > 1 else values[0]
    bits = values > threshold

    hash_value = 0
    for bit in bits.astype(np.uint8).tolist():
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def _read_sampled_frame_hashes(path: Path, sample_frames: int) -> List[int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hashes: List[int] = []

    try:
        if frame_count > 0:
            num = min(sample_frames, frame_count)
            indices = np.unique(np.linspace(0, frame_count - 1, num=num, dtype=np.int64))
            for idx in indices.tolist():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hashes.append(_phash_frame(gray))
        else:
            # Fallback for streams without reliable frame count.
            while len(hashes) < sample_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hashes.append(_phash_frame(gray))
    finally:
        cap.release()

    return hashes


def _aggregate_hashes(hashes: Sequence[int]) -> int | None:
    if not hashes:
        return None

    bit_votes = np.zeros(64, dtype=np.int32)
    for value in hashes:
        for i in range(64):
            bit_votes[i] += (value >> (63 - i)) & 1

    threshold = len(hashes) / 2.0
    agg = 0
    for vote in bit_votes.tolist():
        agg = (agg << 1) | int(vote >= threshold)
    return agg


def compute_video_hashes(paths: Sequence[Path], sample_frames: int) -> Dict[Path, int]:
    result: Dict[Path, int] = {}
    for idx, path in enumerate(paths, start=1):
        frame_hashes = _read_sampled_frame_hashes(path, sample_frames=sample_frames)
        agg = _aggregate_hashes(frame_hashes)
        if agg is None:
            continue
        result[path] = agg
        if idx % 200 == 0:
            print(f"[progress] hashed {idx}/{len(paths)} videos")
    return result


def _prefix_bucket(value: int, prefix_bits: int) -> int:
    if prefix_bits <= 0:
        return value
    if prefix_bits >= 64:
        return value
    return value >> (64 - prefix_bits)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_similar_pairs(
    path_to_hash: Dict[Path, int],
    prefix_bits: int,
    phash_threshold: int,
) -> List[Tuple[Path, Path, int]]:
    buckets: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for path, value in path_to_hash.items():
        buckets[_prefix_bucket(value, prefix_bits)].append((path, value))

    similar: List[Tuple[Path, Path, int]] = []
    seen = set()
    for items in buckets.values():
        if len(items) < 2:
            continue
        for (path_a, hash_a), (path_b, hash_b) in itertools.combinations(items, 2):
            key = tuple(sorted((str(path_a), str(path_b))))
            if key in seen:
                continue
            seen.add(key)
            dist = _hamming(hash_a, hash_b)
            if dist <= phash_threshold:
                similar.append((path_a, path_b, dist))

    similar.sort(key=lambda x: (x[2], str(x[0]), str(x[1])))
    return similar


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    files = find_target_files(root, args.target_name, args.max_files)
    print(f"[scan] found {len(files)} file(s) named {args.target_name} under {root}")
    if not files:
        return

    exact_groups = find_exact_duplicates(files)
    if exact_groups:
        print("\n[exact duplicates]")
        for group_id, group in enumerate(exact_groups, start=1):
            print(f"group {group_id}:")
            for path in group:
                print(f"  {path}")
    else:
        print("\n[exact duplicates] none")

    path_to_hash = compute_video_hashes(files, sample_frames=max(1, args.sample_frames))
    print(f"[scan] computed perceptual hashes for {len(path_to_hash)} file(s)")

    similar_pairs = find_similar_pairs(
        path_to_hash=path_to_hash,
        prefix_bits=max(0, args.prefix_bits),
        phash_threshold=max(0, args.phash_threshold),
    )
    if similar_pairs:
        print("\n[similar pairs]")
        for path_a, path_b, dist in similar_pairs:
            print(f"distance={dist:2d} | {path_a}")
            print(f"             | {path_b}")
    else:
        print("\n[similar pairs] none")


if __name__ == "__main__":
    main()
