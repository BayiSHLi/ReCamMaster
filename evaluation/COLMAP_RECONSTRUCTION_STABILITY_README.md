# COLMAP/PyCOLMAP Reconstruction Stability Guide

This document summarizes why camera trajectory reconstruction may fail frequently, how we fixed it, and how to reproduce stable results.

## 1. Symptom Summary

Typical issues seen in camera extraction:

- The same input video produces different reconstructed frames across runs.
- Some camera videos reconstruct only a few frames (for example, 2/81).
- Different runs choose different image subsets, causing unstable downstream metrics.
- In offline environments, sequential matching with loop detection can crash the process.

## 2. Root Cause Analysis

Main causes identified in this repository setup:

- Incremental SfM non-determinism:
  `pycolmap.incremental_mapping` defaults to random behavior (`random_seed=-1`) and can produce different initializations.
- Multi-thread + GPU introduces extra execution-order variance:
  feature matching and optimization order can drift between runs.
- Environment mismatch:
  running with system Python may not have `pycolmap`, while conda env does.
- Offline loop-detection crash risk:
  `SequentialPairingOptions.loop_detection=True` may trigger vocab-tree download.
  If download fails, process may abort with `SIGABRT` (not a normal Python exception).
- Weak matching setup on hard videos:
  a single matcher/profile can fail to register enough images.

## 3. Implemented Fixes

### 3.1 Deterministic-first mode

Implemented in `evaluation/extract_camera_trajectory.py`:

- Deterministic mode is default (`deterministic=True`).
- Force CPU path and single thread in deterministic mode.
- Set seeds for RANSAC / mapper / triangulation.
- Disable multi-model expansion and tighten model selection behavior.
- Sort output trajectory by `image_name` for stable output ordering.

### 3.2 Robust retry strategy

If the first attempt is weak, run fallback profiles and select the best reconstruction:

- `primary`: requested matcher, default profile.
- `sequential_robust`: larger overlap, more features, guided matching.
- `exhaustive_robust`: exhaustive matching fallback for difficult sequences.

Best model is selected by:

- Number of reconstructed images.
- Number of 3D points.

Early stop is used when target reconstruction ratio is reached.

### 3.3 Offline-safe behavior

- Keep sequential loop detection disabled in default retries.
- Avoid vocab-tree download dependency in offline environment.

### 3.4 Better observability

`camera_trajectory.json` now records:

- deterministic flag
- selected attempt name/index
- per-attempt reconstruction stats
- seed/thread/device metadata

## 4. Key Code Entry Points

### 4.1 Trajectory extractor

File: `evaluation/extract_camera_trajectory.py`

- Deterministic/retry API:

```python
extract_camera_trajectory_from_video(
    video_path,
    output_dir,
    ...,
    deterministic=True,
    random_seed=0,
    enable_retry=True,
    min_reconstruction_ratio=0.2,
)
```

- Retry profiles and selection logic are inside `extract_camera_trajectory_from_video`.
- PyCOLMAP options (feature extraction, matching, incremental mapping) are configured in `_run_pycolmap_sfm`.

### 4.2 Camera evaluation CLI

File: `evaluation/eval_camera.py`

CLI now derives paths from three top-level inputs:

- `--generated_video_root`
- `--video_id` (for example, `video_0`)
- `--save_root`

Derived paths:

- `generated_video_dir = generated_video_root / video_id`
- `trajectory_root = save_root / f"{video_id}_glomap"`
- `output_json = save_root / f"{video_id}_glomap_camera_metrics.json"`

## 5. Reproduction Steps

### 5.1 Activate correct Python environment

Use the environment that contains `pycolmap`.

```bash
conda activate py310
python -c "import pycolmap; print(pycolmap.__version__)"
```

### 5.2 Run stable extraction/evaluation

```bash
python -u evaluation/eval_camera.py \
  --generated_video_root /mnt/hdd/dataset/webvid10m/outputs \
  --video_id video_0 \
  --save_root evaluation/trajectory_test \
  --gt_camera_json example_test_data/cameras/camera_extrinsics.json \
  --camera_pattern 'cam_*.mp4' \
  --frame_skip 1 \
  --max_frames 81 \
  --matcher_type sequential \
  --num_threads 4
```

### 5.3 Verify reproducibility (run twice)

Run the same command twice (change `--save_root` or move output JSON), then compare:

- `trajectory_num_frames_reconstructed`
- `trajectory_selected_attempt`
- `sampled_frame_indices`

Quick check script:

```python
import json
from pathlib import Path

r1 = json.loads(Path('evaluation/trajectory_test/video_0_glomap_camera_metrics.json').read_text())
r2 = json.loads(Path('evaluation/trajectory_test_run2/video_0_glomap_camera_metrics.json').read_text())

a = r1['results'][0]
b = r2['results'][0]
print('rec:', a['trajectory_num_frames_reconstructed'], b['trajectory_num_frames_reconstructed'])
print('attempt:', a.get('trajectory_selected_attempt'), b.get('trajectory_selected_attempt'))
print('frames_equal:', a.get('sampled_frame_indices') == b.get('sampled_frame_indices'))
```

## 6. Recommended Tuning

- For hard videos, keep retry enabled (default).
- If trajectories are too short, increase temporal baseline:
  try `--frame_skip 2` or `--frame_skip 4`.
- If quality is still unstable, keep deterministic mode (default) and do not enable non-deterministic mode.
- If speed is more important than strict reproducibility, use `--trajectory_non_deterministic`.

## 7. Troubleshooting

- `pycolmap not installed`:
  use the conda env with pycolmap; do not use a system Python without dependencies.
- Process aborts during matching in offline machine:
  ensure loop detection is not enabled with remote vocab-tree dependency.
- Very low reconstructed frame ratio:
  keep retry enabled and test `frame_skip`/matcher combinations (`sequential`, fallback `exhaustive`).

## 8. Output Layout

With:

- `--video_id video_0`
- `--save_root evaluation/trajectory_test`

Outputs are written to:

- Trajectory root: `evaluation/trajectory_test/video_0_glomap/`
- Metrics report: `evaluation/trajectory_test/video_0_glomap_camera_metrics.json`
- Per camera trajectory json: `evaluation/trajectory_test/video_0_glomap/cam_xx/camera_trajectory.json`

## 9. Notes

- `global_mapping` is unavailable in current pycolmap environment (`3.13.0`), so pipeline uses incremental mapping path.
- Stable reconstruction does not guarantee high metric quality for all videos; it guarantees better repeatability and higher chance of successful registration.
