# Evaluation Pipeline (ReCamMaster)

This directory provides a unified evaluation pipeline for ReCamMaster outputs.

Camera evaluation is now fixed to the GLOMAP-based pipeline (no backend switch).

## 1. Core Pipeline

The full evaluation workflow is integrated in `evaluation/evaluation.py`, and a single command automatically completes steps 1-5:

1. (Optional) Run `evaluation/inference_webvid.py` to generate multi-camera videos (skipped when `--skip_generation` is set).
2. Scan all `<video_id>/camXX.mp4` folders under `save_dir`.
3. Run `evaluation/eval_camera.py` (fixed GLOMAP/COLMAP trajectory extraction) to compute paper-comparable camera metrics (RotErr/TransErr).
4. Run pair-wise metrics for each camera video:
  - `evaluation/eval_matching.py` -> Mat.Pix
  - `evaluation/eval_clip.py` -> CLIP-V / CLIP-T / CLIP-F
  - `evaluation/eval_fvd.py` -> FID / FVD / FVD-V
  - `evaluation/eval_vbench.py` -> VBench sub-metrics
5. Aggregate all metrics and write a unified JSON report.


## 2. Installation

`eval_camera.py` requires both `colmap` and `glomap` in PATH.

### 1) Environment dependencies

Requires cmake >= 3.28
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1.tar.gz
tar xfvz cmake-3.30.1.tar.gz && cd cmake-3.30.1
./bootstrap && make -j$(nproc) && sudo make install
```

Requires eigen>=3.4.0
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xfvz eigen-3.4.0.tar.gz && cd eigen-3.4.0
mkdir build && cd build
cmake ..
sudo make install
```

Requires ceres>=2.0.0
```bash

git clone https://github.com/ceres-solver/ceres-solver.git
git checkout 2.0.0
git branch 
 
cd ceres-solver
mkdir build && cd build
cmake ..
make -j4
 
sudo make install
```

### 2) Build and install COLMAP

```bash
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build

cmake ..
make -j4
sudo make install
```

### 3) Build and install GOLMAP
```bash
git clone https://github.com/colmap/glomap.git
cd glomap
 
mkdir build && cd build
cmake .. -DSuiteSparse_CHOLMOD_INCLUDE_DIR=/usr/include/suitesparse -DSuiteSparse_CHOLMOD_LIBRARY=/usr/lib/x86_64-linux-gnu/libcholmod.so
make -j4
sudo make install
```

### 4) Verify installation
```
colmap -h
glomap -h
```

## 3. Entry Point

Main script: `evaluation/evaluation.py`

- Key input arguments:
  - `--data_root`: Dataset root (contains `videos/` and metadata CSV, default `0000.csv`).
  - `--save_dir`: Generation output directory; defaults to `<data_root>/outputs` when empty.
  - `--skip_generation`: Skip generation and evaluate existing outputs only.
    - When `--skip_generation` is used,
      the pipeline checks all `<video_id>` folders under `save_dir` (or `<data_root>/outputs`)
      and requires complete non-empty `cam01.mp4` ... `cam10.mp4` files.
      The run fails fast if incomplete folders are found.
  - `--selected_metrics`: Comma-separated metrics to run, from `camera,matching,clip,fvd,vbench`.
  - `--gt_camera_json`: GT camera extrinsics JSON (default `evaluation/camera_extrinsics.json`).
  - `--trajectory_save_root`: Root directory for evaluation outputs (per-video structure, default `results/evaluation`).
  - `--skip_camera_extraction`: Reuse saved trajectory files under `<trajectory_save_root>/<video_id>/camera/<video_id>_glomap/camXX/camera_trajectory.json` and skip GLOMAP/COLMAP extraction.
  - `--eval_show_progress` / `--no-eval_show_progress`: Enable/disable evaluation progress bars (default enabled).
  - `--max_eval_videos`: Maximum number of `<video_id>` folders to evaluate (`-1` means all).
  - `--camera_glob`: Camera-video matching pattern in each video folder (default `cam[0-9][0-9].mp4`).
  - `--output_json`: Final aggregated report path.


## 4. Evaluation Command

Full pipeline (generate first, then evaluate; no `--skip_generation`):

```bash
python evaluation/evaluation.py \
  --data_root /mnt/hdd/dataset/webvid \
  --save_dir /mnt/hdd/dataset/webvid/outputs \
  --selected_metrics camera,matching,clip,fvd,vbench \
  --trajectory_save_root results/evaluation \
  --output_json results/evaluation/pipeline_results.json
```

Skip generation (evaluate existing outputs only):

```bash
python evaluation/evaluation.py \
  --skip_generation \
  --data_root /mnt/hdd/dataset/webvid \
  --save_dir /mnt/hdd/dataset/webvid/outputs \
  --selected_metrics camera,matching,clip,fvd,vbench \
  --trajectory_save_root results/evaluation \
  --output_json results/evaluation/pipeline_results_skip_generation.json
```

Skip camera extraction for faster metric-only debugging (reuse saved trajectories):

```bash
python evaluation/evaluation.py \
  --skip_generation \
  --selected_metrics camera \
  --skip_camera_extraction \
  --data_root /mnt/hdd/dataset/webvid \
  --save_dir /mnt/hdd/dataset/webvid/outputs \
  --trajectory_save_root results/evaluation \
  --output_json results/evaluation/pipeline_results_camera_skip_extract.json
```

Smoke test (quick validation on one video):

```bash
python evaluation/evaluation.py \
  --skip_generation \
  --data_root /mnt/hdd/dataset/webvid \
  --save_dir /mnt/hdd/dataset/webvid/outputs \
  --selected_metrics camera,matching,clip,fvd,vbench \
  --max_eval_videos 1 \
  --output_json results/evaluation/smoke_test_one_sample.json
```

## 5. Minimal Input Convention

Without `--skip_generation`
- Source videos: `<data_root>/videos/<video_id>.mp4`
- GT camera extrinsics: `evaluation/camera_extrinsics.json`
- Metadata (text prompt): `<data_root>/0000.csv` (or set via `--metadata_csv`)

With `--skip_generation`
- Generated videos: `<data_root>/outputs/<video_id>/camXX.mp4`
- GT camera extrinsics: `evaluation/camera_extrinsics.json`
- Metadata (text prompt): `<data_root>/0000.csv` (or set via `--metadata_csv`)


## 6. Outputs

- Final aggregated report: `results/evaluation/pipeline_results.json` (or `--output_json`)
  - Includes: `summary` and `report_paths`
  - `summary.numeric`: `mean/count` for Table1/Table2 primary metrics
  - `summary.auxiliary`: `mean/count` for all auxiliary metrics
  - `report_paths.camera_reports`: per-video mapping of camera report paths
  - `report_paths.pair_metric_reports`: per-video mapping of pair-metric report paths

- Per-video reports under `trajectory_save_root` (default: `results/evaluation`):
  - `results/evaluation/<video_id>/camera/<video_id>_glomap_camera_metrics.json`
  - `results/evaluation/<video_id>/pair_metrics.json`

## 7. Metric Status

- `ok`: successfully computed
- `pending`: interface is connected but some sub-metrics are not implemented yet
- `failed`: runtime/dependency/data error
- `skipped`: missing inputs or intentionally skipped

## 8. Notes

- CLIP may download large model files on first run; you can pre-place them at:
  - `models/DiffSynth-Studio/QualityMetric_reward_pretrained/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin`
  - `models/DiffSynth-Studio/QualityMetric_reward_pretrained/bpe_simple_vocab_16e6.txt.gz`
- Recommended environment: `py310`.
- Camera metric convention in simplified `eval_camera.py`:
  - `camera.RotErr`: radian sum over frames
  - `camera.TransErr`: L2 translation error sum over frames
- `evaluation/eval_camera_complete.py` keeps the archived full/diagnostic implementation.


## 9. Camera Trajectory Visualization

Use the visualization script after camera evaluation is finished:

- Script: `evaluation/utils/visualize_gt_pred_grid.py`
- Input: one trajectory directory like `results/evaluation/<video_id>/camera/<video_id>_glomap`
- Output: grid image(s) comparing GT vs predicted trajectories

Example:

```bash
python evaluation/utils/visualize_gt_pred_grid.py \
  --trajectory_root results/evaluation/10046243/camera/10046243_glomap \
  --output_dir results/evaluation/10046243/camera/trajectory_visualizations
```

By default it renders a 2x5 overlay view (`--overlay`). Use `--no-overlay` for split GT/PRED panels.
