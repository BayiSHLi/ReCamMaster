# GLOMAP/COLMAP 相机轨迹提取 - 快速开始指南

## 安装

### 1. 安装 COLMAP（推荐：Conda）
```bash
# 最简单的方法
conda install -c conda-forge colmap -y
```

### 2. 安装 PyColmap（Python API）
```bash
# 标准版（CPU）
pip install pycolmap

# 或 CUDA 12 加速版
pip install pycolmap-cuda12
```

### 3. 验证安装
```bash
# 检查 COLMAP 命令行工具
colmap -h

# 检查 Python 模块
python -c "import pycolmap; print('OK')"
```

---

## 快速测试（使用示例数据）

### 方案 1: 直接调用 Python 脚本

```bash
cd /home/user/workspace/SHLi/ReCamMaster

# 提取轨迹
python evaluation/extract_camera_trajectory.py \
  example_test_data/videos/1.mp4 \
  evaluation/trajectory_test

# 输出：
# - evaluation/trajectory_test/camera_trajectory.json
```

### 方案 2: 通过评测框架集成

```bash
cd /home/user/workspace/SHLi/ReCamMaster

# Run with trajectory extraction
python evaluation/run_evaluation.py \
  --metrics camera_trajectory \
  --source-video example_test_data/videos/1.mp4 \
  --output-json evaluation/results_trajectory.json
```

---

## 参数配置

### 基础参数
```python
# 在 Python 中直接调用
from evaluation.extract_camera_trajectory import extract_camera_trajectory_from_video

result = extract_camera_trajectory_from_video(
    video_path="your_video.mp4",
    output_dir="output_dir",
    
    # 帧采样参数
    frame_skip=1,        # 每隔 N 帧提取一次（加速）
    max_frames=300,      # 最多提取 N 帧
    resize_short=0,      # 短边缩放（0 表示不缩放）
    
    # SfM 参数
    camera_model="PINHOLE",
    matcher_type="sequential",  # 视频推荐
    use_global_mapper=True,  # 使用 global mapper（GLOMAP 功能）
    num_threads=4,  # 并行线程数
    verbose=True,   # 打印日志
)
```

### 高级参数（通过 run_evaluation.py）
```bash
python evaluation/run_evaluation.py \
  --metrics camera_trajectory \
  --source-video input.mp4 \
  --trajectory-frame-skip 2 \           # 每 2 帧提取 1 个
  --trajectory-max-frames 100 \         # 只提取前 100 帧
  --trajectory-matcher sequential \     # sequential|vocab_tree
  --trajectory-use-global-mapper 1 \    # 使用全局 mapper
  --trajectory-num-threads 8 \          # 8 个线程
  --output-json results.json
```

---

## 输出格式

### 完整输出 JSON 结构
```json
{
  "status": "ok",
  "reason": "",
  "metadata": {
    "num_frames_extracted": 150,
    "num_frames_reconstructed": 148,
    "reconstruction_ratio": 0.9867,
    "num_3d_points": 125000,
    "camera_model": "PINHOLE",
    "use_global_mapper": true
  },
  "camera_intrinsics": {
    "model": "PINHOLE",
    "width": 1280,
    "height": 720,
    "params": [1000.0, 1000.0, 640.0, 360.0]
  },
  "trajectory": [
    {
      "frame_id": 0,
      "image_id": 1,
      "image_name": "frame_000000.jpg",
      "rotation_matrix": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
      ],
      "translation": [0.0, 0.0, 0.0],
      "quaternion": [1.0, 0.0, 0.0, 0.0]
    },
    ...
  ]
}
```

### 保存位置
- **JSON 轨迹**: `{output_dir}/camera_trajectory.json`
- **COLMAP 数据库**: 临时（自动清理）
- **COLMAP 稀疏重建**: 临时（自动清理）

---

## 与 ReCamMaster 现有模块的集成

### 1. 与 Camera Metric (RotErr/TransErr) 结合

```python
# extract_camera_trajectory.py 产生的轨迹
# 可直接用于 eval_camra.py 的 RotErr/TransErr 计算

# 先提取轨迹
pred_trajectory = extract_camera_trajectory_from_video(pred_video)

# 再计算误差
from evaluation.eval_camra import evaluate_camera_metric

error = evaluate_camera_metric(
    gt_camera_json="gt_cameras.json",
    pred_camera_json="pred_cameras_from_colmap.json",  # 由上面转换而来
)
```

### 2. 与 Feature Matching (Mat.Pix) 协同

```python
# COLMAP 的特征提取/匹配步骤已在 trajectory 提取中完成
# 可在后续评测中复用匹配统计信息
```

### 3. 评测完整流程

```bash
# 一键评测：轨迹提取 + 相机误差 + 同步度 + CLIP + FVD

python evaluation/run_evaluation.py \
  --metrics camera_trajectory,camera,matching,clip,fvd \
  --source-video source.mp4 \
  --generated-video generated.mp4 \  # 如果有生成的视频
  --text-prompt "a man dancing" \
  --output-json full_evaluation.json
```

---

## 性能优化建议

### 对于快速演示
```bash
# 只提取开头 30 帧，每 4 帧提取 1 个
python evaluation/extract_camera_trajectory.py \
  video.mp4 output \
  --frame-skip 4 \
  --max-frames 30
```

### 对于高精度
```bash
# 提取所有帧，使用多线程
python evaluation/extract_camera_trajectory.py \
  video.mp4 output \
  --frame-skip 1 \
  --max-frames 0 \        # 0 表示无限制
  --num-threads 8
```

### 对于大规模数据
```bash
# 使用词汇树匹配（需要下载词表）
# 降低分辨率，增加帧间隔
python evaluation/extract_camera_trajectory.py \
  video.mp4 output \
  --resize-short 480 \      # 短边缩放到 480p
  --frame-skip 8 \          # 每 8 帧提取 1 个
  --matcher-type vocab_tree \
  --num-threads 16
```

---

## 故障排除

### 问题 1: "COLMAP not found in PATH"
```bash
# 解决方案 1: 安装 COLMAP
conda install -c conda-forge colmap -y

# 解决方案 2: 检查安装位置
which colmap
colmap -h
```

### 问题 2: "pycolmap not installed"
```bash
pip install pycolmap
```

### 问题 3: "No frames extracted from video"
```bash
# 检查视频文件是否有效
ffprobe video.mp4

# 或手动解帧测试
ffmpeg -i video.mp4 -q:v 2 test_frames/frame_%04d.jpg
```

### 问题 4: "No reconstruction found (0 cameras)"
这通常表示特征匹配或 SfM 失败，可能原因：
- 视频内容不足（纹理不足、运动过快等）
- 分辨率太低
- 特征匹配参数不合适

**尝试调整**:
```bash
# 增加特征数量
colmap feature_extractor --SiftExtraction.max_num_features=8192 ...

# 降低匹配阈值
colmap sequential_matcher --SequentialMatching.overlap=20 ...

# 放松 SfM 约束
colmap mapper --Mapper.abs_pose_min_num_inliers=4 ...
```

### 问题 5: 重建缓慢（GPU 加速）
```bash
# 特征提取启用 GPU（需要 SIFT-GPU）
colmap feature_extractor \
  --SiftExtraction.use_gpu=1 \
  --SiftExtraction.gpu_index=0 ...
```

---

## 常见用例

### 用例 1: 提取真实视频的相机轨迹
```bash
python evaluation/extract_camera_trajectory.py \
  ~/Videos/my_video.mp4 \
  ./my_trajectory_output
```

### 用例 2: 对比生成视频与原视频的相机轨迹
```bash
# 先提取两个轨迹
python evaluation/extract_camera_trajectory.py original.mp4 traj_original
python evaluation/extract_camera_trajectory.py generated.mp4 traj_generated

# 再计算相机误差
python -c "
import json
import numpy as np

with open('traj_original/camera_trajectory.json') as f:
    orig = json.load(f)
with open('traj_generated/camera_trajectory.json') as f:
    gen = json.load(f)

# 比较轨迹...
"
```

### 用例 3: 集成到自动化评测管道
```bash
# day_batch_evaluate.sh
for video in videos/*.mp4; do
  python evaluation/run_evaluation.py \
    --metrics camera_trajectory \
    --source-video "$video" \
    --output-json "results/$(basename $video .mp4).json"
done
```

---

## 参考资源

- **完整文档**: [GLOMAP_RESEARCH.md](./GLOMAP_RESEARCH.md)
- **COLMAP 官网**: https://colmap.github.io/
- **COLMAP 命令行文档**: https://colmap.github.io/cli.html
- **PyColmap API**: https://colmap.github.io/pycolmap/index.html

---

## 已知限制

1. **单相机假设**: 当前实现假设单个相机（同一内参）
   - 可扩展支持多相机
   
2. **无时间戳对齐**: 提取的轨迹是按帧索引排列的
   - 如需与其他模态对齐，需外部注册逻辑

3. **静态场景效果更好**: COLMAP SfM 对有动态物体的视频可能效果不佳
   - 建议使用纹理丰富的场景

4. **内参自动估计精度**: 如有标定内参，建议外部提供

---

## 下一步

1. ✅ 安装 COLMAP 和 pycolmap
2. ✅ 用示例视频测试 `extract_camera_trajectory.py`
3. ✅ 将轨迹与 GT 相机参数对比验证
4. 🔄 针对 ReCamMaster 的实际视频优化参数
5. 🔄 考虑多相机和时间戳对齐功能

---

## 需要帮助？

- 查看完整研究报告: [GLOMAP_RESEARCH.md](./GLOMAP_RESEARCH.md)
- COLMAP 官方讨论区: https://github.com/colmap/colmap/discussions
- 在评测框架中加入 debug 参数: `--verbose 1`
