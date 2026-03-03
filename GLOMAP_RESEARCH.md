# GLOMAP 视频相机轨迹提取研究报告

## 1. 核心信息

### 1.1 GLOMAP 项目状态
- **官方 Repo**: https://github.com/colmap/glomap
- **重要警告**: ⚠️ **GLOMAP 已被弃用（Deprecated）**，功能已完全迁移到最新的 [COLMAP](https://github.com/colmap/colmap) 中
- **替代方案**: 使用 COLMAP 最新版本（3.13.0+），其中 GLOMAP 功能被称为 **"global" mapper**
- **许可**: BSD-3-Clause（同 COLMAP）

### 1.2 GLOMAP 的核心优势
1. **速度快**：相比 COLMAP 增量 SfM，快 1-2 个数量级
2. **质量好**：相同或更优的重建质量
3. **可扩展性好**：支持大规模数据集
4. **输入输出统一**：与 COLMAP 兼容

### 1.3 与 COLMAP 的关系
- GLOMAP 是全局 SfM 算法的研究项目版本
- 最新 COLMAP 已完整集成 GLOMAP 功能
- 建议：**直接使用最新 COLMAP** 而非独立的 GLOMAP

---

## 2. 视频 → 相机轨迹完整工作流

### 2.1 高层流程图
```
视频 (MP4/MOV/...)
  ↓
[子流程：解帧] 
  ↓
图像序列 + 内参 (内参可来自视频元数据或标定)
  ↓
[COLMAP 特征提取] (feature_extractor)
  ↓
COLMAP 数据库 (SQLite .db)
  ↓
[COLMAP 特征匹配] (sequential_matcher 或 vocab_tree_matcher)
  ↓
COLMAP 数据库 (带匹配信息)
  ↓
[COLMAP Mapper (Global mode)] 
  ↓
稀疏 3D 重建 + 相机内外参 (COLMAP 格式)
  ↓
[导出相机轨迹]
  ↓
相机外参 (R, t) 序列 JSON/CSV
```

### 2.2 关键步骤详解

#### 步骤 1: 视频解帧
**输入**: 视频文件（video.mp4）  
**输出**: 帧序列（frame_0000.jpg, frame_0001.jpg, ...）  
**工具**: OpenCV (cv2) 或 FFmpeg

```bash
# FFmpeg 方案
ffmpeg -i video.mp4 -q:v 2 frame_%04d.jpg

# 或 Python/OpenCV
import cv2
cap = cv2.VideoCapture('video.mp4')
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f'frame_{frame_count:04d}.jpg', frame)
    frame_count += 1
```

#### 步骤 2: COLMAP 特征提取
```bash
colmap feature_extractor \
  --database_path ./database.db \
  --image_path ./images
```

**参数说明**:
- `--database_path`: SQLite 数据库存储特征
- `--image_path`: 帧所在目录
- 可选：`--ImageReader.camera_model` (实拍视频通常 PINHOLE 即可)

#### 步骤 3: COLMAP 特征匹配
对于**视频序列**（帧顺序已知），推荐使用 `sequential_matcher`:

```bash
# 快速方案：顺序匹配（适合视频）
colmap sequential_matcher --database_path ./database.db

# 或高精度方案：词汇树匹配（需下载词表）
colmap vocab_tree_matcher \
  --database_path ./database.db \
  --VocabTreeMatching.vocab_tree_path ./vocab_tree.fta
```

#### 步骤 4: COLMAP Global Mapper（核心）
```bash
# 最新 COLMAP（推荐）
colmap mapper \
  --database_path ./database.db \
  --image_path ./images \
  --output_path ./sparse

# 如果仍用独立 GLOMAP（已弃用但可用）
glomap mapper \
  --database_path ./database.db \
  --image_path ./images \
  --output_path ./sparse
```

**COLMAP Mapper 模式**:
- 默认使用增量 mapper（slow_BA, local)
- 可通过配置启用 global mapper（等价于 GLOMAP）
- 或直接用新版 COLMAP 命令 `--Mapper.ba_global_function_tolerance=1e-4`

#### 步骤 5: 提取相机轨迹
**输出位置**: `./sparse/0/` 目录结构：
```
sparse/0/
  ├── cameras.bin       # 内参
  ├── images.bin        # 图像+外参
  ├── points3D.bin      # 3D 点
  └── cameras.txt / images.txt / points3D.txt  # 文本版本
```

**使用 pycolmap 读取轨迹**:
```python
import pycolmap

# 读取重建
reconstruction = pycolmap.Reconstruction('./sparse/0')

# 遍历所有图像的相机参数
camera_trajectory = []
for image_id in sorted(reconstruction.images.keys()):
    image = reconstruction.images[image_id]
    
    camera = reconstruction.cameras[image.camera_id]
    
    # 获取外参 (R, t)
    R = image.qvec2rotmat()  # 从四元数获取旋转矩阵
    t = image.tvec            # 平移向量
    
    camera_trajectory.append({
        'frame': image.name,
        'image_id': image_id,
        'camera_id': image.camera_id,
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'qvec': image.qvec.tolist(),  # 四元数 (w, x, y, z)
    })

# 保存为 JSON
import json
with open('camera_trajectory.json', 'w') as f:
    json.dump(camera_trajectory, f, indent=2)
```

---

## 3. 环境配置与安装

### 3.1 COLMAP 安装（推荐方案）

#### 方案 A: Conda（最简）
```bash
conda install -c conda-forge colmap
pip install pycolmap  # 或 pycolmap-cuda12
```

#### 方案 B: 从源码编译（完整控制）
```bash
# 安装依赖
sudo apt-get install -y \
    cmake git ninja-build build-essential \
    libboost-all-dev libeigen3-dev libflann-dev \
    libfreeimage-dev libmetis-dev libgflags-dev \
    libsuitesparse-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev

# 克隆并构建
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install
```

#### 方案 C: Docker
```bash
docker run -it colmap/colmap:latest
```

### 3.2 PyPyColmap （Python API）
```bash
# 标准版（CPU）
pip install pycolmap

# CUDA 12 加速版
pip install pycolmap-cuda12
```

### 3.3 检验安装
```bash
# 命令行
colmap -h
colmap mapper -h

# Python
python -c "import pycolmap; print(pycolmap.__version__)"
```

---

## 4. ReCamMaster 集成方案

### 4.1 集成架构
基于现有框架扩展一个新的 **`extract_camera_trajectory.py`** 模块：

```python
"""
evaluation/extract_camera_trajectory.py
用途：视频 → 解帧 → COLMAP 特征提取 → SfM → 相机轨迹导出
"""

def extract_camera_trajectory_from_video(
    video_path: str,
    output_dir: str,
    camera_model: str = "PINHOLE",
    matcher_type: str = "sequential",  # 或 "vocab_tree"
    use_gpu: bool = True,
    skip_extraction_if_exists: bool = True,
) -> Dict[str, Any]:
    """
    完整工作流：视频 → 相机轨迹
    
    返回:
        {
            'status': 'ok' | 'failed',
            'camera_trajectory': [
                {
                    'frame_id': 0,
                    'rotation': [[...], [...], [...]],
                    'translation': [...],
                    'qvec': [w, x, y, z]
                },
                ...
            ],
            'camera_intrinsics': {...},
            'num_frames_processed': int,
            'num_frames_reconstructed': int,
            'sparsities': {
                'num_3d_points': int,
                'reconstruction_ratio': float,
            }
        }
    """
```

### 4.2 集成到 run_evaluation.py
```python
# 添加新指标到注册表
from evaluation import extract_camera_trajectory

METRIC_REGISTRY["camera_trajectory"] = extract_camera_trajectory.run

# 命令行示例
# python evaluation/run_evaluation.py \
#   --metrics camera_trajectory \
#   --source-video input_video.mp4 \
#   --output-trajectory outputs/camera_trajectory.json
```

### 4.3 与现有评测框架的对接
- **输入**: 视频文件
- **输出**: 相机轨迹 JSON
- **与 eval_camra.py 协同**: 提取的轨迹可直接用于 RotErr/TransErr 计算
- **与 eval_matching.py 协同**: 匹配步骤可复用

---

## 5. 配置建议与优化

### 5.1 针对视频序列的最佳实践

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **Matcher** | `sequential_matcher` | 视频帧有顺序关系，顺序匹配最高效 |
| **Frame Skip** | 4-8 帧 | 减少冗余，加速特征提取；可根据动作快速程度调整 |
| **Image Resolution** | 720p-1080p | 平衡速度和精度，过小易丢失细节 |
| **Mapper Mode** | `global` | 使用 GLOMAP/global 获得最快速度 |
| **Bundle Adjustment** | Local + Global | 先局部 BA 再全局 BA，质量最佳 |
| **Feature Type** | SIFT / SUPERPOINT | SIFT 稳定（默认），SUPERPOINT 更快（需 ONNX） |

### 5.2 COLMAP 配置文件示例
创建 `colmap_config.txt`：
```ini
[FeatureExtraction]
ImageReader.camera_model=PINHOLE
ImageReader.single_camera=0
SiftExtraction.max_image_size=2048
SiftExtraction.peak_threshold=0.0033

[SequentialMatcher]
SequentialMatching.overlap=10
SequentialMatching.loop_detection=1

[Mapper]
Mapper.num_threads=4
Mapper.ba_global_function_tolerance=1e-4
Mapper.ba_local_max_num_iterations=25
```

### 5.3 性能优化
- **使用 GPU**: 特征提取支持 CUDA (SIFT-GPU)，匹配支持 GPU
- **多线程**: 指定 `--Mapper.num_threads` 利用多核
- **Frame Sampling**: 对长视频，可每隔 N 帧提取一次，后续插值

---

## 6. 输出格式规范

### 6.1 标准相机轨迹格式
```json
{
  "version": "1.0",
  "metadata": {
    "video_source": "example.mp4",
    "total_frames_input": 300,
    "frames_reconstructed": 298,
    "reconstruction_ratio": 0.9933,
    "camera_model": "PINHOLE",
    "num_3d_points": 150000
  },
  "camera_intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 640.0,
    "cy": 360.0,
    "width": 1280,
    "height": 720
  },
  "trajectory": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
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

### 6.2 兼容 ReCamMaster 格式
可转换为现有 `camera_extrinsics.json` 格式（见 example_test_data）

---

## 7. 常见问题与故障排除

### Q1: GLOMAP vs COLMAP global mapper？
**A**: GLOMAP 已弃用，建议使用最新 COLMAP。功能完全一致，官方维护更好。

### Q2: 对于快速运动的视频，如何保证特征匹配质量？
**A**: 
1. 增加 frame overlap（sequential_matcher 的 `overlap` 参数）
2. 使用更稳定的特征描述子（SIFT > SUPERPOINT）
3. 降低帧采样间隔
4. 预处理：去模糊、增强对比度

### Q3: 内参如何获取？
**A**:
- 方案 1: COLMAP 自动估计 (默认 PINHOLE)
- 方案 2: 从视频元数据或标定结果提供
- 方案 3: 融合多视角约束优化（COLMAP 内置）

### Q4: 相机轨迹与 GT 轴不对齐，如何对齐？
**A**: 使用 `eval_camra.py` 中的 `align_poses_umeyama()` 函数做 Umeyama 对齐

### Q5: 重建失败（0 个相机重建），如何调试？
**A**:
```bash
# 检查数据库是否正确创建
sqlite3 database.db "SELECT COUNT(*) FROM images;"

# 查看特征和匹配统计
colmap database_print_tables --database_path database.db

# 尝试更宽松的参数
colmap mapper \
  --database_path database.db \
  --Mapper.abs_pose_min_num_inliers=4 \
  --Mapper.min_focal_length_ratio=0.1 \
  --Mapper.max_focal_length_ratio=10.0
```

---

## 8. 参考资源

### 官方文档
- **COLMAP 官网**: https://colmap.github.io/
- **COLMAP 命令行文档**: https://colmap.github.io/cli.html
- **PyColmap API**: https://colmap.github.io/pycolmap/index.html
- **PyCOLMAP 重建格式**: https://colmap.github.io/format.html

### 论文
- **GLOMAP (ECCV 2024)**: https://arxiv.org/pdf/2407.20219
- **COLMAP (CVPR 2016)**: https://demuc.de/papers/schoenberger2016sfm.pdf

### 相关工具
- **hloc (Hierarchical Localization)**: https://github.com/cvg/Hierarchical-Localization/
  - 支持学习型特征描述子（SuperPoint, DISK, etc.)
  - 推荐用于高质量匹配
  
- **Rerun Visualizer**: https://rerun.io/
  - 专门用于 COLMAP 重建可视化

---

## 9. 下一步行动计划

### 立即可做
1. ✅ 安装最新 COLMAP (`conda install -c conda-forge colmap`)
2. ✅ 验证 pycolmap 可用
3. 使用 `example_test_data/videos/1.mp4` 跑一遍完整流程

### 短期
1. 实现 `extract_camera_trajectory.py` 模块
2. 集成到 `run_evaluation.py`
3. 与 `eval_camra.py` 对接，验证轨迹精度

### 中期
1. 优化针对 ReCamMaster 视频的特征匹配参数
2. 支持从 IP 摄像头、实时视频流提取
3. 对比不同特征类型 (SIFT vs SUPERPOINT) 的性能

---

## 附录：快速命令参考

```bash
# 1. 从视频提取帧
ffmpeg -i video.mp4 -q:v 2 frames/frame_%04d.jpg

# 2. 创建 COLMAP 数据库并提取特征
colmap feature_extractor \
  --database_path database.db \
  --image_path frames

# 3. 特征匹配（视频用 sequential）
colmap sequential_matcher --database_path database.db

# 4. 运行 SfM mapper
colmap mapper \
  --database_path database.db \
  --image_path frames \
  --output_path sparse

# 5. 可视化（可选）
colmap gui --import_path sparse/0

# 6. 导出为文本格式（方便 Python 处理）
colmap model_converter \
  --input_path sparse/0 \
  --output_path sparse_txt \
  --output_format TXT
```
