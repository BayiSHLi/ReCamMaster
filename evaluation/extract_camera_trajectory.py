"""
evaluation/extract_camera_trajectory.py

使用 pycolmap 从视频提取相机轨迹（简单版）。

流程：
1) 视频解帧
2) pycolmap.extract_features
3) pycolmap.match_sequential / match_exhaustive
4) pycolmap.incremental_mapping
5) 读取重建并导出轨迹 JSON
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pycolmap
except ImportError:
    pycolmap = None


def _resolve_attr_or_call(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    if callable(value):
        return value()
    return value


def _to_list(value: Any) -> List[float]:
    if value is None:
        return []
    if callable(value):
        value = value()
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return [float(value)]


def _camera_model_name(camera: Any) -> str:
    model_name = _resolve_attr_or_call(camera, "model_name", None)
    if model_name is not None:
        return str(model_name)

    model = _resolve_attr_or_call(camera, "model", None)
    if model is None:
        return "UNKNOWN"
    if hasattr(model, "name"):
        return str(model.name)
    return str(model)


def _image_pose(image: Any) -> tuple[np.ndarray, np.ndarray, List[float]]:
    if hasattr(image, "qvec2rotmat") and hasattr(image, "tvec"):
        rotation = image.qvec2rotmat()
        translation = _resolve_attr_or_call(image, "tvec", None)
        qvec = _to_list(_resolve_attr_or_call(image, "qvec", None))
        return np.asarray(rotation), np.asarray(translation), qvec

    cam_from_world = _resolve_attr_or_call(image, "cam_from_world", None)
    if cam_from_world is not None:
        rotation_obj = _resolve_attr_or_call(cam_from_world, "rotation", None)
        translation_obj = _resolve_attr_or_call(cam_from_world, "translation", None)
        if rotation_obj is None or translation_obj is None:
            raise AttributeError("cam_from_world does not provide rotation/translation")

        rotation = _resolve_attr_or_call(rotation_obj, "matrix", None)
        quaternion = _resolve_attr_or_call(rotation_obj, "quat", None)
        if rotation is None:
            raise AttributeError("rotation object does not provide matrix")

        return np.asarray(rotation), np.asarray(translation_obj), _to_list(quaternion)

    raise AttributeError("Unable to read pose from pycolmap image object")


def _extract_frames_from_video(
    video_path: str,
    output_dir: Path,
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    resize_short: int = 0,
) -> List[Path]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for frame extraction")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    saved: List[Path] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip == 0:
            if max_frames is not None and len(saved) >= max_frames:
                break

            if resize_short > 0:
                height, width = frame.shape[:2]
                short_side = min(height, width)
                if short_side > resize_short:
                    scale = resize_short / float(short_side)
                    new_width = int(round(width * scale))
                    new_height = int(round(height * scale))
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            frame_path = output_dir / f"frame_{len(saved):06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved.append(frame_path)

        frame_idx += 1

    cap.release()
    return saved


def _run_pycolmap_sfm(
    database_path: Path,
    image_path: Path,
    sparse_path: Path,
    matcher_type: str,
    camera_model: str,
    verbose: bool,
) -> Any:
    if pycolmap is None:
        raise RuntimeError("pycolmap is not installed")

    sparse_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[*] pycolmap.extract_features ...")

    extract_kwargs: Dict[str, Any] = {}
    if hasattr(pycolmap, "ImageReaderOptions"):
        reader_options = pycolmap.ImageReaderOptions()
        if hasattr(reader_options, "camera_model"):
            reader_options.camera_model = camera_model
        extract_kwargs["reader_options"] = reader_options

    if hasattr(pycolmap, "CameraMode") and hasattr(pycolmap.CameraMode, "SINGLE"):
        extract_kwargs["camera_mode"] = pycolmap.CameraMode.SINGLE

    pycolmap.extract_features(database_path, image_path, **extract_kwargs)

    if verbose:
        print(f"[*] pycolmap.match_{matcher_type} ...")

    if matcher_type == "sequential" and hasattr(pycolmap, "match_sequential"):
        pycolmap.match_sequential(database_path)
    elif matcher_type == "vocab_tree" and hasattr(pycolmap, "match_vocabtree"):
        pycolmap.match_vocabtree(database_path)
    else:
        pycolmap.match_exhaustive(database_path)

    if verbose:
        print("[*] pycolmap.incremental_mapping ...")

    maps = pycolmap.incremental_mapping(database_path, image_path, sparse_path)
    if maps is None:
        raise RuntimeError("incremental_mapping returned None")

    if isinstance(maps, dict):
        if len(maps) == 0:
            raise RuntimeError("No reconstruction generated")
        best_index, best_reconstruction = max(
            maps.items(),
            key=lambda item: len(dict(_resolve_attr_or_call(item[1], "images", {}))),
        )
        if verbose:
            print(f"[*] selected reconstruction #{best_index}")
        return best_reconstruction

    return maps


def extract_camera_trajectory_from_video(
    video_path: str,
    output_dir: str,
    camera_model: str = "PINHOLE",
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    resize_short: int = 0,
    matcher_type: str = "sequential",
    use_global_mapper: bool = True,
    num_threads: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    del use_global_mapper
    del num_threads

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if pycolmap is None:
        return {
            "status": "skipped",
            "reason": "pycolmap not installed. Install with: pip install pycolmap",
            "metadata": {},
            "camera_intrinsics": {},
            "trajectory": [],
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frames_dir = tmp_path / "frames"
            database_path = tmp_path / "database.db"
            sparse_path = tmp_path / "sparse"

            if verbose:
                print(f"[*] extracting frames from {video_path}")
            frames = _extract_frames_from_video(
                video_path=video_path,
                output_dir=frames_dir,
                frame_skip=frame_skip,
                max_frames=max_frames,
                resize_short=resize_short,
            )
            if len(frames) == 0:
                return {
                    "status": "failed",
                    "reason": "No frames extracted from video",
                    "metadata": {},
                    "camera_intrinsics": {},
                    "trajectory": [],
                }

            reconstruction = _run_pycolmap_sfm(
                database_path=database_path,
                image_path=frames_dir,
                sparse_path=sparse_path,
                matcher_type=matcher_type,
                camera_model=camera_model,
                verbose=verbose,
            )

            cameras = dict(_resolve_attr_or_call(reconstruction, "cameras", {}))
            if len(cameras) == 0:
                return {
                    "status": "failed",
                    "reason": "No cameras in reconstruction",
                    "metadata": {"num_frames_extracted": len(frames)},
                    "camera_intrinsics": {},
                    "trajectory": [],
                }

            first_camera_id = list(cameras.keys())[0]
            camera = cameras[first_camera_id]
            intrinsics = {
                "model": _camera_model_name(camera),
                "width": int(_resolve_attr_or_call(camera, "width", 0)),
                "height": int(_resolve_attr_or_call(camera, "height", 0)),
                "params": _to_list(_resolve_attr_or_call(camera, "params", [])),
            }

            images = dict(_resolve_attr_or_call(reconstruction, "images", {}))
            trajectory: List[Dict[str, Any]] = []
            for image_id in sorted(images.keys()):
                image = images[image_id]
                rotation, translation, quaternion = _image_pose(image)
                trajectory.append(
                    {
                        "frame_id": len(trajectory),
                        "image_id": int(image_id),
                        "image_name": str(_resolve_attr_or_call(image, "name", "")),
                        "rotation_matrix": rotation.tolist(),
                        "translation": translation.tolist(),
                        "quaternion": quaternion,
                    }
                )

            points3d = dict(_resolve_attr_or_call(reconstruction, "points3D", {}))
            result = {
                "status": "ok",
                "reason": "",
                "metadata": {
                    "num_frames_extracted": len(frames),
                    "num_frames_reconstructed": len(trajectory),
                    "reconstruction_ratio": float(len(trajectory) / len(frames)),
                    "num_3d_points": len(points3d),
                    "camera_model": camera_model,
                    "matcher_type": matcher_type,
                },
                "camera_intrinsics": intrinsics,
                "trajectory": trajectory,
            }

            output_json = output_path / "camera_trajectory.json"
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if verbose:
                print(f"[*] done: {len(trajectory)}/{len(frames)} frames reconstructed")
                print(f"[*] saved trajectory: {output_json}")

            return result

    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"Exception: {exc}",
            "metadata": {},
            "camera_intrinsics": {},
            "trajectory": [],
        }


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    source_video = config.get("source_video", "")
    if not source_video:
        return {
            "status": "skipped",
            "reason": "Missing source_video in config",
            "metrics": {},
        }

    result = extract_camera_trajectory_from_video(
        video_path=source_video,
        output_dir=config.get("trajectory_output_dir", "evaluation/trajectory_outputs"),
        camera_model=config.get("camera_model", "PINHOLE"),
        frame_skip=int(config.get("trajectory_frame_skip", 1)),
        max_frames=(
            int(config.get("trajectory_max_frames"))
            if config.get("trajectory_max_frames")
            else None
        ),
        resize_short=int(config.get("trajectory_resize_short", 0)),
        matcher_type=config.get("trajectory_matcher", "sequential"),
        use_global_mapper=bool(config.get("trajectory_use_global_mapper", True)),
        num_threads=int(config.get("trajectory_num_threads", 4)),
        verbose=bool(config.get("verbose", False)),
    )

    metrics: Dict[str, Any] = {}
    if result["status"] == "ok":
        metrics = {
            "num_frames_extracted": result["metadata"]["num_frames_extracted"],
            "num_frames_reconstructed": result["metadata"]["num_frames_reconstructed"],
            "reconstruction_ratio": result["metadata"]["reconstruction_ratio"],
            "num_3d_points": result["metadata"]["num_3d_points"],
        }

    return {
        "status": result["status"],
        "reason": result["reason"],
        "metrics": metrics,
        "details": result,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "trajectory_output"
        response = extract_camera_trajectory_from_video(
            video_path=video,
            output_dir=out_dir,
            verbose=True,
        )
        print("\n" + "=" * 72)
        print("结果:")
        print(json.dumps({k: v for k, v in response.items() if k != "trajectory"}, indent=2, ensure_ascii=False))
        print(f"轨迹帧数: {len(response['trajectory'])}")
    else:
        print("用法: python evaluation/extract_camera_trajectory.py <video_path> [output_dir]")
