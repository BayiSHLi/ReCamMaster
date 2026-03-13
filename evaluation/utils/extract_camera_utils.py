from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def safe_set_attr(obj: Any, name: str, value: Any) -> None:
    if obj is not None and hasattr(obj, name):
        setattr(obj, name, value)


def resolve_attr_or_call(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    if callable(value):
        return value()
    return value


def to_list(value: Any) -> List[float]:
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


def camera_model_name(camera: Any) -> str:
    model_name = resolve_attr_or_call(camera, "model_name", None)
    if model_name is not None:
        return str(model_name)

    model = resolve_attr_or_call(camera, "model", None)
    if model is None:
        return "UNKNOWN"
    if hasattr(model, "name"):
        return str(model.name)
    return str(model)


def image_pose(image: Any) -> tuple[np.ndarray, np.ndarray, List[float]]:
    if hasattr(image, "qvec2rotmat") and hasattr(image, "tvec"):
        rotation = image.qvec2rotmat()
        translation = resolve_attr_or_call(image, "tvec", None)
        qvec = to_list(resolve_attr_or_call(image, "qvec", None))
        return np.asarray(rotation), np.asarray(translation), qvec

    cam_from_world = resolve_attr_or_call(image, "cam_from_world", None)
    if cam_from_world is not None:
        rotation_obj = resolve_attr_or_call(cam_from_world, "rotation", None)
        translation_obj = resolve_attr_or_call(cam_from_world, "translation", None)
        if rotation_obj is None or translation_obj is None:
            raise AttributeError("cam_from_world does not provide rotation/translation")

        rotation = resolve_attr_or_call(rotation_obj, "matrix", None)
        quaternion = resolve_attr_or_call(rotation_obj, "quat", None)
        if rotation is None:
            raise AttributeError("rotation object does not provide matrix")

        return np.asarray(rotation), np.asarray(translation_obj), to_list(quaternion)

    raise AttributeError("Unable to read pose from image object")


def extract_frames_from_video(
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


def count_reconstruction_stats(reconstruction: Any) -> tuple[int, int]:
    images = dict(resolve_attr_or_call(reconstruction, "images", {}))
    points3d = dict(resolve_attr_or_call(reconstruction, "points3D", {}))
    return len(images), len(points3d)


def save_camera_trajectory_json(output_dir: Path, payload: Dict[str, Any]) -> Path:
    output_json = output_dir / "camera_trajectory.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_json


def to_int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def first_camera_id(cameras: Dict[Any, Any]) -> Any:
    keys = list(cameras.keys())
    return min(keys, key=lambda key: (to_int_or_default(key, 10**9), str(key)))


def build_ok_result_from_colmap_entities(
    num_frames_extracted: int,
    cameras: Dict[Any, Any],
    images: Dict[Any, Any],
    points3d: Optional[Dict[Any, Any]],
    metadata_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    if len(cameras) == 0:
        return {
            "status": "failed",
            "reason": "No cameras in reconstruction",
            "metadata": {"num_frames_extracted": int(num_frames_extracted)},
            "camera_intrinsics": {},
            "trajectory": [],
        }

    first_camera = cameras[first_camera_id(cameras)]
    intrinsics = {
        "model": camera_model_name(first_camera),
        "width": int(resolve_attr_or_call(first_camera, "width", 0)),
        "height": int(resolve_attr_or_call(first_camera, "height", 0)),
        "params": to_list(resolve_attr_or_call(first_camera, "params", [])),
    }

    image_items = sorted(
        images.items(),
        key=lambda item: str(resolve_attr_or_call(item[1], "name", "")),
    )
    trajectory: List[Dict[str, Any]] = []
    for image_id, image in image_items:
        rotation, translation, quaternion = image_pose(image)
        trajectory.append(
            {
                "frame_id": len(trajectory),
                "image_id": to_int_or_default(image_id, len(trajectory) + 1),
                "image_name": str(resolve_attr_or_call(image, "name", "")),
                "rotation_matrix": rotation.tolist(),
                "translation": translation.tolist(),
                "quaternion": quaternion,
            }
        )

    points3d = points3d or {}
    metadata = {
        "num_frames_extracted": int(num_frames_extracted),
        "num_frames_reconstructed": int(len(trajectory)),
        "reconstruction_ratio": (
            float(len(trajectory) / num_frames_extracted) if num_frames_extracted > 0 else 0.0
        ),
        "num_3d_points": int(len(points3d)),
    }
    metadata.update(metadata_overrides)

    return {
        "status": "ok",
        "reason": "",
        "metadata": metadata,
        "camera_intrinsics": intrinsics,
        "trajectory": trajectory,
    }
