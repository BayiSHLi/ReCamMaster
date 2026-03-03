from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
	from evaluation.utils.pose_utils import align_poses_umeyama, evaluate_camera_sequence
except ModuleNotFoundError:
	from utils.pose_utils import align_poses_umeyama, evaluate_camera_sequence


_ROW_PATTERN = re.compile(r"\[([^\]]+)\]")


def _parse_pose_row(row_text: str) -> np.ndarray:
	values = [float(x) for x in row_text.strip().split()]
	if len(values) != 4:
		raise ValueError(f"Expected 4 numbers in pose row, got: {row_text}")
	return np.asarray(values, dtype=np.float64)


def parse_camera_pose(pose_text: str) -> np.ndarray:
	"""
	Parse a pose string like:
	"[r11 r12 r13 0] [r21 r22 r23 0] [r31 r32 r33 0] [tx ty tz 1]"

	Returns world-to-camera 4x4 matrix.
	"""
	rows = _ROW_PATTERN.findall(pose_text)
	if len(rows) != 4:
		raise ValueError(f"Invalid camera pose text: {pose_text}")

	w2c = np.stack([_parse_pose_row(row) for row in rows], axis=0)
	return w2c


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
	"""Convert world-to-camera matrix to camera-to-world matrix."""
	R = w2c[:3, :3]
	t = w2c[3, :3]

	c2w = np.eye(4, dtype=np.float64)
	c2w[:3, :3] = R.T
	c2w[:3, 3] = -R.T @ t
	return c2w


def load_camera_sequence(camera_json: Path, camera_key: str = "cam01") -> np.ndarray:
	"""
	Load camera sequence from json into (T, 4, 4) c2w matrices.
	"""
	with camera_json.open("r", encoding="utf-8") as f:
		payload: Dict[str, Dict[str, str]] = json.load(f)

	frame_items = sorted(payload.items(), key=lambda item: int(item[0].replace("frame", "")))
	poses: List[np.ndarray] = []

	for frame_name, frame_data in frame_items:
		if camera_key not in frame_data:
			raise KeyError(f"Camera key '{camera_key}' missing in {frame_name} of {camera_json}")
		w2c = parse_camera_pose(frame_data[camera_key])
		poses.append(w2c_to_c2w(w2c))

	return np.stack(poses, axis=0)


def evaluate_camera_metric(
	gt_camera_json: str,
	pred_camera_json: str,
	camera_key: str = "cam01",
	align_center_trajectory: bool = True,
) -> Dict[str, float]:
	"""
	Compute camera metrics: RotErr / TransErr.
	"""
	gt_path = Path(gt_camera_json)
	pred_path = Path(pred_camera_json)

	gt_poses = load_camera_sequence(gt_path, camera_key=camera_key)
	pred_poses = load_camera_sequence(pred_path, camera_key=camera_key)

	length = min(len(gt_poses), len(pred_poses))
	if length == 0:
		raise ValueError("No valid poses found in camera sequences")

	gt_poses = gt_poses[:length]
	pred_poses = pred_poses[:length]

	if align_center_trajectory:
		gt_centers = gt_poses[:, :3, 3]
		pred_centers = pred_poses[:, :3, 3]
		scale, rotation, translation = align_poses_umeyama(gt_centers, pred_centers)

		aligned_pred = pred_poses.copy()
		aligned_pred[:, :3, 3] = (scale * (rotation @ pred_centers.T)).T + translation
		pred_poses = aligned_pred

	raw_result = evaluate_camera_sequence(gt_poses, pred_poses)
	return {
		"RotErr": float(raw_result["mean_RotErr"]),
		"TransErr": float(raw_result["mean_TransErr"]),
		"num_frames": int(length),
	}


def run(config: Dict) -> Dict:
	gt_camera_json = config.get("gt_camera_json")
	pred_camera_json = config.get("pred_camera_json")
	camera_key = config.get("camera_key", "cam01")
	align_center_trajectory = bool(config.get("align_camera_centers", False))

	if not gt_camera_json or not pred_camera_json:
		return {
			"status": "skipped",
			"reason": "Missing gt_camera_json or pred_camera_json",
			"metrics": {},
		}

	result = evaluate_camera_metric(
		gt_camera_json=gt_camera_json,
		pred_camera_json=pred_camera_json,
		camera_key=camera_key,
		align_center_trajectory=align_center_trajectory,
	)
	return {
		"status": "ok",
		"reason": "",
		"metrics": result,
	}
