from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from utils.pose_utils import (
	evaluate_camera_sequence,
	evaluate_camera_sequence_cameractrl_paper,
	rescale_translation_with_first_gap,
	to_relative_poses,
)
from extract_camera_trajectory_colmap import extract_camera_trajectory_from_video_colmap


_ROW_PATTERN = re.compile(r"\[([^\]]+)\]")
_CAM_VIDEO_PATTERN = "cam[0-9][0-9].mp4"


def _resolve_gt_camera_json_path(path_value: str) -> Path:
	raw = Path(path_value)
	eval_dir = Path(__file__).resolve().parent
	repo_root = eval_dir.parent

	candidates: List[Path] = []
	if raw.is_absolute():
		candidates.append(raw)
	else:
		candidates.extend([
			Path.cwd() / raw,
			repo_root / raw,
			eval_dir / raw,
			eval_dir / raw.name,
		])

	for candidate in candidates:
		if candidate.is_file():
			return candidate.resolve()

	searched = ", ".join(str(path) for path in candidates)
	raise FileNotFoundError(f"gt_camera_json not found: {path_value}. Searched: {searched}")


def _parse_pose_row(row_text: str) -> np.ndarray:
	values = [float(x) for x in row_text.strip().split()]
	if len(values) != 4:
		raise ValueError(f"Expected 4 numbers in pose row, got: {row_text}")
	return np.asarray(values, dtype=np.float64)


def parse_camera_pose(pose_text: str) -> np.ndarray:
	"""
	Parse a pose string like:
	"[r11 r12 r13 0] [r21 r22 r23 0] [r31 r32 r33 0] [tx ty tz 1]"

	Returns a raw 4x4 pose matrix as stored in json.
	"""
	rows = _ROW_PATTERN.findall(pose_text)
	if len(rows) != 4:
		raise ValueError(f"Invalid camera pose text: {pose_text}")

	w2c = np.stack([_parse_pose_row(row) for row in rows], axis=0)
	return w2c


def gt_pose_to_generation_c2w(raw_pose: np.ndarray) -> np.ndarray:
	"""
	Convert GT json pose into the same c2w convention used by generation.

	This mirrors evaluation/inference_webvid.py::_build_target_trajectories:
	1) transpose pose matrix,
	2) axis reorder [:, [1, 2, 0, 3]],
	3) flip Y axis,
	4) divide translation by 100.
	"""
	pose = np.asarray(raw_pose, dtype=np.float64)
	if pose.shape != (4, 4):
		raise ValueError(f"GT pose must be (4, 4), got {pose.shape}")

	c2w = pose.T.copy()
	c2w = c2w[:, [1, 2, 0, 3]]
	c2w[:3, 1] *= -1.0
	c2w[:3, 3] /= 100.0
	return c2w


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
	"""Convert world-to-camera matrix to camera-to-world matrix."""
	R = w2c[:3, :3]
	t = w2c[3, :3]

	c2w = np.eye(4, dtype=np.float64)
	c2w[:3, :3] = R.T
	c2w[:3, 3] = -R.T @ t
	return c2w


def _estimate_similarity_transform(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
	"""
	Estimate similarity transform mapping src -> dst with:
	dst ~= scale * (rotation @ src) + translation
	"""
	if src_points.shape != dst_points.shape:
		raise ValueError(f"Point shape mismatch: {src_points.shape} vs {dst_points.shape}")
	if src_points.ndim != 2 or src_points.shape[1] != 3:
		raise ValueError(f"Points must be (N,3), got {src_points.shape}")

	n = src_points.shape[0]
	if n == 0:
		return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

	src_mean = src_points.mean(axis=0)
	dst_mean = dst_points.mean(axis=0)
	src_centered = src_points - src_mean
	dst_centered = dst_points - dst_mean

	if n == 1:
		return 1.0, np.eye(3, dtype=np.float64), dst_mean - src_mean

	cov = (dst_centered.T @ src_centered) / float(n)
	u, s_vals, vt = np.linalg.svd(cov)

	diag = np.eye(3, dtype=np.float64)
	if np.linalg.det(u) * np.linalg.det(vt) < 0:
		diag[-1, -1] = -1.0

	rotation = u @ diag @ vt
	src_var = np.sum(src_centered ** 2) / float(n)
	if src_var < 1e-12:
		scale = 1.0
	else:
		scale = float(np.sum(s_vals * np.diag(diag)) / src_var)

	translation = dst_mean - scale * (rotation @ src_mean)
	return scale, rotation, translation


def _align_pred_relative_translation(gt_relative: np.ndarray, pred_relative: np.ndarray) -> Tuple[np.ndarray, float]:
	"""
	Align predicted relative trajectory translations to GT with a global Sim(3)-style fit.

	Only translation vectors are transformed to preserve the original rotation metric.
	"""
	if len(gt_relative) == 0 or len(pred_relative) == 0:
		return pred_relative, 1.0

	scale, rotation, translation = _estimate_similarity_transform(
		src_points=pred_relative[:, :3, 3],
		dst_points=gt_relative[:, :3, 3],
	)

	aligned = pred_relative.copy()
	aligned[:, :3, 3] = scale * (pred_relative[:, :3, 3] @ rotation.T) + translation
	return aligned, float(scale)


def load_camera_sequence(
	camera_json: Path,
	camera_key: str = "cam01",
	parse_mode: str = "pred_colmap",
) -> np.ndarray:
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
		raw_pose = parse_camera_pose(frame_data[camera_key])
		if parse_mode == "gt_generation":
			poses.append(gt_pose_to_generation_c2w(raw_pose))
		elif parse_mode == "pred_colmap":
			poses.append(w2c_to_c2w(raw_pose))
		else:
			raise ValueError(f"Unsupported parse_mode: {parse_mode}")

	return np.stack(poses, axis=0)


def evaluate_camera_metric(
	gt_camera_json: str,
	pred_camera_json: str,
	camera_key: str = "cam01",
) -> Dict[str, Any]:
	"""
	Compute camera metrics: RotErr / TransErr.

	Following CameraCtrl (Sec. 3.4 + Appendix D.5):
	1) Convert GT/Pred trajectories to relative poses by setting frame-0 pose as identity.
	2) Rescale generated trajectory translation by frame0-frame1 translation gap ratio.
	"""
	gt_path = Path(gt_camera_json)
	pred_path = Path(pred_camera_json)

	# Keep GT parsing strictly consistent with generation-time camera transform.
	gt_poses = load_camera_sequence(gt_path, camera_key=camera_key, parse_mode="gt_generation")
	# Pred poses are reconstructed COLMAP extrinsics and remain w2c->c2w converted.
	pred_poses = load_camera_sequence(pred_path, camera_key=camera_key, parse_mode="pred_colmap")

	length = min(len(gt_poses), len(pred_poses))
	if length == 0:
		raise ValueError("No valid poses found in camera sequences")

	gt_poses = gt_poses[:length]
	pred_poses = pred_poses[:length]

	gt_relative = to_relative_poses(gt_poses)
	pred_relative = to_relative_poses(pred_poses)
	pred_relative = rescale_translation_with_first_gap(gt_relative, pred_relative)

	raw_result = evaluate_camera_sequence(gt_relative, pred_relative)
	paper_result = evaluate_camera_sequence_cameractrl_paper(gt_relative, pred_relative)
	aligned_pred_relative, align_scale = _align_pred_relative_translation(gt_relative, to_relative_poses(pred_poses))
	aligned_result = evaluate_camera_sequence(gt_relative, aligned_pred_relative)
	aligned_paper_result = evaluate_camera_sequence_cameractrl_paper(gt_relative, aligned_pred_relative)
	return {
		"RotErr": float(raw_result["mean_RotErr"]),
		"TransErr": float(raw_result["mean_TransErr"]),
		"paper": {
			"method": "cameractrl_cami2v_style",
			"RotErr": float(paper_result["RotErr_rad_sum"]),
			"TransErr": float(paper_result["TransErr_sum"]),
			"RotErr_unit": "radian_sum",
			"TransErr_unit": "l2_sum",
		},
		"align": {
			"method": "sim3_translation_only",
			"RotErr": float(aligned_result["mean_RotErr"]),
			"TransErr": float(aligned_result["mean_TransErr"]),
			"paper": {
				"RotErr": float(aligned_paper_result["RotErr_rad_sum"]),
				"TransErr": float(aligned_paper_result["TransErr_sum"]),
				"RotErr_unit": "radian_sum",
				"TransErr_unit": "l2_sum",
			},
			"scale": float(align_scale),
		},
		"num_frames": int(length),
	}


def run(config: Dict) -> Dict:
	gt_camera_json = config.get("gt_camera_json")
	pred_camera_json = config.get("pred_camera_json")
	camera_key = config.get("camera_key", "cam01")

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
	)
	return {
		"status": "ok",
		"reason": "",
		"metrics": result,
	}


def _extract_frame_id(name: str) -> int:
	# Prefer COLMAP frame naming from extractor: frame_000123.jpg -> 123.
	# This avoids accidentally parsing unrelated digits from parent paths.
	stem = Path(name).stem
	match = re.search(r"frame_(\d+)$", stem)
	if match:
		return int(match.group(1))

	# Fallback: if naming deviates, use the last numeric chunk in the stem.
	fallback_matches = re.findall(r"(\d+)", stem)
	if fallback_matches:
		return int(fallback_matches[-1])
	return 10**9


def _format_pose_text(rotation: np.ndarray, translation: np.ndarray) -> str:
	r = np.asarray(rotation, dtype=np.float64)
	t = np.asarray(translation, dtype=np.float64).reshape(-1)
	if r.shape != (3, 3):
		raise ValueError(f"rotation matrix must be (3, 3), got {r.shape}")
	if t.shape[0] < 3:
		raise ValueError(f"translation vector must have >= 3 elements, got shape {t.shape}")

	rows = [
		f"[{r[0, 0]:.9g} {r[0, 1]:.9g} {r[0, 2]:.9g} 0]",
		f"[{r[1, 0]:.9g} {r[1, 1]:.9g} {r[1, 2]:.9g} 0]",
		f"[{r[2, 0]:.9g} {r[2, 1]:.9g} {r[2, 2]:.9g} 0]",
		f"[{t[0]:.9g} {t[1]:.9g} {t[2]:.9g} 1]",
	]
	return " ".join(rows)


def _build_aligned_eval_camera_json_pair(
	trajectory_payload: Dict[str, Any],
	gt_payload: Dict[str, Dict[str, str]],
	gt_camera_key: str,
	frame_skip: int,
	output_dir: Path,
) -> Tuple[Path, Path, Dict[str, Any]]:
	trajectory = trajectory_payload.get("trajectory", [])
	if not isinstance(trajectory, list) or len(trajectory) == 0:
		raise ValueError("trajectory payload has no valid trajectory entries")

	if frame_skip <= 0:
		raise ValueError(f"frame_skip must be > 0, got {frame_skip}")

	sorted_trajectory = sorted(
		trajectory,
		key=lambda item: _extract_frame_id(str(item.get("image_name", ""))),
	)

	matched_items: List[Tuple[int, int, str, str]] = []
	for item in sorted_trajectory:
		sampled_idx = _extract_frame_id(str(item.get("image_name", "")))
		if sampled_idx >= 10**9:
			continue

		gt_frame_idx = sampled_idx * frame_skip
		gt_frame_name = f"frame{gt_frame_idx}"
		gt_frame_data = gt_payload.get(gt_frame_name)
		if not gt_frame_data or gt_camera_key not in gt_frame_data:
			continue

		rotation = np.asarray(item.get("rotation_matrix"), dtype=np.float64)
		translation = np.asarray(item.get("translation"), dtype=np.float64)
		pred_pose_text = _format_pose_text(rotation, translation)
		gt_pose_text = gt_frame_data[gt_camera_key]
		matched_items.append((sampled_idx, gt_frame_idx, gt_pose_text, pred_pose_text))

	if not matched_items:
		raise ValueError(
			f"No aligned frames between trajectory and GT camera '{gt_camera_key}'. "
			f"Check frame_skip={frame_skip} and GT json coverage."
		)

	gt_eval: Dict[str, Dict[str, str]] = {}
	pred_eval: Dict[str, Dict[str, str]] = {}
	for _, gt_frame_idx, gt_pose_text, pred_pose_text in matched_items:
		# Keep original aligned frame index for traceability.
		frame_key = f"frame{gt_frame_idx}"
		gt_eval[frame_key] = {gt_camera_key: gt_pose_text}
		pred_eval[frame_key] = {gt_camera_key: pred_pose_text}

	output_dir.mkdir(parents=True, exist_ok=True)
	gt_output_json = output_dir / "gt_camera_for_eval.json"
	pred_output_json = output_dir / "pred_camera_for_eval.json"
	with gt_output_json.open("w", encoding="utf-8") as f:
		json.dump(gt_eval, f, indent=2, ensure_ascii=False)
	with pred_output_json.open("w", encoding="utf-8") as f:
		json.dump(pred_eval, f, indent=2, ensure_ascii=False)

	stats = {
		"aligned_pairs": len(matched_items),
		"sampled_frame_indices": [item[0] for item in matched_items],
		"gt_frame_indices": [item[1] for item in matched_items],
	}
	return gt_output_json, pred_output_json, stats


def _gt_camera_key_from_cam_video(cam_video_name: str) -> str:
	# Generated videos are expected to be named cam01.mp4 ... cam10.mp4.
	match = re.match(r"cam(\d{2})\.mp4$", cam_video_name)
	if not match:
		raise ValueError(f"Invalid generated camera video name (expect camNN.mp4): {cam_video_name}")
	return f"cam{int(match.group(1)):02d}"


def evaluate_video_dir_with_glomap(args: argparse.Namespace) -> Dict[str, Any]:
	video_dir = Path(args.generated_video_dir)
	if not video_dir.exists():
		raise FileNotFoundError(f"generated_video_dir not found: {video_dir}")
	if args.frame_skip <= 0:
		raise ValueError(f"--frame_skip must be > 0, got {args.frame_skip}")

	gt_path = _resolve_gt_camera_json_path(args.gt_camera_json)
	args.gt_camera_json = str(gt_path)
	with gt_path.open("r", encoding="utf-8") as f:
		gt_payload: Dict[str, Dict[str, str]] = json.load(f)

	cam_videos = sorted([path for path in video_dir.glob(_CAM_VIDEO_PATTERN) if path.is_file()])
	if not cam_videos:
		raise FileNotFoundError(f"No generated camera videos found in {video_dir} with pattern {_CAM_VIDEO_PATTERN}")

	results: List[Dict[str, Any]] = []
	rot_values: List[float] = []
	trans_values: List[float] = []
	align_rot_values: List[float] = []
	align_trans_values: List[float] = []
	paper_rot_values: List[float] = []
	paper_trans_values: List[float] = []
	align_paper_rot_values: List[float] = []
	align_paper_trans_values: List[float] = []

	for cam_video in cam_videos:
		gt_camera_key = _gt_camera_key_from_cam_video(cam_video.name)
		traj_out_dir = Path(args.trajectory_root) / cam_video.stem

		trajectory_result = extract_camera_trajectory_from_video_colmap(
			video_path=str(cam_video),
			output_dir=str(traj_out_dir),
			camera_model=args.camera_model,
			frame_skip=args.frame_skip,
			max_frames=(None if args.max_frames <= 0 else args.max_frames),
			resize_short=args.resize_short,
			matcher_type=args.matcher_type,
			use_global_mapper=not args.disable_global_mapper,
			num_threads=args.num_threads,
			deterministic=not args.trajectory_non_deterministic,
			random_seed=args.trajectory_random_seed,
			enable_retry=not args.disable_trajectory_retry,
			min_reconstruction_ratio=args.trajectory_min_reconstruction_ratio,
			verbose=args.verbose,
		)

		entry: Dict[str, Any] = {
			"camera_video": str(cam_video),
			"camera_name": cam_video.name,
			"gt_camera_key": gt_camera_key,
			"trajectory_output_dir": str(traj_out_dir),
			"trajectory_status": trajectory_result.get("status", "unknown"),
			"trajectory_reason": trajectory_result.get("reason", ""),
			"trajectory_num_frames_extracted": trajectory_result.get("metadata", {}).get("num_frames_extracted"),
			"trajectory_num_frames_reconstructed": trajectory_result.get("metadata", {}).get("num_frames_reconstructed"),
			"trajectory_mapper_backend": trajectory_result.get("metadata", {}).get("mapper_backend"),
			"trajectory_selected_attempt": trajectory_result.get("metadata", {}).get("selected_attempt_name"),
		}

		if trajectory_result.get("status") != "ok":
			entry["metric_status"] = "skipped"
			entry["metric_reason"] = "trajectory extraction failed"
			entry["metrics"] = {}
			results.append(entry)
			continue

		try:
			gt_eval_json, pred_eval_json, alignment_stats = _build_aligned_eval_camera_json_pair(
				trajectory_payload=trajectory_result,
				gt_payload=gt_payload,
				gt_camera_key=gt_camera_key,
				frame_skip=args.frame_skip,
				output_dir=traj_out_dir,
			)
			metrics = evaluate_camera_metric(
				gt_camera_json=str(gt_eval_json),
				pred_camera_json=str(pred_eval_json),
				camera_key=gt_camera_key,
			)
		except Exception as exc:  # noqa: BLE001
			entry["metric_status"] = "failed"
			entry["metric_reason"] = str(exc)
			entry["metrics"] = {}
			results.append(entry)
			continue

		entry["metric_status"] = "ok"
		entry["metric_reason"] = ""
		entry["gt_eval_camera_json"] = str(gt_eval_json)
		entry["pred_eval_camera_json"] = str(pred_eval_json)
		entry["aligned_pairs"] = alignment_stats["aligned_pairs"]
		entry["sampled_frame_indices"] = alignment_stats["sampled_frame_indices"]
		entry["gt_frame_indices"] = alignment_stats["gt_frame_indices"]
		entry["metrics"] = metrics
		results.append(entry)

		rot_values.append(float(metrics["RotErr"]))
		trans_values.append(float(metrics["TransErr"]))
		paper_metrics = metrics.get("paper", {}) if isinstance(metrics, dict) else {}
		if isinstance(paper_metrics, dict):
			paper_rot = paper_metrics.get("RotErr")
			paper_trans = paper_metrics.get("TransErr")
			if paper_rot is not None:
				paper_rot_values.append(float(paper_rot))
			if paper_trans is not None:
				paper_trans_values.append(float(paper_trans))
		if isinstance(metrics.get("align"), dict):
			align_rot = metrics["align"].get("RotErr")
			align_trans = metrics["align"].get("TransErr")
			if align_rot is not None:
				align_rot_values.append(float(align_rot))
			if align_trans is not None:
				align_trans_values.append(float(align_trans))
			align_paper_metrics = metrics["align"].get("paper")
			if isinstance(align_paper_metrics, dict):
				align_paper_rot = align_paper_metrics.get("RotErr")
				align_paper_trans = align_paper_metrics.get("TransErr")
				if align_paper_rot is not None:
					align_paper_rot_values.append(float(align_paper_rot))
				if align_paper_trans is not None:
					align_paper_trans_values.append(float(align_paper_trans))

	summary = {
		"num_generated_videos": len(cam_videos),
		"num_metric_ok": len(rot_values),
		"mean_RotErr": (float(np.mean(rot_values)) if rot_values else None),
		"mean_TransErr": (float(np.mean(trans_values)) if trans_values else None),
		"mean_paper_RotErr": (float(np.mean(paper_rot_values)) if paper_rot_values else None),
		"mean_paper_TransErr": (float(np.mean(paper_trans_values)) if paper_trans_values else None),
		"mean_align_RotErr": (float(np.mean(align_rot_values)) if align_rot_values else None),
		"mean_align_TransErr": (float(np.mean(align_trans_values)) if align_trans_values else None),
		"mean_align_paper_RotErr": (float(np.mean(align_paper_rot_values)) if align_paper_rot_values else None),
		"mean_align_paper_TransErr": (float(np.mean(align_paper_trans_values)) if align_paper_trans_values else None),
	}

	return {
		"generated_video_dir": str(video_dir),
		"gt_camera_json": args.gt_camera_json,
		"camera_pattern": _CAM_VIDEO_PATTERN,
		"summary": summary,
		"results": results,
	}


def _build_cli_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Evaluate camera metrics (RotErr/TransErr) on video_0 cam01..cam10 videos using GLOMAP/COLMAP trajectory extraction"
	)
	parser.add_argument("--generated_video_root", type=str, default="/mnt/hdd/dataset/webvid10m/outputs")
	parser.add_argument("--video_id", type=str, default="video_2", help="Video folder name under generated_video_root, e.g. video_0")
	parser.add_argument("--save_root", type=str, default="results/evaluation/trajectory_test", help="Root directory to save trajectory outputs and metric report")
	parser.add_argument("--gt_camera_json", type=str, default="evaluation/camera_extrinsics.json")

	# Trajectory extraction options.
	# Keep the same CLI argument name/schema, but default to glomap pipeline.
	parser.add_argument("--trajectory_backend", type=str, default="glomap", choices=["glomap"])
	parser.add_argument("--camera_model", type=str, default="PINHOLE")
	parser.add_argument("--frame_skip", type=int, default=4)
	parser.add_argument("--max_frames", type=int, default=81, help="<=0 means no frame limit")
	parser.add_argument("--resize_short", type=int, default=0)
	parser.add_argument("--matcher_type", type=str, default="sequential", choices=["sequential", "vocab_tree", "exhaustive"])
	parser.add_argument("--num_threads", type=int, default=4)
	parser.add_argument("--trajectory_non_deterministic", action="store_true", help="Disable deterministic trajectory extraction mode")
	parser.add_argument("--trajectory_random_seed", type=int, default=0, help="Base random seed for deterministic mapper and matching")
	parser.add_argument("--disable_trajectory_retry", action="store_true", help="Disable fallback attempts when initial reconstruction is weak")
	parser.add_argument("--trajectory_min_reconstruction_ratio", type=float, default=0.2, help="Target minimum reconstructed frame ratio before stopping retries")
	parser.add_argument("--disable_global_mapper", action="store_true", help="Compatibility flag; ignored when trajectory_backend is glomap/colmap")
	parser.add_argument("--verbose", action="store_true")
	return parser


def _fill_derived_paths(args: argparse.Namespace) -> argparse.Namespace:
	video_id = str(args.video_id)
	args.generated_video_dir = str(Path(args.generated_video_root) / video_id)
	args.trajectory_root = str(Path(args.save_root) / f"{video_id}_glomap")
	args.output_json = str(Path(args.save_root) / f"{video_id}_glomap_camera_metrics.json")
	return args


def main() -> None:
	parser = _build_cli_parser()
	args = _fill_derived_paths(parser.parse_args())

	report = evaluate_video_dir_with_glomap(args)
	report["config"] = vars(args)

	output_path = Path(args.output_json)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=False)

	print(f"[eval_camra] Wrote report to: {output_path}")
	print(f"[eval_camra] Mean RotErr: {report['summary']['mean_RotErr']}")
	print(f"[eval_camra] Mean TransErr: {report['summary']['mean_TransErr']}")
	print(f"[eval_camra] Mean paper RotErr(rad-sum): {report['summary']['mean_paper_RotErr']}")
	print(f"[eval_camra] Mean paper TransErr(l2-sum): {report['summary']['mean_paper_TransErr']}")
	print(f"[eval_camra] Mean align RotErr: {report['summary']['mean_align_RotErr']}")
	print(f"[eval_camra] Mean align TransErr: {report['summary']['mean_align_TransErr']}")
	print(f"[eval_camra] Mean align paper RotErr(rad-sum): {report['summary']['mean_align_paper_RotErr']}")
	print(f"[eval_camra] Mean align paper TransErr(l2-sum): {report['summary']['mean_align_paper_TransErr']}")


if __name__ == "__main__":
	main()
