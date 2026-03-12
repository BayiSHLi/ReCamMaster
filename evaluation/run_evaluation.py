from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from evaluation import eval_camera

try:
	from evaluation import eval_clip, eval_fvd, eval_matching
except ModuleNotFoundError:
	import evaluation.eval_camera as eval_camera  # type: ignore
	import eval_clip  # type: ignore
	import eval_fvd  # type: ignore
	import eval_matching  # type: ignore


MetricRunner = Callable[[Dict[str, Any]], Dict[str, Any]]

_CAM_VIDEO_PATTERN = re.compile(r"^cam_(\d+)\.mp4$")
_VIDEO_DIR_PATTERN = re.compile(r"^video_(\d+)$")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate generated WebVid outputs with selected metrics")
	parser.add_argument(
		"--outputs_root",
		type=str,
		default="/mnt/hdd/dataset/webvid/outputs",
		help="Root dir containing generated outputs, e.g. outputs/video_0/cam_00.mp4",
	)
	parser.add_argument(
		"--output_json",
		type=str,
		default="evaluation/results_webvid_eval.json",
		help="Path to write merged evaluation results",
	)
	parser.add_argument(
		"--selected_metrics",
		type=str,
		default="matching,clip,fvd",
		help="Comma-separated metric names: matching,clip,fvd,camera",
	)
	parser.add_argument("--start_video_idx", type=int, default=0, help="Only evaluate video_N with N >= start_video_idx")
	parser.add_argument("--start_sample_idx", type=int, default=None, help=argparse.SUPPRESS)
	parser.add_argument("--max_videos", type=int, default=-1, help="Max number of video folders to evaluate, -1 for all")
	parser.add_argument("--max_pairs", type=int, default=-1, help="Max number of (video,cam) pairs to evaluate, -1 for all")
	parser.add_argument("--original_name", type=str, default="original.mp4", help="Original/source video filename in each folder")
	parser.add_argument("--camera_glob", type=str, default="cam_*.mp4", help="Glob pattern for generated camera videos")
	parser.add_argument("--strict", action="store_true", help="Fail when expected files are missing instead of skipping")
	parser.add_argument("--worker-log-dir", type=str, default="", help=argparse.SUPPRESS)

	# Camera metric config
	parser.add_argument("--gt_camera_json", type=str, default="", help="GT camera json path for camera metric")
	parser.add_argument("--pred_camera_json", type=str, default="", help="Pred camera json path for camera metric")
	parser.add_argument("--camera_key", type=str, default="cam01", help="Camera key name in camera json files")

	# Matching metric config
	parser.add_argument("--matching_max_frames", type=int, default=32)
	parser.add_argument("--matching_frame_stride", type=int, default=4)
	parser.add_argument("--matching_resize_short", type=int, default=480)
	parser.add_argument("--matching_orb_nfeatures", type=int, default=1200)
	parser.add_argument("--matching_ratio_thresh", type=float, default=0.75)
	parser.add_argument("--matching_conf_thresh", type=float, default=0.5)
	parser.add_argument("--matching_ransac_reproj_threshold", type=float, default=3.0)

	args = parser.parse_args()
	if args.start_sample_idx is not None:
		args.start_video_idx = args.start_sample_idx
	if args.start_video_idx < 0:
		raise ValueError(f"--start_video_idx must be >= 0, got {args.start_video_idx}")
	if args.max_videos == 0 or args.max_pairs == 0:
		raise ValueError("--max_videos/--max_pairs cannot be 0, use -1 for no limit")
	return args


def _get_metric_runners() -> Dict[str, MetricRunner]:
	return {
		"matching": eval_matching.run,
		"clip": eval_clip.run,
		"fvd": eval_fvd.run,
		"camera": eval_camera.run,
	}


def _parse_selected_metrics(raw_metrics: str, runners: Dict[str, MetricRunner]) -> List[str]:
	items = [item.strip().lower() for item in raw_metrics.split(",") if item.strip()]
	if not items:
		raise ValueError("--selected_metrics is empty")
	unknown = [name for name in items if name not in runners]
	if unknown:
		supported = ", ".join(sorted(runners.keys()))
		raise ValueError(f"Unsupported metrics: {unknown}. Supported: {supported}")
	return items


def _parse_video_idx(video_dir: Path) -> float:
	match = _VIDEO_DIR_PATTERN.match(video_dir.name)
	if not match:
		return math.inf
	return int(match.group(1))


def _parse_cam_idx(cam_file: Path) -> Tuple[int, str]:
	match = _CAM_VIDEO_PATTERN.match(cam_file.name)
	if not match:
		return (10**9, cam_file.name)
	return (int(match.group(1)), cam_file.name)


def _discover_video_dirs(outputs_root: Path, start_video_idx: int) -> List[Path]:
	video_dirs = [path for path in outputs_root.iterdir() if path.is_dir() and _VIDEO_DIR_PATTERN.match(path.name)]
	video_dirs.sort(key=_parse_video_idx)
	return [path for path in video_dirs if _parse_video_idx(path) >= start_video_idx]


def _is_number(value: Any) -> bool:
	return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _build_metric_config(args: argparse.Namespace, original_video: Path, generated_video: Path) -> Dict[str, Any]:
	return {
		"source_video": str(original_video),
		"target_video": str(generated_video),
		"generated_video": str(generated_video),
		"gt_camera_json": args.gt_camera_json,
		"pred_camera_json": args.pred_camera_json,
		"camera_key": args.camera_key,
		"matching_max_frames": args.matching_max_frames,
		"matching_frame_stride": args.matching_frame_stride,
		"matching_resize_short": args.matching_resize_short,
		"matching_orb_nfeatures": args.matching_orb_nfeatures,
		"matching_ratio_thresh": args.matching_ratio_thresh,
		"matching_conf_thresh": args.matching_conf_thresh,
		"matching_ransac_reproj_threshold": args.matching_ransac_reproj_threshold,
	}


def _run_metrics_for_pair(
	selected_metrics: List[str],
	runners: Dict[str, MetricRunner],
	metric_config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
	results: Dict[str, Dict[str, Any]] = {}
	for metric_name in selected_metrics:
		runner = runners[metric_name]
		try:
			result = runner(metric_config)
		except Exception as exc:  # noqa: BLE001
			result = {
				"status": "failed",
				"reason": f"Unhandled exception: {exc}",
				"metrics": {},
			}
		results[metric_name] = result
	return results


def _update_aggregates(
	aggregated_values: Dict[str, List[float]],
	status_counter: Dict[str, Dict[str, int]],
	metrics_result: Dict[str, Dict[str, Any]],
) -> None:
	for metric_name, payload in metrics_result.items():
		status = str(payload.get("status", "unknown"))
		counter = status_counter.setdefault(metric_name, {})
		counter[status] = counter.get(status, 0) + 1

		if status != "ok":
			continue

		metrics = payload.get("metrics", {})
		if not isinstance(metrics, dict):
			continue

		for key, value in metrics.items():
			if _is_number(value):
				agg_key = f"{metric_name}.{key}"
				aggregated_values.setdefault(agg_key, []).append(float(value))


def _build_summary(aggregated_values: Dict[str, List[float]]) -> Dict[str, float | int]:
	summary: Dict[str, float | int] = {}
	for key, values in sorted(aggregated_values.items()):
		if not values:
			continue
		summary[f"{key}.mean"] = float(sum(values) / len(values))
		summary[f"{key}.count"] = int(len(values))
	return summary


def main() -> None:
	args = _parse_args()
	outputs_root = Path(args.outputs_root)
	if not outputs_root.exists():
		raise FileNotFoundError(f"outputs_root not found: {outputs_root}")

	runners = _get_metric_runners()
	selected_metrics = _parse_selected_metrics(args.selected_metrics, runners)

	video_dirs = _discover_video_dirs(outputs_root, start_video_idx=args.start_video_idx)
	if args.max_videos > 0:
		video_dirs = video_dirs[: args.max_videos]

	aggregated_values: Dict[str, List[float]] = {}
	status_counter: Dict[str, Dict[str, int]] = {}

	results: List[Dict[str, Any]] = []
	processed_pair_count = 0
	evaluated_pair_count = 0
	skipped_pair_count = 0

	for video_dir in video_dirs:
		video_result: Dict[str, Any] = {
			"video_dir": str(video_dir),
			"video_name": video_dir.name,
			"video_index": _parse_video_idx(video_dir),
			"pairs": [],
			"warnings": [],
		}

		original_video = video_dir / args.original_name
		if not original_video.exists():
			msg = f"missing original video: {original_video}"
			if args.strict:
				raise FileNotFoundError(msg)
			video_result["warnings"].append(msg)
			results.append(video_result)
			continue

		camera_videos = [path for path in video_dir.glob(args.camera_glob) if path.is_file()]
		camera_videos.sort(key=_parse_cam_idx)

		if not camera_videos:
			msg = f"no camera videos found with glob '{args.camera_glob}'"
			if args.strict:
				raise FileNotFoundError(f"{msg} under {video_dir}")
			video_result["warnings"].append(msg)
			results.append(video_result)
			continue

		for cam_video in camera_videos:
			if args.max_pairs > 0 and processed_pair_count >= args.max_pairs:
				break

			processed_pair_count += 1

			metric_config = _build_metric_config(args, original_video=original_video, generated_video=cam_video)
			metrics_result = _run_metrics_for_pair(
				selected_metrics=selected_metrics,
				runners=runners,
				metric_config=metric_config,
			)

			pair_has_ok_metric = any(payload.get("status") == "ok" for payload in metrics_result.values())
			if pair_has_ok_metric:
				evaluated_pair_count += 1
			else:
				skipped_pair_count += 1

			_update_aggregates(
				aggregated_values=aggregated_values,
				status_counter=status_counter,
				metrics_result=metrics_result,
			)

			video_result["pairs"].append(
				{
					"camera_video": str(cam_video),
					"camera_name": cam_video.name,
					"original_video": str(original_video),
					"metrics": metrics_result,
				}
			)

		results.append(video_result)

		if args.max_pairs > 0 and processed_pair_count >= args.max_pairs:
			break

	summary = _build_summary(aggregated_values)
	report: Dict[str, Any] = {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"selected_metrics": selected_metrics,
		"outputs_root": str(outputs_root),
		"config": {
			"start_video_idx": args.start_video_idx,
			"max_videos": args.max_videos,
			"max_pairs": args.max_pairs,
			"original_name": args.original_name,
			"camera_glob": args.camera_glob,
		},
		"counts": {
			"discovered_videos": len(video_dirs),
			"videos_with_results": len(results),
			"processed_pairs": processed_pair_count,
			"evaluated_pairs": evaluated_pair_count,
			"skipped_pairs": skipped_pair_count,
		},
		"status_counter": status_counter,
		"summary": summary,
		"results": results,
	}

	output_path = Path(args.output_json)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=False)

	print(f"[run_evaluation] Wrote report: {output_path}")
	print(f"[run_evaluation] Evaluated pairs: {evaluated_pair_count}")
	print(f"[run_evaluation] Selected metrics: {selected_metrics}")


if __name__ == "__main__":
	main()
