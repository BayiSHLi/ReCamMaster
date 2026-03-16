from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def _log(msg: str) -> None:
    """Print a timestamped progress line; always flushed so background logs are readable."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


# Keep summary.numeric aligned with ReCamMaster Table1/Table2 core metrics.
_PRIMARY_METRIC_KEYS = {
    "camera.RotErr",
    "camera.TransErr",
    "matching.Mat.Pix_total",
    "clip.CLIP-V",
    "clip.CLIP-T",
    "clip.CLIP-F",
    "fvd.FID",
    "fvd.FVD",
    "fvd.FVD-V",
    "vbench.Aesthetic Quality",
    "vbench.Imaging Quality",
    "vbench.Temporal Flickering",
    "vbench.Motion Smoothness",
    "vbench.Subject Consistency",
    "vbench.Background Consistency",
}

_EXPECTED_CAMERA_FILES = [f"cam{i:02d}.mp4" for i in range(1, 11)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified WebVid evaluation pipeline: generation + camera/matching/CLIP/FVD/VBench metrics"
    )

    # Step 1: generation (reuses evaluation/inference_webvid.py)
    parser.add_argument("--data_root", type=str, default="/mnt/hdd/dataset/webvid", help="Dataset root with videos/ and metadata CSV")
    parser.add_argument("--metadata_csv", type=str, default="0000.csv", help="Metadata CSV file under data_root")
    parser.add_argument("--save_dir", type=str, default="", help="Generation output root. Empty means <data_root>/outputs")
    parser.add_argument("--ckpt_path", type=str, default="./models/ReCamMaster/checkpoints/step20000.ckpt")
    parser.add_argument("--max_samples", type=int, default=20, help="Max dataset samples to process, -1 for all")
    parser.add_argument("--start_sample_idx", type=int, default=0)
    parser.add_argument("--dataset_shuffle", action="store_true")
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--writer_threads", type=int, default=2)
    parser.add_argument("--show_progress", action="store_true", help="Show per-video denoising progress")
    parser.add_argument("--worker_log_dir", type=str, default="evaluation/logs/workers")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation and only run camera evaluation on existing outputs")

    # Step 2: evaluation settings
    parser.add_argument(
        "--selected_metrics",
        type=str,
        default="camera,matching,clip,fvd,vbench",
        help="Comma-separated metrics to run: camera,matching,clip,fvd,vbench",
    )
    parser.add_argument("--gt_camera_json", type=str, default="evaluation/camera_extrinsics.json")
    parser.add_argument(
        "--trajectory_save_root",
        type=str,
        default="results/evaluation",
        help="Root directory for per-video evaluation reports",
    )
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=81, help="<=0 means no limit")
    parser.add_argument("--resize_short", type=int, default=0)
    parser.add_argument("--matcher_type", type=str, default="sequential", choices=["sequential", "vocab_tree", "exhaustive"])
    parser.add_argument("--camera_model", type=str, default="PINHOLE")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--trajectory_non_deterministic", action="store_true")
    parser.add_argument("--trajectory_random_seed", type=int, default=0)
    parser.add_argument("--disable_trajectory_retry", action="store_true")
    parser.add_argument("--trajectory_min_reconstruction_ratio", type=float, default=0.2)
    parser.add_argument(
        "--skip_camera_extraction",
        action="store_true",
        help="Reuse saved per-camera trajectory json files and skip running GLOMAP/COLMAP extraction",
    )
    parser.add_argument("--max_eval_videos", type=int, default=-1, help="Max generated video folders to evaluate, -1 for all")
    parser.add_argument("--camera_glob", type=str, default="cam[0-9][0-9].mp4", help="Camera video glob in each generated video folder")

    # Pair metric configs (matching / CLIP / FVD / VBench)
    parser.add_argument("--matching_max_frames", type=int, default=32)
    parser.add_argument("--matching_frame_stride", type=int, default=4)
    parser.add_argument("--matching_resize_short", type=int, default=480)
    parser.add_argument("--matching_orb_nfeatures", type=int, default=1200)
    parser.add_argument("--matching_ratio_thresh", type=float, default=0.75)
    parser.add_argument("--matching_conf_thresh", type=float, default=0.5)
    parser.add_argument("--matching_ransac_reproj_threshold", type=float, default=3.0)

    parser.add_argument("--clip_frame_stride", type=int, default=8)
    parser.add_argument("--clip_max_frames", type=int, default=16)
    parser.add_argument("--clip_cache_dir", type=str, default="models")
    parser.add_argument("--clip_device", type=str, default="cuda")

    parser.add_argument("--fvd_frame_stride", type=int, default=4)
    parser.add_argument("--fvd_max_frames", type=int, default=48)
    parser.add_argument("--fvd_resize_short", type=int, default=256)
    parser.add_argument("--fvd_clip_len", type=int, default=16)
    parser.add_argument("--fvd_clip_stride", type=int, default=8)
    parser.add_argument("--fvd_view_clip_len", type=int, default=8)
    parser.add_argument("--fvd_device", type=str, default="cuda")
    parser.add_argument("--fvd_frame_batch_size", type=int, default=16)
    parser.add_argument("--fvd_video_batch_size", type=int, default=4)

    parser.add_argument("--vbench_enable_diffsynth_image_metrics", action="store_true", help="Enable diffsynth-backed Aesthetic/Imaging quality subset")
    parser.add_argument("--vbench_frame_stride", type=int, default=8)
    parser.add_argument("--vbench_max_frames", type=int, default=8)
    parser.add_argument("--vbench_cache_dir", type=str, default="models")
    parser.add_argument("--vbench_device", type=str, default="cuda")

    # Output
    parser.add_argument("--output_json", type=str, default="results/evaluation/pipeline_results.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--eval_show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars for evaluation stages",
    )
    return parser.parse_args()


def _run_command(command: List[str], cwd: Path) -> None:
    print("[cmd]", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def _iter_progress(
    iterable,
    *,
    enabled: bool,
    desc: str,
    total: int | None = None,
    leave: bool = True,
):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


def _resolve_existing_path(path_value: str, repo_root: Path, eval_dir: Path) -> Path:
    raw = Path(path_value)
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


def _normalize_video_key(raw_key: str) -> str:
    key = str(raw_key).strip()
    if key.lower().endswith(".mp4"):
        key = key[:-4]
    return key.rstrip("+").strip()


def _load_prompt_lookup(data_root: Path, metadata_csv: str) -> Dict[str, str]:
    csv_path = data_root / metadata_csv
    if not csv_path.is_file():
        return {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    text_column = "name" if "name" in rows[0] else ("text" if "text" in rows[0] else None)
    if text_column is None:
        return {}

    id_column = None
    for candidate in ("videoid", "video_id", "id", "file_name", "filename"):
        if candidate in rows[0]:
            id_column = candidate
            break

    if id_column is None:
        return {}

    id_to_text: Dict[str, str] = {}
    for row in rows:
        text = str(row.get(text_column, "")).strip()
        if not text:
            continue
        key = _normalize_video_key(str(row.get(id_column, "")))
        if key:
            id_to_text[key] = text
    return id_to_text


def _resolve_source_video_path(data_root: Path, video_id: str) -> Path | None:
    video_id = _normalize_video_key(video_id)
    for folder_name in ("videos", "video"):
        video_root = data_root / folder_name
        candidates = [
            video_root / f"{video_id}+.mp4",
            video_root / f"{video_id}.mp4",
        ]
        for path in candidates:
            if path.is_file():
                return path
    return None


def _parse_selected_metrics(raw: str) -> List[str]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("--selected_metrics cannot be empty")
    supported = {"camera", "matching", "clip", "fvd", "vbench"}
    unknown = [item for item in items if item not in supported]
    if unknown:
        raise ValueError(f"Unsupported metric names: {unknown}. Supported: {sorted(supported)}")
    return items


def _load_pair_metric_runners(eval_dir: Path) -> Dict[str, Any]:
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    runners: Dict[str, Any] = {}
    module_names = {
        "matching": "eval_matching",
        "clip": "eval_clip",
        "fvd": "eval_fvd",
        "vbench": "eval_vbench",
    }
    for metric_name, module_name in module_names.items():
        try:
            module = importlib.import_module(module_name)
            runner = getattr(module, "run", None)
            if callable(runner):
                runners[metric_name] = runner
        except Exception:
            continue
    return runners


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_numeric_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for key, value in metrics.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_numeric_metrics(value, flat_key))
            continue
        if _is_number(value):
            flattened[flat_key] = float(value)
    return flattened


def _update_numeric_aggregates(aggregated: Dict[str, List[float]], metric_name: str, payload: Dict[str, Any]) -> None:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return

    flattened = _flatten_numeric_metrics(metrics)
    for key, value in flattened.items():
        aggregated.setdefault(f"{metric_name}.{key}", []).append(float(value))


def _split_primary_auxiliary_aggregates(
    aggregated: Dict[str, List[float]],
) -> tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    primary: Dict[str, List[float]] = {}
    auxiliary: Dict[str, List[float]] = {}
    for key, values in aggregated.items():
        if key in _PRIMARY_METRIC_KEYS:
            primary[key] = values
        else:
            auxiliary[key] = values
    return primary, auxiliary


def _build_numeric_summary(aggregated: Dict[str, List[float]]) -> Dict[str, Dict[str, float | int]]:
    means: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for key, values in sorted(aggregated.items()):
        if not values:
            continue
        means[key] = float(statistics.fmean(values))
        counts[key] = int(len(values))
    return {
        "mean": means,
        "count": counts,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _build_pair_metric_config(args: argparse.Namespace, source_video: Path, generated_video: Path, text_prompt: str) -> Dict[str, Any]:
    return {
        "source_video": str(source_video),
        "target_video": str(generated_video),
        "generated_video": str(generated_video),
        "text_prompt": text_prompt,
        "matching_max_frames": args.matching_max_frames,
        "matching_frame_stride": args.matching_frame_stride,
        "matching_resize_short": args.matching_resize_short,
        "matching_orb_nfeatures": args.matching_orb_nfeatures,
        "matching_ratio_thresh": args.matching_ratio_thresh,
        "matching_conf_thresh": args.matching_conf_thresh,
        "matching_ransac_reproj_threshold": args.matching_ransac_reproj_threshold,
        "clip_frame_stride": args.clip_frame_stride,
        "clip_max_frames": args.clip_max_frames,
        "clip_cache_dir": args.clip_cache_dir,
        "clip_device": args.clip_device,
        "fvd_frame_stride": args.fvd_frame_stride,
        "fvd_max_frames": args.fvd_max_frames,
        "fvd_resize_short": args.fvd_resize_short,
        "fvd_clip_len": args.fvd_clip_len,
        "fvd_clip_stride": args.fvd_clip_stride,
        "fvd_view_clip_len": args.fvd_view_clip_len,
        "fvd_device": args.fvd_device,
        "fvd_frame_batch_size": args.fvd_frame_batch_size,
        "fvd_video_batch_size": args.fvd_video_batch_size,
        "vbench_enable_diffsynth_image_metrics": args.vbench_enable_diffsynth_image_metrics,
        "vbench_frame_stride": args.vbench_frame_stride,
        "vbench_max_frames": args.vbench_max_frames,
        "vbench_cache_dir": args.vbench_cache_dir,
        "vbench_device": args.vbench_device,
    }


def _build_generation_command(args: argparse.Namespace, eval_dir: Path, save_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(eval_dir / "inference_webvid.py"),
        "--data_root",
        args.data_root,
        "--metadata_csv",
        args.metadata_csv,
        "--save_dir",
        str(save_dir),
        "--ckpt_path",
        args.ckpt_path,
        "--max_samples",
        str(args.max_samples),
        "--start_sample_idx",
        str(args.start_sample_idx),
        "--dataset_seed",
        str(args.dataset_seed),
        "--num_frames",
        str(args.num_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--cfg_scale",
        str(args.cfg_scale),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--seed",
        str(args.seed),
        "--num_gpus",
        str(args.num_gpus),
        "--writer_threads",
        str(args.writer_threads),
        "--worker-log-dir",
        args.worker_log_dir,
    ]

    if args.dataset_shuffle:
        cmd.append("--dataset_shuffle")
    if args.gpu_ids.strip():
        cmd.extend(["--gpu_ids", args.gpu_ids])
    if args.show_progress:
        cmd.append("--show_progress")
    else:
        cmd.append("--no_show_progress")

    return cmd


def _build_camera_eval_command(
    args: argparse.Namespace,
    save_dir: Path,
    video_id: str,
    camera_save_root: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent) / "eval_camera.py"),
        "--generated_video_root",
        str(save_dir),
        "--video_id",
        video_id,
        "--save_root",
        str(camera_save_root),
        "--gt_camera_json",
        args.gt_camera_json,
        "--camera_model",
        args.camera_model,
        "--frame_skip",
        str(args.frame_skip),
        "--max_frames",
        str(args.max_frames),
        "--resize_short",
        str(args.resize_short),
        "--matcher_type",
        args.matcher_type,
        "--num_threads",
        str(args.num_threads),
        "--trajectory_random_seed",
        str(args.trajectory_random_seed),
        "--trajectory_min_reconstruction_ratio",
        str(args.trajectory_min_reconstruction_ratio),
    ]

    if args.trajectory_non_deterministic:
        cmd.append("--trajectory_non_deterministic")
    if args.disable_trajectory_retry:
        cmd.append("--disable_trajectory_retry")
    if args.skip_camera_extraction:
        cmd.append("--skip_camera_extraction")
    if args.verbose:
        cmd.append("--verbose")

    return cmd


def _discover_generated_video_dirs(save_dir: Path) -> List[Path]:
    if not save_dir.is_dir():
        return []

    dirs: List[Path] = []
    for child in sorted(save_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if any(child.glob("cam[0-9][0-9].mp4")):
            dirs.append(child)
    return dirs


def _discover_output_video_dirs_for_check(save_dir: Path) -> List[Path]:
    if not save_dir.is_dir():
        return []
    return [
        child
        for child in sorted(save_dir.iterdir(), key=lambda p: p.name)
        if child.is_dir() and not child.name.startswith(".")
    ]


def _check_camera_generation_completeness(save_dir: Path) -> Dict[str, Any]:
    video_dirs = _discover_output_video_dirs_for_check(save_dir)
    incomplete_items: List[Dict[str, Any]] = []

    for video_dir in video_dirs:
        missing: List[str] = []
        empty: List[str] = []
        for camera_file in _EXPECTED_CAMERA_FILES:
            file_path = video_dir / camera_file
            if not file_path.is_file():
                missing.append(camera_file)
                continue
            try:
                if file_path.stat().st_size <= 0:
                    empty.append(camera_file)
            except OSError:
                empty.append(camera_file)

        if missing or empty:
            incomplete_items.append(
                {
                    "video_id": video_dir.name,
                    "missing": missing,
                    "empty": empty,
                }
            )

    return {
        "checked_root": str(save_dir),
        "expected_camera_files": list(_EXPECTED_CAMERA_FILES),
        "num_checked_video_dirs": len(video_dirs),
        "num_incomplete_video_dirs": len(incomplete_items),
        "incomplete_video_dirs": incomplete_items,
    }


def _build_incomplete_lookup(skip_generation_check: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    if not skip_generation_check:
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}
    for item in skip_generation_check.get("incomplete_video_dirs", []):
        video_id = str(item.get("video_id", "")).strip()
        if not video_id:
            continue
        lookup[video_id] = {
            "video_id": video_id,
            "missing": list(item.get("missing", [])),
            "empty": list(item.get("empty", [])),
        }
    return lookup


def _load_video_report(camera_save_root: Path, video_id: str) -> Dict[str, Any]:
    report_path = camera_save_root / f"{video_id}_glomap_camera_metrics.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Camera report not found for {video_id}: {report_path}")
    with report_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["report_path"] = str(report_path)
    return payload


def _summarize_camera_reports(camera_reports: List[Dict[str, Any]], aggregated: Dict[str, List[float]], status_counter: Dict[str, Dict[str, int]]) -> None:
    for report in camera_reports:
        for item in report.get("results", []):
            status = str(item.get("metric_status", "unknown"))
            status_counter.setdefault("camera", {}).setdefault(status, 0)
            status_counter["camera"][status] += 1
            payload = {
                "metrics": item.get("metrics", {}),
            }
            if status == "ok":
                _update_numeric_aggregates(aggregated, "camera", payload)


def main() -> None:
    args = _parse_args()
    eval_dir = Path(__file__).resolve().parent
    repo_root = eval_dir.parent
    args.gt_camera_json = str(_resolve_existing_path(args.gt_camera_json, repo_root=repo_root, eval_dir=eval_dir))
    data_root = Path(args.data_root)
    selected_metrics = _parse_selected_metrics(args.selected_metrics)

    save_dir = Path(args.save_dir) if args.save_dir else data_root / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    skip_generation_check: Dict[str, Any] | None = None
    incomplete_lookup: Dict[str, Dict[str, Any]] = {}
    skipped_incomplete_samples: List[Dict[str, Any]] = []
    if args.skip_generation:
        skip_generation_check = _check_camera_generation_completeness(save_dir)
        incomplete_lookup = _build_incomplete_lookup(skip_generation_check)
        if skip_generation_check["num_incomplete_video_dirs"] > 0:
            examples = skip_generation_check["incomplete_video_dirs"][:5]
            details = "; ".join(
                f"{item['video_id']}(missing={item['missing']}, empty={item['empty']})"
                for item in examples
            )
            print(
                "[pipeline] skip_generation completeness check warning: expected cam01-cam10.mp4 in each output video folder. "
                f"Will skip incomplete folders: {skip_generation_check['num_incomplete_video_dirs']}/{skip_generation_check['num_checked_video_dirs']}. "
                f"Examples: {details}"
            )
        else:
            print(
                "[pipeline] skip_generation completeness check passed: "
                f"{skip_generation_check['num_checked_video_dirs']} video folders contain complete cam01-cam10.mp4 files"
            )

    results_root = Path(args.trajectory_save_root)
    results_root.mkdir(parents=True, exist_ok=True)
    prompt_lookup = _load_prompt_lookup(data_root=data_root, metadata_csv=args.metadata_csv)

    generation_cmd: List[str] | None = None
    if not args.skip_generation:
        generation_cmd = _build_generation_command(args, eval_dir=eval_dir, save_dir=save_dir)
        _run_command(generation_cmd, cwd=repo_root)

    video_dirs = _discover_generated_video_dirs(save_dir)
    if args.max_eval_videos > 0:
        video_dirs = video_dirs[: args.max_eval_videos]

    if args.skip_generation and incomplete_lookup:
        complete_video_dirs: List[Path] = []
        for video_dir in video_dirs:
            if video_dir.name in incomplete_lookup:
                item = incomplete_lookup[video_dir.name]
                skipped_incomplete_samples.append(
                    {
                        "video_id": item["video_id"],
                        "reason": "incomplete_generated_cameras",
                        "missing": item["missing"],
                        "empty": item["empty"],
                    }
                )
            else:
                complete_video_dirs.append(video_dir)
        video_dirs = complete_video_dirs

    if not video_dirs:
        if args.skip_generation:
            print(
                "[pipeline] no complete video folders found after skip_generation completeness filtering; "
                "writing empty evaluation report"
            )
        else:
            raise RuntimeError(f"No generated video folders found under {save_dir}")

    camera_reports: List[Dict[str, Any]] = []
    camera_report_paths: Dict[str, str] = {}
    if "camera" in selected_metrics:
        n_cam = len(video_dirs)
        _log(f"[camera] Starting camera evaluation for {n_cam} video(s)")
        for cam_idx, video_dir in enumerate(_iter_progress(
            video_dirs,
            enabled=args.eval_show_progress,
            desc="camera videos",
            total=n_cam,
        ), start=1):
            video_id = video_dir.name
            _log(f"[camera] [{cam_idx}/{n_cam}] Processing: {video_id}")
            video_result_dir = results_root / video_id
            camera_save_root = video_result_dir / "camera"
            cmd = _build_camera_eval_command(
                args,
                save_dir=save_dir,
                video_id=video_id,
                camera_save_root=camera_save_root,
            )
            try:
                _run_command(cmd, cwd=repo_root)
                report = _load_video_report(camera_save_root=camera_save_root, video_id=video_id)
                camera_reports.append(report)
                camera_report_paths[video_id] = report["report_path"]
                _log(f"[camera] [{cam_idx}/{n_cam}] Done:    {video_id}")
            except Exception as exc:  # noqa: BLE001
                _log(f"[camera] [{cam_idx}/{n_cam}] FAILED:  {video_id} -- {exc}")
        _log(f"[camera] Finished camera evaluation ({len(camera_reports)}/{n_cam} succeeded)")

    pair_metric_names = [name for name in selected_metrics if name != "camera"]
    pair_runners = _load_pair_metric_runners(eval_dir=eval_dir)

    pair_report_paths: Dict[str, str] = {}
    aggregated_values: Dict[str, List[float]] = {}
    status_counter: Dict[str, Dict[str, int]] = {}

    _summarize_camera_reports(camera_reports=camera_reports, aggregated=aggregated_values, status_counter=status_counter)

    if pair_metric_names:
        n_pair = len(video_dirs)
        _log(f"[pair] Starting pair-metric evaluation ({','.join(pair_metric_names)}) for {n_pair} video(s)")
        for pair_idx, video_dir in enumerate(_iter_progress(
            video_dirs,
            enabled=args.eval_show_progress,
            desc="pair-metric videos",
            total=n_pair,
        ), start=1):
            video_id = video_dir.name
            _log(f"[pair] [{pair_idx}/{n_pair}] Processing: {video_id}")
            source_video = _resolve_source_video_path(data_root=data_root, video_id=video_id)
            video_result_dir = results_root / video_id
            pair_report_path = video_result_dir / "pair_metrics.json"
            if source_video is None:
                _log(f"[pair] [{pair_idx}/{n_pair}] SKIP (no source video): {video_id}")
                pair_payload = {
                    "video_id": video_id,
                    "status": "skipped",
                    "reason": f"source video not found under {data_root}/videos or {data_root}/video",
                    "pairs": [],
                }
                _write_json(pair_report_path, pair_payload)
                pair_report_paths[video_id] = str(pair_report_path)
                continue

            text_prompt = prompt_lookup.get(_normalize_video_key(video_id), "")
            cam_videos = sorted([path for path in video_dir.glob(args.camera_glob) if path.is_file()], key=lambda p: p.name)
            n_cam_videos = len(cam_videos)

            per_video_pairs: List[Dict[str, Any]] = []
            for cv_idx, cam_video in enumerate(_iter_progress(
                cam_videos,
                enabled=args.eval_show_progress,
                desc=f"{video_id} camera pairs",
                total=n_cam_videos,
                leave=False,
            ), start=1):
                _log(f"[pair] [{pair_idx}/{n_pair}] {video_id}  cam [{cv_idx}/{n_cam_videos}] {cam_video.name}")
                metric_config = _build_pair_metric_config(
                    args=args,
                    source_video=source_video,
                    generated_video=cam_video,
                    text_prompt=text_prompt,
                )
                metrics_payload: Dict[str, Any] = {}

                for metric_name in _iter_progress(
                    pair_metric_names,
                    enabled=args.eval_show_progress,
                    desc=f"{video_id}/{cam_video.name} metrics",
                    total=len(pair_metric_names),
                    leave=False,
                ):
                    runner = pair_runners.get(metric_name)
                    if runner is None:
                        payload = {
                            "status": "pending",
                            "reason": f"Runner not found for metric '{metric_name}'",
                            "metrics": {},
                        }
                    else:
                        try:
                            payload = runner(metric_config)
                        except Exception as exc:  # noqa: BLE001
                            payload = {
                                "status": "failed",
                                "reason": f"Unhandled exception: {exc}",
                                "metrics": {},
                            }

                    status = str(payload.get("status", "unknown"))
                    status_counter.setdefault(metric_name, {}).setdefault(status, 0)
                    status_counter[metric_name][status] += 1
                    if status == "ok":
                        _update_numeric_aggregates(aggregated_values, metric_name, payload)
                    else:
                        _log(f"[pair] [{pair_idx}/{n_pair}] {video_id}/{cam_video.name}  {metric_name}: {status}")
                    metrics_payload[metric_name] = payload

                per_video_pairs.append(
                    {
                        "camera_video": str(cam_video),
                        "camera_name": cam_video.name,
                        "source_video": str(source_video),
                        "metrics": metrics_payload,
                    }
                )

            pair_payload = {
                "video_id": video_id,
                "status": "ok",
                "reason": "",
                "pairs": per_video_pairs,
            }
            _write_json(pair_report_path, pair_payload)
            pair_report_paths[video_id] = str(pair_report_path)
            _log(f"[pair] [{pair_idx}/{n_pair}] Done:    {video_id} ({n_cam_videos} cameras)")
        _log(f"[pair] Finished pair-metric evaluation ({n_pair} video(s))")

    primary_aggregated, auxiliary_aggregated = _split_primary_auxiliary_aggregates(aggregated_values)

    summary = {
        "num_video_dirs": len(video_dirs),
        "numeric": _build_numeric_summary(primary_aggregated),
        "auxiliary": _build_numeric_summary(auxiliary_aggregated),
        "status_counter": status_counter,
    }

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            **vars(args),
            "save_dir": str(save_dir),
            "trajectory_save_root": str(results_root),
            "selected_metrics": selected_metrics,
        },
        "generation": {
            "skipped": bool(args.skip_generation),
            "command": (generation_cmd or []),
        },
        "skip_generation_check": skip_generation_check,
        "skipped_samples": {
            "num_skipped_incomplete": len(skipped_incomplete_samples),
            "incomplete_samples": skipped_incomplete_samples,
        },
        "summary": summary,
        "report_paths": {
            "camera_reports": camera_report_paths,
            "pair_metric_reports": pair_report_paths,
        },
    }

    output_path = Path(args.output_json)
    _write_json(output_path, output)

    _log(f"[pipeline] Wrote report: {output_path}")
    _log(f"[pipeline] num_video_dirs: {summary['num_video_dirs']}")
    _log(f"[pipeline] selected_metrics: {selected_metrics}")
    primary_means = summary["numeric"].get("mean", {})
    for k, v in sorted(primary_means.items()):
        _log(f"[pipeline] {k}: {v:.6f}")


if __name__ == "__main__":
    main()
