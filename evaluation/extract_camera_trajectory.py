"""
Backward-compatible trajectory extraction entry.

Use these dedicated scripts for debugging:
- evaluation/extract_camera_trajectory_colmap.py
- evaluation/extract_camera_trajectory_flowmap.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from evaluation.extract_camera_trajectory_colmap import (
        extract_camera_trajectory_from_video_colmap,
    )
    from evaluation.extract_camera_trajectory_common import as_bool
    from evaluation.extract_camera_trajectory_flowmap import (
        extract_camera_trajectory_from_video_flowmap,
    )
except ModuleNotFoundError:
    from extract_camera_trajectory_colmap import extract_camera_trajectory_from_video_colmap
    from extract_camera_trajectory_common import as_bool
    from extract_camera_trajectory_flowmap import extract_camera_trajectory_from_video_flowmap


def extract_camera_trajectory_from_video(
    video_path: str,
    output_dir: str,
    trajectory_backend: str = "glomap",
    camera_model: str = "PINHOLE",
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    resize_short: int = 0,
    matcher_type: str = "sequential",
    use_global_mapper: bool = True,
    num_threads: int = 4,
    deterministic: bool = True,
    random_seed: int = 0,
    enable_retry: bool = True,
    min_reconstruction_ratio: float = 0.2,
    flowmap_repo_dir: str = "",
    flowmap_max_steps: int = 500,
    flowmap_checkpoint: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    backend = str(trajectory_backend).strip().lower()

    if backend == "flowmap":
        return extract_camera_trajectory_from_video_flowmap(
            video_path=video_path,
            output_dir=output_dir,
            frame_skip=frame_skip,
            max_frames=max_frames,
            resize_short=resize_short,
            flowmap_repo_dir=flowmap_repo_dir,
            flowmap_max_steps=flowmap_max_steps,
            flowmap_checkpoint=flowmap_checkpoint,
            verbose=verbose,
        )

    if backend in {"pycolmap", "colmap", "glomap"}:
        return extract_camera_trajectory_from_video_colmap(
            video_path=video_path,
            output_dir=output_dir,
            camera_model=camera_model,
            frame_skip=frame_skip,
            max_frames=max_frames,
            resize_short=resize_short,
            matcher_type=matcher_type,
            use_global_mapper=use_global_mapper,
            num_threads=num_threads,
            deterministic=deterministic,
            random_seed=random_seed,
            enable_retry=enable_retry,
            min_reconstruction_ratio=min_reconstruction_ratio,
            verbose=verbose,
        )

    return {
        "status": "failed",
        "reason": f"Unsupported trajectory_backend: {trajectory_backend}",
        "metadata": {"supported_backends": ["pycolmap", "flowmap"]},
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
        trajectory_backend=config.get("trajectory_backend", "pycolmap"),
        camera_model=config.get("camera_model", "PINHOLE"),
        frame_skip=int(config.get("trajectory_frame_skip", 1)),
        max_frames=(
            int(config.get("trajectory_max_frames"))
            if config.get("trajectory_max_frames")
            else None
        ),
        resize_short=int(config.get("trajectory_resize_short", 0)),
        matcher_type=config.get("trajectory_matcher", "sequential"),
        use_global_mapper=as_bool(config.get("trajectory_use_global_mapper", True), default=True),
        num_threads=int(config.get("trajectory_num_threads", 4)),
        deterministic=as_bool(config.get("trajectory_deterministic", True), default=True),
        random_seed=int(config.get("trajectory_random_seed", 0)),
        enable_retry=as_bool(config.get("trajectory_retry", True), default=True),
        min_reconstruction_ratio=float(config.get("trajectory_min_reconstruction_ratio", 0.2)),
        flowmap_repo_dir=str(config.get("trajectory_flowmap_repo_dir", "")),
        flowmap_max_steps=int(config.get("trajectory_flowmap_max_steps", 500)),
        flowmap_checkpoint=str(config.get("trajectory_flowmap_checkpoint", "")),
        verbose=as_bool(config.get("verbose", False), default=False),
    )

    metrics: Dict[str, Any] = {}
    if result["status"] == "ok":
        metadata = result.get("metadata", {})
        metrics = {
            "num_frames_extracted": metadata.get("num_frames_extracted"),
            "num_frames_reconstructed": metadata.get("num_frames_reconstructed"),
            "reconstruction_ratio": metadata.get("reconstruction_ratio"),
            "num_3d_points": metadata.get("num_3d_points"),
        }

    return {
        "status": result["status"],
        "reason": result["reason"],
        "metrics": metrics,
        "details": result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract camera trajectory")
    parser.add_argument("video_path", type=str)
    parser.add_argument("output_dir", type=str, nargs="?", default="trajectory_output")
    parser.add_argument("--trajectory_backend", type=str, default="pycolmap", choices=["pycolmap", "flowmap"])
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="<=0 means no frame limit")
    parser.add_argument("--resize_short", type=int, default=0)
    parser.add_argument("--flowmap_repo_dir", type=str, default="")
    parser.add_argument("--flowmap_max_steps", type=int, default=500)
    parser.add_argument("--flowmap_checkpoint", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    response = extract_camera_trajectory_from_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        trajectory_backend=args.trajectory_backend,
        frame_skip=args.frame_skip,
        max_frames=(None if args.max_frames <= 0 else args.max_frames),
        resize_short=args.resize_short,
        flowmap_repo_dir=args.flowmap_repo_dir,
        flowmap_max_steps=args.flowmap_max_steps,
        flowmap_checkpoint=args.flowmap_checkpoint,
        verbose=args.verbose,
    )
    print(json.dumps({k: v for k, v in response.items() if k != "trajectory"}, indent=2, ensure_ascii=False))
    print(f"trajectory_frames={len(response['trajectory'])}")
