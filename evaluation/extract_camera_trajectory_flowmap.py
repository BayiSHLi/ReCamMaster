from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from evaluation.extract_camera_trajectory_common import (
        as_bool,
        build_ok_result_from_colmap_entities,
        extract_frames_from_video,
        save_camera_trajectory_json,
    )
except ModuleNotFoundError:
    from extract_camera_trajectory_common import (
        as_bool,
        build_ok_result_from_colmap_entities,
        extract_frames_from_video,
        save_camera_trajectory_json,
    )


def _resolve_flowmap_repo_dir(flowmap_repo_dir: str) -> Path:
    candidates: List[Path] = []
    if flowmap_repo_dir:
        candidates.append(Path(flowmap_repo_dir).expanduser())

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend([repo_root / "third_party" / "flowmap", repo_root.parent / "flowmap"])

    for candidate in candidates:
        if (candidate / "flowmap" / "overfit.py").exists() and (candidate / "config" / "overfit.yaml").exists():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates) if candidates else "<none>"
    raise FileNotFoundError(
        "FlowMap repository not found. Set flowmap_repo_dir or place FlowMap at one of: "
        f"{searched}"
    )


def _run_flowmap_overfit(
    frames_dir: Path,
    flowmap_colmap_dir: Path,
    flowmap_repo_dir: Path,
    flowmap_max_steps: int,
    flowmap_checkpoint: str,
    verbose: bool,
) -> None:
    if flowmap_colmap_dir.exists():
        shutil.rmtree(flowmap_colmap_dir)

    cmd = [
        sys.executable,
        "-m",
        "flowmap.overfit",
        "dataset=images",
        f"dataset.images.root={frames_dir}",
        "wandb.mode=disabled",
        f"local_save_root={flowmap_colmap_dir}",
        f"trainer.max_steps={max(1, int(flowmap_max_steps))}",
        "trainer.val_check_interval=1000000",
        "frame_sampler.start=0",
        "frame_sampler.step=1",
        "frame_sampler.num_frames=null",
        "loss=[flow]",
    ]
    if flowmap_checkpoint:
        cmd.append(f"checkpoint.load={flowmap_checkpoint}")
    else:
        cmd.append("checkpoint.load=null")

    process = subprocess.run(
        cmd,
        cwd=str(flowmap_repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = process.stdout or ""
    if process.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        raise RuntimeError(
            "FlowMap overfit failed with non-zero exit code "
            f"({process.returncode}). Last log lines:\n{tail}"
        )

    if verbose and output.strip():
        print("\n".join(output.splitlines()[-20:]))


def extract_camera_trajectory_from_video_flowmap(
    video_path: str,
    output_dir: str,
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    resize_short: int = 0,
    flowmap_repo_dir: str = "",
    flowmap_max_steps: int = 500,
    flowmap_checkpoint: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir) / "frames"
            frames = extract_frames_from_video(
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

            repo_dir = _resolve_flowmap_repo_dir(flowmap_repo_dir)
            flowmap_colmap_dir = output_path / "flowmap_colmap"
            _run_flowmap_overfit(
                frames_dir=frames_dir,
                flowmap_colmap_dir=flowmap_colmap_dir,
                flowmap_repo_dir=repo_dir,
                flowmap_max_steps=flowmap_max_steps,
                flowmap_checkpoint=flowmap_checkpoint,
                verbose=verbose,
            )

            sparse_model_dir = flowmap_colmap_dir / "sparse" / "0"
            if not sparse_model_dir.exists():
                raise FileNotFoundError(f"FlowMap output is missing sparse model: {sparse_model_dir}")

            repo_dir_str = str(repo_dir)
            if repo_dir_str not in sys.path:
                sys.path.insert(0, repo_dir_str)

            read_write_model = importlib.import_module("flowmap.third_party.colmap.read_write_model")
            model = read_write_model.read_model(path=str(sparse_model_dir), ext="")
            if model is None:
                raise RuntimeError(f"Failed to read FlowMap sparse model from: {sparse_model_dir}")

            cameras, images, points3d = model
            points3d = points3d or {}

            result = build_ok_result_from_colmap_entities(
                num_frames_extracted=len(frames),
                cameras=dict(cameras),
                images=dict(images),
                points3d=dict(points3d),
                metadata_overrides={
                    "camera_model": "PINHOLE",
                    "matcher_type": "flowmap",
                    "use_global_mapper_requested": False,
                    "mapper_backend": "flowmap.overfit",
                    "deterministic": False,
                    "random_seed_base": None,
                    "num_threads_requested": None,
                    "num_threads_used": None,
                    "use_gpu": True,
                    "min_reconstruction_ratio_target": None,
                    "min_reconstruction_frames_target": None,
                    "attempts": [
                        {
                            "attempt_index": 0,
                            "attempt_name": "flowmap_primary",
                            "matcher_type": "flowmap",
                            "num_frames_reconstructed": int(len(images)),
                            "num_3d_points": int(len(points3d)),
                            "mapper_backend": "flowmap.overfit",
                            "status": "ok",
                            "reason": "",
                        }
                    ],
                    "selected_attempt_index": 0,
                    "selected_attempt_name": "flowmap_primary",
                    "trajectory_backend": "flowmap",
                    "flowmap_repo_dir": str(repo_dir),
                    "flowmap_max_steps": int(max(1, flowmap_max_steps)),
                    "flowmap_checkpoint": (flowmap_checkpoint if flowmap_checkpoint else None),
                    "flowmap_output_dir": str(flowmap_colmap_dir),
                },
            )
            save_camera_trajectory_json(output_path, result)
            return result
    except Exception as exc:  # noqa: BLE001
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

    result = extract_camera_trajectory_from_video_flowmap(
        video_path=source_video,
        output_dir=config.get("trajectory_output_dir", "evaluation/trajectory_outputs"),
        frame_skip=int(config.get("trajectory_frame_skip", 1)),
        max_frames=(
            int(config.get("trajectory_max_frames"))
            if config.get("trajectory_max_frames")
            else None
        ),
        resize_short=int(config.get("trajectory_resize_short", 0)),
        flowmap_repo_dir=str(config.get("trajectory_flowmap_repo_dir", "")),
        flowmap_max_steps=int(config.get("trajectory_flowmap_max_steps", 500)),
        flowmap_checkpoint=str(config.get("trajectory_flowmap_checkpoint", "")),
        verbose=as_bool(config.get("verbose", False), default=False),
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
    import argparse

    parser = argparse.ArgumentParser(description="Extract camera trajectory with FlowMap")
    parser.add_argument("--video_path", type=str, default="/mnt/hdd/dataset/webvid10m/outputs/video_0/cam01.mp4")
    parser.add_argument("--output_dir", type=str, default="evaluation/trajectory_output")
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=81, help="<=0 means no frame limit")
    parser.add_argument("--resize_short", type=int, default=384)
    parser.add_argument("--flowmap_repo_dir", type=str, default="~/workspace/SHLi/flowmap/")
    parser.add_argument("--flowmap_max_steps", type=int, default=200)
    parser.add_argument("--flowmap_checkpoint", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    response = extract_camera_trajectory_from_video_flowmap(
        video_path=args.video_path,
        output_dir=args.output_dir,
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
