from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

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


@dataclass
class _ParsedCamera:
    camera_id: int
    model_name: str
    width: int
    height: int
    params: List[float]


@dataclass
class _ParsedImage:
    image_id: int
    camera_id: int
    name: str
    qvec: List[float]
    tvec: List[float]

    def qvec2rotmat(self) -> List[List[float]]:
        qw, qx, qy, qz = self.qvec
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm <= 0.0:
            raise ValueError("Invalid quaternion with zero norm")

        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm

        return [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]


def _as_cli_bool(flag: bool) -> str:
    return "1" if bool(flag) else "0"


def _ensure_command_available(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f"Required command is not available in PATH: {command}")


def _run_command(command: Sequence[str], verbose: bool) -> None:
    if verbose:
        print("[*]", " ".join(command))

    completed = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = (completed.stdout or "").strip()
    if verbose and output:
        print(output)

    if completed.returncode != 0:
        tail_lines = output.splitlines()[-40:]
        tail_text = "\n".join(tail_lines)
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}"
            + (f"\n{tail_text}" if tail_text else "")
        )


def _iter_non_comment_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def _parse_cameras_txt(path: Path) -> Dict[int, _ParsedCamera]:
    cameras: Dict[int, _ParsedCamera] = {}
    for line in _iter_non_comment_lines(path):
        tokens = line.split()
        if len(tokens) < 5:
            continue

        camera_id = int(tokens[0])
        cameras[camera_id] = _ParsedCamera(
            camera_id=camera_id,
            model_name=str(tokens[1]),
            width=int(tokens[2]),
            height=int(tokens[3]),
            params=[float(v) for v in tokens[4:]],
        )

    return cameras


def _parse_images_txt(path: Path) -> Dict[int, _ParsedImage]:
    images: Dict[int, _ParsedImage] = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    idx = 0
    while idx < len(lines):
        tokens = lines[idx].split()
        if len(tokens) >= 10:
            image_id = int(tokens[0])
            qvec = [float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])]
            tvec = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
            camera_id = int(tokens[8])
            image_name = " ".join(tokens[9:])
            images[image_id] = _ParsedImage(
                image_id=image_id,
                camera_id=camera_id,
                name=image_name,
                qvec=qvec,
                tvec=tvec,
            )

        # images.txt uses two lines per image; the second line stores 2D tracks.
        idx += 2

    return images


def _parse_points3d_txt(path: Path) -> Dict[int, bool]:
    points3d: Dict[int, bool] = {}
    for line in _iter_non_comment_lines(path):
        tokens = line.split()
        if len(tokens) < 1:
            continue
        points3d[int(tokens[0])] = True
    return points3d


def _find_model_dir(base_path: Path) -> Path:
    def _is_model_dir(candidate: Path) -> bool:
        return (
            (candidate / "cameras.bin").exists()
            and (candidate / "images.bin").exists()
            or (candidate / "cameras.txt").exists()
            and (candidate / "images.txt").exists()
        )

    candidates: List[Path] = [base_path, base_path / "0"]
    for child in sorted(base_path.iterdir()) if base_path.exists() else []:
        if child.is_dir():
            candidates.append(child)

    for candidate in candidates:
        if candidate.exists() and _is_model_dir(candidate):
            return candidate

    raise RuntimeError(f"No COLMAP model files found under: {base_path}")


def _ensure_text_model(model_dir: Path, verbose: bool) -> Path:
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    points3d_txt = model_dir / "points3D.txt"
    if cameras_txt.exists() and images_txt.exists() and points3d_txt.exists():
        return model_dir

    output_txt_dir = model_dir / "txt"
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(output_txt_dir),
            "--output_type",
            "TXT",
        ],
        verbose=verbose,
    )
    return output_txt_dir


def _run_colmap_glomap_sfm(
    database_path: Path,
    image_path: Path,
    sparse_path: Path,
    matcher_type: str,
    camera_model: str,
    num_threads: int,
    max_num_features: int,
    guided_matching: bool,
    sequential_overlap: int,
    sequential_loop_detection: bool,
    use_gpu: bool,
    verbose: bool,
) -> tuple[Dict[int, _ParsedCamera], Dict[int, _ParsedImage], Dict[int, bool], str]:
    _ensure_command_available("colmap")
    _ensure_command_available("glomap")

    sparse_path.mkdir(parents=True, exist_ok=True)

    _run_command(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            str(camera_model),
            "--SiftExtraction.num_threads",
            str(num_threads),
            "--SiftExtraction.use_gpu",
            _as_cli_bool(use_gpu),
            "--SiftExtraction.max_num_features",
            str(max_num_features),
        ],
        verbose=verbose,
    )

    normalized_matcher_type = str(matcher_type).strip().lower()
    matcher_command = "exhaustive_matcher"
    matcher_args: List[str] = []
    if normalized_matcher_type == "sequential":
        matcher_command = "sequential_matcher"
        matcher_args.extend(
            [
                "--SequentialMatching.overlap",
                str(sequential_overlap),
                "--SequentialMatching.loop_detection",
                _as_cli_bool(sequential_loop_detection),
            ]
        )
    elif normalized_matcher_type == "vocab_tree":
        # Keep the retry flow robust when no vocab tree file is available.
        if verbose:
            print("[*] vocab_tree matcher requested but unsupported without a tree file; fallback to exhaustive")

    _run_command(
        [
            "colmap",
            matcher_command,
            "--database_path",
            str(database_path),
            "--SiftMatching.num_threads",
            str(num_threads),
            "--SiftMatching.use_gpu",
            _as_cli_bool(use_gpu),
            "--SiftMatching.guided_matching",
            _as_cli_bool(guided_matching),
            *matcher_args,
        ],
        verbose=verbose,
    )

    _run_command(
        [
            "glomap",
            "mapper",
            "--database_path",
            str(database_path),
            "--output_path",
            str(sparse_path),
        ],
        verbose=verbose,
    )

    model_dir = _find_model_dir(sparse_path)
    text_model_dir = _ensure_text_model(model_dir, verbose=verbose)

    cameras = _parse_cameras_txt(text_model_dir / "cameras.txt")
    images = _parse_images_txt(text_model_dir / "images.txt")
    points3d = _parse_points3d_txt(text_model_dir / "points3D.txt")

    if len(cameras) == 0 or len(images) == 0:
        raise RuntimeError("Reconstruction output is empty")

    return cameras, images, points3d, "glomap.mapper"


def extract_camera_trajectory_from_video_colmap(
    video_path: str,
    output_dir: str,
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
    verbose: bool = False,
) -> Dict[str, Any]:
    num_threads = max(1, int(num_threads))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    missing_commands = [cmd for cmd in ("colmap", "glomap") if shutil.which(cmd) is None]
    if missing_commands:
        return {
            "status": "skipped",
            "reason": "Missing required commands: " + ", ".join(missing_commands),
            "metadata": {},
            "camera_intrinsics": {},
            "trajectory": [],
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frames_dir = tmp_path / "frames"

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

            use_gpu = not deterministic
            seed_base = int(random_seed)
            min_reconstruction_frames = max(8, int(round(len(frames) * float(min_reconstruction_ratio))))

            attempts: List[Dict[str, Any]] = [
                {
                    "name": "primary",
                    "matcher_type": matcher_type,
                    "max_num_features": 8192,
                    "guided_matching": False,
                    "sequential_overlap": 10,
                    "sequential_loop_detection": False,
                    "random_seed": seed_base,
                }
            ]
            if enable_retry:
                attempts.extend(
                    [
                        {
                            "name": "sequential_robust",
                            "matcher_type": "sequential",
                            "max_num_features": 16384,
                            "guided_matching": True,
                            "sequential_overlap": 24,
                            "sequential_loop_detection": False,
                            "random_seed": seed_base + 1,
                        },
                        {
                            "name": "exhaustive_robust",
                            "matcher_type": "exhaustive",
                            "max_num_features": 16384,
                            "guided_matching": True,
                            "sequential_overlap": 10,
                            "sequential_loop_detection": False,
                            "random_seed": seed_base + 2,
                        },
                    ]
                )

            best_cameras: Dict[int, _ParsedCamera] = {}
            best_images: Dict[int, _ParsedImage] = {}
            best_points3d: Dict[int, bool] = {}
            best_mapper_backend = ""
            best_score = (-1, -1)
            best_attempt_index = -1
            attempt_summaries: List[Dict[str, Any]] = []

            for attempt_idx, attempt in enumerate(attempts):
                try:
                    cameras, images, points3d, mapper_backend = _run_colmap_glomap_sfm(
                        database_path=tmp_path / f"database_{attempt_idx:02d}.db",
                        image_path=frames_dir,
                        sparse_path=tmp_path / f"sparse_{attempt_idx:02d}",
                        matcher_type=str(attempt["matcher_type"]),
                        camera_model=camera_model,
                        num_threads=(1 if deterministic else num_threads),
                        max_num_features=int(attempt["max_num_features"]),
                        guided_matching=bool(attempt["guided_matching"]),
                        sequential_overlap=int(attempt["sequential_overlap"]),
                        sequential_loop_detection=bool(attempt["sequential_loop_detection"]),
                        use_gpu=bool(use_gpu),
                        verbose=verbose,
                    )
                    num_images = len(images)
                    num_points = len(points3d)
                    attempt_summaries.append(
                        {
                            "attempt_index": attempt_idx,
                            "attempt_name": attempt["name"],
                            "matcher_type": attempt["matcher_type"],
                            "num_frames_reconstructed": num_images,
                            "num_3d_points": num_points,
                            "mapper_backend": mapper_backend,
                            "status": "ok",
                            "reason": "",
                        }
                    )

                    if (num_images, num_points) > best_score:
                        best_cameras = cameras
                        best_images = images
                        best_points3d = points3d
                        best_mapper_backend = mapper_backend
                        best_score = (num_images, num_points)
                        best_attempt_index = attempt_idx

                    if num_images >= min_reconstruction_frames:
                        break
                except Exception as exc:  # noqa: BLE001
                    attempt_summaries.append(
                        {
                            "attempt_index": attempt_idx,
                            "attempt_name": attempt["name"],
                            "matcher_type": attempt["matcher_type"],
                            "num_frames_reconstructed": 0,
                            "num_3d_points": 0,
                            "mapper_backend": "",
                            "status": "failed",
                            "reason": str(exc),
                        }
                    )

            if len(best_images) == 0:
                return {
                    "status": "failed",
                    "reason": "All reconstruction attempts failed",
                    "metadata": {
                        "num_frames_extracted": len(frames),
                        "attempts": attempt_summaries,
                        "deterministic": bool(deterministic),
                        "random_seed_base": seed_base,
                    },
                    "camera_intrinsics": {},
                    "trajectory": [],
                }

            if not use_global_mapper and verbose:
                print("[*] use_global_mapper=False is ignored in this backend; glomap.mapper is always used")

            result = build_ok_result_from_colmap_entities(
                num_frames_extracted=len(frames),
                cameras=best_cameras,
                images=best_images,
                points3d=best_points3d,
                metadata_overrides={
                    "camera_model": camera_model,
                    "matcher_type": matcher_type,
                    "use_global_mapper_requested": bool(use_global_mapper),
                    "mapper_backend": best_mapper_backend,
                    "deterministic": bool(deterministic),
                    "random_seed_base": seed_base,
                    "num_threads_requested": int(num_threads),
                    "num_threads_used": (1 if deterministic else int(num_threads)),
                    "use_gpu": bool(use_gpu),
                    "min_reconstruction_ratio_target": float(min_reconstruction_ratio),
                    "min_reconstruction_frames_target": int(min_reconstruction_frames),
                    "attempts": attempt_summaries,
                    "selected_attempt_index": int(best_attempt_index),
                    "selected_attempt_name": attempts[best_attempt_index]["name"],
                    "trajectory_backend": "glomap",
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

    result = extract_camera_trajectory_from_video_colmap(
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
        use_global_mapper=as_bool(config.get("trajectory_use_global_mapper", True), default=True),
        num_threads=int(config.get("trajectory_num_threads", 4)),
        deterministic=as_bool(config.get("trajectory_deterministic", True), default=True),
        random_seed=int(config.get("trajectory_random_seed", 0)),
        enable_retry=as_bool(config.get("trajectory_retry", True), default=True),
        min_reconstruction_ratio=float(config.get("trajectory_min_reconstruction_ratio", 0.2)),
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

    parser = argparse.ArgumentParser(description="Extract camera trajectory with COLMAP + GLOMAP")
    parser.add_argument("--video_path", type=str, default="/mnt/hdd/dataset/webvid10m/outputs/video_0/cam01.mp4")
    parser.add_argument("--output_dir", type=str, default="evaluation/trajectory_output")
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=81, help="<=0 means no frame limit")
    parser.add_argument("--resize_short", type=int, default=0)
    parser.add_argument("--matcher_type", type=str, default="sequential", choices=["sequential", "vocab_tree", "exhaustive"])
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--trajectory_non_deterministic", action="store_true")
    parser.add_argument("--disable_retry", action="store_true")
    parser.add_argument("--min_reconstruction_ratio", type=float, default=0.2)
    parser.add_argument("--disable_global_mapper", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    response = extract_camera_trajectory_from_video_colmap(
        video_path=args.video_path,
        output_dir=args.output_dir,
        frame_skip=args.frame_skip,
        max_frames=(None if args.max_frames <= 0 else args.max_frames),
        resize_short=args.resize_short,
        matcher_type=args.matcher_type,
        use_global_mapper=not args.disable_global_mapper,
        num_threads=args.num_threads,
        deterministic=not args.trajectory_non_deterministic,
        random_seed=args.random_seed,
        enable_retry=not args.disable_retry,
        min_reconstruction_ratio=args.min_reconstruction_ratio,
        verbose=args.verbose,
    )
    print(json.dumps({k: v for k, v in response.items() if k != "trajectory"}, indent=2, ensure_ascii=False))
    print(f"trajectory_frames={len(response['trajectory'])}")
