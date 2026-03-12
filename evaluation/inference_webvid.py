from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torch import nn
from typing import Any, Callable, Dict, List
from diffsynth import ModelManager, WanVideoReCamMasterPipeline
# from diffsynth.extensions.ImageQualityMetric import config
from einops import rearrange
from torch.utils.data import Dataset
import av
import imageio.v2 as imageio
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torchvision
from torchvision.transforms import v2


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
	sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))




def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run ReCamMaster evaluation metrics")

	parser.add_argument(
		"--metrics",
		type=str,
		default="camera,matching,clip,fvd",
		help="Comma-separated metrics to run: camera,matching,clip,fvd",
	)
	
	parser.add_argument("--data_root", type=str, default="/mnt/hdd/dataset/webvid", help="Root directory containing videos/ and 0000.csv")
	parser.add_argument("--metadata_csv", type=str, default="0000.csv", help="CSV filename under data_root; column 'name' is used as text prompt")
	parser.add_argument("--gt_camera_json", type=str, default="", help="Path to GT camera extrinsics json")
	parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to evaluate for each video")
	parser.add_argument("--ckpt_path", type=str, default="./models/ReCamMaster/checkpoints/step20000.ckpt", help="Path to ReCamMaster checkpoint")
	parser.add_argument("--max_samples", type=int, default=20, help="Max number of videos from dataset to calucate metrics. Set to -1 to use all videos.")
	parser.add_argument("--start_sample_idx", type=int, default=0, help="Start reading dataset from this global sample index (0-based)")
	parser.add_argument("--dataset_shuffle", action="store_true", help="Enable dataset shuffle (disabled by default for deterministic reading)")
	
	parser.add_argument("--save_dir", type=str, default="", help="Directory to save generated videos and intermediate results. Empty means <data_root>/outputs")
	parser.add_argument(
		"--output-json",
		type=str,
		default="evaluation/results.json",
		help="Output path for evaluation report JSON",
	)
	parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale for inference")
	parser.add_argument("--height", type=int, default=480, help="Height to resize input videos to")
	parser.add_argument("--width", type=int, default=832, help="Width to resize input videos to")
	parser.add_argument("--dataset_seed", type=int, default=42, help="Dataset shuffle seed (used only when --dataset_shuffle is enabled)")
	parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps for generation")
	parser.add_argument("--seed", type=int, default=0, help="Random seed used in generation")
	parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs to use for parallel generation. -1 means all visible GPUs")
	parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1. Empty means using first num_gpus GPUs")
	parser.add_argument("--writer_threads", type=int, default=2, help="Thread count for asynchronous video writing")
	parser.add_argument("--show_progress", action="store_true", help="Show per-video denoising progress bars")
	parser.add_argument("--no_show_progress", action="store_false", dest="show_progress", help="Disable denoising progress bars")
	parser.set_defaults(show_progress=True)
	parser.add_argument("--worker-log-dir", dest="worker_log_dir", type=str, default="evaluation/logs/workers", help="Directory for per-worker log files")
	return parser.parse_args()


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class WebVidDataset(Dataset):
	def __init__(self, data_root, save_dir, num_samples, csv_file="0000.csv", max_num_frames=81, shuffle=True, seed=42, buffer_size=10000, 
			  frame_interval=1, num_frames=81, height=480, width=832, shard_rank=0, num_shards=1, 
			  start_sample_idx=0):
		self.num_samples = num_samples
		self.max_num_frames = max_num_frames
		self.shuffle = shuffle
		self.seed = seed
		self.save_dir = save_dir
		self.buffer_size = buffer_size
		self.frame_interval = frame_interval
		self.num_frames = num_frames
		self.height = height
		self.width = width
		self.shard_rank = shard_rank
		self.num_shards = max(1, num_shards)
		self.start_sample_idx = max(0, int(start_sample_idx))

		self.data_root = Path(data_root)
		self.video_root = self.data_root / "videos"
		
		self.csv_path = self.data_root / csv_file
		if not self.csv_path.is_file():
			raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")

		self.data_paths = sorted(self.video_root.glob("*.mp4"))
		if not self.data_paths:
			raise FileNotFoundError(f"No .mp4 files found in {self.video_root}")

		self.samples = self._build_samples()
		if not self.samples:
			raise RuntimeError(
				f"No valid samples were resolved from {self.video_root} and {self.csv_path}. "
				"Please check video ids and CSV columns."
			)
		self.index_map = self._build_index_map()

		self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
			v2.ToDtype(torch.float32, scale=True),
			v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

	def _build_index_map(self) -> List[tuple[int, int]]:
		ordered_indices = np.arange(len(self.samples), dtype=np.int64)
		if self.shuffle:
			rng = np.random.default_rng(self.seed)
			rng.shuffle(ordered_indices)

		mapped: List[tuple[int, int]] = []
		for raw_idx, sample_idx in enumerate(ordered_indices.tolist()):
			if raw_idx < self.start_sample_idx:
				continue
			if (raw_idx - self.start_sample_idx) % self.num_shards != self.shard_rank:
				continue
			mapped.append((raw_idx, int(sample_idx)))

		if self.num_samples != -1:
			mapped = mapped[: self.num_samples]
		return mapped

	@staticmethod
	def _normalize_video_key(raw_key: str) -> str:
		key = str(raw_key).strip()
		if key.lower().endswith(".mp4"):
			key = key[:-4]
		# Local WebVid files may end with '+', e.g., '<id>+.mp4'.
		return key.rstrip("+").strip()

	def _load_csv_rows(self) -> List[Dict[str, str]]:
		rows: List[Dict[str, str]] = []
		with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
			reader = csv.DictReader(f)
			for row in reader:
				rows.append({k: (v if v is not None else "") for k, v in row.items()})
		return rows

	def _build_samples(self) -> List[Dict[str, Any]]:
		rows = self._load_csv_rows()
		if not rows:
			return []

		text_column = "name" if "name" in rows[0] else "text"
		if text_column not in rows[0]:
			raise KeyError(f"CSV {self.csv_path} must contain a 'name' column (or fallback 'text').")

		id_column = None
		for candidate in ("videoid", "video_id", "id", "file_name", "filename"):
			if candidate in rows[0]:
				id_column = candidate
				break

		if id_column is not None:
			id_to_text: Dict[str, str] = {}
			for row in rows:
				text = str(row.get(text_column, "")).strip()
				if not text:
					continue
				video_key = self._normalize_video_key(str(row.get(id_column, "")))
				if video_key:
					id_to_text[video_key] = text

			resolved: List[Dict[str, Any]] = []
			for video_path in self.data_paths:
				video_key = self._normalize_video_key(video_path.stem)
				text = id_to_text.get(video_key)
				if text is None:
					continue
				resolved.append({"video_path": video_path, "text": text})

			if resolved:
				if len(resolved) < len(self.data_paths):
					print(
						f"[WebVidDataset] matched {len(resolved)}/{len(self.data_paths)} videos by id column '{id_column}'."
					)
				return resolved

		# Fallback: pair by sorted order when id mapping is unavailable.
		texts = [str(row.get(text_column, "")).strip() for row in rows]
		texts = [item for item in texts if item]
		n = min(len(self.data_paths), len(texts))
		if n == 0:
			return []
		if n < len(self.data_paths):
			print(f"[WebVidDataset] CSV text rows ({len(texts)}) < videos ({len(self.data_paths)}); using first {n} pairs.")
		return [{"video_path": self.data_paths[i], "text": texts[i]} for i in range(n)]

	def crop_and_resize(self, frames: torch.Tensor):
		_, _, height, width = frames.shape
		scale = max(self.width / width, self.height / height)
		frames = torchvision.transforms.functional.resize(
			frames,
			(round(height * scale), round(width * scale)),
			interpolation=torchvision.transforms.InterpolationMode.BILINEAR
		)
		return frames

	def read_video(self, video_path: Path) -> torch.Tensor | None:
		container = None
		try:
			container = av.open(str(video_path))
			video_stream = container.streams.video[0]
			video_stream.thread_type = "AUTO"
		except Exception:
			if container is not None:
				try:
					container.close()
				except Exception:
					pass
			return None

		required_span = (self.num_frames - 1) * self.frame_interval + 1
		decoded_frames: List[torch.Tensor] = []
		try:
			for frame in container.decode(video=0):
				frame_rgb = frame.to_ndarray(format="rgb24")
				decoded_frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))
				if self.max_num_frames > 0 and len(decoded_frames) >= self.max_num_frames:
					break
		except Exception:
			decoded_frames = []
		finally:
			try:
				container.close()
			except Exception:
				pass

		effective_total = len(decoded_frames)
		if effective_total < required_span:
			return None

		start_frame_id = max(0, (effective_total - required_span) // 2)
		frames = [decoded_frames[start_frame_id + i * self.frame_interval] for i in range(self.num_frames)]

		video = torch.stack(frames, dim=0)
		video = self.crop_and_resize(video)
		video = self.frame_process(video)
		video = rearrange(video, "t c h w -> c t h w")
		return video
	
	def __len__(self):
		return len(self.index_map)

	def __getitem__(self, idx: int):
		if idx < 0 or idx >= len(self.index_map):
			raise IndexError(idx)

		raw_idx, sample_idx = self.index_map[idx]
		sample = self.samples[sample_idx]
		video_path = sample["video_path"]
		text = sample["text"]
		video_id = self._normalize_video_key(video_path.stem)

		try:
			frames = self.read_video(video_path)
			if frames is None:
				return None
		except Exception:
			return None

		return {
			"video": frames,
			"text": text,
			"index": raw_idx,
			"video_id": video_id,
			"video_path": str(video_path),
		}


def parse_matrix(matrix_str):
	rows = matrix_str.strip().split('] [')
	matrix = []
	for row in rows:
		row = row.replace('[', '').replace(']', '')
		matrix.append(list(map(float, row.split())))
	return np.array(matrix)


def get_relative_pose(cam_params):
	abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
	abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

	cam_to_origin = 0
	target_cam_c2w = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, -cam_to_origin],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])
	abs2rel = target_cam_c2w @ abs_w2cs[0]
	ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
	ret_poses = np.array(ret_poses, dtype=np.float32)
	return ret_poses


NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def _save_video_fast(frames, save_path: Path, fps: int = 30, lock_path: Path | None = None) -> None:
	writer = imageio.get_writer(str(save_path), fps=fps, quality=9)
	try:
		for frame in frames:
			writer.append_data(np.asarray(frame))
	finally:
		writer.close()
		if lock_path is not None:
			try:
				lock_path.unlink(missing_ok=True)
			except Exception:
				pass


def _is_valid_generated_video(video_path: Path) -> bool:
	return video_path.is_file() and video_path.stat().st_size > 0


def _get_missing_camera_indices(save_dir: Path, num_cameras: int) -> List[int]:
	missing = []
	for i in range(num_cameras):
		candidate = save_dir / f"cam{i+1:02d}.mp4"
		if not _is_valid_generated_video(candidate):
			missing.append(i)
	return missing


def _pid_exists(pid: int) -> bool:
	if pid <= 0:
		return False
	try:
		os.kill(pid, 0)
		return True
	except ProcessLookupError:
		return False
	except PermissionError:
		return True


def _read_lock_pid(lock_path: Path) -> int | None:
	try:
		text = lock_path.read_text(encoding="utf-8").strip()
		if text.startswith("pid="):
			return int(text.split("=", 1)[1])
	except Exception:
		return None
	return None


def _try_acquire_generation_lock(lock_path: Path) -> bool:
	for _ in range(2):
		try:
			fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
			with os.fdopen(fd, "w") as f:
				f.write(f"pid={os.getpid()}\n")
			return True
		except FileExistsError:
			owner_pid = _read_lock_pid(lock_path)
			if owner_pid is None or not _pid_exists(owner_pid):
				try:
					lock_path.unlink(missing_ok=True)
				except Exception:
					pass
				continue
			return False
	return False


def _silent_progress(iterable):
	return iterable


def _build_named_progress_bar(desc: str):
	def _progress(iterable):
		return tqdm(iterable, desc=desc, leave=False, file=sys.stdout)
	return _progress


def _resolve_worker_log_dir(worker_log_dir: str, run_tag: str) -> Path:
	log_dir = Path(worker_log_dir)
	name = log_dir.name
	if name.startswith("run_"):
		suffix = name[len("run_"):]
		if len(suffix) == 15 and suffix[8] == "_" and suffix.replace("_", "").isdigit():
			return log_dir
	return log_dir / f"run_{run_tag}"


def _setup_worker_log(rank: int, device_id: int, args: argparse.Namespace):
	log_dir = Path(args.worker_log_dir)
	log_dir.mkdir(parents=True, exist_ok=True)
	run_tag = getattr(args, "run_tag", datetime.now().strftime("%Y%m%d_%H%M%S"))
	log_path = log_dir / f"run_{run_tag}_worker{rank}_gpu{device_id}.log"
	log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
	sys.stdout = log_fh
	sys.stderr = log_fh
	print(f"[Worker {rank}] logging redirected to: {log_path}")
	return log_fh, log_path


def _split_max_samples(max_samples: int, rank: int, world_size: int) -> int:
	if max_samples < 0:
		return -1
	base = max_samples // world_size
	extra = max_samples % world_size
	return base + (1 if rank < extra else 0)


def _resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
	available = torch.cuda.device_count()
	if available == 0:
		raise RuntimeError("No CUDA device found. This script currently requires at least one GPU.")

	if args.gpu_ids.strip():
		gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
	else:
		use_n = available if args.num_gpus <= 0 else min(args.num_gpus, available)
		gpu_ids = list(range(use_n))

	if not gpu_ids:
		raise ValueError("No GPU ids resolved. Please check --num_gpus / --gpu_ids.")
	for gpu_id in gpu_ids:
		if gpu_id < 0 or gpu_id >= available:
			raise ValueError(f"GPU id {gpu_id} is out of range. Available device count: {available}")
	return gpu_ids


def _build_target_trajectories(num_frames: int) -> List[torch.Tensor]:
	tgt_camera_path = "./example_test_data/cameras/camera_extrinsics.json"
	with open(tgt_camera_path, 'r') as file:
		cam_data = json.load(file)
	trajs: List[torch.Tensor] = []
	for cam_type in cam_data["frame0"].keys():
		cam_idx = list(range(num_frames))[::4]
		traj = [parse_matrix(cam_data[f"frame{idx}"][cam_type]) for idx in cam_idx]
		traj = np.stack(traj).transpose(0, 2, 1)
		c2ws = []
		for c2w in traj:
			c2w = c2w[:, [1, 2, 0, 3]]
			c2w[:3, 1] *= -1.
			c2w[:3, 3] /= 100
			c2ws.append(c2w)
		tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
		relative_poses = []
		for i in range(len(tgt_cam_params)):
			relative_pose = get_relative_pose([tgt_cam_params[0], tgt_cam_params[i]])
			relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
		pose_embedding = torch.stack(relative_poses, dim=0)
		pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
		trajs.append(pose_embedding.to(torch.bfloat16))
	return trajs


def _init_pipeline(args: argparse.Namespace, device: str) -> WanVideoReCamMasterPipeline:
	model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
	model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
	pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=device)

	dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
	for block in pipe.dit.blocks:
		block.cam_encoder = nn.Linear(12, dim)
		block.projector = nn.Linear(dim, dim)
		block.cam_encoder.weight.data.zero_()
		block.cam_encoder.bias.data.zero_()
		block.projector.weight = nn.Parameter(torch.eye(dim))
		block.projector.bias = nn.Parameter(torch.zeros(dim))

	state_dict = torch.load(args.ckpt_path, map_location="cpu")
	pipe.dit.load_state_dict(state_dict, strict=True)
	pipe.to(device)
	pipe.to(dtype=torch.bfloat16)
	pipe.eval()
	return pipe


def _run_worker(rank: int, args: argparse.Namespace, gpu_ids: List[int]) -> None:
	device_id = gpu_ids[rank]
	device = f"cuda:{device_id}"
	torch.cuda.set_device(device_id)
	log_fh, log_path = _setup_worker_log(rank, device_id, args)

	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True
	torch.backends.cudnn.benchmark = True

	local_samples = _split_max_samples(args.max_samples, rank, len(gpu_ids))
	pipe = _init_pipeline(args, device=device)
	trajs = [traj.to(device=device, dtype=torch.bfloat16, non_blocking=True) for traj in _build_target_trajectories(args.num_frames)]

	dataset = WebVidDataset(
		args.data_root,
		args.save_dir,
		num_samples=local_samples,
		csv_file=args.metadata_csv,
		max_num_frames=args.num_frames,
		shuffle=args.dataset_shuffle,
		seed=args.dataset_seed,
		frame_interval=1,
		num_frames=81,
		height=args.height,
		width=args.width,
		shard_rank=rank,
		num_shards=len(gpu_ids),
		start_sample_idx=args.start_sample_idx,
	)

	print(
		f"[Worker {rank}] device={device}, local_samples={local_samples}, trajectories={len(trajs)}, "
		f"start_sample_idx={args.start_sample_idx}, dataset_shuffle={args.dataset_shuffle}"
	)

	pending = []
	skipped_samples = 0
	resumed_samples = 0
	generated_videos = 0
	with ThreadPoolExecutor(max_workers=max(1, args.writer_threads)) as writer_pool:
		with torch.inference_mode():
			for sample in dataset:
				if sample is None:
					continue
				target_text = sample["text"]
				idx = sample["index"]
				video_id = sample["video_id"]

				save_dir = Path(args.save_dir) / video_id
				save_dir.mkdir(parents=True, exist_ok=True)
				missing_cam_indices = _get_missing_camera_indices(save_dir, len(trajs))

				if not missing_cam_indices:
					skipped_samples += 1
					continue
				if len(missing_cam_indices) < len(trajs):
					resumed_samples += 1

				source_video = sample["video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16, non_blocking=True)

				for i in missing_cam_indices:
					target_camera = trajs[i]
					save_path = save_dir / f"cam{i+1:02d}.mp4"
					if _is_valid_generated_video(save_path):
						continue
					lock_path = save_dir / f"cam{i+1:02d}.lock"
					if not _try_acquire_generation_lock(lock_path):
						continue

					progress_cmd = _silent_progress
					if args.show_progress:
						progress_desc = f"[w{rank}] {video_id}/cam{i+1:02d}.mp4"
						progress_cmd = _build_named_progress_bar(progress_desc)

					pipe_kwargs = dict(
						prompt=target_text,
						negative_prompt=NEGATIVE_PROMPT,
						source_video=source_video,
						target_camera=target_camera,
						cfg_scale=args.cfg_scale,
						num_inference_steps=args.num_inference_steps,
						seed=args.seed,
						rand_device=device,
						tiled=True,
						height=args.height,
						width=args.width,
						num_frames=args.num_frames,
						progress_bar_cmd=progress_cmd,
					)

					try:
						video = pipe(**pipe_kwargs)
					except Exception:
						try:
							lock_path.unlink(missing_ok=True)
						except Exception:
							pass
						raise
					pending.append(writer_pool.submit(_save_video_fast, video, save_path, 30, lock_path))
					generated_videos += 1

				del source_video

				if len(pending) >= max(2, args.writer_threads * 4):
					for future in pending[:args.writer_threads]:
						future.result()
					pending = pending[args.writer_threads:]

		for future in pending:
			future.result()

	print(
		f"[Worker {rank}] finished: generated={generated_videos}, "
		f"skipped_samples={skipped_samples}, resumed_samples={resumed_samples}"
	)
	try:
		log_fh.flush()
		log_fh.close()
	except Exception:
		pass


def main() -> None:
	args = _parse_args()
	if args.start_sample_idx < 0:
		raise ValueError(f"--start_sample_idx must be >= 0, got {args.start_sample_idx}")
	if not args.save_dir:
		args.save_dir = str(Path(args.data_root) / "outputs")

	args.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	args.worker_log_dir = str(_resolve_worker_log_dir(args.worker_log_dir, args.run_tag))

	gpu_ids = _resolve_gpu_ids(args)
	world_size = len(gpu_ids)
	print(f"Using {world_size} GPU(s): {gpu_ids}")
	print(f"Worker log dir: {args.worker_log_dir}")
	if world_size > 1:
		print(f"Per-worker logs: {Path(args.worker_log_dir) / f'run_{args.run_tag}_worker<RANK>_gpu<ID>.log'}")

	if world_size == 1:
		_run_worker(0, args, gpu_ids)
	else:
		mp.spawn(_run_worker, args=(args, gpu_ids), nprocs=world_size, join=True)


if __name__ == "__main__":
	main()
