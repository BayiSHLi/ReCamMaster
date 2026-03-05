from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torch import nn
from typing import Any, Callable, Dict, List
from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video
# from diffsynth.extensions.ImageQualityMetric import config
from einops import rearrange
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset
from torchcodec.decoders import VideoDecoder
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

try:
	from evaluation import eval_camra, eval_clip, eval_fvd, eval_matching
except ModuleNotFoundError:
	import eval_camra
	import eval_clip
	import eval_fvd
	import eval_matching


MetricRunner = Callable[[Dict[str, Any]], Dict[str, Any]]


METRIC_REGISTRY: Dict[str, MetricRunner] = {
	"camera": eval_camra.run,
	"matching": eval_matching.run,
	"clip": eval_clip.run,
	"fvd": eval_fvd.run,
}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run ReCamMaster evaluation metrics")

	parser.add_argument(
		"--metrics",
		type=str,
		default="camera,matching,clip,fvd",
		help="Comma-separated metrics to run: camera,matching,clip,fvd",
	)
	
	parser.add_argument("--data_root", type=str, default="/mnt/hdd/dataset/webvid10m/", help="Root directory containing video files and camera jsons")
	parser.add_argument("--gt_camera_json", type=str, default="", help="Path to GT camera extrinsics json")
	parser.add_argument(
		"--align_camera_centers",
		action="store_true",
		help="Apply Umeyama alignment on camera centers before RotErr/TransErr",
	)
	parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to evaluate for each video")
	parser.add_argument("--ckpt_path", type=str, default="./models/ReCamMaster/checkpoints/step20000.ckpt", help="Path to ReCamMaster checkpoint")
	parser.add_argument("--max_samples", type=int, default=20, help="Max number of videos from dataset to calucate metrics. Set to -1 to use all videos.")
	parser.add_argument("--save_dir", type=str, default="/mnt/hdd/dataset/webvid10m/outputs", help="Directory to save generated videos and intermediate results")
	parser.add_argument(
		"--output-json",
		type=str,
		default="evaluation/results.json",
		help="Output path for evaluation report JSON",
	)
	parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale for inference")
	parser.add_argument("--height", type=int, default=480, help="Height to resize input videos to")
	parser.add_argument("--width", type=int, default=832, help="Width to resize input videos to")
	parser.add_argument("--start_sample_idx", type=int, default=0, help="Start reading dataset from this global sample index (0-based)")
	parser.add_argument("--dataset_shuffle", action="store_true", help="Enable dataset shuffle (disabled by default for deterministic reading)")
	parser.add_argument("--dataset_seed", type=int, default=42, help="Dataset shuffle seed (used only when --dataset_shuffle is enabled)")
	parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps for generation")
	parser.add_argument("--seed", type=int, default=0, help="Random seed used in generation")
	parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs to use for parallel generation. -1 means all visible GPUs")
	parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1. Empty means using first num_gpus GPUs")
	parser.add_argument("--writer_threads", type=int, default=2, help="Thread count for asynchronous video writing")
	parser.add_argument("--save_original_video", action="store_true", help="Save decoded source/original videos for each sample")
	parser.add_argument("--show_progress", action="store_true", help="Show per-video denoising progress bars")
	parser.add_argument("--no_show_progress", action="store_false", dest="show_progress", help="Disable denoising progress bars")
	parser.set_defaults(show_progress=True)
	parser.add_argument("--worker-log-dir", dest="worker_log_dir", type=str, default="evaluation/logs/workers", help="Directory for per-worker log files")
	return parser.parse_args()


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
	return {
        "data_root": args.data_root,
        "gt_camera_json": args.gt_camera_json,
        "align_camera_centers": args.align_camera_centers,
        "max_samples": args.max_samples,
        "save_dir": args.save_dir,
        "height": args.height,
        "width": args.width,
	}	


def _select_metrics(metric_text: str) -> List[str]:
	selected = [name.strip() for name in metric_text.split(",") if name.strip()]
	unknown = [name for name in selected if name not in METRIC_REGISTRY]
	if unknown:
		raise ValueError(f"Unknown metric(s): {unknown}. Available: {list(METRIC_REGISTRY.keys())}")
	return selected


def _run_metrics(metric_names: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
	details: Dict[str, Any] = {}
	summary_metrics: Dict[str, Any] = {}

	for name in metric_names:
		result = METRIC_REGISTRY[name](config)
		details[name] = result

		metrics = result.get("metrics", {})
		if isinstance(metrics, dict):
			summary_metrics.update(metrics)

	return {
		"details": details,
		"summary": summary_metrics,
	}


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class WebVidDataset(IterableDataset):
	def __init__(self, data_root, save_dir, num_samples, max_num_frames=81, shuffle=True, seed=42, buffer_size=10000, 
			  frame_interval=1, num_frames=81, height=480, width=832, shard_rank=0, num_shards=1, save_original_video=False,
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
		self.save_original_video_flag = save_original_video
		self.start_sample_idx = max(0, int(start_sample_idx))

		data_path = Path(data_root) / "data/train-*.parquet"
		self.dataset = load_dataset(
			"parquet",
			data_files=str(data_path),
			streaming=True,
		)["train"]
		if self.shuffle:
			self.dataset = self.dataset.shuffle(
                buffer_size=self.buffer_size,
                seed=self.seed,
            )

		self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

	def crop_and_resize(self, frames: torch.Tensor):
		_, _, height, width = frames.shape
		scale = max(self.width / width, self.height / height)
		frames = torchvision.transforms.functional.resize(
			frames,
			(round(height * scale), round(width * scale)),
			interpolation=torchvision.transforms.InterpolationMode.BILINEAR
		)
		return frames
	
	def decode_video(self, video:VideoDecoder) -> torch.Tensor:
		if video.metadata.num_frames < self.max_num_frames or video.metadata.num_frames - 1 < self.frame_interval * (self.num_frames - 1):
			return None
		try:
			fps = video.metadata.fps
		except:
			fps = 30  # fallback
		frames = []
		for i in range(self.max_num_frames):
			frame = video[i]  
			frames.append(frame)
		frames = torch.stack(frames)  # (T, C, H, W)

		original_frames = frames.clone()

		frames = self.crop_and_resize(frames)
		frames = self.frame_process(frames)  # (T, C, H, W)
		frames = rearrange(frames, "T C H W -> C T H W")

		return frames, original_frames

	def save_original_video(self, video_tensor: torch.Tensor, save_path: Path, fps: int = 30):
		video_tensor = video_tensor.permute(0, 2, 3, 1)  # (T, H, W, C)
		video_tensor = video_tensor.to(torch.uint8).cpu().numpy()
		save_path.parent.mkdir(parents=True, exist_ok=True)
		save_video(video_tensor, save_path, fps=fps)

	def __len__(self):
		return self.num_samples
	
	# def __getitem__(self, idx):
	# 	video = self.eval_dataset[idx]["video"]
	# 	text = self.eval_dataset[idx]["text"]
	# 	num_frames = len(video)

	# 	# try:
	# 	# 	fps = video.metadata.fps
	# 	# except:
	# 	# 	fps = 30  # Default FPS if not available
	# 	frames = []
	# 	for i in range(num_frames):
	# 		frame = self.crop_and_resize(video[i])
	# 		frame = self.frame_process(frame)

	# 		frames.append(frame)
	# 	frames = torch.stack(frames, dim=0)  # (num_frames, C, H, W)
	# 	frames = frames.permute(1, 0, 2, 3)  # (C, num_frames, H, W)


		# return {"video": frames, "text": text}

	def __iter__(self):
		count = 0

		for raw_idx, sample in enumerate(self.dataset):
			if self.num_samples != -1 and count >= self.num_samples:
				break
			if raw_idx < self.start_sample_idx:
				continue
			if (raw_idx - self.start_sample_idx) % self.num_shards != self.shard_rank:
				continue

			video = sample["video"]
			text = sample["text"]

			try:
				frames, original_frames = self.decode_video(video)
				if frames is None:
					continue
				if self.save_original_video_flag:
					self.save_original_video(original_frames, Path(self.save_dir) / f"video_{raw_idx}" / "original.mp4", fps=30)
			except Exception:
				continue

			yield {
				"video": frames,
				"text": text,
				"index": raw_idx,
			}

			count += 1	


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
		candidate = save_dir / f"cam_{i:02d}.mp4"
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
		max_num_frames=args.num_frames,
		shuffle=args.dataset_shuffle,
		seed=args.dataset_seed,
		frame_interval=4,
		num_frames=21,
		height=args.height,
		width=args.width,
		shard_rank=rank,
		num_shards=len(gpu_ids),
		save_original_video=args.save_original_video,
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
				target_text = sample["text"]
				idx = sample["index"]

				save_dir = Path(args.save_dir) / f"video_{idx}"
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
					save_path = save_dir / f"cam_{i:02d}.mp4"
					if _is_valid_generated_video(save_path):
						continue
					lock_path = save_dir / f"cam_{i:02d}.lock"
					if not _try_acquire_generation_lock(lock_path):
						continue

					progress_cmd = _silent_progress
					if args.show_progress:
						progress_desc = f"[w{rank}] video_{idx}/cam_{i:02d}.mp4"
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
	selected_metrics = _select_metrics(args.metrics)
	config = _build_config(args)

	gpu_ids = _resolve_gpu_ids(args)
	world_size = len(gpu_ids)
	args.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	print(f"Using {world_size} GPU(s): {gpu_ids}")
	if world_size > 1:
		print(f"Per-worker logs: {Path(args.worker_log_dir) / f'run_{args.run_tag}_worker<RANK>_gpu<ID>.log'}")

	if world_size == 1:
		_run_worker(0, args, gpu_ids)
	else:
		mp.spawn(_run_worker, args=(args, gpu_ids), nprocs=world_size, join=True)

	# run_result = _run_metrics(selected_metrics, config)

	# report = {
	# 	"timestamp": datetime.now(timezone.utc).isoformat(),
	# 	"selected_metrics": selected_metrics,
	# 	"summary": run_result["summary"],
	# 	"details": run_result["details"],
	# }

	# output_path = Path(args.output_json)
	# output_path.parent.mkdir(parents=True, exist_ok=True)
	# with output_path.open("w", encoding="utf-8") as f:
	# 	json.dump(report, f, indent=2, ensure_ascii=False)

	# print("=" * 72)
	# print("ReCamMaster Evaluation Report")
	# print("=" * 72)
	# print(f"Selected metrics: {', '.join(selected_metrics)}")
	# print("Summary:")
	# for key, value in report["summary"].items():
	# 	print(f"  - {key}: {value}")
	# print(f"Report saved to: {output_path}")


if __name__ == "__main__":
	main()
