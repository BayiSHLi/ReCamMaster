from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq
import glob
import numpy as np
from torch import nn
from typing import Any, Callable, Dict, List
from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video
# from diffsynth.extensions.ImageQualityMetric import config
from einops import rearrange
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset
from torchcodec import VideoDecoder

import torch
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

	return parser.parse_args()


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
	return {
        "data_root": args.data_root,
        "gt_camera_json": args.gt_camera_json,
        "align_camera_centers": args.align_camera_centers,
        "max_samples": args.max_samples,
        "save_dir": args.save_dir,
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


class WebVidDataset(torch.utils.data.Dataset):
	def __init__(self, data_root, num_samples, max_num_frames=81, shuffle=True, seed=42, buffer_size=10000, 
			  frame_interval=1, num_frames=81, height=480, width=832,):
		self.num_samples = num_samples
		self.max_num_frames = max_num_frames
		self.shuffle = shuffle
		self.seed = seed
		self.buffer_size = buffer_size
		self.frame_interval = frame_interval
		self.num_frames = num_frames
		self.height = height
		self.width = width

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
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

	def crop_and_resize(self, image):
		width, height = image.size
		scale = max(self.width / width, self.height / height)
		image = torchvision.transforms.functional.resize(
			image,
			(round(height*scale), round(width*scale)),
			interpolation=torchvision.transforms.InterpolationMode.BILINEAR
		)
		return image
	
	def decode_video(self, video:VideoDecoder) -> torch.Tensor:
		if video.metadata.num_frames < self.max_num_frames or video.metadata.num_frames - 1 < self.frame_interval * (self.num_frames - 1):
			return None
		try:
			fps = video.metadata.fps
		except:
			fps = 30  # fallback
		frames = []
		for i in range(self.max_num_frames):
			frame = video[i]  # (H, W, C)
			frames.append(frame)
		frames = torch.stack(frames)
		
		frames = []
		for frame_id in range(self.num_frames):
			frame = video.get_frame(frame_id * self.frame_interval)
			frame = Image.fromarray(frame)
			frame = self.crop_and_resize(frame)
			frame = self.frame_process(frame)
			frames.append(frame)
		frames = torch.stack(frames, dim=0)  # (num_frames, C, H, W)
		frames = frames.permute(1, 0, 2, 3)  # (C, num_frames, H, W)
		return frames


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

		for sample in self.dataset:
			if count >= self.num_samples:
				break

			video = sample["video"]
			text = sample["text"]

			try:
				frames = self.decode_video(video)
				if frames is None:
					continue
			except Exception:
				continue

			yield {
				"video": frames,
				"text": text,
				"index": count,
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

def main() -> None:
	args = _parse_args()
	selected_metrics = _select_metrics(args.metrics)
	config = _build_config(args)

	# 1. Load Wan2.1 pre-trained models
	model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
	model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
	pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

	# 2. Initialize additional modules introduced in ReCamMaster
	dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
	for block in pipe.dit.blocks:
		block.cam_encoder = nn.Linear(12, dim)
		block.projector = nn.Linear(dim, dim)
		block.cam_encoder.weight.data.zero_()
		block.cam_encoder.bias.data.zero_()
		block.projector.weight = nn.Parameter(torch.eye(dim))
		block.projector.bias = nn.Parameter(torch.zeros(dim))
	
	# 3. Load ReCamMaster checkpoint
	state_dict = torch.load(args.ckpt_path, map_location="cpu")
	pipe.dit.load_state_dict(state_dict, strict=True)
	pipe.to("cuda")
	pipe.to(dtype=torch.bfloat16)

	# 4. Prepare test data (source video, target camera, target trajectory)
	num_frames = args.num_frames
	dataset = WebVidDataset(
		args.data_root, 
		num_samples=args.max_samples, 
		max_num_frames=num_frames, 
		frame_interval=4, 
		num_frames=21
		)
	dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0
    )

	
	tgt_camera_path = "./example_test_data/cameras/camera_extrinsics.json"
	with open(tgt_camera_path, 'r') as file:
		cam_data = json.load(file)
	trajs = []
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
		pose_embedding = torch.stack(relative_poses, dim=0)  # 21x3x4
		pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
		trajs.append(pose_embedding.to(torch.bfloat16))

	print(f"Loaded {args.max_samples} videos and {len(trajs)} target camera trajectories for evaluation.")


	# 5. Run inference to generate predictions for evaluation
	for sample in dataset:
		source_video = sample["video"].squeeze(0)  # (C, T, H, W)
		target_text = sample["text"][0]
		idx = sample["index"]
		for i, target_camera in enumerate(trajs):
			save_dir = Path(args.save_dir) / f"video_{idx}"
			os.makedirs(save_dir, exist_ok=True)
			save_path = save_dir / f"cam_{i:02d}.mp4"
			video = pipe(
				prompt=target_text,
				negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
				source_video=source_video,
				target_camera=target_camera,
				cfg_scale=args.cfg_scale,
				num_inference_steps=50,
				seed=0, tiled=True
			)
			save_video(video, save_path, fps=30)

	run_result = _run_metrics(selected_metrics, config)

	report = {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"selected_metrics": selected_metrics,
		"summary": run_result["summary"],
		"details": run_result["details"],
	}

	output_path = Path(args.output_json)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=False)

	print("=" * 72)
	print("ReCamMaster Evaluation Report")
	print("=" * 72)
	print(f"Selected metrics: {', '.join(selected_metrics)}")
	print("Summary:")
	for key, value in report["summary"].items():
		print(f"  - {key}: {value}")
	print(f"Report saved to: {output_path}")


if __name__ == "__main__":
	main()
