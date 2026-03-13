from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.video import R3D_18_Weights, r3d_18


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
_KINETICS_STD = (0.22803, 0.22145, 0.216989)

_FID_EXTRACTOR_CACHE: Dict[str, nn.Module] = {}
_FVD_EXTRACTOR_CACHE: Dict[str, nn.Module] = {}


class _InceptionFeatureExtractor(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		base = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
		base.fc = nn.Identity()
		base.aux_logits = False
		base.AuxLogits = None
		self.backbone = base

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		if features.ndim > 2:
			return features.flatten(start_dim=1)
		return features


class _R3DFeatureExtractor(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		base = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
		self.backbone = nn.Sequential(*list(base.children())[:-1])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		return features.flatten(start_dim=1)


def _resolve_device(requested_device: str) -> str:
	requested = str(requested_device).strip().lower()
	if requested not in {"cpu", "cuda"}:
		requested = "cuda"
	if requested == "cuda" and not torch.cuda.is_available():
		return "cpu"
	return requested


def _load_fid_extractor(device: str) -> nn.Module:
	model = _FID_EXTRACTOR_CACHE.get(device)
	if model is None:
		model = _InceptionFeatureExtractor().to(device)
		model.eval()
		for param in model.parameters():
			param.requires_grad_(False)
		_FID_EXTRACTOR_CACHE[device] = model
	return model


def _load_fvd_extractor(device: str) -> nn.Module:
	model = _FVD_EXTRACTOR_CACHE.get(device)
	if model is None:
		model = _R3DFeatureExtractor().to(device)
		model.eval()
		for param in model.parameters():
			param.requires_grad_(False)
		_FVD_EXTRACTOR_CACHE[device] = model
	return model


def _sample_video_frames(video_path: Path, max_frames: int, frame_stride: int, resize_short: int) -> List[np.ndarray]:
	if not video_path.is_file():
		raise FileNotFoundError(f"Video not found: {video_path}")

	container = av.open(str(video_path))
	stream = container.streams.video[0]
	frames: List[np.ndarray] = []

	for idx, frame in enumerate(container.decode(stream)):
		if idx % max(1, frame_stride) != 0:
			continue
		rgb = frame.to_ndarray(format="rgb24")

		if resize_short > 0:
			h, w = rgb.shape[:2]
			short_side = min(h, w)
			if short_side > resize_short:
				scale = resize_short / float(short_side)
				new_w = int(round(w * scale))
				new_h = int(round(h * scale))
				rgb = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BICUBIC))

		frames.append(rgb)
		if len(frames) >= max_frames:
			break

	container.close()
	return frames


def _build_start_indices(frame_count: int, clip_len: int, clip_stride: int) -> List[int]:
	if frame_count <= 0:
		return []
	clip_len = max(1, clip_len)
	clip_stride = max(1, clip_stride)
	if frame_count <= clip_len:
		return [0]
	starts = list(range(0, frame_count - clip_len + 1, clip_stride))
	if not starts:
		starts = [0]
	if starts[-1] != frame_count - clip_len:
		starts.append(frame_count - clip_len)
	return starts


def _build_clips(frames_rgb: Sequence[np.ndarray], clip_len: int, starts: Sequence[int]) -> List[np.ndarray]:
	clips: List[np.ndarray] = []
	if not frames_rgb:
		return clips
	n = len(frames_rgb)
	clip_len = max(1, clip_len)
	for start in starts:
		end = start + clip_len
		window = list(frames_rgb[start:min(end, n)])
		while len(window) < clip_len:
			window.append(window[-1])
		clips.append(np.stack(window, axis=0))
	return clips


def _normalize_image_batch(batch: torch.Tensor) -> torch.Tensor:
	mean = torch.tensor(_IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
	std = torch.tensor(_IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
	return (batch - mean) / std


def _normalize_video_batch(batch: torch.Tensor) -> torch.Tensor:
	mean = torch.tensor(_KINETICS_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1, 1)
	std = torch.tensor(_KINETICS_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1, 1)
	return (batch - mean) / std


def _extract_fid_features(
	frames_rgb: Sequence[np.ndarray],
	model: nn.Module,
	device: str,
	batch_size: int,
) -> np.ndarray:
	if not frames_rgb:
		return np.zeros((0, 2048), dtype=np.float64)

	features: List[np.ndarray] = []
	batch_size = max(1, int(batch_size))
	with torch.no_grad():
		for start in range(0, len(frames_rgb), batch_size):
			chunk = frames_rgb[start:start + batch_size]
			tensor = torch.from_numpy(np.stack(chunk, axis=0)).to(device=device, dtype=torch.float32) / 255.0
			tensor = tensor.permute(0, 3, 1, 2).contiguous()
			tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
			tensor = _normalize_image_batch(tensor)
			chunk_features = model(tensor).float().cpu().numpy().astype(np.float64)
			features.append(chunk_features)
	if not features:
		return np.zeros((0, 2048), dtype=np.float64)
	return np.concatenate(features, axis=0)


def _extract_fvd_features(
	clips_rgb: Sequence[np.ndarray],
	model: nn.Module,
	device: str,
	batch_size: int,
) -> np.ndarray:
	if not clips_rgb:
		return np.zeros((0, 512), dtype=np.float64)

	features: List[np.ndarray] = []
	batch_size = max(1, int(batch_size))
	with torch.no_grad():
		for start in range(0, len(clips_rgb), batch_size):
			chunk = clips_rgb[start:start + batch_size]
			tensor = torch.from_numpy(np.stack(chunk, axis=0)).to(device=device, dtype=torch.float32) / 255.0
			tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
			b, c, t, h, w = tensor.shape
			resized = F.interpolate(
				tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
				size=(112, 112),
				mode="bilinear",
				align_corners=False,
			)
			tensor = resized.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4).contiguous()
			tensor = _normalize_video_batch(tensor)
			chunk_features = model(tensor).float().cpu().numpy().astype(np.float64)
			features.append(chunk_features)
	if not features:
		return np.zeros((0, 512), dtype=np.float64)
	return np.concatenate(features, axis=0)


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
	sym = 0.5 * (matrix + matrix.T)
	eigvals, eigvecs = np.linalg.eigh(sym)
	eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
	sqrt_eigvals = np.sqrt(eigvals)
	return (eigvecs * sqrt_eigvals) @ eigvecs.T


def _mean_cov(features: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
	if features.ndim != 2:
		raise ValueError(f"Expected 2D features, got shape {features.shape}")
	mu = features.mean(axis=0)
	d = features.shape[1]
	if features.shape[0] <= 1:
		sigma = np.eye(d, dtype=np.float64) * eps
	else:
		sigma = np.cov(features, rowvar=False)
		sigma = np.asarray(sigma, dtype=np.float64)
		sigma += np.eye(d, dtype=np.float64) * eps
	return mu, sigma


def _frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
	if features_a.shape[0] == 0 or features_b.shape[0] == 0:
		raise ValueError("Empty feature set")

	mu_a, sigma_a = _mean_cov(features_a)
	mu_b, sigma_b = _mean_cov(features_b)

	diff = mu_a - mu_b
	sqrt_sigma_a = _sqrtm_psd(sigma_a)
	covmean = _sqrtm_psd(sqrt_sigma_a @ sigma_b @ sqrt_sigma_a)

	distance = float(diff @ diff + np.trace(sigma_a + sigma_b - 2.0 * covmean))
	if not np.isfinite(distance):
		raise ValueError("Non-finite Fréchet distance")
	return max(distance, 0.0)


def _compute_fid(
	src_frames: Sequence[np.ndarray],
	tgt_frames: Sequence[np.ndarray],
	model: nn.Module,
	device: str,
	batch_size: int,
) -> float:
	src_features = _extract_fid_features(src_frames, model=model, device=device, batch_size=batch_size)
	tgt_features = _extract_fid_features(tgt_frames, model=model, device=device, batch_size=batch_size)
	return _frechet_distance(src_features, tgt_features)


def _compute_fvd(
	src_frames: Sequence[np.ndarray],
	tgt_frames: Sequence[np.ndarray],
	model: nn.Module,
	device: str,
	batch_size: int,
	clip_len: int,
	clip_stride: int,
) -> float:
	src_starts = _build_start_indices(len(src_frames), clip_len=clip_len, clip_stride=clip_stride)
	tgt_starts = _build_start_indices(len(tgt_frames), clip_len=clip_len, clip_stride=clip_stride)
	src_clips = _build_clips(src_frames, clip_len=clip_len, starts=src_starts)
	tgt_clips = _build_clips(tgt_frames, clip_len=clip_len, starts=tgt_starts)
	src_features = _extract_fvd_features(src_clips, model=model, device=device, batch_size=batch_size)
	tgt_features = _extract_fvd_features(tgt_clips, model=model, device=device, batch_size=batch_size)
	return _frechet_distance(src_features, tgt_features)


def _compute_fvd_v(
	src_frames: Sequence[np.ndarray],
	tgt_frames: Sequence[np.ndarray],
	model: nn.Module,
	device: str,
	batch_size: int,
	clip_len: int,
	clip_stride: int,
) -> float:
	shared_len = min(len(src_frames), len(tgt_frames))
	if shared_len <= 0:
		raise ValueError("No shared frames between source and target for FVD-V")

	src_shared = src_frames[:shared_len]
	tgt_shared = tgt_frames[:shared_len]
	starts = _build_start_indices(shared_len, clip_len=clip_len, clip_stride=clip_stride)
	src_clips = _build_clips(src_shared, clip_len=clip_len, starts=starts)
	tgt_clips = _build_clips(tgt_shared, clip_len=clip_len, starts=starts)

	src_features = _extract_fvd_features(src_clips, model=model, device=device, batch_size=batch_size)
	tgt_features = _extract_fvd_features(tgt_clips, model=model, device=device, batch_size=batch_size)
	return _frechet_distance(src_features, tgt_features)


def run(config: Dict[str, Any]) -> Dict[str, Any]:
	required_inputs = ["source_video", "target_video"]
	missing = [name for name in required_inputs if not config.get(name)]
	if missing:
		return {
			"status": "skipped",
			"reason": f"Missing required inputs: {', '.join(missing)}",
			"metrics": {},
		}

	source_video = Path(str(config["source_video"]))
	target_video = Path(str(config["target_video"]))

	frame_stride = int(config.get("fvd_frame_stride", 4))
	max_frames = int(config.get("fvd_max_frames", 48))
	resize_short = int(config.get("fvd_resize_short", 256))
	clip_len = int(config.get("fvd_clip_len", 16))
	clip_stride = int(config.get("fvd_clip_stride", 8))
	view_clip_len = int(config.get("fvd_view_clip_len", max(2, clip_len // 2)))
	device = _resolve_device(str(config.get("fvd_device", "cuda")))
	fid_batch_size = int(config.get("fvd_frame_batch_size", 16))
	fvd_batch_size = int(config.get("fvd_video_batch_size", 4))

	try:
		src_frames = _sample_video_frames(
			source_video,
			max_frames=max_frames,
			frame_stride=frame_stride,
			resize_short=resize_short,
		)
		tgt_frames = _sample_video_frames(
			target_video,
			max_frames=max_frames,
			frame_stride=frame_stride,
			resize_short=resize_short,
		)
		if not src_frames or not tgt_frames:
			raise ValueError("No decoded frames for source/target")

		fid_model = _load_fid_extractor(device)
		fvd_model = _load_fvd_extractor(device)

		fid_value = _compute_fid(
			src_frames,
			tgt_frames,
			model=fid_model,
			device=device,
			batch_size=fid_batch_size,
		)
		fvd_value = _compute_fvd(
			src_frames,
			tgt_frames,
			model=fvd_model,
			device=device,
			batch_size=fvd_batch_size,
			clip_len=clip_len,
			clip_stride=clip_stride,
		)
		fvd_v_value = _compute_fvd_v(
			src_frames,
			tgt_frames,
			model=fvd_model,
			device=device,
			batch_size=fvd_batch_size,
			clip_len=view_clip_len,
			clip_stride=clip_stride,
		)

		return {
			"status": "ok",
			"reason": (
				"Using torchvision pretrained backbones: "
				"Inception-V3 (FID) and R3D-18/Kinetics-400 (FVD, FVD-V)"
			),
			"metrics": {
				"FID": float(fid_value),
				"FVD": float(fvd_value),
				"FVD-V": float(fvd_v_value),
			},
		}
	except Exception as exc:  # noqa: BLE001
		return {
			"status": "failed",
			"reason": f"FVD/FID evaluation failed: {exc}",
			"metrics": {},
		}
