from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import av
import numpy as np
from PIL import Image


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


def _frame_feature(frame_rgb: np.ndarray) -> np.ndarray:
	frame = frame_rgb.astype(np.float32) / 255.0
	mean_rgb = frame.mean(axis=(0, 1))
	std_rgb = frame.std(axis=(0, 1))

	gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
	gy, gx = np.gradient(gray)
	grad = np.sqrt(gx * gx + gy * gy)
	grad_stats = np.array([grad.mean(), grad.std()], dtype=np.float64)

	hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
	return np.concatenate([mean_rgb, std_rgb, grad_stats, hist.astype(np.float64)], axis=0)


def _frame_features(frames_rgb: List[np.ndarray]) -> np.ndarray:
	if not frames_rgb:
		return np.zeros((0, 24), dtype=np.float64)
	features = [_frame_feature(frame) for frame in frames_rgb]
	return np.stack(features, axis=0).astype(np.float64)


def _build_clip_features(frames_rgb: List[np.ndarray], clip_len: int, clip_stride: int) -> np.ndarray:
	if not frames_rgb:
		return np.zeros((0, 51), dtype=np.float64)

	n = len(frames_rgb)
	if n < clip_len:
		windows = [(0, n)]
	else:
		windows = [(start, start + clip_len) for start in range(0, n - clip_len + 1, max(1, clip_stride))]
		if not windows:
			windows = [(0, n)]

	clip_features: List[np.ndarray] = []
	for start, end in windows:
		window_frames = frames_rgb[start:end]
		frame_feat = _frame_features(window_frames)

		arr = np.stack(window_frames, axis=0).astype(np.float32) / 255.0
		if arr.shape[0] >= 2:
			motion = np.mean(np.abs(np.diff(arr, axis=0)), axis=(1, 2, 3))
			accel = np.abs(np.diff(motion)) if motion.shape[0] >= 2 else np.zeros((0,), dtype=np.float32)
			motion_stats = np.array(
				[
					float(motion.mean()),
					float(motion.std()),
					float(accel.mean()) if accel.size > 0 else 0.0,
				],
				dtype=np.float64,
			)
		else:
			motion_stats = np.zeros((3,), dtype=np.float64)

		clip_feature = np.concatenate([frame_feat.mean(axis=0), frame_feat.std(axis=0), motion_stats], axis=0)
		clip_features.append(clip_feature)

	return np.stack(clip_features, axis=0).astype(np.float64)


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

		src_frame_feat = _frame_features(src_frames)
		tgt_frame_feat = _frame_features(tgt_frames)
		fid_value = _frechet_distance(src_frame_feat, tgt_frame_feat)

		src_clip_feat = _build_clip_features(src_frames, clip_len=clip_len, clip_stride=clip_stride)
		tgt_clip_feat = _build_clip_features(tgt_frames, clip_len=clip_len, clip_stride=clip_stride)
		fvd_value = _frechet_distance(src_clip_feat, tgt_clip_feat)

		src_delta_feat = np.diff(src_frame_feat, axis=0)
		tgt_delta_feat = np.diff(tgt_frame_feat, axis=0)
		if src_delta_feat.shape[0] == 0:
			src_delta_feat = src_frame_feat[:1]
		if tgt_delta_feat.shape[0] == 0:
			tgt_delta_feat = tgt_frame_feat[:1]
		fvd_v_value = _frechet_distance(src_delta_feat, tgt_delta_feat)

		return {
			"status": "ok",
			"reason": "Using deterministic lightweight feature backend for FID/FVD/FVD-V",
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
