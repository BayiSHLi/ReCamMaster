from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import av
import torch
from PIL import Image


def _sample_video_frames(video_path: Path, max_frames: int, frame_stride: int) -> List[Image.Image]:
	if not video_path.is_file():
		raise FileNotFoundError(f"Video not found: {video_path}")

	container = av.open(str(video_path))
	stream = container.streams.video[0]
	frames: List[Image.Image] = []

	for idx, frame in enumerate(container.decode(stream)):
		if idx % max(1, frame_stride) != 0:
			continue
		rgb = frame.to_ndarray(format="rgb24")
		frames.append(Image.fromarray(rgb))
		if len(frames) >= max_frames:
			break

	container.close()
	return frames


def _resolve_device(requested_device: str) -> str:
	requested = str(requested_device).lower()
	if requested == "cuda" and not torch.cuda.is_available():
		return "cpu"
	return requested


def _load_clip_scorer(cache_dir: str, device: str) -> Any:
	from diffsynth.extensions.ImageQualityMetric import (
		download_preference_model,
		load_preference_model,
	)

	clip_path = download_preference_model("CLIP", cache_dir=cache_dir)
	return load_preference_model(model_name="CLIP", device=device, path=clip_path)


def _encode_image_features(scorer: Any, images: List[Image.Image]) -> torch.Tensor:
	if not images:
		return torch.empty((0, 1), dtype=torch.float32)

	device = scorer.device
	batch = torch.stack([scorer.preprocess_val(image) for image in images], dim=0).to(device=device, non_blocking=True)
	with torch.no_grad():
		features = scorer.model.encode_image(batch, normalize=True)
	return features.float().cpu()


def _encode_text_feature(scorer: Any, prompt: str) -> torch.Tensor:
	device = scorer.device
	tokens = scorer.tokenizer([prompt]).to(device=device, non_blocking=True)
	with torch.no_grad():
		feature = scorer.model.encode_text(tokens, normalize=True)
	return feature.float().cpu()[0]


def _safe_mean(values: torch.Tensor) -> float | None:
	if values.numel() == 0:
		return None
	return float(values.mean().item())


def run(config: Dict[str, Any]) -> Dict[str, Any]:
	required_inputs = ["generated_video"]
	missing = [name for name in required_inputs if not config.get(name)]
	if missing:
		return {
			"status": "skipped",
			"reason": f"Missing required inputs: {', '.join(missing)}",
			"metrics": {},
		}

	source_video = str(config.get("source_video", "")).strip()
	target_video = str(config.get("target_video", config["generated_video"])).strip()
	prompt = str(config.get("text_prompt", "")).strip()

	frame_stride = int(config.get("clip_frame_stride", 8))
	max_frames = int(config.get("clip_max_frames", 16))
	cache_dir = str(config.get("clip_cache_dir", "models"))
	device = _resolve_device(str(config.get("clip_device", "cuda")))

	notes: List[str] = []
	metrics: Dict[str, float | None] = {
		"CLIP-V": None,
		"CLIP-T": None,
		"CLIP-F": None,
	}

	try:
		scorer = _load_clip_scorer(cache_dir=cache_dir, device=device)
		tgt_frames = _sample_video_frames(Path(target_video), max_frames=max_frames, frame_stride=frame_stride)
		if not tgt_frames:
			raise ValueError("No frames decoded from target video")

		tgt_features = _encode_image_features(scorer, tgt_frames)

		if source_video:
			src_frames = _sample_video_frames(Path(source_video), max_frames=max_frames, frame_stride=frame_stride)
			if src_frames:
				src_features = _encode_image_features(scorer, src_frames)
				pair_count = min(src_features.shape[0], tgt_features.shape[0])
				if pair_count > 0:
					sims = (src_features[:pair_count] * tgt_features[:pair_count]).sum(dim=1)
					metrics["CLIP-V"] = _safe_mean(sims)
				else:
					notes.append("CLIP-V unavailable (no aligned frame pairs)")
			else:
				notes.append("CLIP-V unavailable (no decoded source frames)")
		else:
			notes.append("CLIP-V unavailable (missing source_video)")

		if prompt:
			text_feature = _encode_text_feature(scorer, prompt)
			sims_text = tgt_features @ text_feature
			metrics["CLIP-T"] = _safe_mean(sims_text)
		else:
			notes.append("CLIP-T unavailable (empty text_prompt)")

		if tgt_features.shape[0] >= 2:
			adj_sims = (tgt_features[:-1] * tgt_features[1:]).sum(dim=1)
			metrics["CLIP-F"] = _safe_mean(adj_sims)
		else:
			notes.append("CLIP-F unavailable (need at least 2 frames)")

	except Exception as exc:  # noqa: BLE001
		return {
			"status": "failed",
			"reason": f"CLIP metric evaluation failed: {exc}",
			"metrics": metrics,
		}

	numeric_count = sum(1 for value in metrics.values() if isinstance(value, float))
	status = "ok" if numeric_count > 0 else "pending"
	return {
		"status": status,
		"reason": " | ".join(notes),
		"metrics": metrics,
	}
