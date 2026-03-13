from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List


def _empty_metrics() -> Dict[str, float | None]:
    return {
        "Aesthetic Quality": None,
        "Imaging Quality": None,
        "Temporal Flickering": None,
        "Motion Smoothness": None,
        "Subject Consistency": None,
        "Background Consistency": None,
    }


def _build_default_pending(reason: str) -> Dict[str, Any]:
    return {
        "status": "pending",
        "reason": reason,
        "metrics": _empty_metrics(),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _score01_to_percent(score01: float | None, gamma: float = 0.35) -> float | None:
    if score01 is None:
        return None
    x = _clamp(float(score01), 0.0, 1.0)
    # Gamma < 1 expands high-quality range to match VBench's percentage-style reporting.
    return float(100.0 * (x ** gamma))


def _normalize_raw_score_to_percent(raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    x = float(raw_score)
    if not math.isfinite(x):
        return None

    # Common score ranges used by quality/reward models.
    if -1.0 <= x <= 1.0:
        return float((x + 1.0) * 50.0)
    if 0.0 <= x <= 10.0:
        return float(x * 10.0)
    if 0.0 <= x <= 100.0:
        return float(x)

    # Fallback for logit-like or unconstrained outputs.
    return float(100.0 / (1.0 + math.exp(-x)))


def _sample_video_rgb_frames(video_path: str, max_frames: int, frame_stride: int) -> List[Any]:
    import av
    import numpy as np

    frames: List[Any] = []
    container = av.open(video_path)
    stream = container.streams.video[0]

    for idx, frame in enumerate(container.decode(stream)):
        if idx % max(1, frame_stride) != 0:
            continue
        rgb = frame.to_ndarray(format="rgb24")
        frames.append(np.asarray(rgb))
        if len(frames) >= max_frames:
            break

    container.close()
    return frames


def _compute_temporal_flickering(frames_rgb: List[Any]) -> float | None:
    import numpy as np

    if len(frames_rgb) < 3:
        return None

    luminance_means: List[float] = []
    for frame in frames_rgb:
        # Use luminance to estimate frame-level exposure/brightness oscillation.
        gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        luminance_means.append(float(np.mean(gray) / 255.0))

    means = np.asarray(luminance_means, dtype=np.float64)
    second_diff = np.diff(means, n=2)
    if second_diff.size == 0:
        return None

    jitter = float(np.mean(np.abs(second_diff)))
    score = 1.0 / (1.0 + 30.0 * jitter)
    return float(max(0.0, min(1.0, score)))


def _compute_motion_smoothness(frames_rgb: List[Any]) -> float | None:
    try:
        import cv2
    except Exception:
        return None

    import numpy as np

    if len(frames_rgb) < 4:
        return None

    velocities: List[Any] = []
    prev_gray = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for curr_rgb in frames_rgb[1:]:
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        # Median flow is robust to local outliers/occlusions.
        velocities.append(np.median(flow.reshape(-1, 2), axis=0))
        prev_gray = curr_gray

    vel = np.asarray(velocities, dtype=np.float64)
    if vel.shape[0] < 2:
        return None

    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(np.diff(vel, axis=0), axis=1)
    speed_mean = float(np.mean(speed))
    accel_mean = float(np.mean(accel))

    # Lower acceleration relative to speed implies smoother motion.
    jitter_ratio = accel_mean / (speed_mean + 1e-6)
    score = 1.0 / (1.0 + jitter_ratio)
    return float(max(0.0, min(1.0, score)))


def _build_region_masks(height: int, width: int) -> tuple[Any, Any]:
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    # Elliptical center prior for subject-like region.
    ry = max(1.0, height * 0.28)
    rx = max(1.0, width * 0.22)
    ellipse = ((yy - cy) ** 2) / (ry * ry) + ((xx - cx) ** 2) / (rx * rx)
    subject_mask = ellipse <= 1.0
    background_mask = ~subject_mask
    return subject_mask, background_mask


def _apply_region_mask(frame_rgb: Any, region_mask: Any) -> Any:
    import numpy as np

    masked = np.full_like(frame_rgb, 127, dtype=np.uint8)
    masked[region_mask] = frame_rgb[region_mask]
    return masked


def _mean_adjacent_cosine(features: Any) -> float | None:
    import numpy as np

    if features is None or len(features) < 2:
        return None
    a = features[:-1]
    b = features[1:]
    sims = np.sum(a * b, axis=1)
    return float(np.mean(sims))


def _load_diffsynth_model(
    model_name: str,
    cache_dir: str,
    requested_device: str,
) -> Any:
    import torch
    from diffsynth.extensions.ImageQualityMetric import (
        download_preference_model,
        load_preference_model,
    )

    device = requested_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Force the official loading path: missing assets must be downloaded.
    paths = download_preference_model(model_name, cache_dir=cache_dir)

    return load_preference_model(model_name=model_name, device=device, path=paths)


def _encode_clip_features_for_images(clip_scorer: Any, pil_images: List[Any]) -> Any:
    import torch

    device = clip_scorer.device
    batch = torch.stack([clip_scorer.preprocess_val(img) for img in pil_images], dim=0).to(device=device, non_blocking=True)
    with torch.no_grad():
        feats = clip_scorer.model.encode_image(batch, normalize=True)
    return feats.float().cpu().numpy()


def _compute_region_consistency_clip(frames_rgb: List[Any], clip_scorer: Any, region: str) -> float | None:
    from PIL import Image

    if len(frames_rgb) < 2:
        return None

    h, w = frames_rgb[0].shape[:2]
    subject_mask, background_mask = _build_region_masks(h, w)
    use_mask = subject_mask if region == "subject" else background_mask

    masked_pil = [Image.fromarray(_apply_region_mask(frame, use_mask)) for frame in frames_rgb]
    feats = _encode_clip_features_for_images(clip_scorer=clip_scorer, pil_images=masked_pil)
    sim = _mean_adjacent_cosine(feats)
    if sim is None:
        return None
    return _clamp((sim + 1.0) * 50.0, 0.0, 100.0)


def _score_image_quality_subset(
    frames_rgb: List[Any],
    prompt: str,
    cache_dir: str,
    requested_device: str,
) -> Dict[str, float]:
    import torch
    from PIL import Image

    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"

    prompt = prompt.strip() or "high quality video"
    pil_frames = [Image.fromarray(frame) for frame in frames_rgb]

    aesthetic_model = _load_diffsynth_model(
        model_name="Aesthetic",
        cache_dir=cache_dir,
        requested_device=requested_device,
    )
    aesthetic_scores = aesthetic_model.score(images=pil_frames, prompt=prompt)
    aesthetic = float(statistics.fmean(float(x) for x in aesthetic_scores))

    mps_model = _load_diffsynth_model(
        model_name="MPS",
        cache_dir=cache_dir,
        requested_device=requested_device,
    )
    imaging_scores = mps_model.score(images=pil_frames, prompt=prompt)
    imaging = float(statistics.fmean(float(x) for x in imaging_scores))

    return {
        "Aesthetic Quality": _normalize_raw_score_to_percent(aesthetic),
        "Imaging Quality": _normalize_raw_score_to_percent(imaging),
    }


def _compute_consistency_subset(
    frames_rgb: List[Any],
    cache_dir: str,
    requested_device: str,
) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {
        "Subject Consistency": None,
        "Background Consistency": None,
    }

    clip_scorer = _load_diffsynth_model(
        model_name="CLIP",
        cache_dir=cache_dir,
        requested_device=requested_device,
    )
    metrics["Subject Consistency"] = _compute_region_consistency_clip(frames_rgb, clip_scorer, region="subject")
    metrics["Background Consistency"] = _compute_region_consistency_clip(frames_rgb, clip_scorer, region="background")
    return metrics


def _build_result(metrics: Dict[str, float | None], notes: List[str]) -> Dict[str, Any]:
    required_keys = [
        "Aesthetic Quality",
        "Imaging Quality",
        "Temporal Flickering",
        "Motion Smoothness",
        "Subject Consistency",
        "Background Consistency",
    ]
    all_ready = all(
        isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
        for key in required_keys
    )
    status = "ok" if all_ready else "failed"
    return {
        "status": status,
        "reason": " | ".join(notes) if notes else "",
        "metrics": metrics,
    }


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    generated_video = str(config.get("generated_video", "")).strip()
    if not generated_video:
        return _build_default_pending("Missing 'generated_video' in config")

    metrics: Dict[str, float | None] = _empty_metrics()
    notes: List[str] = []

    try:
        frame_stride = int(config.get("vbench_frame_stride", 8))
        max_frames = int(config.get("vbench_max_frames", 8))
        frames_rgb = _sample_video_rgb_frames(
            video_path=generated_video,
            max_frames=max_frames,
            frame_stride=frame_stride,
        )
    except Exception as exc:  # noqa: BLE001
        return _build_default_pending(f"Failed to decode video frames: {exc}")

    if not frames_rgb:
        return _build_default_pending("No frames decoded from generated video")

    temporal_flickering = _compute_temporal_flickering(frames_rgb)
    if temporal_flickering is not None:
        metrics["Temporal Flickering"] = _score01_to_percent(temporal_flickering, gamma=0.35)
    else:
        notes.append("Temporal Flickering unavailable (need at least 3 sampled frames)")

    motion_smoothness = _compute_motion_smoothness(frames_rgb)
    if motion_smoothness is not None:
        metrics["Motion Smoothness"] = _score01_to_percent(motion_smoothness, gamma=0.35)
    else:
        notes.append("Motion Smoothness unavailable (cv2 missing or insufficient sampled frames)")

    prompt = str(config.get("text_prompt", ""))
    cache_dir = str(config.get("vbench_cache_dir", "models"))
    requested_device = str(config.get("vbench_device", "cuda")).lower()
    # Strict mode: force official diffsynth models without fallback proxies.
    try:
        image_metrics = _score_image_quality_subset(
            frames_rgb=frames_rgb,
            prompt=prompt,
            cache_dir=cache_dir,
            requested_device=requested_device,
        )
        metrics.update(image_metrics)
        notes.append("Aesthetic/Imaging computed with diffsynth models")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "reason": f"Aesthetic/Imaging diffsynth computation failed: {exc}",
            "metrics": metrics,
        }

    try:
        consistency_metrics = _compute_consistency_subset(
            frames_rgb=frames_rgb,
            cache_dir=cache_dir,
            requested_device=requested_device,
        )
        metrics.update(consistency_metrics)
        if consistency_metrics.get("Subject Consistency") is not None and consistency_metrics.get("Background Consistency") is not None:
            notes.append("Subject/Background consistency computed")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "reason": f"Subject/Background diffsynth computation failed: {exc}",
            "metrics": metrics,
        }

    return _build_result(metrics=metrics, notes=notes)
