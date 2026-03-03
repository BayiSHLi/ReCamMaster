from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
	import cv2
except Exception:
	cv2 = None


def _sample_video_frames(video_path: Path, max_frames: int, frame_stride: int, resize_short: int) -> List[np.ndarray]:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise ValueError(f"Failed to open video: {video_path}")

	frames: List[np.ndarray] = []
	frame_idx = 0

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		if frame_idx % frame_stride == 0:
			h, w = frame.shape[:2]
			short_side = min(h, w)
			if resize_short > 0 and short_side > resize_short:
				scale = resize_short / float(short_side)
				new_w = int(round(w * scale))
				new_h = int(round(h * scale))
				frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
			frames.append(frame)
			if len(frames) >= max_frames:
				break
		frame_idx += 1

	cap.release()
	return frames


def _match_frame_pair(
	src_bgr: np.ndarray,
	tgt_bgr: np.ndarray,
	orb_nfeatures: int,
	ratio_thresh: float,
	ransac_reproj_threshold: float,
) -> Tuple[int, int, int, float]:
	src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
	tgt_gray = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY)

	detector = cv2.ORB_create(nfeatures=orb_nfeatures)
	kp1, des1 = detector.detectAndCompute(src_gray, None)
	kp2, des2 = detector.detectAndCompute(tgt_gray, None)

	if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
		return 0, 0, 0, 0.0

	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
	knn = matcher.knnMatch(des1, des2, k=2)

	good = []
	for pair in knn:
		if len(pair) < 2:
			continue
		m, n = pair
		if m.distance < ratio_thresh * n.distance:
			good.append(m)

	good_count = len(good)
	if good_count < 4:
		return len(kp1), len(kp2), good_count, 0.0

	src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
	tgt_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
	_, inlier_mask = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, ransac_reproj_threshold)

	if inlier_mask is None:
		return len(kp1), len(kp2), good_count, 0.0

	inliers = int(inlier_mask.ravel().sum())
	confidence = float(inliers / max(good_count, 1))
	return len(kp1), len(kp2), inliers, confidence


def evaluate_matching_metric(
	source_video: str,
	target_video: str,
	max_frames: int = 32,
	frame_stride: int = 4,
	resize_short: int = 480,
	orb_nfeatures: int = 1200,
	ratio_thresh: float = 0.75,
	conf_thresh: float = 0.5,
	ransac_reproj_threshold: float = 3.0,
) -> Dict:
	source_path = Path(source_video)
	target_path = Path(target_video)

	if not source_path.exists():
		raise FileNotFoundError(f"Source video not found: {source_video}")
	if not target_path.exists():
		raise FileNotFoundError(f"Target video not found: {target_video}")

	src_frames = _sample_video_frames(source_path, max_frames=max_frames, frame_stride=frame_stride, resize_short=resize_short)
	tgt_frames = _sample_video_frames(target_path, max_frames=max_frames, frame_stride=frame_stride, resize_short=resize_short)

	pair_count = min(len(src_frames), len(tgt_frames))
	if pair_count == 0:
		raise ValueError("No frame pairs available for matching evaluation")

	mat_pix_values: List[float] = []
	confidences: List[float] = []

	for idx in range(pair_count):
		_, _, inliers, confidence = _match_frame_pair(
			src_frames[idx],
			tgt_frames[idx],
			orb_nfeatures=orb_nfeatures,
			ratio_thresh=ratio_thresh,
			ransac_reproj_threshold=ransac_reproj_threshold,
		)
		confidences.append(confidence)
		mat_pix_values.append(float(inliers) if confidence >= conf_thresh else 0.0)

	mat_pix_array = np.asarray(mat_pix_values, dtype=np.float64)
	conf_array = np.asarray(confidences, dtype=np.float64)

	return {
		"Mat.Pix": float(mat_pix_array.mean()),
		"Mat.Pix_total": float(mat_pix_array.sum()),
		"mean_confidence": float(conf_array.mean()),
		"valid_pair_ratio": float((conf_array >= conf_thresh).mean()),
		"num_pairs": int(pair_count),
		"confidence_threshold": float(conf_thresh),
	}


def run(config: Dict) -> Dict:
	if cv2 is None:
		return {
			"status": "skipped",
			"reason": "OpenCV (cv2) is not available in current environment",
			"metrics": {},
		}

	required_inputs = ["source_video", "target_video"]
	missing = [name for name in required_inputs if not config.get(name)]

	if missing:
		return {
			"status": "skipped",
			"reason": f"Missing required inputs: {', '.join(missing)}",
			"metrics": {},
		}

	try:
		result = evaluate_matching_metric(
			source_video=config["source_video"],
			target_video=config["target_video"],
			max_frames=int(config.get("matching_max_frames", 32)),
			frame_stride=int(config.get("matching_frame_stride", 4)),
			resize_short=int(config.get("matching_resize_short", 480)),
			orb_nfeatures=int(config.get("matching_orb_nfeatures", 1200)),
			ratio_thresh=float(config.get("matching_ratio_thresh", 0.75)),
			conf_thresh=float(config.get("matching_conf_thresh", 0.5)),
			ransac_reproj_threshold=float(config.get("matching_ransac_reproj_threshold", 3.0)),
		)
	except Exception as exc:
		return {
			"status": "failed",
			"reason": str(exc),
			"metrics": {},
		}

	return {
		"status": "ok",
		"reason": "",
		"metrics": result,
	}
