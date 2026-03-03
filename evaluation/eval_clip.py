from __future__ import annotations

from typing import Dict


def run(config: Dict) -> Dict:
	"""
	Placeholder for CLIP-based metrics:
	- CLIP-V: source-target frame similarity at same timestamp
	- CLIP-T: frame-text similarity
	- CLIP-F: adjacent-frame temporal similarity
	"""
	required_inputs = ["generated_video"]
	missing = [name for name in required_inputs if not config.get(name)]

	if missing:
		return {
			"status": "skipped",
			"reason": f"Missing required inputs: {', '.join(missing)}",
			"metrics": {},
		}

	return {
		"status": "pending",
		"reason": "CLIP metric implementation pending (frame sampling + CLIP encoder)",
		"metrics": {
			"CLIP-V": None,
			"CLIP-T": None,
			"CLIP-F": None,
		},
	}
