from __future__ import annotations

from typing import Dict


def run(config: Dict) -> Dict:
	"""
	Placeholder for visual quality metrics:
	- FID
	- FVD
	- FVD-V (SV4D setting)
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
		"reason": "FVD/FID implementation pending (feature extractor + stats)",
		"metrics": {
			"FID": None,
			"FVD": None,
			"FVD-V": None,
		},
	}
