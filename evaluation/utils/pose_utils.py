import numpy as np

def rotation_error(R_gt, R_pred):
    """
    R_gt, R_pred: (3,3)
    return rotation error in degrees
    """
    R_rel = R_gt.T @ R_pred
    trace = np.trace(R_rel)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle = np.arccos(trace)
    return np.degrees(angle)


def translation_error(t_gt, t_pred):
    """
    t_gt, t_pred: (3,)
    """
    return np.linalg.norm(t_gt - t_pred)


def evaluate_camera_sequence(gt_poses, pred_poses):
    """
    gt_poses: (T,4,4)
    pred_poses: (T,4,4)
    """
    T = gt_poses.shape[0]

    rot_errors = []
    trans_errors = []

    for t in range(T):
        R_gt = gt_poses[t][:3, :3]
        t_gt = gt_poses[t][:3, 3]

        R_pred = pred_poses[t][:3, :3]
        t_pred = pred_poses[t][:3, 3]

        rot_errors.append(rotation_error(R_gt, R_pred))
        trans_errors.append(translation_error(t_gt, t_pred))

    return {
        "mean_RotErr": np.mean(rot_errors),
        "mean_TransErr": np.mean(trans_errors),
        "frame_RotErr": np.array(rot_errors),
        "frame_TransErr": np.array(trans_errors)
    }


def evaluate_camera_sequence_cameractrl_paper(gt_poses, pred_poses):
    """
    CameraCtrl/CamI2V-compatible metric aggregation.

    RotErr: per-frame rotation error in radians, summed across frames.
    TransErr: per-frame L2 translation error, summed across frames.
    """
    gt_poses = np.asarray(gt_poses, dtype=np.float64)
    pred_poses = np.asarray(pred_poses, dtype=np.float64)
    if gt_poses.shape != pred_poses.shape:
        raise ValueError(f"gt/pred shape mismatch: {gt_poses.shape} vs {pred_poses.shape}")
    if gt_poses.ndim != 3 or gt_poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (T,4,4), got {gt_poses.shape}")

    T = gt_poses.shape[0]
    rot_errors_rad = []
    trans_errors = []

    for t in range(T):
        R_gt = gt_poses[t][:3, :3]
        t_gt = gt_poses[t][:3, 3]

        R_pred = pred_poses[t][:3, :3]
        t_pred = pred_poses[t][:3, 3]

        # Same angular definition as CameraCtrl/CamI2V: arccos((tr(R_gt^T R_pred)-1)/2).
        cos_theta = np.clip((np.trace(R_gt.T @ R_pred) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors_rad.append(float(np.arccos(cos_theta)))
        trans_errors.append(float(np.linalg.norm(t_gt - t_pred)))

    rot_errors_rad = np.asarray(rot_errors_rad, dtype=np.float64)
    trans_errors = np.asarray(trans_errors, dtype=np.float64)
    return {
        "RotErr_rad_sum": float(np.sum(rot_errors_rad)),
        "TransErr_sum": float(np.sum(trans_errors)),
        "frame_RotErr_rad": rot_errors_rad,
        "frame_TransErr": trans_errors,
        "num_frames": int(T),
    }

def to_relative_poses(poses):
    """
    Convert absolute c2w poses to relative poses by setting frame 0 to identity.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    if poses.shape[0] == 0:
        raise ValueError("poses is empty")

    base_inv = np.linalg.inv(poses[0])
    relative = np.stack([base_inv @ pose for pose in poses], axis=0)
    return relative


def rescale_translation_with_first_gap(gt_relative_poses, pred_relative_poses):
    """
    Rescale generated trajectory translation using the frame-0 to frame-1 gap.

    This follows CameraCtrl Appendix D.5 postprocessing for COLMAP scale ambiguity.
    """
    gt_relative_poses = np.asarray(gt_relative_poses, dtype=np.float64)
    pred_relative_poses = np.asarray(pred_relative_poses, dtype=np.float64)

    if gt_relative_poses.shape[0] < 2 or pred_relative_poses.shape[0] < 2:
        raise ValueError("Need at least 2 frames to estimate CameraCtrl translation scale")

    gt_gap = np.linalg.norm(gt_relative_poses[1, :3, 3] - gt_relative_poses[0, :3, 3])
    pred_gap = np.linalg.norm(pred_relative_poses[1, :3, 3] - pred_relative_poses[0, :3, 3])
    if pred_gap <= 1e-12:
        raise ValueError("Predicted translation gap between first two frames is near zero")

    scale = gt_gap / pred_gap
    scaled_pred = pred_relative_poses.copy()
    scaled_pred[:, :3, 3] *= scale
    return scaled_pred