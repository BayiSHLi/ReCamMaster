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

def align_poses_umeyama(gt_centers, pred_centers):
    """
    Align predicted camera centers to GT centers using Umeyama.
    gt_centers: (T,3)
    pred_centers: (T,3)
    """
    mu_gt = np.mean(gt_centers, axis=0)
    mu_pred = np.mean(pred_centers, axis=0)

    X = gt_centers - mu_gt
    Y = pred_centers - mu_pred

    cov = Y.T @ X / len(X)
    U, S, Vt = np.linalg.svd(cov)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = U @ Vt

    scale = np.trace(np.diag(S)) / np.sum(Y**2)

    t = mu_gt - scale * R @ mu_pred

    return scale, R, t