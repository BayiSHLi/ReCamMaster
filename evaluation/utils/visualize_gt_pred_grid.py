from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


VISCAM_TRANSFORM_MATRIX = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
ROW_PATTERN = re.compile(r"\[([^\]]+)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize GT/PRED camera trajectories for each video directory. "
            "With --overlay (default), output is 2x5 PRED panels with GT overlaid; "
            "with --no-overlay, output is split 4x5 view (GT1-5, PRED1-5, GT6-10, PRED6-10)."
        )
    )
    parser.add_argument(
        "--trajectory_root",
        type=str,
        default="evaluation/trajectory_test/video_0_glomap",
        help=(
            "Path to one video trajectory directory (contains cam01..cam10) or "
            "a root that contains multiple such video directories."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/trajectory_test/trajectory_visualizations",
        help="Directory to save generated grid images.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to overlay GT trajectory on each PRED subplot (default: enabled).",
    )
    return parser.parse_args()


def parse_camera_pose_viscam(matrix_str: str) -> np.ndarray:
    # Keep behavior aligned with vis_cam.parse_matrix for compatibility.
    rows = matrix_str.strip().split("] [")
    matrix: List[List[float]] = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        values = [float(x) for x in row.split()]
        if len(values) == 3:
            values = values + [0.0]
        if len(values) != 4:
            raise ValueError(f"Invalid camera row values ({len(values)}): {row}")
        matrix.append(values)
    if len(matrix) != 4:
        raise ValueError(f"Invalid camera pose text: {matrix_str}")
    return np.asarray(matrix, dtype=np.float64)


def parse_camera_pose_eval(pose_text: str) -> np.ndarray:
    rows = ROW_PATTERN.findall(pose_text)
    if len(rows) != 4:
        raise ValueError(f"Invalid camera pose text: {pose_text}")
    matrix = np.stack(
        [np.asarray([float(x) for x in row.strip().split()], dtype=np.float64) for row in rows],
        axis=0,
    )
    return matrix


def w2c_to_c2w_eval(w2c: np.ndarray) -> np.ndarray:
    rotation = w2c[:3, :3]
    translation = w2c[3, :3]

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ translation
    return c2w


def _get_c2w_like_viscam(w2cs: Sequence[np.ndarray], relative_c2w: bool = True) -> np.ndarray:
    if len(w2cs) == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)

    if relative_c2w:
        target_cam_c2w = np.eye(4, dtype=np.float64)
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]

    transformed = [VISCAM_TRANSFORM_MATRIX @ pose for pose in ret_poses]
    return np.asarray(transformed, dtype=np.float64)


def _frame_sort_key(frame_name: str) -> int:
    match = re.search(r"(\d+)", frame_name)
    return int(match.group(1)) if match else 10**9


def load_camera_sequence_gt_vis(camera_json: Path, camera_key: str) -> Tuple[np.ndarray, np.ndarray]:
    with camera_json.open("r", encoding="utf-8") as f:
        payload: Dict[str, Dict[str, str]] = json.load(f)

    frame_items = sorted(payload.items(), key=lambda item: _frame_sort_key(item[0]))
    pose_texts: List[str] = []
    frame_indices: List[int] = []

    for frame_name, frame_data in frame_items:
        if camera_key not in frame_data:
            raise KeyError(f"Camera key '{camera_key}' missing in {frame_name}: {camera_json}")
        frame_idx = _frame_sort_key(frame_name)
        if frame_idx >= 10**9:
            continue
        pose_texts.append(frame_data[camera_key])
        frame_indices.append(frame_idx)

    if not pose_texts:
        return np.zeros((0, 4, 4), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    parsed = [parse_camera_pose_viscam(text) for text in pose_texts]
    cameras = np.transpose(np.stack(parsed, axis=0), (0, 2, 1))

    w2cs: List[np.ndarray] = []
    for cam in cameras:
        if cam.shape[0] == 3:
            cam = np.vstack((cam, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)))
        cam = cam[:, [1, 2, 0, 3]]
        cam[:3, 1] *= -1.0
        w2cs.append(np.linalg.inv(cam))

    c2ws = _get_c2w_like_viscam(w2cs, relative_c2w=True)
    if len(c2ws) > 0:
        scale = max(float(np.max(np.abs(c2w[:3, 3]))) for c2w in c2ws)
        if scale > 1e-3:
            c2ws[:, :3, 3] /= scale

    return c2ws, np.asarray(frame_indices, dtype=np.int64)


def load_camera_sequence_pred_eval(camera_json: Path, camera_key: str) -> Tuple[np.ndarray, np.ndarray]:
    with camera_json.open("r", encoding="utf-8") as f:
        payload: Dict[str, Dict[str, str]] = json.load(f)

    frame_items = sorted(payload.items(), key=lambda item: _frame_sort_key(item[0]))
    poses: List[np.ndarray] = []
    frame_indices: List[int] = []

    for frame_name, frame_data in frame_items:
        if camera_key not in frame_data:
            raise KeyError(f"Camera key '{camera_key}' missing in {frame_name}: {camera_json}")
        frame_idx = _frame_sort_key(frame_name)
        if frame_idx >= 10**9:
            continue
        w2c = parse_camera_pose_eval(frame_data[camera_key])
        poses.append(w2c_to_c2w_eval(w2c))
        frame_indices.append(frame_idx)

    if not poses:
        return np.zeros((0, 4, 4), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.stack(poses, axis=0), np.asarray(frame_indices, dtype=np.int64)


def _align_by_frame_indices(
    poses_a: np.ndarray,
    frame_indices_a: np.ndarray,
    poses_b: np.ndarray,
    frame_indices_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(poses_a) == 0 or len(poses_b) == 0:
        return (
            np.zeros((0, 4, 4), dtype=np.float64),
            np.zeros((0, 4, 4), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    idx_map_b = {int(fid): idx for idx, fid in enumerate(frame_indices_b.tolist())}
    aligned_a: List[np.ndarray] = []
    aligned_b: List[np.ndarray] = []
    aligned_ids: List[int] = []

    for a_idx, fid in enumerate(frame_indices_a.tolist()):
        b_idx = idx_map_b.get(int(fid))
        if b_idx is None:
            continue
        aligned_a.append(poses_a[a_idx])
        aligned_b.append(poses_b[b_idx])
        aligned_ids.append(int(fid))

    if not aligned_a:
        return (
            np.zeros((0, 4, 4), dtype=np.float64),
            np.zeros((0, 4, 4), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(aligned_a, axis=0),
        np.stack(aligned_b, axis=0),
        np.asarray(aligned_ids, dtype=np.int64),
    )


def _select_by_frame_indices(
    poses: np.ndarray,
    frame_indices: np.ndarray,
    target_frame_indices: np.ndarray,
) -> np.ndarray:
    if len(poses) == 0 or len(target_frame_indices) == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)

    idx_map = {int(fid): idx for idx, fid in enumerate(frame_indices.tolist())}
    selected: List[np.ndarray] = []
    for fid in target_frame_indices.tolist():
        idx = idx_map.get(int(fid))
        if idx is not None:
            selected.append(poses[idx])

    if not selected:
        return np.zeros((0, 4, 4), dtype=np.float64)
    return np.stack(selected, axis=0)


def _compute_motion_scale_from_pairs(gt_poses: np.ndarray, pred_poses: np.ndarray) -> float:
    """
    Estimate an isotropic scale factor so PRED translation motion magnitude matches GT.

    The factor is computed from frame-aligned pairs and applied on PRED translations
    around the first frame anchor for visualization only.
    """
    if len(gt_poses) < 2 or len(pred_poses) < 2:
        return 1.0

    gt_centers = gt_poses[:, :3, 3]
    pred_centers = pred_poses[:, :3, 3]

    gt_motion = gt_centers - gt_centers[0]
    pred_motion = pred_centers - pred_centers[0]

    gt_amp = float(np.max(np.linalg.norm(gt_motion, axis=1)))
    pred_amp = float(np.max(np.linalg.norm(pred_motion, axis=1)))
    if pred_amp <= 1e-12:
        return 1.0
    return gt_amp / pred_amp


def _rescale_pred_translation_motion(pred_poses: np.ndarray, scale: float) -> np.ndarray:
    if pred_poses is None or len(pred_poses) == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)

    scaled = pred_poses.copy()
    origin = scaled[0, :3, 3].copy()
    scaled[:, :3, 3] = origin + scale * (scaled[:, :3, 3] - origin)
    return scaled


def _estimate_similarity_transform(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    # Estimate similarity transform mapping src -> dst.
    # dst ~= scale * (rotation @ src) + translation.
    if src_points.shape != dst_points.shape:
        raise ValueError(f"Point shape mismatch: {src_points.shape} vs {dst_points.shape}")
    if src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError(f"Points must be (N,3), got {src_points.shape}")

    n = src_points.shape[0]
    if n == 0:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    if n == 1:
        return 1.0, np.eye(3, dtype=np.float64), dst_mean - src_mean

    cov = (dst_centered.T @ src_centered) / float(n)
    u, s_vals, vt = np.linalg.svd(cov)

    diag = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        diag[-1, -1] = -1.0

    rotation = u @ diag @ vt
    src_var = np.sum(src_centered ** 2) / float(n)
    if src_var < 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(s_vals * np.diag(diag)) / src_var)

    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def _apply_similarity_to_poses(
    poses: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    if poses is None or len(poses) == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)

    mapped = poses.copy()
    for i in range(len(mapped)):
        mapped[i, :3, :3] = rotation @ mapped[i, :3, :3]
        mapped[i, :3, 3] = scale * (rotation @ mapped[i, :3, 3]) + translation
    return mapped


def _collect_video_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    # Single video directory mode.
    if (root / "cam01").is_dir():
        return [root]

    # Multi-video directory mode.
    video_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "cam01").is_dir()]
    if not video_dirs:
        raise FileNotFoundError(
            f"No video directories found under {root}. Expect subdirs containing cam01..cam10."
        )
    return video_dirs


def _compute_global_limits(trajectory_sets: Sequence[np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    centers_list = [poses[:, :3, 3] for poses in trajectory_sets if poses is not None and len(poses) > 0]
    if not centers_list:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    centers = np.concatenate(centers_list, axis=0)
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    margin = 0.1 * span

    xlim = (float(mins[0] - margin[0]), float(maxs[0] + margin[0]))
    ylim = (float(mins[1] - margin[1]), float(maxs[1] + margin[1]))
    zlim = (float(mins[2] - margin[2]), float(maxs[2] + margin[2]))
    return xlim, ylim, zlim


def _plot_single_traj(
    ax,
    poses: np.ndarray,
    frame_indices: np.ndarray,
    cmap,
    frame_norm,
    title: str,
    xlim,
    ylim,
    zlim,
    overlay_poses: Optional[np.ndarray] = None,
    overlay_frame_indices: Optional[np.ndarray] = None,
    overlay_cmap=None,
    overlay_frame_norm=None,
) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.view_init(elev=22, azim=-58)

    if poses is None or len(poses) == 0:
        ax.text2D(0.28, 0.48, "No data", transform=ax.transAxes, fontsize=9)
        return

    if (
        overlay_poses is not None
        and len(overlay_poses) > 0
        and overlay_cmap is not None
        and overlay_frame_norm is not None
    ):
        _plot_time_colored_camera_pyramids(
            ax=ax,
            poses=overlay_poses,
            frame_indices=overlay_frame_indices,
            cmap=overlay_cmap,
            frame_norm=overlay_frame_norm,
        )

    _plot_time_colored_camera_pyramids(
        ax=ax,
        poses=poses,
        frame_indices=frame_indices,
        cmap=cmap,
        frame_norm=frame_norm,
    )

    if frame_indices is not None and len(frame_indices) > 0:
        ax.text2D(
            0.02,
            0.02,
            f"frame[{int(frame_indices[0])}->{int(frame_indices[-1])}], N={len(frame_indices)}",
            transform=ax.transAxes,
            fontsize=7,
        )


def _extrinsic2pyramid(
    ax,
    extrinsic: np.ndarray,
    color,
    hw_ratio: float = 9 / 16,
    base_xval: float = 0.08,
    zval: float = 0.15,
) -> None:
    vertex_std = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [base_xval, -base_xval * hw_ratio, zval, 1.0],
            [base_xval, base_xval * hw_ratio, zval, 1.0],
            [-base_xval, base_xval * hw_ratio, zval, 1.0],
            [-base_xval, -base_xval * hw_ratio, zval, 1.0],
        ],
        dtype=np.float64,
    )
    vertex_transformed = vertex_std @ extrinsic.T
    meshes = [
        [vertex_transformed[0, :-1], vertex_transformed[1, :-1], vertex_transformed[2, :-1]],
        [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
        [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
        [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
        [
            vertex_transformed[1, :-1],
            vertex_transformed[2, :-1],
            vertex_transformed[3, :-1],
            vertex_transformed[4, :-1],
        ],
    ]
    ax.add_collection3d(
        Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
    )


def _plot_time_colored_camera_pyramids(
    ax,
    poses: np.ndarray,
    frame_indices: np.ndarray,
    cmap,
    frame_norm,
) -> None:
    if poses is None or len(poses) == 0:
        return

    for pose_idx, pose in enumerate(poses):
        if frame_indices is not None and pose_idx < len(frame_indices):
            frame_value = int(frame_indices[pose_idx])
        else:
            frame_value = int(pose_idx)
        color = cmap(frame_norm(frame_value))
        _extrinsic2pyramid(ax=ax, extrinsic=pose, color=color)


def _build_frame_index_norm(frame_indices_list: Sequence[np.ndarray]) -> mpl.colors.Normalize:
    valid_arrays = [arr for arr in frame_indices_list if arr is not None and len(arr) > 0]
    if not valid_arrays:
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    min_idx = int(min(int(arr.min()) for arr in valid_arrays))
    max_idx = int(max(int(arr.max()) for arr in valid_arrays))
    if min_idx == max_idx:
        max_idx = min_idx + 1
    return mpl.colors.Normalize(vmin=float(min_idx), vmax=float(max_idx))


def _truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def _add_left_dual_colorbars(
    fig,
    gt_frame_norm: mpl.colors.Normalize,
    pred_frame_norm: mpl.colors.Normalize,
    gt_cmap,
    pred_cmap,
) -> None:
    # Put both legends on the far-left side: GT (top), PRED (bottom).
    gt_cax = fig.add_axes([0.012, 0.56, 0.016, 0.30])
    pred_cax = fig.add_axes([0.012, 0.13, 0.016, 0.30])

    gt_cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=gt_frame_norm, cmap=gt_cmap),
        cax=gt_cax,
        orientation="vertical",
    )
    gt_cb.set_label("GT Frame Index", fontsize=9)
    gt_cb.ax.tick_params(labelsize=8)

    pred_cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=pred_frame_norm, cmap=pred_cmap),
        cax=pred_cax,
        orientation="vertical",
    )
    pred_cb.set_label("PRED Frame Index", fontsize=9)
    pred_cb.ax.tick_params(labelsize=8)


def render_video_grid(
    video_dir: Path,
    output_dir: Path,
    dpi: int,
    overlay: bool,
) -> Path:
    cams_gt: List[np.ndarray] = [None] * 10
    cams_pred: List[np.ndarray] = [None] * 10
    cams_gt_ids: List[np.ndarray] = [None] * 10
    cams_pred_ids: List[np.ndarray] = [None] * 10

    for traj_num in range(1, 11):
        cam_dir = video_dir / f"cam{traj_num:02d}"
        gt_json = cam_dir / "gt_camera_for_eval.json"
        pred_json = cam_dir / "pred_camera_for_eval.json"
        camera_key = f"cam{traj_num:02d}"

        if not gt_json.exists() or not pred_json.exists():
            continue

        # GT visual flow stays unchanged (vis_cam-compatible parsing and transform).
        gt_vis_poses, gt_vis_indices = load_camera_sequence_gt_vis(gt_json, camera_key=camera_key)

        # PRED uses raw COLMAP output path only (parse + w2c->c2w), then is aligned to GT coords.
        pred_raw_poses, pred_raw_indices = load_camera_sequence_pred_eval(pred_json, camera_key=camera_key)

        # Align PRED to GT with a similarity transform estimated from frame-matched camera centers.
        gt_for_align, pred_for_align, _ = _align_by_frame_indices(
            gt_vis_poses,
            gt_vis_indices,
            pred_raw_poses,
            pred_raw_indices,
        )

        if len(gt_for_align) > 0 and len(pred_for_align) > 0:
            pred_scale, pred_rot, pred_trans = _estimate_similarity_transform(
                src_points=pred_for_align[:, :3, 3],
                dst_points=gt_for_align[:, :3, 3],
            )
            pred_aligned_poses = _apply_similarity_to_poses(
                poses=pred_raw_poses,
                scale=pred_scale,
                rotation=pred_rot,
                translation=pred_trans,
            )
        else:
            pred_aligned_poses = pred_raw_poses

        cams_gt[traj_num - 1] = gt_vis_poses
        cams_pred[traj_num - 1] = pred_aligned_poses
        cams_gt_ids[traj_num - 1] = gt_vis_indices
        cams_pred_ids[traj_num - 1] = pred_raw_indices

    vis_xlim, vis_ylim, vis_zlim = _compute_global_limits(cams_gt + cams_pred)

    # Use two clearly separated color families to improve GT/PRED discrimination.
    gt_cmap = _truncate_colormap(mpl.cm.Blues, minval=0.25, maxval=1.0)
    pred_cmap = _truncate_colormap(mpl.cm.YlOrRd, minval=0.20, maxval=1.0)
    gt_frame_norm = _build_frame_index_norm(cams_gt_ids)
    pred_frame_norm = _build_frame_index_norm(cams_pred_ids)

    if overlay:
        fig = plt.figure(figsize=(24, 10))
        fig.suptitle(
            f"{video_dir.name} | 2x5 overlay view | PRED (aligned) + GT overlay",
            fontsize=16,
            y=0.99,
        )

        for col in range(5):
            top_idx = col
            bottom_idx = col + 5

            ax = fig.add_subplot(2, 5, 1 + col, projection="3d")
            _plot_single_traj(
                ax=ax,
                poses=cams_pred[top_idx],
                frame_indices=cams_pred_ids[top_idx],
                cmap=pred_cmap,
                frame_norm=pred_frame_norm,
                title=f"cam{top_idx + 1:02d} PRED(aligned)+GT",
                xlim=vis_xlim,
                ylim=vis_ylim,
                zlim=vis_zlim,
                overlay_poses=cams_gt[top_idx],
                overlay_frame_indices=cams_gt_ids[top_idx],
                overlay_cmap=gt_cmap,
                overlay_frame_norm=gt_frame_norm,
            )

            ax = fig.add_subplot(2, 5, 6 + col, projection="3d")
            _plot_single_traj(
                ax=ax,
                poses=cams_pred[bottom_idx],
                frame_indices=cams_pred_ids[bottom_idx],
                cmap=pred_cmap,
                frame_norm=pred_frame_norm,
                title=f"cam{bottom_idx + 1:02d} PRED(aligned)+GT",
                xlim=vis_xlim,
                ylim=vis_ylim,
                zlim=vis_zlim,
                overlay_poses=cams_gt[bottom_idx],
                overlay_frame_indices=cams_gt_ids[bottom_idx],
                overlay_cmap=gt_cmap,
                overlay_frame_norm=gt_frame_norm,
            )
    else:
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle(
            f"{video_dir.name} | 4x5 split view | GT (vis flow) and PRED (aligned to GT)",
            fontsize=16,
            y=0.99,
        )

        for col in range(5):
            top_idx = col
            bottom_idx = col + 5

            ax = fig.add_subplot(4, 5, 1 + col, projection="3d")
            _plot_single_traj(
                ax=ax,
                poses=cams_gt[top_idx],
                frame_indices=cams_gt_ids[top_idx],
                cmap=gt_cmap,
                frame_norm=gt_frame_norm,
                title=f"cam{top_idx + 1:02d} GT",
                xlim=vis_xlim,
                ylim=vis_ylim,
                zlim=vis_zlim,
            )

            ax = fig.add_subplot(4, 5, 6 + col, projection="3d")
            _plot_single_traj(
                ax=ax,
                poses=cams_pred[top_idx],
                frame_indices=cams_pred_ids[top_idx],
                cmap=pred_cmap,
                frame_norm=pred_frame_norm,
                title=f"cam{top_idx + 1:02d} PRED(aligned)",
                xlim=vis_xlim,
                ylim=vis_ylim,
                zlim=vis_zlim,
            )

            ax = fig.add_subplot(4, 5, 11 + col, projection="3d")
            _plot_single_traj(
                ax=ax,
                poses=cams_gt[bottom_idx],
                frame_indices=cams_gt_ids[bottom_idx],
                cmap=gt_cmap,
                frame_norm=gt_frame_norm,
                title=f"cam{bottom_idx + 1:02d} GT",
                xlim=vis_xlim,
                ylim=vis_ylim,
                zlim=vis_zlim,
            )

            ax = fig.add_subplot(4, 5, 16 + col, projection="3d")
            _plot_single_traj(
                ax=ax,
                poses=cams_pred[bottom_idx],
                frame_indices=cams_pred_ids[bottom_idx],
                cmap=pred_cmap,
                frame_norm=pred_frame_norm,
                title=f"cam{bottom_idx + 1:02d} PRED(aligned)",
                xlim=vis_xlim,
                ylim=vis_ylim,
                zlim=vis_zlim,
            )

    _add_left_dual_colorbars(
        fig,
        gt_frame_norm=gt_frame_norm,
        pred_frame_norm=pred_frame_norm,
        gt_cmap=gt_cmap,
        pred_cmap=pred_cmap,
    )
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.04, top=0.93, wspace=0.16, hspace=0.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_dir.name}_gt_pred_split_grid.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    root = Path(args.trajectory_root)
    output_dir = Path(args.output_dir)

    video_dirs = _collect_video_dirs(root)
    print(f"[visualize_gt_pred_grid] Found {len(video_dirs)} video dir(s)")

    for video_dir in video_dirs:
        out_path = render_video_grid(
            video_dir=video_dir,
            output_dir=output_dir,
            dpi=args.dpi,
            overlay=args.overlay,
        )
        print(f"[visualize_gt_pred_grid] Saved: {out_path}")


if __name__ == "__main__":
    main()
