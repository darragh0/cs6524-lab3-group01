"""Step 6: semi-dense colored reconstruction from tracked image pixels."""

import csv
from pathlib import Path

import cv2
import numpy as np

from mypylibs.essential import _read_calibration


def _read_poses(motion_csv, n_frames):
    rotations = [np.eye(3) for _ in range(n_frames)]
    centers = [np.zeros(3) for _ in range(n_frames)]
    with Path(motion_csv).open(newline="") as stream:
        for row in csv.DictReader(stream):
            j = int(row["frame_j"])
            if j >= n_frames:
                continue
            rotations[j] = np.array([
                [float(row[f"pose_R_{r}{c}"]) for c in range(3)]
                for r in range(3)
            ])
            centers[j] = np.array([float(row[f"pose_{axis}"]) for axis in "xyz"])
    return rotations, centers


def _write_ply(path, points, colors):
    vertices = np.empty(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with Path(path).open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def _initial_flow_points(frame1, frame2, k, d, r1, c1, r2, c2,
                         pixel_step, max_depth):
    image1 = cv2.undistort(frame1, k, d, None, k)
    image2 = cv2.undistort(frame2, k, d, None, k)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    flow12 = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    flow21 = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    height, width = gray1.shape
    yy, xx = np.mgrid[4:height - 4:pixel_step, 4:width - 4:pixel_step]
    p1 = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    motion = flow12[yy, xx].reshape(-1, 2)
    p2 = p1 + motion
    qx, qy = np.rint(p2[:, 0]).astype(int), np.rint(p2[:, 1]).astype(int)
    inside = (qx >= 1) & (qx < width - 1) & (qy >= 1) & (qy < height - 1)
    fb_error = np.full(len(p1), np.inf)
    valid_ids = np.flatnonzero(inside)
    reverse = flow21[qy[valid_ids], qx[valid_ids]]
    fb_error[valid_ids] = np.linalg.norm(motion[valid_ids] + reverse, axis=1)
    keep = inside & (fb_error < 1.5) & (np.linalg.norm(motion, axis=1) > 0.5)
    p1, p2 = p1[keep], p2[keep]
    if len(p1) < 8:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    w2c1 = np.column_stack((r1.T, -r1.T @ c1))
    w2c2 = np.column_stack((r2.T, -r2.T @ c2))
    homogeneous = cv2.triangulatePoints(k @ w2c1, k @ w2c2, p1.T, p2.T)
    finite = np.isfinite(homogeneous).all(axis=0) & (np.abs(homogeneous[3]) > 1e-4)
    homogeneous, p1, p2 = homogeneous[:, finite], p1[finite], p2[finite]
    if homogeneous.shape[1] < 8:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    bounded = np.isfinite(xyz).all(axis=1) & (np.max(np.abs(xyz), axis=1) < 1e6)
    xyz, p1, p2 = xyz[bounded], p1[bounded], p2[bounded]
    if len(xyz) < 8:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    camera1 = (r1.T @ (xyz - c1).T).T
    camera2 = (r2.T @ (xyz - c2).T).T
    positive = (camera1[:, 2] > 1e-6) & (camera2[:, 2] > 1e-6)
    xyz, p1, p2 = xyz[positive], p1[positive], p2[positive]
    camera1, camera2 = camera1[positive], camera2[positive]
    if len(xyz) < 8:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    projected1 = (k @ camera1.T).T
    projected2 = (k @ camera2.T).T
    projected1 = projected1[:, :2] / projected1[:, 2:3]
    projected2 = projected2[:, :2] / projected2[:, 2:3]
    error = np.maximum(np.linalg.norm(projected1 - p1, axis=1),
                       np.linalg.norm(projected2 - p2, axis=1))
    depth = np.linalg.norm(xyz - c1, axis=1)
    good = (np.isfinite(error) & (error < 2.5)
            & (depth > 0.02) & (depth < max_depth))
    colors = image1[np.rint(p1[good, 1]).astype(int),
                    np.rint(p1[good, 0]).astype(int)][:, ::-1].copy()
    return xyz[good], colors


def reconstruct_dense_cloud(video_path, k_path, d_path, motion_csv, output_path,
                            step5_points=None, frame_step=5, pixel_step=8,
                            max_depth=250.0):
    """Triangulate sampled dense-flow tracks and write a colored PLY point cloud.

    Step 5's refined multi-view points are added when supplied. The output scale
    remains arbitrary because the camera translation from an Essential Matrix
    is known only up to scale.
    """
    video_path = Path(video_path)
    k, d = _read_calibration(k_path, d_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotations, centers = _read_poses(motion_csv, n_frames)
    all_points, all_colors = [], []
    step = max(1, int(frame_step))

    for first in range(0, n_frames - step, step):
        second = first + step
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        ok1, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, second)
        ok2, frame2 = cap.read()
        if not ok1 or not ok2:
            continue
        if (np.linalg.norm(centers[second] - centers[first]) < 1e-8
                and np.linalg.norm(rotations[second] - rotations[first]) < 1e-8):
            continue
        points, colors = _initial_flow_points(
            frame1, frame2, k, d, rotations[first], centers[first],
            rotations[second], centers[second], pixel_step, max_depth,
        )
        if len(points):
            all_points.append(points)
            all_colors.append(colors)
        if first % (step * 20) == 0:
            print(f"Step 6 {video_path.name}: processed frames {first}-{second}")
    cap.release()

    # Add the multi-frame refined feature points from step 5, colored from the
    # corresponding window's first image when their projections are visible.
    if step5_points is not None and Path(step5_points).exists():
        sparse_points, sparse_colors = [], []
        sparse_cap = cv2.VideoCapture(str(video_path))
        frame_cache = {}
        with Path(step5_points).open(newline="") as stream:
            for row in csv.DictReader(stream):
                frame = int(row["window_start"])
                point = np.array([float(row[a]) for a in "xyz"])
                if frame >= n_frames:
                    continue
                camera_point = rotations[frame].T @ (point - centers[frame])
                if camera_point[2] <= 0:
                    continue
                pixel = k @ camera_point
                x, y = np.rint(pixel[:2] / pixel[2]).astype(int)
                if frame not in frame_cache:
                    sparse_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = sparse_cap.read()
                    frame_cache[frame] = (
                        cv2.undistort(image, k, d, None, k) if ok else None
                    )
                image = frame_cache[frame]
                if image is not None and 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    sparse_points.append(point)
                    sparse_colors.append(image[y, x, ::-1])
        sparse_cap.release()
        if sparse_points:
            all_points.append(np.asarray(sparse_points))
            all_colors.append(np.asarray(sparse_colors, dtype=np.uint8))

    if not all_points:
        raise RuntimeError(f"No valid 3D points reconstructed for {video_path}.")
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    _write_ply(output_path, points, colors)
    return {"points": len(points), "output": str(output_path)}
