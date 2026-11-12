"""Step 4: estimate 3D points from two consecutive frames and measure reprojection error."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Mat = NDArray[np.float64]


@dataclass(frozen=True)
class TriangulationResult:
    points_3d: Mat  # (M, 3), in camera-1 coordinates
    keep: NDArray[np.bool_]  # (N,) which input matches survived filtering
    errors: Mat  # (M,) per-point reprojection error in pixels (mean of both views)
    mean_error: float


def projection_matrix(K: Mat, R: Mat, t: Mat) -> Mat:
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(P: Mat, points_3d: Mat) -> Mat:
    homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    projected = (P @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def triangulate(
    pts1: Mat,
    pts2: Mat,
    K: Mat,
    R: Mat,
    t: Mat,
    max_error: float = 4.0,
) -> TriangulationResult:
    """Triangulate matched pixels; camera 1 is the origin, camera 2 is at (R, t) relative to it.

    pts1/pts2 are (N, 2) matched pixel coordinates from undistorted frames.
    Points behind either camera or with reprojection error above max_error are dropped.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    P1 = projection_matrix(K, np.eye(3), np.zeros(3))
    P2 = projection_matrix(K, R, t)

    homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (homogeneous[:3] / homogeneous[3]).T

    depth1 = points_3d[:, 2]
    depth2 = (points_3d @ R.T + t.reshape(1, 3))[:, 2]
    in_front = (depth1 > 0) & (depth2 > 0)

    err1 = np.linalg.norm(project(P1, points_3d) - pts1, axis=1)
    err2 = np.linalg.norm(project(P2, points_3d) - pts2, axis=1)
    errors = (err1 + err2) / 2

    keep = in_front & (errors < max_error)
    kept_errors = errors[keep]
    mean_error = float(kept_errors.mean()) if kept_errors.size else float("nan")

    return TriangulationResult(points_3d[keep], keep, kept_errors, mean_error)
