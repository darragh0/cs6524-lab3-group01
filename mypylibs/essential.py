"""Adjacent-frame Essential Matrix estimation for Lab 3, step 3.

The output translation is unit length per pair: monocular essential-matrix
recovery cannot determine metric scale. Consecutive translations are composed
to provide a relative trajectory, not a metric reconstruction.
"""

import csv
from pathlib import Path

import cv2
import numpy as np


def _read_calibration(k_path, d_path):
    k = np.loadtxt(str(k_path), delimiter=",").astype(np.float64).reshape(3, 3)
    # Lab 2 exports sometimes include a named header (k1,k2,p1,p2,k3).
    with Path(d_path).open() as stream:
        first_line = stream.readline().strip()
    try:
        [float(value) for value in first_line.split(",")]
        skiprows = 0
    except ValueError:
        skiprows = 1
    distortion = np.loadtxt(str(d_path), delimiter=",", skiprows=skiprows).astype(np.float64).reshape(-1)
    if not np.isfinite(k).all() or not np.isfinite(distortion).all():
        raise ValueError("Calibration files must contain finite numeric values.")
    if abs(np.linalg.det(k)) < 1e-12 or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("K.csv must be a valid 3x3 camera intrinsic matrix.")
    return k, distortion


def _make_detector():
    # SIFT is more robust to the scale and viewpoint changes in the lab videos.
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=3000), cv2.NORM_L2
    return cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING


def _match_points(gray_a, gray_b, detector, norm):
    key_a, desc_a = detector.detectAndCompute(gray_a, None)
    key_b, desc_b = detector.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    # Keep spatially diverse matches: best descriptor matches are used first.
    good.sort(key=lambda match: match.distance)
    good = good[:1500]
    return (np.float32([key_a[m.queryIdx].pt for m in good]),
            np.float32([key_b[m.trainIdx].pt for m in good]))


def _estimate_pair(gray_a, gray_b, k):
    detector, norm = _make_detector()
    points_a, points_b = _match_points(gray_a, gray_b, detector, norm)
    empty = (np.eye(3), np.zeros((3, 1)), None, 0, len(points_a))
    if len(points_a) < 8:
        return empty
    essential, mask = cv2.findEssentialMat(
        points_a, points_b, cameraMatrix=k, method=cv2.RANSAC,
        prob=0.999, threshold=1.0,
    )
    if essential is None:
        return empty
    # OpenCV can return several stacked 3x3 candidates; recoverPose handles
    # the first candidate while honoring the RANSAC inlier mask.
    essential = essential[:3, :3]
    inliers, rotation, translation, pose_mask = cv2.recoverPose(
        essential, points_a, points_b, cameraMatrix=k, mask=mask
    )
    return rotation, translation.reshape(3), essential, int(inliers), len(points_a)


def estimate_video_motion(video_path, k_path, d_path, output_dir):
    """Undistort a video, estimate E/R/t for adjacent frames, and save results.

    Writes ``pair_motion.csv`` (one row per pair) and ``trajectory.npz`` in
    ``output_dir``. Failed pairs are marked invalid with identity/zero motion.
    """
    video_path, output_dir = Path(video_path), Path(output_dir)
    k, distortion = _read_calibration(k_path, d_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detector, norm = _make_detector()
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"Video contains no readable frames: {video_path}")

    def prepare(image):
        corrected = cv2.undistort(image, k, distortion, None, k)
        return cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    previous = prepare(frame)
    frames = 1
    rotations, translations, essentials, records = [], [], [], []
    cumulative_r, cumulative_t = np.eye(3), np.zeros(3)
    pair_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current = prepare(frame)
        points_a, points_b = _match_points(previous, current, detector, norm)
        matches = len(points_a)
        essential = None
        inliers = 0
        rotation, translation = np.eye(3), np.zeros(3)
        valid = False
        if matches >= 8:
            essential, mask = cv2.findEssentialMat(
                points_a, points_b, cameraMatrix=k, method=cv2.RANSAC,
                prob=0.999, threshold=1.0,
            )
            if essential is not None:
                essential = essential[:3, :3]
                inliers, rotation, t, _ = cv2.recoverPose(
                    essential, points_a, points_b, cameraMatrix=k, mask=mask
                )
                translation = t.reshape(3)
                valid = inliers >= 8

        if valid:
            # Camera-to-world pose composition from the relative transform
            # returned by recoverPose (scale remains unknown).
            cumulative_t = cumulative_t - cumulative_r @ rotation.T @ translation
            cumulative_r = cumulative_r @ rotation.T
            rotations.append(rotation)
            translations.append(translation)
            essentials.append(essential)
        else:
            rotations.append(np.eye(3))
            translations.append(np.zeros(3))
            essentials.append(np.full((3, 3), np.nan))

        records.append([
            pair_index, pair_index + 1, int(valid), matches, int(inliers),
            *rotation.reshape(-1).tolist(), *translation.tolist(),
            *cumulative_r.reshape(-1).tolist(), *cumulative_t.tolist(),
            *(essential.reshape(-1).tolist() if essential is not None else [np.nan] * 9),
        ])
        previous = current
        frames += 1
        pair_index += 1

    cap.release()
    header = (['frame_i', 'frame_j', 'valid', 'matches', 'inliers']
              + [f'R_{r}{c}' for r in range(3) for c in range(3)]
              + ['t_x', 't_y', 't_z']
              + [f'pose_R_{r}{c}' for r in range(3) for c in range(3)]
              + ['pose_x', 'pose_y', 'pose_z']
              + [f'E_{r}{c}' for r in range(3) for c in range(3)])
    with (output_dir / 'pair_motion.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(records)
    np.savez_compressed(
        output_dir / 'trajectory.npz',
        rotations=np.asarray(rotations, dtype=np.float64),
        translations=np.asarray(translations, dtype=np.float64),
        essential_matrices=np.asarray(essentials, dtype=np.float64),
        intrinsic_matrix=k,
        distortion_coefficients=distortion,
    )
    return {
        'frames': frames,
        'pairs': len(records),
        'valid_pairs': sum(row[2] for row in records),
    }
