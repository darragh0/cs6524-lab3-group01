
"""Step 5: fixed-pose, five-frame multi-view point refinement.

Uses Step 3 relative poses as initial estimates. This is NOT bundle adjustment:
camera poses and the monocular translation-scale ambiguity are not corrected.
"""
from pathlib import Path
import csv
import cv2
import numpy as np
from scipy.optimize import least_squares


def _projection(k, r_cw, c_world):
    r_wc = r_cw.T
    return k @ np.column_stack((r_wc, -r_wc @ c_world))


def _initial_point(observations, projections):
    rows = []
    for frame, xy in observations:
        p = projections[frame]
        rows.extend((xy[0] * p[2] - p[0], xy[1] * p[2] - p[1]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1]
    if abs(h[3]) < 1e-10:
        return None
    x = h[:3] / h[3]
    return x if np.isfinite(x).all() else None


def _residual(x, observations, projections):
    result = []
    h = np.append(x, 1.0)
    for frame, xy in observations:
        q = projections[frame] @ h
        if abs(q[2]) < 1e-9:
            result.extend((1e4, 1e4))
        else:
            result.extend((q[:2] / q[2] - xy).tolist())
    return np.asarray(result)


def _positive_depth(x, observations, rotations, centers):
    return all((rotations[i].T @ (x - centers[i]))[2] > 1e-6 for i, _ in observations)


def _frame_features(frame, k, d, sift):
    gray = cv2.cvtColor(cv2.undistort(frame, k, d, None, k), cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def optimize_video_five_frames(video_path, trajectory_path, output_dir,
                               window_size=5, window_stride=5,
                               max_tracks=120, min_track_length=3):
    """Refine points observed in >=3 frames in each five-frame window.

    Writes point positions and reprojection-error statistics. Independent windows
    have no guaranteed consistent scale or alignment across failed pose pairs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(trajectory_path)
    k = data['intrinsic_matrix']
    d = data['distortion_coefficients']
    rel_r = data['rotations']
    rel_t = data['translations']
    n_frames = len(rel_r) + 1
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'Cannot open {video_path}')

    # Step 3's relative transforms: X_next = R X_previous + t.
    # Convert to camera-to-world rotations and world camera centres.
    rotations = [np.eye(3)]
    centers = [np.zeros(3)]
    for r, t in zip(rel_r, rel_t):
        rotations.append(rotations[-1] @ r.T)
        centers.append(centers[-1] - rotations[-1] @ t)
    rotations = np.asarray(rotations)
    centers = np.asarray(centers)
    projections = [_projection(k, rotations[i], centers[i]) for i in range(n_frames)]

    sift = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    all_points = []
    stats = []

    for start in range(0, n_frames - window_size + 1, window_stride):
        features = []
        readable = True
        for idx in range(start, start + window_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                readable = False
                break
            features.append(_frame_features(frame, k, d, sift))
        if not readable:
            continue

        # Track descriptor indices, not rounded pixel coordinates. A track
        # continues only if its feature index matches in the next frame.
        tracks = []
        active = {}
        for j in range(window_size - 1):
            kp_a, des_a = features[j]
            kp_b, des_b = features[j + 1]
            if des_a is None or des_b is None or len(des_b) < 2:
                active = {}
                continue
            candidates = matcher.knnMatch(des_a, des_b, k=2)
            good = [m for pair in candidates if len(pair) == 2
                    for m, n in [pair] if m.distance < 0.75 * n.distance]
            good.sort(key=lambda m: m.distance)
            used_next = set()
            new_active = {}
            for m in good:
                if m.trainIdx in used_next:
                    continue
                used_next.add(m.trainIdx)
                obs = (start + j + 1, np.asarray(kp_b[m.trainIdx].pt))
                if m.queryIdx in active:
                    track_id = active[m.queryIdx]
                    tracks[track_id].append(obs)
                else:
                    track_id = len(tracks)
                    tracks.append([(start + j, np.asarray(kp_a[m.queryIdx].pt)), obs])
                new_active[m.trainIdx] = track_id
            active = new_active

        tracks = [track for track in tracks if len(track) >= min_track_length]
        # Limit work for CI; prefer longer tracks.
        tracks.sort(key=len, reverse=True)
        tracks = tracks[:max_tracks]
        before, after, kept = [], [], 0
        for track in tracks:
            x0 = _initial_point(track, projections)
            if x0 is None or not _positive_depth(x0, track, rotations, centers):
                continue
            initial = _residual(x0, track, projections)
            if np.sqrt(np.mean(initial ** 2)) > 20:
                continue
            fit = least_squares(_residual, x0, args=(track, projections),
                                max_nfev=35, method='lm')
            if not np.isfinite(fit.x).all() or not _positive_depth(fit.x, track, rotations, centers):
                continue
            e0 = float(np.sqrt(np.mean(initial ** 2)))
            e1 = float(np.sqrt(np.mean(_residual(fit.x, track, projections) ** 2)))
            if e1 > 5:
                continue
            before.append(e0)
            after.append(e1)
            all_points.append((start, *fit.x))
            kept += 1
        stats.append((start, start + window_size - 1, len(tracks), kept,
                      float(np.mean(before)) if before else float('nan'),
                      float(np.mean(after)) if after else float('nan')))
        print(f'Step 5 frames {start}-{start + window_size - 1}: '
              f'{kept}/{len(tracks)} points, '
              f'RMS {stats[-1][4]:.3f} -> {stats[-1][5]:.3f} px')

    cap.release()
    with (output_dir / 'five_frame_stats.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_frame', 'end_frame', 'tracks', 'points_kept',
                         'rms_before_px', 'rms_after_px'])
        writer.writerows(stats)
    with (output_dir / 'five_frame_points.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['window_start', 'x', 'y', 'z'])
        writer.writerows(all_points)
    return {'windows': len(stats), 'points': len(all_points), 'output': str(output_dir)}
