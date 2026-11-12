
"""Run keypoint matching (step 2), adjacent-frame motion (step 3),
triangulation (step 4), five-frame optimisation (step 5), dense
reconstruction (step 6) and the six-run sensitivity analysis (step 7).
"""

from pathlib import Path

import cv2
import numpy as np

from mypylibs.essential import estimate_video_motion
from mypylibs.keypoint_matching import match_keypoints, visualise_matches
from mypylibs.triangulation import triangulate
from mypylibs.multiframe_sfm import optimize_video_five_frames
from mypylibs.dense import reconstruct_dense_cloud
from mypylibs.sensitivity import FOCAL_SCALES, UNDISTORT, write_calibration


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "in"
OUTPUTS = ROOT / "out"


def process_keypoint_matches(video_name, video_path, num_frames=30):
    """Run step 2 on sampled frames and save match visualisations."""

    output_folder = OUTPUTS / f"{video_name}_matches"
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video contains no frames: {video_path}")

    frame_indices = np.unique(np.linspace(
        0,
        total_frames - 1,
        min(num_frames, total_frames),
        dtype=int
    ))

    previous_frame = None
    previous_index = None

    match_counts = []
    displacements = []

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()

        if not ok:
            continue

        if previous_frame is not None:

            points_a, points_b = match_keypoints(
                previous_frame, frame
            )

            count = len(points_a)
            match_counts.append(count)

            median_displacement = (
                float(np.median(
                    np.linalg.norm(points_b - points_a, axis=1)
                ))
                if count else 0.0
            )

            if count:
                displacements.append(median_displacement)

            visualisation = visualise_matches(
                previous_frame,
                frame,
                points_a,
                points_b,
                max_matches=100
            )

            cv2.putText(
                visualisation,
                f"Frames {previous_index}-{frame_index} | Matches: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            output_path = output_folder / (
                f"matches_{previous_index:04d}_{frame_index:04d}.jpg"
            )

            cv2.imwrite(str(output_path), visualisation)

            print(
                f"Step 2 {video_name} | "
                f"frames {previous_index}-{frame_index} | "
                f"matches: {count} | "
                f"median displacement: {median_displacement:.2f} px"
            )

        previous_frame = frame
        previous_index = int(frame_index)

    cap.release()

    if match_counts:
        print(
            f"Step 2 {video_name}: {len(match_counts)} sampled pairs, "
            f"mean matches {np.mean(match_counts):.1f}, "
            f"range {min(match_counts)}-{max(match_counts)}"
        )

        if displacements:
            print(
                f"Mean median displacement: "
                f"{np.mean(displacements):.2f} px"
            )

    print(f"Step 2 visualisations: {output_folder}")


def report_triangulation(video_name, k, pair_matches):
    """Run step 4 and report reprojection error statistics."""

    errors = []
    kept = []

    for _, rotation, translation, points_a, points_b in pair_matches:

        result = triangulate(
            points_a,
            points_b,
            k,
            rotation,
            translation
        )

        if len(result.points_3d):
            errors.append(result.mean_error)
            kept.append(len(result.points_3d))

    if not errors:
        print(f"Step 4 {video_name}: no pairs triangulated")
        return

    print(
        f"Step 4 {video_name}: "
        f"{len(errors)} pairs triangulated, "
        f"reprojection error median {np.median(errors):.3f} px "
        f"(max {np.max(errors):.3f}), "
        f"median {np.median(kept):.0f} points per pair"
    )


def reconstruct(video, run, k_path, d_path, run_dir):
    """Run steps 3-6 for one camera setting and write out/<run>.ply."""

    # Step 3
    summary = estimate_video_motion(video, k_path, d_path, run_dir)
    print(
        f"Step 3 {run}: "
        f"{summary['frames']} frames, "
        f"{summary['pairs']} adjacent pairs, "
        f"{summary['valid_pairs']} recovered motions"
    )

    # Step 4
    k = np.loadtxt(str(k_path), delimiter=",").reshape(3, 3)
    report_triangulation(run, k, summary["pair_matches"])

    # Step 5
    step5 = optimize_video_five_frames(
        video,
        run_dir / "trajectory.npz",
        run_dir,
        window_size=5,
        window_stride=5
    )
    print(
        f"Step 5 {run}: "
        f"{step5['windows']} windows, "
        f"{step5['points']} refined points"
    )

    # Step 6
    cloud = reconstruct_dense_cloud(
        video,
        k_path,
        d_path,
        run_dir / "pair_motion.csv",
        OUTPUTS / f"{run}.ply",
        step5_points=run_dir / "five_frame_points.csv",
    )
    print(f"Step 6 {run}: wrote {cloud['points']} colored points to {cloud['output']}")


def main():

    videos = {
        "a": next(
            (p for p in (
                INPUTS / "a.mp4",
                INPUTS / "spinCamera.mp4"
            ) if p.exists()),
            None
        ),
        "b": next(
            (p for p in (
                INPUTS / "b.mp4",
                INPUTS / "spinHorse.mp4"
            ) if p.exists()),
            None
        )
    }

    missing = [
        name for name, video in videos.items()
        if video is None
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing videos: {', '.join(missing)}"
        )

    k_path = next(
        (p for p in (
            INPUTS / "K.csv",
            INPUTS / "k.csv"
        ) if p.exists()),
        None
    )

    d_path = next(
        (p for p in (
            INPUTS / "D.csv",
            INPUTS / "d.csv"
        ) if p.exists()),
        None
    )

    if k_path is None or d_path is None:
        raise FileNotFoundError(
            "Missing camera calibration files."
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    for label, video in videos.items():

        # Step 2
        process_keypoint_matches(
            video.stem,
            video
        )

        # Step 7: run steps 3-6 under each focal scale, with and without undistortion
        for distortion_code, undistort in UNDISTORT.items():
            for focal_code, focal_scale in FOCAL_SCALES.items():
                run = f"{label}_{focal_code}{distortion_code}"
                run_dir = OUTPUTS / "runs" / run
                run_k, run_d = write_calibration(
                    k_path, d_path, focal_scale, undistort, run_dir
                )
                reconstruct(video, run, run_k, run_d, run_dir)

if __name__ == "__main__":
    main()
