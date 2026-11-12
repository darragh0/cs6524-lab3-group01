"""Run keypoint matching (step 2) and adjacent-frame motion (step 3)."""

from pathlib import Path

import cv2
import numpy as np

from mypylibs.essential import estimate_video_motion
from mypylibs.keypoint_matching import match_keypoints, visualise_matches


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "in"
OUTPUTS = ROOT / "out"


def process_keypoint_matches(video_name, video_path, num_frames=30):
    """Run step 2 on evenly spaced frame samples and save match visualisations."""
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
        0, total_frames - 1, min(num_frames, total_frames), dtype=int
    ))
    previous_frame = previous_index = None
    match_counts, displacements = [], []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        if previous_frame is not None:
            points_a, points_b = match_keypoints(previous_frame, frame)
            count = len(points_a)
            match_counts.append(count)
            median_displacement = (
                float(np.median(np.linalg.norm(points_b - points_a, axis=1)))
                if count else 0.0
            )
            if count:
                displacements.append(median_displacement)
            visualisation = visualise_matches(
                previous_frame, frame, points_a, points_b, max_matches=100
            )
            cv2.putText(
                visualisation,
                f"Frames {previous_index}-{frame_index} | Matches: {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2,
            )
            output_path = output_folder / f"matches_{previous_index:04d}_{frame_index:04d}.jpg"
            cv2.imwrite(str(output_path), visualisation)
            print(
                f"Step 2 {video_name} | frames {previous_index}-{frame_index} | "
                f"matches: {count} | median displacement: {median_displacement:.2f} px"
            )
        previous_frame, previous_index = frame, int(frame_index)
    cap.release()

    if match_counts:
        print(
            f"Step 2 {video_name}: {len(match_counts)} sampled pairs, "
            f"mean matches {np.mean(match_counts):.1f}, "
            f"range {min(match_counts)}-{max(match_counts)}, "
            f"mean median displacement "
            f"{np.mean(displacements):.2f} px" if displacements else
            f"Step 2 {video_name}: {len(match_counts)} sampled pairs; no matches found"
        )
    print(f"Step 2 visualisations: {output_folder}")


def main():
    videos = {
        "a": next((p for p in (INPUTS / "a.mp4", INPUTS / "spinCamera.mp4") if p.exists()), None),
        "b": next((p for p in (INPUTS / "b.mp4", INPUTS / "spinHorse.mp4") if p.exists()), None),
    }
    missing = [name for name, video in videos.items() if video is None]
    if missing:
        raise FileNotFoundError(f"Missing video(s) for {', '.join(missing)} in {INPUTS}.")

    k_path = next((p for p in (INPUTS / "K.csv", INPUTS / "k.csv") if p.exists()), None)
    d_path = next((p for p in (INPUTS / "D.csv", INPUTS / "d.csv") if p.exists()), None)
    if k_path is None or d_path is None:
        raise FileNotFoundError("Step 3 requires the camera calibration files in in/K.csv and in/D.csv.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for label, video in videos.items():
        process_keypoint_matches(video.stem, video)
        print(f"Estimating Step 3 camera motion for {video.name}...")
        summary = estimate_video_motion(video, k_path, d_path, OUTPUTS / label)
        print(
            f"Step 3 {video.name}: {summary['frames']} frames, "
            f"{summary['pairs']} adjacent pairs, "
            f"{summary['valid_pairs']} recovered motions"
        )


if __name__ == "__main__":
    main()
