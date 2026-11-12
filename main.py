
import cv2
import os
import numpy as np

from mypylibs.keypoint_matching import (
    match_keypoints,
    visualise_matches
)


def process_video(video_name, num_frames=30):

    video_path = f"in/{video_name}.mp4"

    # Create separate output folder for each video
    output_folder = f"out/{video_name}_matches"
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select evenly spaced frames
    frame_indices = np.unique(
        np.linspace(
            0,
            total_frames - 1,
            min(num_frames, total_frames),
            dtype=int
        )
    )

    previous_frame = None
    previous_index = None

    match_counts = []
    displacements = []

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, current_frame = cap.read()

        if not ret:
            continue

        if previous_frame is not None:

            # Match SIFT keypoints
            pts1, pts2 = match_keypoints(
                previous_frame,
                current_frame
            )

            number_of_matches = len(pts1)
            match_counts.append(number_of_matches)

            # Calculate displacement
            if number_of_matches > 0:

                distances = np.linalg.norm(
                    pts2 - pts1,
                    axis=1
                )

                median_displacement = np.median(distances)
                displacements.append(median_displacement)

            else:
                median_displacement = 0

            print(
                f"Video {video_name} | "
                f"Frames {previous_index}-{frame_index} | "
                f"Matches: {number_of_matches} | "
                f"Median displacement: {median_displacement:.2f} px"
            )

            # Generate matching visualisation
            visualisation = visualise_matches(
                previous_frame,
                current_frame,
                pts1,
                pts2,
                max_matches=100
            )

            # Add frame information
            cv2.putText(
                visualisation,
                f"Frames {previous_index}-{frame_index} | Matches: {number_of_matches}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Save image for this frame pair
            output_path = os.path.join(
                output_folder,
                f"matches_{previous_index:04d}_{frame_index:04d}.jpg"
            )

            cv2.imwrite(output_path, visualisation)

        previous_frame = current_frame
        previous_index = frame_index

    cap.release()

    # Print summary
    if match_counts:

        print(f"\n--- {video_name} Summary ---")
        print(f"Total video frames: {total_frames}")
        print(f"Selected frames: {len(frame_indices)}")
        print(f"Pairs processed: {len(match_counts)}")
        print(f"Average matches: {np.mean(match_counts):.2f}")
        print(f"Minimum matches: {np.min(match_counts)}")
        print(f"Maximum matches: {np.max(match_counts)}")

        if displacements:
            print(
                f"Average median displacement: "
                f"{np.mean(displacements):.2f} px"
            )

        print(f"Images saved to: {output_folder}\n")


def main():

    os.makedirs("out", exist_ok=True)

    process_video("spinCamera", num_frames=30)
    process_video("spinHorse", num_frames=30)


if __name__ == "__main__":
    main()
