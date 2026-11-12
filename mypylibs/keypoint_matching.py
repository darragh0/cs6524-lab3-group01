
import cv2
import numpy as np


def match_keypoints(frame1, frame2):

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Handle frames without descriptors
    if des1 is None or des2 is None:
        return np.empty((0, 2)), np.empty((0, 2))

    # Create brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Find two nearest matches for each descriptor
    matches = matcher.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []

    for pair in matches:

        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Extract matching pixel coordinates
    pts1 = np.float32([
        kp1[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 2)

    pts2 = np.float32([
        kp2[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 2)

    return pts1, pts2



def visualise_matches(frame1, frame2, pts1, pts2, max_matches=100):

    # Get frame dimensions
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    # Create canvas for both frames
    canvas = np.zeros(
        (max(h1, h2), w1 + w2, 3),
        dtype=np.uint8
    )

    # Place frames side by side
    canvas[:h1, :w1] = frame1
    canvas[:h2, w1:w1 + w2] = frame2

    # Limit number of displayed matches
    if len(pts1) > max_matches:

        indices = np.linspace(
            0,
            len(pts1) - 1,
            max_matches,
            dtype=int
        )

        pts1 = pts1[indices]
        pts2 = pts2[indices]

    # Draw matching keypoints
    for p1, p2 in zip(pts1, pts2):

        x1, y1 = np.round(p1).astype(int)
        x2, y2 = np.round(p2).astype(int)

        # Offset second frame horizontally
        x2 += w1

        # Draw points
        cv2.circle(canvas, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (x2, y2), 3, (0, 255, 0), -1)

        # Draw connecting line
        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1
        )

    # Return image instead of saving it
    return canvas
