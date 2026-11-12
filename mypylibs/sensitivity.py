"""Step 7: camera settings for the six sensitivity-analysis runs."""

from pathlib import Path

import numpy as np

from mypylibs.essential import _read_calibration

# File-name code -> focal length multiplier.
FOCAL_SCALES = {"08": 0.8, "10": 1.0, "12": 1.2}
# File-name code -> whether frames are undistorted.
UNDISTORT = {"u": True, "d": False}


def write_calibration(
    k_path: Path, d_path: Path, focal_scale: float, undistort: bool, folder: Path
) -> tuple[Path, Path]:
    """Write a copy of K/D for one run and return their paths.

    fx and fy are multiplied by focal_scale. Skipping undistortion is done by
    zeroing the distortion coefficients, which makes cv2.undistort a no-op.
    """
    k, d = _read_calibration(k_path, d_path)
    k[0, 0] *= focal_scale
    k[1, 1] *= focal_scale
    if not undistort:
        d = np.zeros_like(d)

    folder.mkdir(parents=True, exist_ok=True)
    run_k, run_d = folder / "K.csv", folder / "D.csv"
    np.savetxt(run_k, k, delimiter=",")
    np.savetxt(run_d, d.reshape(1, -1), delimiter=",")
    return run_k, run_d
