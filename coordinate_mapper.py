import os
import platform
import re
import subprocess
import sys

import cv2
import numpy as np

# ============================================================
# SMARTIR - CAMERA -> DESKTOP COORDINATE MAPPER
#
# Preferred calibration data (written by calibration.py):
#
#   calibration.npz
#       homography       3x3 camera -> desktop
#       camera_corners   screen polygon in camera space
#       camera_points    measured reference points
#       desktop_points   their known desktop coordinates
#       frame_width/...  capture size the calibration was made at
#       screen_width/... desktop size the calibration was made at
#
# Legacy data (older calibrations):
#
#   screen_corners.npy
#       four camera-space screen corners only; the homography is
#       rebuilt by mapping those corners to the desktop rectangle
#
# The homography always belongs to the capture size it was made
# at. set_frame_size() rescales it if the camera later delivers a
# different frame size (same aspect ratio required).
# ============================================================

CALIBRATION_FILE = "screen_corners.npy"
HOMOGRAPHY_FILE = "calibration.npz"

FALLBACK_WIDTH = 1280
FALLBACK_HEIGHT = 720
CLAMP_COORDINATES = True

# Warn when the frame aspect ratio changes by more than this.
ASPECT_WARN_RATIO = 0.015


def _screen_size_from_xrandr():
    """Parse `xrandr` for the current mode (works without tkinter)."""
    if platform.system() != "Linux":
        return None

    try:
        result = subprocess.run(
            ["xrandr"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None

    match = re.search(r"(\d+)x(\d+)\s+\d+\.\d+\*", result.stdout or "")

    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def get_screen_size():
    """
    Detect the desktop resolution.

    Returns (width, height, detected).
    `detected` is False when we had to fall back to a guess.
    """
    if sys.platform.startswith("win"):
        try:
            import ctypes

            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)

            if width > 0 and height > 0:
                return int(width), int(height), True

        except Exception:
            pass

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        root.destroy()

        if width > 0 and height > 0:
            return int(width), int(height), True

    except Exception:
        pass

    xrandr_size = _screen_size_from_xrandr()

    if xrandr_size is not None:
        return xrandr_size[0], xrandr_size[1], True

    print(
        "WARNING: Could not determine desktop resolution. "
        f"Using {FALLBACK_WIDTH}x{FALLBACK_HEIGHT}."
    )

    return FALLBACK_WIDTH, FALLBACK_HEIGHT, False


def _homography_from_corners(corners, screen_width, screen_height):
    """Legacy path: four camera corners -> desktop rectangle."""
    desktop_corners = np.array(
        [
            [0, 0],
            [screen_width - 1, 0],
            [screen_width - 1, screen_height - 1],
            [0, screen_height - 1],
        ],
        dtype=np.float32,
    )

    matrix, _ = cv2.findHomography(
        np.asarray(corners, dtype=np.float32),
        desktop_corners,
        cv2.RANSAC,
    )

    if matrix is None:
        raise RuntimeError(
            "Could not calculate the 3x3 homography matrix."
        )

    return matrix.astype(np.float32)


class CoordinateMapper:
    def __init__(self, calibration_file=CALIBRATION_FILE,
                 homography_file=HOMOGRAPHY_FILE):
        self.calibration_file = calibration_file
        self.homography_file = homography_file

        self._h_src = None          # homography at calibration frame size
        self._corners_src = None    # screen polygon at that frame size
        self._src_frame_size = None  # (w, h) or None when unknown
        self._cur_frame_size = None
        self._aspect_warned = False

        live_width, live_height, live_detected = get_screen_size()
        stored_screen = None

        # ----------------------------------------------------
        # Load calibration data
        # ----------------------------------------------------
        if os.path.exists(homography_file):
            data = np.load(homography_file)

            self._h_src = np.asarray(
                data["homography"], dtype=np.float32
            )

            self._corners_src = np.asarray(
                data["camera_corners"], dtype=np.float32
            ).reshape(4, 2)

            self._src_frame_size = (
                int(data["frame_width"]),
                int(data["frame_height"]),
            )

            stored_screen = (
                int(data["screen_width"]),
                int(data["screen_height"]),
            )

            print(f"\nLoaded calibration: {homography_file}")

        elif os.path.exists(calibration_file):
            corners = np.asarray(
                np.load(calibration_file),
                dtype=np.float32,
            ).reshape(4, 2)

            self._corners_src = corners
            self._src_frame_size = None

            print(
                f"\nLoaded legacy calibration: {calibration_file}\n"
                "(no stored homography - run calibration.py for the "
                "new point-grid calibration.)"
            )

        else:
            raise FileNotFoundError(
                f"'{calibration_file}' and '{homography_file}' were not "
                "found. Run calibration.py first."
            )

        # ----------------------------------------------------
        # Desktop size
        #
        # The homography was fitted against the desktop size at
        # calibration time, so that size is authoritative. Only
        # fall back to the live size for legacy calibrations.
        # ----------------------------------------------------
        if stored_screen is not None:
            self.screen_width, self.screen_height = stored_screen

            if live_detected and (
                live_width != stored_screen[0]
                or live_height != stored_screen[1]
            ):
                print(
                    "WARNING: desktop resolution changed since "
                    f"calibration (stored {stored_screen[0]}x"
                    f"{stored_screen[1]}, now {live_width}x"
                    f"{live_height}).\n"
                    "         Using the stored size - recalibrate "
                    "after changing resolution."
                )

        elif live_detected:
            self.screen_width, self.screen_height = (
                live_width,
                live_height,
            )

        else:
            self.screen_width = FALLBACK_WIDTH
            self.screen_height = FALLBACK_HEIGHT

        # ----------------------------------------------------
        # Homography at the calibration frame size
        # ----------------------------------------------------
        if self._h_src is None:
            self._h_src = _homography_from_corners(
                self._corners_src,
                self.screen_width,
                self.screen_height,
            )

        # ----------------------------------------------------
        # Adapt to the calibration frame size (identity for now;
        # the real camera frame size is applied by
        # set_frame_size() as soon as a frame arrives)
        # ----------------------------------------------------
        if self._src_frame_size is None:
            self._cur_frame_size = None
            self.matrix = self._h_src
            self.camera_corners = self._corners_src
        else:
            self.set_frame_size(*self._src_frame_size)

    # --------------------------------------------------------
    # Frame size adaptation
    # --------------------------------------------------------

    def set_frame_size(self, width, height):
        """
        Keep the homography valid for the frame size the camera
        currently delivers.

        H' = S . H  where S scales old camera pixels to new ones.
        """
        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:
            return

        if self._cur_frame_size == (width, height):
            return

        if self._src_frame_size is None:
            # Legacy calibration: assume it was taken at this size.
            self._src_frame_size = (width, height)
            self._cur_frame_size = (width, height)
            self.matrix = self._h_src
            self.camera_corners = self._corners_src
            return

        src_width, src_height = self._src_frame_size

        if src_width <= 0 or src_height <= 0:
            return

        src_aspect = src_width / float(src_height)
        cur_aspect = width / float(height)
        aspect_mismatch = (
            abs(src_aspect - cur_aspect) / src_aspect
            > ASPECT_WARN_RATIO
        )

        if aspect_mismatch:
            if not self._aspect_warned:
                self._aspect_warned = True
                print(
                    "WARNING: camera frame aspect ratio changed since "
                    f"calibration ({src_width}x{src_height} -> "
                    f"{width}x{height}).\n"
                    "         Mapping will be wrong - recalibrate."
                )

            # Do not guess a scale for a different aspect ratio. Keep
            # the calibration geometry untouched rather than silently
            # stretching the mapping.
            self._cur_frame_size = (width, height)
            self.matrix = self._h_src
            self.camera_corners = self._corners_src
            return

        scale_x = width / float(src_width)
        scale_y = height / float(src_height)

        # p_new = S . p_old  (homogeneous scale matrix)
        # desktop = H . p_old = H . S^-1 . p_new
        # => H_new = H . S^-1
        inverse_scale = np.array(
            [
                [1.0 / scale_x, 0.0, 0.0],
                [0.0, 1.0 / scale_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.matrix = (
            self._h_src.astype(np.float32) @ inverse_scale
        ).astype(np.float32)

        self.camera_corners = (
            self._corners_src
            * np.array([scale_x, scale_y], dtype=np.float32)
        ).astype(np.float32)

        self._cur_frame_size = (width, height)

    # --------------------------------------------------------
    # Transformations
    # --------------------------------------------------------

    def camera_to_screen(self, camera_x, camera_y):
        point = np.array(
            [[[float(camera_x), float(camera_y)]]],
            dtype=np.float32,
        )

        result = cv2.perspectiveTransform(point, self.matrix)[0, 0]

        x = float(result[0])
        y = float(result[1])

        if CLAMP_COORDINATES:
            x = max(0, min(self.screen_width - 1, x))
            y = max(0, min(self.screen_height - 1, y))

        return x, y

    def camera_to_screen_int(self, camera_x, camera_y):
        x, y = self.camera_to_screen(camera_x, camera_y)
        return int(round(x)), int(round(y))

    def inside_screen(self, camera_x, camera_y):
        """
        Check whether a camera coordinate is inside the calibrated
        screen polygon.
        """
        polygon = self.camera_corners.reshape((-1, 1, 2))

        return (
            cv2.pointPolygonTest(
                polygon,
                (float(camera_x), float(camera_y)),
                False,
            )
            >= 0
        )

    def print_info(self):
        print("\n" + "=" * 65)
        print("SMARTIR COORDINATE TRANSFORMATION")
        print("=" * 65)

        print(
            f"Desktop: {self.screen_width} x {self.screen_height}"
        )

        if self._src_frame_size is not None:
            print(
                "Calibration capture size: "
                f"{self._src_frame_size[0]} x {self._src_frame_size[1]}"
            )

        if self._cur_frame_size is not None:
            print(
                "Current capture size: "
                f"{self._cur_frame_size[0]} x {self._cur_frame_size[1]}"
            )

        names = [
            "TOP LEFT",
            "TOP RIGHT",
            "BOTTOM RIGHT",
            "BOTTOM LEFT",
        ]

        print("\nCalibrated camera corners:")

        for name, point in zip(names, self.camera_corners):
            print(
                f"  {name:<14}"
                f"({point[0]:.2f}, {point[1]:.2f})"
            )

        print("\n3x3 HOMOGRAPHY MATRIX:")
        print(
            np.array2string(
                self.matrix,
                precision=6,
                suppress_small=True,
            )
        )
        print("=" * 65)


def main():
    try:
        mapper = CoordinateMapper()
        mapper.print_info()

        print("\nManual test mode.")
        print("Enter Q to quit.")

        while True:
            x_text = input("\nCamera X: ").strip()

            if x_text.lower() == "q":
                break

            y_text = input("Camera Y: ").strip()

            if y_text.lower() == "q":
                break

            try:
                camera_x = float(x_text)
                camera_y = float(y_text)

                x, y = mapper.camera_to_screen(camera_x, camera_y)

                inside = mapper.inside_screen(camera_x, camera_y)

                print(
                    f"Screen coordinate: ({x:.2f}, {y:.2f})"
                )
                print(f"Inside calibrated screen: {inside}")

            except ValueError:
                print("Please enter valid numbers.")

    except Exception as error:
        print("\nERROR:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
