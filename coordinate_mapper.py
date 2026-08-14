import os
import sys
import cv2
import numpy as np

# ============================================================
# SMARTIR - CAMERA -> DESKTOP COORDINATE MAPPER
# ============================================================

CALIBRATION_FILE = "screen_corners.npy"
FALLBACK_WIDTH = 1280
FALLBACK_HEIGHT = 720
CLAMP_COORDINATES = True


def get_screen_size():
    if sys.platform.startswith("win"):
        try:
            import ctypes

            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)

            if width > 0 and height > 0:
                return int(width), int(height)
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
            return int(width), int(height)
    except Exception:
        pass

    print(
        "WARNING: Could not determine desktop resolution. "
        f"Using {FALLBACK_WIDTH}x{FALLBACK_HEIGHT}."
    )

    return FALLBACK_WIDTH, FALLBACK_HEIGHT


class CoordinateMapper:
    def __init__(self, calibration_file=CALIBRATION_FILE):
        self.calibration_file = calibration_file

        self.screen_width, self.screen_height = get_screen_size()

        self.camera_corners = self._load_corners()

        self.matrix = self._calculate_homography()

    def _load_corners(self):
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(
                f"'{self.calibration_file}' was not found. "
                "Run calibration.py first."
            )

        corners = np.asarray(
            np.load(self.calibration_file),
            dtype=np.float32
        ).reshape(4, 2)

        if corners.shape != (4, 2):
            raise ValueError(
                "Calibration must contain four (x, y) corners."
            )

        return corners

    def _calculate_homography(self):
        desktop_corners = np.array(
            [
                [0, 0],
                [self.screen_width - 1, 0],
                [self.screen_width - 1, self.screen_height - 1],
                [0, self.screen_height - 1],
            ],
            dtype=np.float32
        )

        matrix, _ = cv2.findHomography(
            self.camera_corners,
            desktop_corners,
            cv2.RANSAC
        )

        if matrix is None:
            raise RuntimeError(
                "Could not calculate the 3x3 homography matrix."
            )

        return matrix

    def camera_to_screen(self, camera_x, camera_y):
        point = np.array(
            [[[float(camera_x), float(camera_y)]]],
            dtype=np.float32
        )

        result = cv2.perspectiveTransform(
            point,
            self.matrix
        )[0, 0]

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
        point = np.array(
            [[[float(camera_x), float(camera_y)]]],
            dtype=np.float32
        )

        polygon = self.camera_corners.reshape((-1, 1, 2))

        return cv2.pointPolygonTest(
            polygon,
            (float(point[0, 0, 0]), float(point[0, 0, 1])),
            False
        ) >= 0

    def print_info(self):
        print("\n" + "=" * 65)
        print("SMARTIR COORDINATE TRANSFORMATION")
        print("=" * 65)

        print(
            f"Desktop: "
            f"{self.screen_width} x {self.screen_height}"
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
                suppress_small=True
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

                x, y = mapper.camera_to_screen(
                    camera_x,
                    camera_y
                )

                inside = mapper.inside_screen(
                    camera_x,
                    camera_y
                )

                print(
                    f"Screen coordinate: "
                    f"({x:.2f}, {y:.2f})"
                )

                print(
                    f"Inside calibrated screen: {inside}"
                )

            except ValueError:
                print("Please enter valid numbers.")

    except Exception as error:
        print("\nERROR:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
