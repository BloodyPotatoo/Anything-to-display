"""Small script that exercises calibration.py's helper functions so the
automatic calibration logic can be verified without a projector."""

import cv2
import numpy as np

from calibration import (
    _detect_screen,
    _find_light_points,
    _resolve_exposure_scale,
    build_grid_points,
)


def test_resolve_exposure_scale():
    assert _resolve_exposure_scale(40000, 0) == 1.0
    assert _resolve_exposure_scale(80000, 0) == 2.0
    assert _resolve_exposure_scale(0, 16) == 1.0
    assert _resolve_exposure_scale(0, 32) == 2.0
    # anything above 1.6M uS clamps to the 40.0 ceiling
    assert _resolve_exposure_scale(1600000, 160) == 40.0


def test_find_light_points_detects_a_blob():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(frame, (50, 50), 15, (255, 255, 255), -1)
    points = _find_light_points(frame)
    assert len(points) == 1
    cx, cy = points[0]
    assert abs(cx - 50) < 2
    assert abs(cy - 50) < 2


def test_build_grid_points():
    pts = build_grid_points(1280, 720)
    assert len(pts) == 9
    # row-major, top-left first
    assert abs(pts[0][0]) < 1
    assert abs(pts[0][1]) < 1
    assert abs(pts[-1][0] - 1279) < 1
    assert abs(pts[-1][1] - 719) < 1


def test_detect_screen_with_synthetic_projector_frame():
    width, height = 640, 480
    frames = []
    for color in ((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 150), (440, 330), color, -1)
        # a static desk area identical across all four phases
        frame[330:400, 50:100] = (80, 80, 80)
        frames.append(frame)

    corners, capture_w, capture_h = _detect_screen(
        width, height, frames[0].shape[1], frames[0].shape[0]
    )

    assert corners.shape == (4, 2)
    xs = corners[:, 0]
    ys = corners[:, 1]
    assert xs.min() < 300 and xs.max() > 340
    assert ys.min() < 250 and ys.max() > 290
    assert capture_w == width
    assert capture_h == height


if __name__ == "__main__":
    test_resolve_exposure_scale()
    print("test_resolve_exposure_scale ... ok")

    test_find_light_points_detects_a_blob()
    print("test_find_light_points_detects_a_blob ... ok")

    test_build_grid_points()
    print("test_build_grid_points ... ok")

    test_detect_screen_with_synthetic_projector_frame()
    print("test_detect_screen_with_synthetic_projector_frame ... ok")

    print("\nAll checks passed.")
