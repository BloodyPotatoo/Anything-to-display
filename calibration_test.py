"""$pytest -v calibration_test.py

Unit tests for the automatic calibration logic. Functional, no camera
or projector required.
"""

import cv2
import numpy as np

from calibration import (
    _compute_homography,
    _derive_screen_corners,
    _screen_corners_from_frames,
    _find_light_points,
    _resolve_exposure_scale,
    build_grid_points,
    order_corners_tlbr,
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
    # The function returns a list of (cx, cy) tuples.
    assert len(points) == 1
    cx, cy = points[0]
    assert abs(cx - 50) < 2
    assert abs(cy - 50) < 2


def test_find_light_points_returns_circles():
    # A large white disc on a black frame yields one blob centroid.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 30, (255, 255, 255), -1)
    points = _find_light_points(frame)
    assert len(points) == 1
    cx, cy = points[0]
    assert abs(cx - 100) < 3
    assert abs(cy - 100) < 3


def test_build_grid_points():
    pts = build_grid_points(1280, 720)
    assert len(pts) == 9
    # row-major, top-left first
    assert abs(pts[0][0]) < 1
    assert abs(pts[0][1]) < 1
    assert abs(pts[-1][0] - 1279) <= 1
    assert abs(pts[-1][1] - 719) <= 1


def test_compute_homography_uses_correct_desktop_pairs():
    # The camera sees the desktop rectangle as a quadrilateral. Build
    # the measured pairs by inverse-mapping the desktop grid, then make
    # sure the fitted homography maps camera -> desktop (not camera ->
    # camera, which is what happened when the desktop pairs were built
    # from the camera points themselves).
    cam_corners = np.array(
        [[100, 80], [560, 120], [520, 430], [80, 390]],
        dtype=np.float32,
    )
    desk_corners = np.array(
        [[0, 0], [1919, 0], [1919, 1079], [0, 1079]],
        dtype=np.float32,
    )
    h_true = cv2.getPerspectiveTransform(cam_corners, desk_corners)

    grid = np.array(build_grid_points(1920, 1080), dtype=np.float32)
    cam_pts = cv2.perspectiveTransform(
        grid.reshape(-1, 1, 2), np.linalg.inv(h_true)
    ).reshape(-1, 2)
    pairs = [(tuple(c), tuple(d)) for c, d in zip(cam_pts, grid)]

    homography, error = _compute_homography(pairs, 1920, 1080)

    assert homography is not None
    assert error < 0.01

    mapped = cv2.perspectiveTransform(
        cam_pts.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    assert np.allclose(mapped, grid, atol=0.5)


def test_derive_screen_corners_round_trip():
    cam_corners = np.array(
        [[100, 80], [560, 120], [520, 430], [80, 390]],
        dtype=np.float32,
    )
    desk_corners = np.array(
        [[0, 0], [1919, 0], [1919, 1079], [0, 1079]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(cam_corners, desk_corners)

    corners = _derive_screen_corners(homography, 1920, 1080)

    assert corners.shape == (4, 2)
    assert np.allclose(corners, cam_corners, atol=0.5)

    back = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    assert np.allclose(back, desk_corners, atol=0.5)


def test_order_corners_tlbr():
    rect = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        dtype=np.float32,
    )
    shuffled = rect[[2, 0, 3, 1]]

    ordered = order_corners_tlbr(shuffled)

    assert np.allclose(ordered, rect)


def test_detect_screen_with_synthetic_projector_frame():
    # Build a synthetic "camera view" where only a projected rectangle
    # changes colour between the RGBW phases. The rest of the frame
    # (the desk) is identical in every phase.
    width, height = 640, 480
    frames = []

    for color in ((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # a projector screen region in the centre (in camera space)
        cv2.rectangle(frame, (200, 150), (440, 330), color, -1)
        # the desk area (unchanged across phases)
        frame[330:400, 50:100] = (80, 80, 80)
        frames.append(frame)

    image_w, image_h = frames[0].shape[1], frames[0].shape[0]
    corners = _screen_corners_from_frames(frames, image_w, image_h)

    assert corners.shape == (4, 2)
    xs = corners[:, 0]
    ys = corners[:, 1]
    # The screen region should project to a rectangle roughly where
    # the synthetic projector region was.
    assert xs.min() < 300 and xs.max() > 340
    assert ys.min() < 250 and ys.max() > 290
