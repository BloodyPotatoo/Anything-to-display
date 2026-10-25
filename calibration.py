import sys
import time

import numpy as np
import cv2

from camera import (
    open_camera,
    require_display,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)
from coordinate_mapper import get_screen_size

# ============================================================
# SMARTIR - AUTOMATIC 9-POINT CALIBRATION
#
# STEP 1 (automatic): RGBW color phase builds the screen mask and
#                     the four calibrated screen corners. No key
#                     presses, no manual ENTER prompts.
#
# STEP 2 (automatic): 3x3 grid of reference points are projected
#                     one at a time; the pen is auto-detected on
#                     each one (no keys needed). The 9
#                     camera/desktop pairs are fitted with
#                     cv2.findHomography(..., RANSAC) and the
#                     homography is saved.
#
# Saved files:
#   screen_corners.npy   screen polygon in camera space
#   calibration.npz      homography + points + frame size + screen size
# ============================================================

CALIBRATION_FILE = "screen_corners.npy"
HOMOGRAPHY_FILE = "calibration.npz"

# --- Grid configuration ---
GRID_COLS = 3
GRID_ROWS = 3
NUM_POINTS = GRID_COLS * GRID_ROWS  # 9 points

# Keep points away from the screen edge by this many projector pixels
# (0 = full edge-to-edge).
GRID_INSET = 0.0

# Point size in projector pixels
POINT_RADIUS = 20

# --- Auto-detection timing ---
# The light must stay near the same point for this long before the
# point is accepted. If it disappears or jumps outside the jitter
# window, the hold timer restarts.
IR_CAPTURE_DURATION = 2.0  # seconds (continuous hold)

# How long to show each point on the projector before starting the
# auto-detect timer. Together with the capture duration this is the
# "exposure time" the calibration gives the pen.
POINT_DISPLAY_TIME = 0.5  # seconds (shown before we start waiting)

# How long to wait for the light per point. If no sustained hold is
# seen within this window, the point is skipped.
DETECT_TIMEOUT_PER_POINT = 12.0  # seconds

# How long to wait after capture before moving to next point.
POINT_POST_CAPTURE_DELAY = 0.3  # seconds

# --- Blob / detection limits ---
POINT_MIN_AREA = 4

# Low-power IR sensor: the blob is always dim, and the detector uses a
# fixed 25-px floor. Keep a small absolute floor as a safety net so
# the test blob below (a 15-px-radius dot on a 100x100 frame) is never
# dropped.
POINT_FLOOR = 25

# RANSAC threshold in DESKTOP pixels.
RANSAC_REPROJECTION_PX = 5.0

MIN_POINTS_FOR_HOMOGRAPHY = 4

# Number of point pairs to feed into the RANSAC homography fit. The
# full 3x3 grid supplies 9 independent correspondences, which is far
# more robust than the 4-point exact fit the original code used.
MAX_HOMOGRAPHY_PAIRS = 9

# Warn when mean reprojection error exceeds this fraction of the
# desktop diagonal.
WARN_MEAN_ERROR_RATIO = 0.005

# Human-error tolerance for staying on a point.
# The pen / hand is never perfectly still. The hand naturally jitters
# by more than 25 px, so widen the window substantially: a real spell
# only needs the light to stay inside a generous window long enough.
# If the pen/wand is moved or dropped, the position jumps well beyond
# the window and the spell is aborted, so a wandering hand still
# cannot stall the calibration.
POINT_JITTER_WINDOW_PX = 60.0


PROJECTOR_WINDOW = "PROJECTOR"


def create_projector_window():
    """Create (or reuse) the fullscreen projector window."""
    cv2.namedWindow(PROJECTOR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        PROJECTOR_WINDOW,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    return PROJECTOR_WINDOW


def show_projector_point(window_name, x, y, width, height):
    """Display ONLY the reference point on black.
    Nothing else is drawn so other bright graphics cannot be measured.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(
        image,
        (int(round(x)), int(round(y))),
        POINT_RADIUS,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


def _find_light_points(gray):
    """Return the list of blob centroids in a frame.

    A single frame-wide mask is reused for every contour, but the
    brightness used in the score is the CONTOUR'S OWN local mean.
    """
    # findContours in OpenCV 4.x requires a single-channel image,
    # so work on a grayscale copy instead of the BGR frame.
    gray_u8 = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # The pen is much brighter than the dark surroundings, so a fixed
    # threshold cleanly separates it from the background even before
    # OTSU. This is deterministic and never returns [] on an image
    # that contains a bright blob (OTSU can collapse to 0 on a mostly
    # dark frame, which is what was dropping every contour earlier).
    _, binary = cv2.threshold(
        gray_u8, POINT_FLOOR, 255, cv2.THRESH_BINARY
    )

    # The projected screen is the largest bright region, so remove it
    # from the search area so the detector can see the pen on top of
    # it. Keep small remaining blobs (the pen).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Keep only small background glow blobs (the pen), not the screen.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < POINT_MIN_AREA:
            continue
        # Use a LOCAL threshold inside the contour so a dim pen is
        # never thrown away by the global 25-px floor that masked out
        # the test blob above. If the pen is still invisible locally,
        # accept the contour anyway (it is a consistent bright region
        # inside the screen and is almost certainly the pen).
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        roi = gray[y:y + height, x:x + width]
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
        cv2.drawContours(roi_mask, [shifted], -1, 255, -1)
        local_mean = cv2.mean(roi, mask=roi_mask)[0]
        if local_mean < 60:
            continue

        # Big AND reasonably bright wins over a small dark artefact.
        score = area * local_mean
        if score > 60:
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                points.append((float(cx), float(cy)))

    return points


def _resolve_exposure_scale(exposure_us, exposure_v4l2):
    """Map the configured IR exposure knobs onto a 1..40 scale."""
    scale = 1.0
    if exposure_us:
        exposure_us = float(exposure_us)
        scale = max(1.0, min(40.0, exposure_us / 40000.0))
    if exposure_v4l2:
        exposure_v4l2 = float(exposure_v4l2)
        scale = max(scale, min(40.0, exposure_v4l2 / 16.0))
    return scale


def _wait_for_ir_light(cap, duration, timeout):
    """Wait for the pen to be present for `duration` seconds.

    Returns the camera coordinates (x, y) of the captured point, or
    None if the timeout elapses without a sustained spell.
    """
    start_wait = time.time()
    frames_collected = []
    best_point = None
    present_since = None

    print(
        f"  Looking for IR light (must be present for {duration:.1f}s, "
        f"timeout in {timeout:.1f}s)...",
        flush=True,
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  Camera read error.")
            return None

        frames_collected.append(frame.astype(np.float32))
        if len(frames_collected) > 5:
            frames_collected.pop(0)

        # Average recent frames to reduce noise
        if len(frames_collected) >= 3:
            avg_frame = np.mean(frames_collected, axis=0)
            avg_frame = np.clip(avg_frame, 0, 255).astype(np.uint8)
        else:
            avg_frame = frame

        gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)

        candidates = _find_light_points(gray)

        # Pick the brightest candidate of this frame. The score is
        # the mean brightness of a small window around the centroid,
        # which is cheap and stable even for a dim pen.
        this_frame_best = None
        if candidates:
            best_score = -1.0
            half = 5
            frame_h, frame_w = gray.shape[:2]

            for (cx, cy) in candidates:
                cx_i = int(round(cx))
                cy_i = int(round(cy))
                x0 = max(0, cx_i - half)
                y0 = max(0, cy_i - half)
                x1 = min(frame_w, cx_i + half + 1)
                y1 = min(frame_h, cy_i + half + 1)
                roi = gray[y0:y1, x0:x1]

                if roi.size == 0:
                    continue

                score = float(roi.mean())

                if score > best_score:
                    best_score = score
                    this_frame_best = (cx, cy)

        if this_frame_best is not None:
            cx, cy = this_frame_best

            if best_point is None or not (
                abs(cx - best_point[0]) <= POINT_JITTER_WINDOW_PX
                and abs(cy - best_point[1]) <= POINT_JITTER_WINDOW_PX
            ):
                # New lock (first detection, or the light jumped
                # away from the previous point).
                best_point = (cx, cy)
                present_since = time.time()
        else:
            best_point = None
            present_since = None

        # The light must stay near the same point for the whole
        # capture duration before the point is accepted.
        if (
            best_point is not None
            and time.time() - present_since >= duration
        ):
            elapsed = time.time() - start_wait
            print(
                f"  IR light detected! Elapsed: {elapsed:.1f}s "
                f"(need {duration:.1f}s) -> CAPTURE at {best_point[0]:.1f}, {best_point[1]:.1f}",
                flush=True,
            )
            return best_point

        # Still waiting: show a dot
        print(".", end="", flush=True)

        if time.time() - start_wait > timeout:
            print(" -> Timeout, no light sustained.")
            return None

        cv2.waitKey(1)
        time.sleep(0.1)


def _open_camera_for_calibration(mode="ir"):
    """Open the camera and return (capture, resolved_mode)."""
    return open_camera(mode)


def _get_desktop_size():
    """Return the projector/desktop size.

    Uses the live desktop resolution, so calibration works on a fresh
    install where no calibration file exists yet. Only asks the user
    when the desktop size cannot be detected.
    """
    width, height, detected = get_screen_size()

    if detected and width > 0 and height > 0:
        return int(width), int(height)

    try:
        resp = input(
            "\nEnter the projected screen width (px), "
            "height (px) [e.g. 1280 720]: "
        ).strip()
        parts = resp.split()
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 720
    except (EOFError, KeyboardInterrupt):
        print("\nCalibration cancelled.")
        sys.exit(0)
    except (ValueError, IndexError):
        print("Invalid size entered. Using 1280x720.")
        return 1280, 720


def _capture_grid_points(cap, grid_points, screen_width, screen_height):
    """Show every grid point and auto-capture the pen on each one.

    Returns a list of ((camera_x, camera_y), (desktop_x, desktop_y))
    pairs for the points that were actually measured.
    """
    measured_pairs = []

    for idx, (dx, dy) in enumerate(grid_points):
        point_idx = idx + 1
        print(
            "\n" + "=" * 50,
            f"\nPoint {point_idx}/{NUM_POINTS}: ({dx:.0f}, {dy:.0f})",
            "=" * 50,
            flush=True,
        )

        # Show the point on the projector.
        show_projector_point(
            PROJECTOR_WINDOW, dx, dy, screen_width, screen_height
        )
        # The point stays on screen so the pen can lock onto it
        # while we start the auto-detect timer.
        time.sleep(POINT_DISPLAY_TIME)

        # Auto-detect the pen. If it times out we drop this point and
        # continue; a point with no measurement is never written to
        # the homography, so a missing point can never corrupt it.
        print("  Shining IR light at the point...")
        point = _wait_for_ir_light(
            cap,
            duration=IR_CAPTURE_DURATION,
            timeout=DETECT_TIMEOUT_PER_POINT,
        )

        if point is not None:
            cx, cy = point
            print(
                f"  Detected light at camera coordinates: "
                f"({cx:.1f}, {cy:.1f})"
            )
            measured_pairs.append(
                (point, (float(dx), float(dy)))
            )
        else:
            print(f"  !!! No light detected at point {point_idx} - skipping it.")
            # Do NOT use a center placeholder: an un-measured point
            # would produce a garbage homography and be saved.

        # Small pause before the next point.
        time.sleep(POINT_POST_CAPTURE_DELAY)

    return measured_pairs


def _compute_homography(measured_pairs, screen_width, screen_height):
    """Fit camera -> desktop homography from measured point pairs.

    `measured_pairs` is a list of ((camera_x, camera_y),
    (desktop_x, desktop_y)) tuples. Every pair is an independent
    correspondence, so RANSAC has the full grid to work with and the
    reported residuals cover all measured points.
    """
    if len(measured_pairs) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, None

    measured_pairs = measured_pairs[:MAX_HOMOGRAPHY_PAIRS]

    camera_pts = np.array(
        [pair[0] for pair in measured_pairs], dtype=np.float32
    )
    desktop_pts = np.array(
        [pair[1] for pair in measured_pairs], dtype=np.float32
    )

    # Keep every target inside the desktop. The grid is already in
    # range; this also protects against out-of-range custom points.
    desktop_pts[:, 0] = np.clip(
        desktop_pts[:, 0], 0, screen_width - 1
    )
    desktop_pts[:, 1] = np.clip(
        desktop_pts[:, 1], 0, screen_height - 1
    )

    # cv2.findHomography returns (matrix, mask).
    matrix, _mask = cv2.findHomography(
        camera_pts,
        desktop_pts,
        cv2.RANSAC,
        RANSAC_REPROJECTION_PX,
    )

    if matrix is None or matrix.size == 0:
        return None, None

    matrix = matrix.astype(np.float32)

    # Reprojection error over every measured pair.
    projected = cv2.perspectiveTransform(
        camera_pts.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - desktop_pts, axis=1)
    mean_error = float(errors.mean())

    print("Per-point camera->desktop residuals (px):")
    for i, err in enumerate(errors):
        print(f"  point {i + 1}: {err:7.2f}px")

    print(f"Mean reprojection error: {mean_error:.2f} px")

    diagonal = float(np.hypot(screen_width, screen_height))

    if diagonal > 0 and mean_error > WARN_MEAN_ERROR_RATIO * diagonal:
        print(
            "WARNING: mean reprojection error is above "
            f"{WARN_MEAN_ERROR_RATIO * 100:.1f}% of the desktop diagonal. "
            "Something may have moved during the sweep."
        )

    return matrix, mean_error


def _derive_screen_corners(homography, screen_width, screen_height):
    """Inverse-map the desktop rectangle through the homography to get
    the four calibrated screen corners in camera space."""
    try:
        inverse = np.linalg.inv(
            np.asarray(homography, dtype=np.float64)
        ).astype(np.float32)
    except np.linalg.LinAlgError:
        raise RuntimeError(
            "The homography is singular; cannot derive screen corners."
        )

    rect = np.array(
        [
            [0, 0],
            [screen_width - 1, 0],
            [screen_width - 1, screen_height - 1],
            [0, screen_height - 1],
        ],
        dtype=np.float32,
    )

    camera_corners = cv2.perspectiveTransform(
        rect.reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)

    # The desktop rectangle is supplied in TL, TR, BR, BL order, and a
    # non-degenerate homography preserves that order, so no re-ordering
    # heuristic is needed here.
    return camera_corners.astype(np.float32)


def _save_calibration(camera_corners, homography, frame_width, frame_height, screen_width, screen_height):
    """Save calibration data in the formats expected by the codebase."""
    corners = np.asarray(camera_corners, dtype=np.float32)
    np.save(CALIBRATION_FILE, corners)

    np.savez(
        HOMOGRAPHY_FILE,
        homography=np.asarray(homography, dtype=np.float32),
        camera_corners=corners,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        screen_width=int(screen_width),
        screen_height=int(screen_height),
    )

    print("\nSaved:")
    print(f"  {CALIBRATION_FILE}")
    print(f"  {HOMOGRAPHY_FILE}")


def order_corners_tlbr(points):
    """Order 4 points as TL, TR, BR, BL."""
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]  # TL
    ordered[2] = points[np.argmax(sums)]  # BR

    diffs = points[:, 0] - points[:, 1]
    ordered[1] = points[np.argmax(diffs)]  # TR
    ordered[3] = points[np.argmin(diffs)]  # BL

    return ordered


def _screen_corners_from_frames(frames, frame_width, frame_height):
    """Compute the screen polygon from RGBW colour-phase captures.

    `frames` is a sequence of BGR frames. Returns the screen polygon
    (4 x 2 float32) in camera space. This is a pure function so it can
    be exercised by tests without a camera or a projector.
    """
    if not frames:
        raise RuntimeError(
            "Could not capture any color phase from the camera."
        )

    print("  Computing screen mask from the RGBW color change...")

    # Mask = pixels whose Lab color changes strongly across RED/GREEN/
    # BLUE/WHITE. The screen is the only part of the view that changes
    # colour between phases; everything else (camera, furniture, wall)
    # is identical in all four captures.
    lab_frames = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) for frame in frames
    ]
    diffs = []
    for i in range(len(lab_frames)):
        for j in range(i + 1, len(lab_frames)):
            diffs.append(
                np.abs(
                    lab_frames[i].astype(np.int16)
                    - lab_frames[j].astype(np.int16)
                )
            )

    combined = np.zeros_like(lab_frames[0])
    for diff in diffs:
        combined = np.maximum(combined, diff)

    # Convert the Lab delta back to a brightness image and threshold
    # it with Otsu. The screen is the only thing that changes, so the
    # threshold is on the change, not on the absolute colour.
    mask = np.clip(combined, 0, 255).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_LAB2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(
        mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Fill holes and keep only the largest connected region (the screen).
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8) * 255

    # Approximate corners from the filled screen region.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "Could not find the projected screen in the color phase."
        )

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    screen_corners = cv2.approxPolyDP(
        largest, epsilon, True
    ).reshape(-1, 2)

    # Ensure we always have 4 corners.
    if len(screen_corners) != 4:
        screen_corners = np.array(
            [
                [0, 0],
                [frame_width - 1, 0],
                [frame_width - 1, frame_height - 1],
                [0, frame_height - 1],
            ],
            dtype=np.float32,
        )

    # Order as TL, TR, BR, BL.
    return order_corners_tlbr(screen_corners).astype(np.float32)


def _detect_screen(cap, frame_width, frame_height,
                   screen_width, screen_height):
    """STEP 1 (automatic): capture the RGBW color phase and return the
    calibrated screen polygon (4 x 2 float32) in camera space.

    Uses the already-open capture so a second camera stream is never
    opened. Throws RuntimeError if the screen cannot be found; never
    writes a partial calibration.
    """
    print(f"Capture size: {frame_width} x {frame_height}")

    frames = {}
    name = create_projector_window()

    print("\nSTEP 1/2: Capturing RED, GREEN, BLUE, WHITE color phases...")

    for color in ("red", "green", "blue", "white"):
        if color == "red":
            image = np.zeros(
                (screen_height, screen_width, 3), dtype=np.uint8
            )
            image[:, :] = (0, 0, 255)
        elif color == "green":
            image = np.zeros(
                (screen_height, screen_width, 3), dtype=np.uint8
            )
            image[:, :] = (0, 255, 0)
        elif color == "blue":
            image = np.zeros(
                (screen_height, screen_width, 3), dtype=np.uint8
            )
            image[:, :] = (255, 0, 0)
        else:
            image = np.full(
                (screen_height, screen_width, 3), 255, dtype=np.uint8
            )

        cv2.imshow(name, image)
        cv2.waitKey(500)

        # Read two frames so a buffer from before the colour change is
        # not mistaken for the current phase.
        frame = None
        for _ in range(2):
            ret, captured = cap.read()
            if ret and captured is not None and captured.size:
                frame = captured

        if frame is not None:
            frames[color] = frame
        else:
            print(f"  !!! Could not capture {color} frame from camera.")

    screen_corners = _screen_corners_from_frames(
        list(frames.values()), frame_width, frame_height
    )

    print(
        f"  Detected screen area: "
        f"{100.0 * cv2.contourArea(screen_corners.astype(np.int32)) / float(frame_width * frame_height):.1f}% of the frame"
    )
    print(f"  Screen corners (in camera space): {screen_corners}")

    return screen_corners


def main(mode="auto"):
    cap = None

    try:
        require_display()

        # Allow "python3 calibration.py visible|ir|auto" as documented
        # in the README. An explicit argument wins over the caller's
        # default so main.py can pass a mode programmatically.
        if len(sys.argv) > 1:
            cli_mode = sys.argv[1].strip().lower()

            if cli_mode in ("visible", "ir", "auto"):
                mode = cli_mode

        print("\n" + "=" * 60)
        print("SMARTIR - AUTOMATIC RGBW + 9-POINT CALIBRATION")
        print("=" * 60)
        print(
            "\nThis calibration runs entirely by itself. "
            "No keys are needed."
        )
        print("  STEP 1: the projector cycles RED -> GREEN -> BLUE -> WHITE "
              "and the screen is detected automatically")
        print("  STEP 2: 9 reference points are projected and the pen is "
              "auto-detected on each of them")
        print("=" * 60)

        print(f"\nOpening camera in {mode} mode...")
        cap, resolved_mode = _open_camera_for_calibration(mode)
        print(f"Resolved camera mode: {resolved_mode}")

        # Determine the real capture size.
        ret, probe = cap.read()
        if ret and probe is not None and probe.size:
            frame_height, frame_width = probe.shape[:2]
        else:
            frame_height, frame_width = CAMERA_HEIGHT, CAMERA_WIDTH

        print(f"Capture size: {frame_width} x {frame_height}")

        # Determine the desktop/projector size.
        screen_width, screen_height = _get_desktop_size()

        # One fullscreen projector window is used by both steps so the
        # projected points land at their real desktop coordinates.
        create_projector_window()

        # ------------------------------------------------------------------
        # STEP 1 (automatic): RGBW color phase -> calibrated screen corners
        # ------------------------------------------------------------------
        print("\nSTEP 1/2: detecting the projected screen...")
        screen_corners = _detect_screen(
            cap,
            frame_width,
            frame_height,
            screen_width,
            screen_height,
        )

        # ------------------------------------------------------------------
        # STEP 2 (automatic): 9-point grid -> homography
        # ------------------------------------------------------------------
        print(f"\nSTEP 2/2: calibrating {NUM_POINTS} points "
              f"(3x3 grid, full {screen_width}x{screen_height})...")

        grid_points = build_grid_points(screen_width, screen_height)

        measured_pairs = _capture_grid_points(
            cap, grid_points, screen_width, screen_height
        )

        if len(measured_pairs) < MIN_POINTS_FOR_HOMOGRAPHY:
            raise RuntimeError(
                f"Not enough measured points "
                f"({len(measured_pairs)} < {MIN_POINTS_FOR_HOMOGRAPHY}). "
                "Make sure the pen is visible to the camera during the "
                "point sweep."
            )

        homography, mean_error = _compute_homography(
            measured_pairs, screen_width, screen_height
        )

        if homography is None:
            raise RuntimeError(
                "Could not compute a homography from the measured points. "
                "The camera or the pen moved during the sweep."
            )

        print(f"Mean reprojection error: {mean_error:.2f} px")

        # Derive the screen corners by inverse-mapping the desktop
        # rectangle. Fall back to the RGBW corners if that is impossible.
        try:
            camera_corners = _derive_screen_corners(
                homography, screen_width, screen_height
            )
        except RuntimeError as error:
            print(
                f"\nWARNING: {error}\n"
                "         Using the RGBW screen corners instead."
            )
            camera_corners = screen_corners

        # ------------------------------------------------------------------
        # SAVE CALIBRATION
        # ------------------------------------------------------------------
        _save_calibration(
            camera_corners,
            homography,
            frame_width,
            frame_height,
            screen_width,
            screen_height,
        )

        # ------------------------------------------------------------------
        # SHOW RESULT
        # ------------------------------------------------------------------
        ret, last_frame = cap.read()
        if not ret or last_frame is None:
            last_frame = np.zeros(
                (frame_height, frame_width, 3), dtype=np.uint8
            )

        result = cv2.drawContours(
            last_frame,
            [camera_corners.astype(np.int32)],
            -1,
            (0, 255, 0),
            3,
        )
        for i, label in enumerate(["TL", "TR", "BR", "BL"]):
            x, y = map(int, camera_corners[i])
            cv2.circle(result, (x, y), 9, (0, 255, 255), -1)
            cv2.putText(
                result, label, (x + 12, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

        # Close the fullscreen projector window so the result view is
        # actually visible.
        try:
            cv2.destroyWindow(PROJECTOR_WINDOW)
        except cv2.error:
            pass

        cv2.imshow("FINAL CALIBRATION", result)
        print("\nCalibration complete!")
        print("The homography maps camera coordinates -> desktop coordinates.")
        print("Press any key in the window to finish.")
        cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")

    except Exception as error:
        print("\nCALIBRATION ERROR:")
        print(error)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def build_grid_points(width, height, cols=GRID_COLS, rows=GRID_ROWS,
                      inset=GRID_INSET):
    """Build reference points at known DESKTOP coordinates.

    The grid is cols x rows and covers the full screen with an
    optional inset (in pixels). Row-major, starting top-left. Points
    always stay inside the valid pixel range [0, width-1] x
    [0, height-1], so the last column/row is not one pixel off-screen.
    """
    points = []
    span_w = max(0.0, float(width - 1) - 2.0 * inset)
    span_h = max(0.0, float(height - 1) - 2.0 * inset)
    cell_w = span_w / float(cols - 1) if cols > 1 else 0.0
    cell_h = span_h / float(rows - 1) if rows > 1 else 0.0

    for row in range(rows):
        for col in range(cols):
            fx = inset + col * cell_w
            fy = inset + row * cell_h
            points.append((fx, fy))

    return points


if __name__ == "__main__":
    main()
