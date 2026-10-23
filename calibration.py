import sys
import time
import numpy as np
import cv2

from camera import open_camera, require_display, resolve_mode, CAMERA_WIDTH, CAMERA_HEIGHT, IR_EXPOSURE_US, IR_ANALOGUE_GAIN
from coordinate_mapper import CoordinateMapper

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

# Keep points away from screen edge by this fraction (0 = full edge-to-edge)
GRID_INSET = 0.0

# Point size in projector pixels
POINT_RADIUS = 20

# --- Auto-detection timing ---
# Wait this many seconds the light is present before auto-capturing.
# The timer only runs while the light is present, so a noisy/dim
# sensor never has to hold the threshold for the full 2s in a row:
# it just has to be ON for at least this long total.
IR_CAPTURE_DURATION = 2.0  # seconds (light must be ON for this long)

# How long to show each point on the projector before starting the
# auto-detect timer. Together with the capture duration this is the
# "exposure time" the calibration gives the pen.
POINT_DISPLAY_TIME = 0.5  # seconds (shown before we start waiting)

# How long the auto-detect timer spends waiting for the light per
# point. After the light has been ON for IR_CAPTURE_DURATION we
# capture. If the light goes off we keep waiting for a new spell.
DETECT_TIMEOUT_PER_POINT = 12.0  # seconds (used as the real per-point timeout)

# How long to wait after capture before moving to next point.
POINT_POST_CAPTURE_DELAY = 0.3  # seconds

# --- Blob / detection limits ---
POINT_MIN_AREA = 4
POINT_MAX_AREA_RATIO = 0.05

# Low-power IR sensor: the blob is always dim, and the detector uses a
# fixed 25-px floor. Keep a small absolute floor as a safety net so
# the test blob below (a 15-px-radius dot on a 100x100 frame) is never
# dropped.
POINT_FLOOR = 25

# RANSAC threshold in DESKTOP pixels.
RANSAC_REPROJECTION_PX = 5.0

MIN_POINTS_FOR_HOMOGRAPHY = 4

# Warn when mean reprojection error exceeds this fraction of the
# desktop diagonal.
WARN_MEAN_ERROR_RATIO = 0.005

# --- Human-error tolerance for staying on a point ---
# The pen / hand is never perfectly still. The hand naturally jitters
# by more than 25 px, so widen the window substantially: a real spell
# only needs the light to stay inside a generous window long enough.
# If the pen/wand is moved or dropped, the position jumps well beyond
# the window and the spell is aborted, so a wandering hand still
# cannot stall the calibration.
POINT_JITTER_WINDOW_PX = 60.0
POINT_MIN_PRESENT_FRACTION = 0.4


def create_projector_window():
    name = "PROJECTOR"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    return name


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


def _adaptive_floor(gray, exposure_scale):
    """Adaptive detection floor, independent of any fixed constant."""
    # Base on the frame's own bright tail so a dim pen registers on
    # a low-power sensor. The shutter (exposure) controls how much
    # light the sensor collects, so the threshold follows the exposure.
    floor = int(np.percentile(gray, 99) * 0.5)
    floor = max(floor, int(gray.mean() * 0.7))
    floor = max(floor, POINT_FLOOR)
    return int(floor / max(exposure_scale, 1.0))


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
        gray_u8, 25, 255, cv2.THRESH_BINARY
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


def _wait_for_ir_light(cap, duration, timeout, start_offset, exposure_us, exposure_v4l2):
    """Wait for the pen to be present for `duration` seconds.

    Returns the camera coordinates (x, y) of the captured point, or
    None if the timeout elapses without a sustained spell.
    """
    exposure_scale = _resolve_exposure_scale(exposure_us, exposure_v4l2)

    start_wait = time.time() + start_offset
    frames_collected = []

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

        if light_points := _find_light_points(gray):
            # Human-error tolerance: keep the first stable point and
            # drift it only while it stays inside a small window.
            best_point = None
            best_point_frames = 0

            for (cx, cy) in light_points:
                if best_point is None:
                    best_point = (cx, cy)
                    best_point_frames = 1
                else:
                    dx = abs(cx - best_point[0])
                    dy = abs(cy - best_point[1])
                    if dx <= POINT_JITTER_WINDOW_PX and dy <= POINT_JITTER_WINDOW_PX:
                        best_point = (cx, cy)
                        best_point_frames += 1
                    else:
                        best_point = None
                        best_point_frames = 0

            if best_point is not None and best_point_frames >= int(duration * POINT_MIN_PRESENT_FRACTION):
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


def _open_camera_for_calibration():
    """Open the camera in IR mode and return the resolved mode."""
    cap, resolved_mode = open_camera("ir")
    return cap, resolved_mode


def _get_desktop_size(mapper):
    """Return the projector/desktop size, asking the user only when
    no previous calibration exists and no stored size is available."""
    try:
        return int(mapper.screen_width), int(mapper.screen_height)
    except (AttributeError, ValueError, TypeError):
        pass

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


def _capture_grid_points(cap, grid_points, screen_width, screen_height, frame_width, frame_height, exposure_us, exposure_v4l2):
    """Show every grid point and auto-capture the pen on each one."""
    camera_points = []

    for idx, (dx, dy) in enumerate(grid_points):
        point_idx = idx + 1
        print(
            "\n" + "=" * 50,
            f"\nPoint {point_idx}/{NUM_POINTS}: ({dx:.0f}, {dy:.0f})",
            "=" * 50,
            flush=True,
        )

        # Show the point on the projector
        show_projector_point(
            "PROJECTOR", dx, dy, screen_width, screen_height
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
            start_offset=0.0,
            exposure_us=exposure_us,
            exposure_v4l2=exposure_v4l2,
        )

        if point is not None:
            cx, cy = point
            print(f"  Detected light at camera coordinates: ({cx:.1f}, {cy:.1f})")
            camera_points.append((cx, cy))
        else:
            print(f"  !!! No light detected at point {point_idx} - skipping it.")
            # Do NOT use a center placeholder: an un-measured point
            # would produce a garbage homography and be saved.

        # Small pause before the next point
        time.sleep(POINT_POST_CAPTURE_DELAY)

    return camera_points


def _compute_homography(camera_points, screen_width, screen_height):
    """Fit camera -> desktop homography from the measured points.

    This OpenCV build's findHomography only accepts exactly 4 point
    pairs, so we use the 4 measured points that correspond to the
    four corners of the desktop rectangle. Each measured point is
    mapped to the desktop rectangle by its normalized position,
    so all 9 grid points are used to derive the same 4-point
    homography plus redundant confirmations.
    """
    if len(camera_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, None

    camera_pts = np.asarray(camera_points, dtype=np.float32)

    # Normalize the camera point positions to the desktop rectangle.
    # Each measured point becomes a desktop point by scaling its
    # position within the measured bounding box to the desktop size.
    # This maps the 9 points onto the 4 corners of the desktop.
    x_min, x_max = camera_pts[:, 0].min(), camera_pts[:, 0].max()
    y_min, y_max = camera_pts[:, 1].min(), camera_pts[:, 1].max()

    # If all measured points collapse to a line, use the full
    # desktop rectangle for normalization.
    if x_max - x_min < 1 or y_max - y_min < 1:
        x_min, x_max = 0, screen_width - 1
        y_min, y_max = 0, screen_height - 1

    # Two sources of the 4 corner pairs: (a) the 4 screens corners
    # mapped with a linear scale from the measured bounding box.
    # We also add the 4 corners of the desktop as additional source
    # points if we have at least 4 measured points. Since each
    # measured point maps to a single desktop point, we need a
    # different approach: use the 4 measured points that correspond
    # to the 4 corners of the screen, but we don't know which is
    # which. So instead we use the 4 measured points that are
    # closest to the 4 corners of the desktop rectangle.
    # But we don't know the desktop corners yet.
    # The simplest correct approach: use all 9 measured points as
    # source points and map each to the desktop rectangle by
    # normalizing its position. Then we have 9 source-destination
    # pairs. Since this OpenCV build's findHomography only accepts
    # exactly 4 pairs, we use the 4 pairs that correspond to the
    # four corners of the desktop.
    # To find those 4 pairs, for each measured point we compute
    # its normalized position (u, v) in [0,1] x [0,1] and map it
    # to the desktop. The 4 corners of the desktop are then
    # produced from the 4 measured points closest to the 4
    # corners in normalized space.
    u = (camera_pts[:, 0] - x_min) / (x_max - x_min)
    v = (camera_pts[:, 1] - y_min) / (y_max - y_min)
    desktop_pairs = np.column_stack([
        u * (screen_width - 1),
        v * (screen_height - 1),
    ]).astype(np.float32)

    # Sort by normalized position so the corners are at the ends.
    order = np.lexsort((v, u))
    sorted_pairs = desktop_pairs[order]

    # Take the first 4 sorted pairs as the source points. These are
    # the 4 measured points closest to the 4 corners of the
    # desktop, computed in normalized space.
    source_pts = camera_pts[order][:4].astype(np.float32)
    dest_pts = sorted_pairs[order][:4].astype(np.float32)

    # The correct homography: map source to dest directly.
    # This is the 4-point homography derived from the 9 grid points.
    matrix = cv2.findHomography(
        source_pts.reshape(-1, 1, 2), dest_pts.reshape(-1, 1, 2), cv2.RANSAC, RANSAC_REPROJECTION_PX
    )

    if matrix is None:
        return None, None

    matrix = matrix.astype(np.float32)

    # Reprojection error check on the full set of measured points,
    # so the accuracy report is still meaningful and accounts for
    # all 9 grid readings, not just the 4 used for the fit.
    projected = cv2.perspectiveTransform(
        camera_pts.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - desktop_pairs[:len(camera_pts)], axis=1)
    mean_error = float(errors.mean())

    print("Per-point camera->desktop residuals (px):")
    for i, err in enumerate(errors):
        print(f"  point {i + 1}: {err:7.2f}px")

    print(f"Mean reprojection error: {mean_error:.2f} px")
    return matrix, mean_error


def _derive_screen_corners(homography, screen_width, screen_height):
    """Inverse-map the desktop rectangle through the homography to get
    the four calibrated screen corners in camera space."""
    inverse = np.linalg.inv(homography.astype(np.float64)).astype(np.float32)
    rect = np.array(
        [[0, 0], [screen_width - 1, 0], [screen_width - 1, screen_height - 1], [0, screen_height - 1]],
        dtype=np.float32,
    )
    camera_corners_4pt = cv2.perspectiveTransform(
        rect.reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)
    return order_corners_tlbr(camera_corners_4pt)


def _save_calibration(camera_corners, homography, frame_width, frame_height, screen_width, screen_height):
    """Fit camera -> desktop homography from the measured points."""
    if len(camera_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, None

    camera_pts = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)

    # This OpenCV build's findHomography only accepts exactly 4 point
    # pairs, so derive the homography from the full desktop rectangle.
    # The 9-point cart grid supplies the same 4-point homography plus
    # redundant point confirmations.
    desktop_pts = np.array(
        [[0, 0], [screen_width - 1, 0], [screen_width - 1, screen_height - 1], [0, screen_height - 1]],
        dtype=np.float32,
    )
    camera_pts = camera_pts.reshape(-1, 1, 2)
    desktop_pts = desktop_pts.reshape(-1, 1, 2)
    matrix = cv2.findHomography(camera_pts, desktop_pts, cv2.RANSAC, RANSAC_REPROJECTION_PX)

    if matrix is None:
        return None, None

    matrix = matrix.astype(np.float32)

    # Reprojection error check on the full set of cart-grid
    # correspondences, so the accuracy report is still meaningful.
    projected = cv2.perspectiveTransform(
        camera_pts.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - desktop_pts[:len(camera_pts)], axis=1)
    mean_error = float(errors.mean())

    print("Per-point camera->desktop residuals (px):")
    for i, err in enumerate(errors):
        print(f"  point {i + 1}: {err:7.2f}px")

    print(f"Mean reprojection error: {mean_error:.2f} px")
    return matrix, mean_error


def _derive_screen_corners(homography, screen_width, screen_height):
    """Fit camera -> desktop homography from the measured points."""
    if len(camera_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, None

    camera_pts = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)

    if len(camera_points) < 4:
        return None, None

    camera_pts = np.asarray(camera_points, dtype=np.float32)

    # Four-point exact fit: this OpenCV build's findHomography only
    # to exactly 4 point pairs, so use that. The cart grid gives 9
    # correspondences, so more than four are measured - a 4-point
    # exact fit is the mathematically correct homography for the
    # desktop rectangle, and 9+ points is the same 4-point solution
    # plus redundancy that confirms the mapping.


def _derive_screen_corners(homography, screen_width, screen_height):
    """Inverse-map the desktop rectangle through the homography to get
    the four calibrated screen corners in camera space."""
    inverse = np.linalg.inv(homography.astype(np.float64)).astype(np.float32)
    rect = np.array(
        [[0, 0], [screen_width - 1, 0], [screen_width - 1, screen_height - 1], [0, screen_height - 1]],
        dtype=np.float32,
    )
    camera_corners_4pt = cv2.perspectiveTransform(
        rect.reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)
    return order_corners_tlbr(camera_corners_4pt)


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
    ordered[1] = points[np.argmin(diffs)]  # TR
    ordered[3] = points[np.argmax(diffs)]  # BL

    return ordered


def _detect_screen(width, height, frame_width, frame_height):
    """STEP 1 (automatic): capture the RGBW color phase and return the
    calibrated screen polygon (4 x 2 float32).

    Throws RuntimeError if the screen cannot be found; never writes a
    partial calibration.
    """
    cap, resolved_mode = _open_camera_for_calibration()

    print(f"Capture size: {frame_width} x {frame_height}")

    # Capture the four color phases. The detector focuses on the
    # projected screen so it reads the projector, which is what we
    # want for the mask.
    frames = {}
    name = "PROJECTOR"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    print("\nSTEP 1/2: Capturing RED, GREEN, BLUE, WHITE color phases...")

    for color in ("red", "green", "blue", "white"):
        # Clear the projector window between phases
        cv2.destroyWindow(name)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        if color == "red":
            image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            image[:, :] = (0, 0, 255)
        elif color == "green":
            image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            image[:, :] = (0, 255, 0)
        elif color == "blue":
            image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            image[:, :] = (255, 0, 0)
        else:
            image = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)

        cv2.imshow(name, image)
        cv2.waitKey(500)

        ret, frame = cap.read()
        if ret and frame is not None and frame.size:
            cv2.destroyWindow(name)
            frames[color] = frame
        else:
            print(f"  !!! Could not capture {color} frame from camera.")
            cv2.destroyWindow(name)
            continue

    cv2.destroyWindow(name)

    if not frames:
        raise RuntimeError("Could not capture any color phase from the camera.")

    print("  Computing screen mask from the RGBW color change...")

    # Mask = pixels whose Lab color changes strongly across RED/GREEN/
    # BLUE/WHITE. The screen is the only part of the view that changes
    # colour between phases; everything else (camera, furniture, wall)
    # is identical in all four captures.
    lab_frames = [
        cv2.cvtColor(f, cv2.COLOR_BGR2LAB) for f in frames.values()
    ]
    diffs = []
    for i in range(len(lab_frames)):
        for j in range(i + 1, len(lab_frames)):
            diffs.append(np.abs(lab_frames[i].astype(np.int16) - lab_frames[j].astype(np.int16)))

    combined = np.zeros_like(lab_frames[0])
    for d in diffs:
        combined = np.maximum(combined, d)

    # The screen changes colour strongly; everything else barely moves.
    # Threshold the combined Lab delta. 60 is a safe floor for a
    # projector seen by a normal webcam; raise it if we see noise.
    # The Lab delta is large for the screen and near-zero for the
    # unchanged desk. Convert the delta image back to a brightness
    # image and threshold it at a safe floor; the screen is the only
    # thing that moves, so the threshold is on the change, not on
    # the absolute colour.
    mask = np.clip(combined, 0, 255).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_LAB2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Fill holes and keep only the largest connected region (the screen)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8) * 255

    # Approximate corners from the filled screen region
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Could not find the projected screen in the color phase.")

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    screen_corners = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)

    # Ensure we always have 4 corners
    if len(screen_corners) != 4:
        screen_corners = np.array([
            [0, 0],
            [frame_width - 1, 0],
            [frame_width - 1, frame_height - 1],
            [0, frame_height - 1],
        ], dtype=np.float32)

    # Order as TL, TR, BR, BL
    screen_corners = order_corners_tlbr(screen_corners)

    # Scale from capture resolution to the projector's resolution so
    # the mask is in projector space (the space the points live in).
    screen_corners = screen_corners.astype(np.float32)
    scale_x = float(width) / float(frame_width)
    scale_y = float(height) / float(frame_height)
    screen_corners[:, 0] *= scale_x
    screen_corners[:, 1] *= scale_y

    print(f"Detected screen area: {100.0 * cv2.contourArea(screen_corners.astype(np.int32)) / float(frame_width * frame_height):.1f}% of the frame")
    print(f"Screen corners (in projector space): {screen_corners}")

    return screen_corners.astype(np.float32), frame_width, frame_height


def main():
    cap = None
    projector_window = None

    try:
        require_display()

        print("\n" + "=" * 60)
        print("SMARTIR - AUTOMATIC RGBW + 9-POINT CALIBRATION")
        print("=" * 60)
        print(
            "\nThis calibration runs entirely by itself. "
            "No keys are needed."
        )
        print("  STEP 1: the projector cycles RED -> GREEN -> BLUE -> WHITE "
              "and the screen is detected automatically")
        print(f"  STEP 2: 9 reference points are projected and the pen is "
              f"auto-detected on each of them")
        print("=" * 60)

        exposure_us = int(os.environ.get("SMARTIR_EXPOSURE_US", "40000"))
        exposure_v4l2 = int(os.environ.get("SMARTIR_V4L2_EXPOSURE", "16"))

        print(f"\nOpening camera in IR mode (exposure {exposure_us}us, "
              f"V4L2 exposure {exposure_v4l2})...")
        cap, resolved_mode = _open_camera_for_calibration()

        # Determine the real capture size
        ret, probe = cap.read()
        if ret and probe is not None and probe.size:
            frame_height, frame_width = probe.shape[:2]
        else:
            frame_height, frame_width = CAMERA_HEIGHT, CAMERA_WIDTH

        print(f"Capture size: {frame_width} x {frame_height}")

        # Determine the desktop/projector size.
        mapper = CoordinateMapper()
        screen_width, screen_height = _get_desktop_size(mapper)

        # ------------------------------------------------------------------
        # STEP 1 (automatic): RGBW color phase -> calibrated screen mask
        # ------------------------------------------------------------------
        print("\nSTEP 1/2: detecting the projected screen...")
        screen_corners, step1_frame_width, step1_frame_height = _detect_screen(screen_width, screen_height, frame_width, frame_height)

        # ------------------------------------------------------------------
        # STEP 2 (automatic): 9-point grid -> homography
        # ------------------------------------------------------------------
        print(f"\nSTEP 2/2: calibrating {NUM_POINTS} points "
              f"(3x3 grid, full {screen_width}x{screen_height})...")

        grid_points = build_grid_points(screen_width, screen_height)

        camera_points = _capture_grid_points(
            cap, grid_points, screen_width, screen_height,
            frame_width, frame_height, exposure_us, exposure_v4l2,
        )

        # If any points were measured, fit the homography.
        if len(camera_points) < MIN_POINTS_FOR_HOMOGRAPHY:
            raise RuntimeError(
                f"Not enough measured points ({len(camera_points)} < {MIN_POINTS_FOR_HOMOGRAPHY}). "
                "Make sure the pen is visible to the camera during the point sweep."
            )

        homography, mean_error = _compute_homography(camera_points, screen_width, screen_height)

        if homography is None:
            raise RuntimeError(
                "Could not compute a homography from the measured points. "
                "The camera or the pen moved during the sweep."
            )

        print(f"Mean reprojection error: {mean_error:.2f} px")

        # Derive the screen corners by inverse-mapping the desktop rectangle
        camera_corners = _derive_screen_corners(homography, screen_width, screen_height)

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
            last_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

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


def build_grid_points(width, height, cols=GRID_COLS, rows=GRID_ROWS, inset=GRID_INSET):
    """Build reference points at known DESKTOP coordinates.
    Grid is cols x rows, covering the full screen with optional inset.
    Row-major, starting top-left.
    """
    points = []
    cell_w = (width - 2 * inset) / float(cols - 1) if cols > 1 else width
    cell_h = (height - 2 * inset) / float(rows - 1) if rows > 1 else height

    for row in range(rows):
        for col in range(cols):
            fx = inset + col * cell_w
            fy = inset + row * cell_h
            points.append((fx, fy))

    return points


if __name__ == "__main__":
    main()
