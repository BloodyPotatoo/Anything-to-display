import sys
import time

import cv2
import numpy as np

from camera import (
    open_camera,
    require_display,
    resolve_mode,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)
from coordinate_mapper import get_screen_size

# ============================================================
# SMARTIR - SCREEN CALIBRATION
#
# Two steps, in order:
#
#   STEP 1 - RGBW SCREEN DETECTION (where is the screen?)
#       The projector flashes RED -> GREEN -> BLUE -> WHITE.
#       Pixels that change colour strongly across those four
#       captures belong to the projected screen.
#       -> screen mask + four approximate corners
#       -> used to restrict the search area in step 2 and as a
#          fallback if step 2 fails.
#
#   STEP 2 - POINT-GRID HOMOGRAPHY (how does camera space map
#             to desktop space?)
#       The projector displays N reference points, one at a time,
#       at known desktop coordinates. Each point is measured in
#       camera space. This is the same idea as Johnny Chung Lee's
#       IR whiteboard calibration: real point correspondences,
#       fitted with RANSAC, instead of guessing a rectangle.
#       -> 3x3 homography + reprojection error report
#       -> camera-space screen corners derived from the homography
#
# Saved files:
#   screen_corners.npy  four camera-space corners (screen polygon)
#   calibration.npz     homography, points, frame size, screen size
#
# Run:
#   python calibration.py            (asks for the mode)
#   python calibration.py visible
#   python calibration.py ir
# ============================================================

CALIBRATION_FILE = "screen_corners.npy"
HOMOGRAPHY_FILE = "calibration.npz"

AVERAGE_FRAMES = 20
FRAME_DELAY = 0.03

# ------------------------------------------------------------
# STEP 1 settings - RGBW screen detection
# ------------------------------------------------------------
MORPH_KERNEL_SIZE = 7

# Minimum Lab colour-change score (absolute floor; also makes a
# static scene produce an empty mask).
RGBW_MIN_CHANGE = 18

# Morphological closing kernel used to join gaps in the screen.
RGBW_CLOSE_KERNEL = 31

# Ignore very small disconnected components.
RGBW_MIN_COMPONENT_RATIO = 0.015

# Contour approximation epsilon.
RGBW_APPROX_EPSILON = 0.02

# The screen should occupy at least this much of the camera frame.
MIN_SCREEN_AREA_RATIO = 0.015

# ------------------------------------------------------------
# STEP 2 settings - point-grid homography calibration
# ------------------------------------------------------------
GRID_COLS = 3
GRID_ROWS = 3

# Keep the reference points this far inside the screen edge
# (fraction of screen size) so they are never clipped.
GRID_INSET = 0.07

# Projected point radius, in projector pixels.
POINT_RADIUS = 16

# Time to let the projector/camera settle before measuring.
POINT_SETTLE_TIME = 0.30

# Frames averaged per measurement (noise reduction).
POINT_AVERAGE_FRAMES = 8

# Captures attempted per reference point.
POINT_ATTEMPTS = 2

# Blob size limits for one reference point, in camera pixels.
POINT_MIN_AREA = 4
POINT_MAX_AREA_RATIO = 0.05

# Absolute brightness floor for a projected point. The projector
# "black" background is never actually black, so this separates
# the dot from the screen without depending on a histogram alone.
POINT_FLOOR = 120

# RANSAC threshold in DESKTOP pixels (the homography is fitted
# from camera points to desktop points).
RANSAC_REPROJECTION_PX = 5.0

MIN_POINTS_FOR_HOMOGRAPHY = 4

# Warn when mean reprojection error exceeds this fraction of the
# desktop diagonal.
WARN_MEAN_ERROR_RATIO = 0.005

CALIBRATION_COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "WHITE": (255, 255, 255),
}


# ============================================================
# Shared helpers
# ============================================================

def create_projector_window():
    name = "PROJECTOR"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    return name


def show_color(window_name, color, width=1280, height=720):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


def confirm(title, lines=()):
    """Ask the user to continue in the terminal."""
    print("\n" + "=" * 60)
    print(title)

    for line in lines:
        print(line)

    print("=" * 60)

    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer != "q"


def wait_for_confirmation(color_name):
    return confirm(
        f"SHOW {color_name}",
        (
            "Make sure the projector covers the intended screen area.",
            "Press ENTER in this terminal to capture.",
            "Type Q and press ENTER to quit.",
        ),
    )


def capture_average_frame(cap, frame_count=None, quiet=False):
    if frame_count is None:
        frame_count = AVERAGE_FRAMES

    frames = []

    if not quiet:
        print(f"Capturing {frame_count} frames...")

    for index in range(frame_count):
        ret, frame = cap.read()

        if ret and frame is not None and frame.size:
            frames.append(frame.astype(np.float32))

        cv2.waitKey(1)
        time.sleep(FRAME_DELAY)

        if not quiet:
            print(
                f"\rFrame {index + 1}/{frame_count}",
                end="",
                flush=True,
            )

    if not quiet:
        print()

    if not frames:
        raise RuntimeError("Could not capture any camera frames.")

    average = np.mean(frames, axis=0)
    return np.clip(average, 0, 255).astype(np.uint8)


def resize_to_reference(frame, reference_shape):
    height, width = reference_shape[:2]

    if frame.shape[:2] == (height, width):
        return frame

    return cv2.resize(frame, (width, height))


def desktop_rectangle(width, height):
    """The four desktop corners, in TL, TR, BR, BL order."""
    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


# ============================================================
# STEP 1 - RGBW temporal screen detection
# ============================================================

def build_rgbw_screen_mask(captures):
    """
    Detect the physical projected screen from the FOUR captures.

    Core idea:
        The camera is fixed.
        The projector changes:
            RED -> GREEN -> BLUE -> WHITE

        Therefore pixels belonging to the projector screen change
        strongly in colour across the four frames, while most of the
        room/background stays comparatively stable.

    We measure the per-pixel range in CIE-Lab chroma (a*, b*) and
    brightness.
    """
    names = ["RED", "GREEN", "BLUE", "WHITE"]

    frames = [
        captures[name].astype(np.uint8)
        for name in names
    ]

    frames = [
        resize_to_reference(frame, frames[0].shape)
        for frame in frames
    ]

    lab_frames = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        for frame in frames
    ]

    # OpenCV Lab:
    #   L = lightness
    #   a = green <-> red
    #   b = blue <-> yellow
    a_stack = np.stack(
        [lab[:, :, 1] for lab in lab_frames],
        axis=0,
    )
    b_stack = np.stack(
        [lab[:, :, 2] for lab in lab_frames],
        axis=0,
    )
    l_stack = np.stack(
        [lab[:, :, 0] for lab in lab_frames],
        axis=0,
    )

    a_range = np.ptp(a_stack, axis=0)
    b_range = np.ptp(b_stack, axis=0)
    l_range = np.ptp(l_stack, axis=0)

    # Chroma variation is the primary signal.
    chroma_range = np.sqrt(
        a_range * a_range +
        b_range * b_range
    )

    change_score = chroma_range + 0.35 * l_range

    change_score = cv2.GaussianBlur(
        change_score.astype(np.float32),
        (5, 5),
        0,
    )

    # Adaptive threshold via Otsu on the normalized change map.
    #
    # A plain high percentile BREAKS DOWN when the screen covers
    # more than (100 - percentile) % of the frame: the threshold
    # then lands inside the screen's own score distribution and
    # only scattered noise pixels survive the morphological open.
    # Otsu splits "changes a lot" from "barely changes" no matter
    # how much of the frame the screen occupies.
    score_min = float(change_score.min())
    score_max = float(change_score.max())

    if score_max > score_min:
        normalized = cv2.normalize(
            change_score,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        otsu_normalized, _ = cv2.threshold(
            normalized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Map the Otsu level back into raw score units.
        otsu_threshold = score_min + (
            float(otsu_normalized) / 255.0
        ) * (score_max - score_min)
    else:
        # Completely static scene: nothing changed at all.
        otsu_threshold = score_max

    threshold = max(RGBW_MIN_CHANGE, float(otsu_threshold))

    mask = np.where(
        change_score >= threshold,
        255,
        0,
    ).astype(np.uint8)

    mask = clean_mask(mask)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (RGBW_CLOSE_KERNEL, RGBW_CLOSE_KERNEL),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return mask, change_score, threshold

    frame_area = mask.shape[0] * mask.shape[1]

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= frame_area * RGBW_MIN_COMPONENT_RATIO:
            candidates.append((area, contour))

    if not candidates:
        return mask, change_score, threshold

    candidates.sort(key=lambda item: item[0], reverse=True)

    best_contour = None
    best_score = -1.0

    for area, contour in candidates[:10]:
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        polygon = cv2.approxPolyDP(
            contour,
            RGBW_APPROX_EPSILON * perimeter,
            True,
        )

        # A 4-corner contour gets a strong preference.
        quad_bonus = 5.0 if len(polygon) == 4 else 0.0

        score = area / frame_area + quad_bonus

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        best_contour = candidates[0][1]

    final_mask = np.zeros_like(mask)

    cv2.drawContours(
        final_mask,
        [best_contour],
        -1,
        255,
        -1,
    )

    return final_mask, change_score, threshold


def clean_mask(mask):
    """Clean binary calibration masks."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_screen_corners(mask):
    """
    Find a quadrilateral screen from the final RGBW mask.

    We strongly prefer a true 4-point contour. If the contour is not
    exactly four points, we use a convex hull and try again.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    frame_area = mask.shape[0] * mask.shape[1]

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    for contour in contours[:10]:
        area = cv2.contourArea(contour)

        if area < frame_area * MIN_SCREEN_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)

        for epsilon_ratio in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04):
            polygon = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True,
            )

            if len(polygon) == 4:
                points = polygon.reshape(4, 2).astype(np.float32)

                h, w = mask.shape[:2]

                if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= w):
                    continue
                if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= h):
                    continue

                return order_corners(points)

        # Try convex hull if the raw contour was fragmented.
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)

        for epsilon_ratio in (0.005, 0.01, 0.02, 0.03, 0.05):
            polygon = cv2.approxPolyDP(
                hull,
                epsilon_ratio * hull_perimeter,
                True,
            )

            if len(polygon) == 4:
                points = polygon.reshape(4, 2).astype(np.float32)

                h, w = mask.shape[:2]

                if (
                    np.all(points[:, 0] >= 0)
                    and np.all(points[:, 0] < w)
                    and np.all(points[:, 1] >= 0)
                    and np.all(points[:, 1] < h)
                ):
                    return order_corners(points)

    return None


def order_corners(points):
    """Return TL, TR, BR, BL in that order."""
    points = np.asarray(points, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = points[:, 0] - points[:, 1]

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmax(differences)]
    ordered[3] = points[np.argmin(differences)]

    return ordered


def rough_homography(corners, screen_width, screen_height):
    """
    Camera -> desktop homography from the four RGBW corners.
    Used only to predict where the grid points will appear.
    """
    if corners is None:
        return None

    rect = desktop_rectangle(screen_width, screen_height)

    matrix, _ = cv2.findHomography(
        np.asarray(corners, dtype=np.float32),
        rect,
        0,
    )

    if matrix is None:
        return None

    return matrix.astype(np.float32)


def homography_from_corners(corners, screen_width, screen_height):
    """Fallback homography: map the four screen corners to the
    four desktop corners."""
    return rough_homography(corners, screen_width, screen_height)


# ============================================================
# STEP 2 - point-grid homography calibration
# ============================================================

def build_grid_points(width, height, cols=GRID_COLS, rows=GRID_ROWS,
                      inset=GRID_INSET):
    """
    Reference points at known DESKTOP coordinates.
    Row-major, starting top-left. Kept GRID_INSET away from the
    screen edge so they are never clipped by the projector.
    """
    points = []

    for row in range(rows):
        for col in range(cols):
            fx = inset + (1.0 - 2.0 * inset) * (
                col / float(cols - 1)
            )
            fy = inset + (1.0 - 2.0 * inset) * (
                row / float(rows - 1)
            )

            points.append(
                (fx * (width - 1), fy * (height - 1))
            )

    return points


def show_projector_point(window_name, x, y, screen_width, screen_height):
    """
    Draw ONLY the reference point on black. Nothing else is
    drawn on the projector - any other bright graphic could be
    measured as a second point.
    """
    image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

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


def _predict_roi(homography, x, y, frame_width, frame_height, half):
    """
    Where (approximately) does desktop point (x, y) appear in the
    camera frame? Returns an inclusive-exclusive box, or None to
    search the whole frame.
    """
    point = np.array([[[float(x), float(y)]]], dtype=np.float32)

    projected = cv2.perspectiveTransform(point, homography)[0, 0]

    px = float(projected[0])
    py = float(projected[1])

    # Wildly wrong prediction -> fall back to a full-frame search.
    if px < -half or py < -half:
        return None

    if px > frame_width + half or py > frame_height + half:
        return None

    x0 = max(0, int(round(px - half)))
    y0 = max(0, int(round(py - half)))
    x1 = min(frame_width, int(round(px + half)))
    y1 = min(frame_height, int(round(py + half)))

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)


def _weighted_centroid(roi_gray, binary, contour):
    """Intensity-weighted center of a blob (sub-pixel)."""
    x, y, width, height = cv2.boundingRect(contour)

    if width <= 0 or height <= 0:
        return None

    sub_gray = roi_gray[y:y + height, x:x + width].astype(np.float32)

    sub_mask = np.zeros((height, width), dtype=np.uint8)
    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)

    cv2.drawContours(sub_mask, [shifted], -1, 255, -1)

    weights = sub_gray * (sub_mask > 0)

    total = float(weights.sum())

    if total <= 0.0:
        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            return None

        return (
            x + moments["m10"] / moments["m00"],
            y + moments["m01"] / moments["m00"],
        )

    ys, xs = np.nonzero(weights)

    if ys.size == 0:
        return None

    values = weights[ys, xs]

    cx = float((xs * values).sum() / values.sum())
    cy = float((ys * values).sum() / values.sum())

    return (cx + x, cy + y)


def find_point_in_frame(frame, roi=None, screen_mask=None):
    """
    Locate one projected reference point.

    roi       : optional (x0, y0, x1, y1) search window
    screen_mask : optional binary mask (search only the screen)

    Returns (x, y) in full-frame camera coordinates, or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_height, frame_width = gray.shape[:2]

    if roi is None:
        x0, y0, x1, y1 = 0, 0, frame_width, frame_height
    else:
        x0, y0, x1, y1 = roi

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame_width, x1)
    y1 = min(frame_height, y1)

    if x1 <= x0 or y1 <= y0:
        return None

    roi_gray = gray[y0:y1, x0:x1]

    work = roi_gray

    if screen_mask is not None:
        region = screen_mask[y0:y1, x0:x1]

        if region.size == 0 or not np.any(region):
            return None

        work = cv2.bitwise_and(roi_gray, roi_gray, mask=region)

    # Otsu separates the dot from the (not really black) screen;
    # the absolute floor additionally rejects dim reflections and
    # ambient IR that Otsu alone would sometimes accept.
    otsu_value, _ = cv2.threshold(
        work,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    threshold_value = max(int(otsu_value), POINT_FLOOR)
    threshold_value = min(threshold_value, 255)

    binary = cv2.threshold(
        work,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    region_area = float((y1 - y0) * (x1 - x0))
    max_area = region_area * POINT_MAX_AREA_RATIO

    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < POINT_MIN_AREA or area > max_area:
            continue

        score = area * cv2.mean(work, mask=_contour_mask(work, contour))[0]

        if score > best_score:
            best_score = score
            best = contour

    if best is None:
        return None

    centroid = _weighted_centroid(work, binary, best)

    if centroid is None:
        return None

    return (centroid[0] + x0, centroid[1] + y0)


def _contour_mask(image, contour):
    """Full-frame binary mask for one contour."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, 255, -1)

    return mask


def capture_point_grid(cap, window_name, points, screen_width,
                       screen_height, homography, screen_mask,
                       frame_size):
    """
    Project every reference point in turn and measure it.

    Returns (camera_points, desktop_points, missing_indices, last_frame).
    camera_points[i] corresponds to desktop_points[i].
    """
    frame_width, frame_height = frame_size
    half = max(48, int(0.12 * frame_width))

    camera_points = []
    desktop_points = []
    missing = []
    last_frame = None

    total = len(points)

    print("\n" + "=" * 60)
    print("MEASURING REFERENCE POINTS")
    print("=" * 60)

    for index, (dx, dy) in enumerate(points):
        print(
            f"Point {index + 1}/{total} "
            f"desktop ({dx:.0f}, {dy:.0f}) ... ",
            end="",
            flush=True,
        )

        show_projector_point(
            window_name,
            dx,
            dy,
            screen_width,
            screen_height,
        )

        time.sleep(POINT_SETTLE_TIME)

        roi = None

        if homography is not None:
            roi = _predict_roi(
                homography,
                dx,
                dy,
                frame_width,
                frame_height,
                half,
            )

        found = None

        for attempt in range(POINT_ATTEMPTS):
            frame = capture_average_frame(
                cap,
                frame_count=POINT_AVERAGE_FRAMES,
                quiet=True,
            )

            last_frame = frame

            # Search plans, most specific first.
            plans = []

            if roi is not None and screen_mask is not None:
                plans.append((roi, screen_mask))

            if screen_mask is not None:
                plans.append((None, screen_mask))

            plans.append((None, None))

            for plan_roi, plan_mask in plans:
                found = find_point_in_frame(frame, plan_roi, plan_mask)

                if found is not None:
                    break

            if found is not None:
                break

            time.sleep(0.1)

        if found is not None:
            camera_points.append(found)
            desktop_points.append((dx, dy))
            print(f"camera ({found[0]:.1f}, {found[1]:.1f})  OK")
        else:
            missing.append(index)
            print("NOT FOUND")

    print("=" * 60)

    camera_array = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)
    desktop_array = np.asarray(desktop_points, dtype=np.float32).reshape(-1, 2)

    return camera_array, desktop_array, missing, last_frame


def fit_homography(camera_points, desktop_points):
    """
    Fit camera -> desktop with RANSAC.

    Returns (H, errors, inliers):
        H       : 3x3 float32 homography or None
        errors  : per-point reprojection error in DESKTOP pixels
        inliers : boolean mask of points used by RANSAC
    """
    camera_points = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)
    desktop_points = np.asarray(desktop_points, dtype=np.float32).reshape(-1, 2)

    if camera_points.shape[0] < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, None, None

    matrix, inlier_mask = cv2.findHomography(
        camera_points,
        desktop_points,
        cv2.RANSAC,
        RANSAC_REPROJECTION_PX,
    )

    if matrix is None:
        return None, None, None

    matrix = matrix.astype(np.float32)

    projected = cv2.perspectiveTransform(
        camera_points.reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - desktop_points,
        axis=1,
    )

    if inlier_mask is not None:
        inliers = inlier_mask.ravel().astype(bool)
    else:
        inliers = np.ones(errors.shape[0], dtype=bool)

    if not np.any(inliers):
        inliers[:] = True

    return matrix, errors, inliers


def camera_corners_from_homography(matrix, screen_width, screen_height):
    """
    Inverse-map the four desktop corners into camera space.
    These become the screen polygon used for detection masking.
    """
    inverse = np.linalg.inv(matrix.astype(np.float64)).astype(np.float32)

    rect = desktop_rectangle(screen_width, screen_height)

    corners = cv2.perspectiveTransform(
        rect.reshape(-1, 1, 2),
        inverse,
    ).reshape(-1, 2)

    return np.asarray(corners, dtype=np.float32)


def report_reprojection(errors, inliers, screen_width, screen_height,
                        camera_points=None):
    """Print a readable accuracy report. Returns the mean error."""
    if errors is None:
        return None

    used = errors[inliers]

    mean_error = float(used.mean())
    max_error = float(used.max())

    diagonal = float(np.hypot(screen_width, screen_height))
    ratio = mean_error / diagonal if diagonal else 0.0

    print("\nCalibration accuracy (desktop pixels):")
    print(f"  points measured : {int(inliers.sum())} / {int(inliers.size)}")
    print(f"  mean error      : {mean_error:.2f} px")
    print(f"  max  error      : {max_error:.2f} px")
    print(f"  mean / diagonal : {ratio * 100:.3f} %")

    if camera_points is not None:
        print("\nPer-point residuals:")

        for index, error in enumerate(errors):
            marker = "" if inliers[index] else "   (rejected as outlier)"
            print(f"  point {index + 1:2d}: {error:7.2f} px{marker}")

    if ratio > WARN_MEAN_ERROR_RATIO:
        print(
            "\nWARNING: reprojection error is high.\n"
            "  - make sure nothing moved between captures\n"
            "  - recalibrate in a darker room\n"
            "  - check that the projector really fills the screen"
        )

    return mean_error


# ============================================================
# Saving
# ============================================================

def save_calibration(camera_corners, homography, camera_points,
                     desktop_points, frame_width, frame_height,
                     screen_width, screen_height):
    corners = np.asarray(camera_corners, dtype=np.float32)

    np.save(CALIBRATION_FILE, corners)

    np.savez(
        HOMOGRAPHY_FILE,
        homography=np.asarray(homography, dtype=np.float32),
        camera_corners=corners,
        camera_points=np.asarray(
            camera_points, dtype=np.float32
        ).reshape(-1, 2),
        desktop_points=np.asarray(
            desktop_points, dtype=np.float32
        ).reshape(-1, 2),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        screen_width=int(screen_width),
        screen_height=int(screen_height),
    )

    print("\nSaved:")
    print(f"  {CALIBRATION_FILE}")
    print(f"  {HOMOGRAPHY_FILE}")


def draw_result(frame, corners, camera_points=None):
    """Overlay the calibrated screen polygon (and measured points)."""
    output = frame.copy()

    labels = ["TL", "TR", "BR", "BL"]
    polygon = corners.astype(np.int32)

    cv2.polylines(output, [polygon], True, (0, 255, 0), 3)

    for label, point in zip(labels, corners):
        x, y = map(int, point)

        cv2.circle(output, (x, y), 9, (0, 255, 255), -1)
        cv2.putText(
            output,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    if camera_points is not None:
        for point in np.asarray(camera_points).reshape(-1, 2):
            cv2.circle(
                output,
                (int(round(point[0])), int(round(point[1]))),
                5,
                (255, 0, 255),
                -1,
            )

    return output


# ============================================================
# MAIN
# ============================================================

def _read_mode():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        return mode if mode in ("visible", "ir", "auto") else "visible"

    try:
        answer = input(
            "\nCalibration camera mode [visible/ir/auto] "
            "(default visible): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "visible"

    if answer in ("visible", "ir", "auto"):
        return answer

    return "visible"


def main():
    cap = None

    try:
        # Bail out with instructions (instead of letting the Qt
        # backend abort the process) when there is no display.
        require_display()

        print("\n" + "=" * 70)
        print("SMARTIR SCREEN CALIBRATION")
        print("STEP 1: RGBW screen detection")
        print("STEP 2: point-grid homography (real correspondences)")
        print("=" * 70)

        mode = _read_mode()

        cap, resolved_mode = open_camera(mode)

        projector_window = create_projector_window()

        # Learn the real capture size (it may differ from the
        # requested one on some cameras).
        ret, probe = cap.read()

        if ret and probe is not None and probe.size:
            frame_height, frame_width = probe.shape[:2]
        else:
            frame_height, frame_width = CAMERA_HEIGHT, CAMERA_WIDTH

        print(f"\nCapture size: {frame_width} x {frame_height}")

        # Desktop/projector size. Everything the projector draws is
        # made at this exact aspect ratio so fullscreen display maps
        # 1:1 onto the screen.
        screen_width, screen_height, screen_detected = get_screen_size()

        if not screen_detected:
            print(
                "\nWARNING: desktop size could not be detected; "
                f"assuming {screen_width} x {screen_height}.\n"
                "         Make sure this matches the resolution the "
                "projector actually displays."
            )

        # ----------------------------------------------------
        # STEP 1 - RGBW screen detection
        # ----------------------------------------------------
        corners = None
        screen_mask = None

        if resolved_mode == "ir":
            print(
                "\nIR mode: skipping the RGBW colour phase.\n"
                "(An IR-sensitive camera cannot see projector "
                "colours. The point sweep below still works if "
                "the reference points are visible to the camera.)"
            )
        else:
            captures = {}

            for name, color in CALIBRATION_COLORS.items():
                show_color(
                    projector_window,
                    color,
                    width=screen_width,
                    height=screen_height,
                )

                if not wait_for_confirmation(name):
                    print("Calibration cancelled.")
                    return

                captures[name] = capture_average_frame(cap)
                frame_height, frame_width = captures[name].shape[:2]
                print(f"{name} capture complete.")

            print("\nBuilding RGBW screen-change mask...")
            print(
                "Detecting pixels whose colour changes across "
                "RED -> GREEN -> BLUE -> WHITE."
            )

            screen_mask, change_score, change_threshold = (
                build_rgbw_screen_mask(captures)
            )

            coverage = 100.0 * float(np.count_nonzero(screen_mask)) / (
                screen_mask.shape[0] * screen_mask.shape[1]
            )

            print(f"Adaptive Lab-change threshold: {change_threshold:.2f}")
            print(f"Detected screen area: {coverage:.1f} % of the frame")

            corners = find_screen_corners(screen_mask)

            if corners is None:
                print(
                    "\nWARNING: no clean screen quadrilateral found.\n"
                    "         The point sweep will search the whole frame."
                )
            else:
                print("\nApproximate screen corners (RGBW):")

                names = ["TOP LEFT", "TOP RIGHT", "BOTTOM RIGHT",
                         "BOTTOM LEFT"]

                for name, point in zip(names, corners):
                    print(f"  {name:<14}: ({point[0]:.1f}, {point[1]:.1f})")

        # ----------------------------------------------------
        # STEP 2 - point-grid homography
        # ----------------------------------------------------
        points = build_grid_points(screen_width, screen_height)

        print(
            f"\nReference grid: {GRID_COLS} x {GRID_ROWS} = "
            f"{len(points)} points on a {screen_width} x {screen_height} "
            "desktop"
        )

        if resolved_mode == "ir":
            print(
                "NOTE: if this camera has an IR-pass filter it cannot "
                "see the projector, and the point sweep will fail."
            )

        if not confirm(
            "STEP 2: POINT-GRID SWEEP",
            (
                f"The projector will display {len(points)} points, one "
                "at a time,",
                "and each one is measured automatically.",
                "",
                "Do NOT move the camera, projector or screen.",
                "Press ENTER in this terminal to start.",
                "Type Q and press ENTER to quit.",
            ),
        ):
            print("Calibration cancelled.")
            return

        h0 = rough_homography(corners, screen_width, screen_height)

        camera_points, desktop_points, missing, last_frame = (
            capture_point_grid(
                cap,
                projector_window,
                points,
                screen_width,
                screen_height,
                h0,
                screen_mask,
                (frame_width, frame_height),
            )
        )

        if last_frame is not None:
            frame_height, frame_width = last_frame.shape[:2]

        if missing:
            print(
                f"\n{len(missing)} point(s) could not be measured: "
                + ", ".join(str(i + 1) for i in missing)
            )

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------
        homography, errors, inliers = fit_homography(
            camera_points,
            desktop_points,
        )

        used_camera = camera_points
        used_desktop = desktop_points

        if homography is None:
            if corners is None:
                raise RuntimeError(
                    "Calibration failed: not enough reference points "
                    "were measured and no RGBW screen was found.\n"
                    "Check that the projector is visible to the camera "
                    "and try again in a darker room."
                )

            print(
                "\nWARNING: not enough valid reference points.\n"
                "         Falling back to the 4-corner RGBW estimate."
            )

            homography = homography_from_corners(
                corners,
                screen_width,
                screen_height,
            )

            if homography is None:
                raise RuntimeError(
                    "Calibration failed: could not build a homography."
                )

            used_camera = np.zeros((0, 2), dtype=np.float32)
            used_desktop = np.zeros((0, 2), dtype=np.float32)
            errors = None
            inliers = None
        else:
            report_reprojection(
                errors,
                inliers,
                screen_width,
                screen_height,
                camera_points=used_camera,
            )

            # Keep only the points RANSAC accepted.
            if inliers is not None and not np.all(inliers):
                used_camera = used_camera[inliers]
                used_desktop = used_desktop[inliers]

        camera_corners = camera_corners_from_homography(
            homography,
            screen_width,
            screen_height,
        )

        save_calibration(
            camera_corners,
            homography,
            used_camera,
            used_desktop,
            frame_width,
            frame_height,
            screen_width,
            screen_height,
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------
        if last_frame is None:
            ret, last_frame = cap.read()

            if not ret or last_frame is None:
                last_frame = np.zeros(
                    (frame_height, frame_width, 3),
                    dtype=np.uint8,
                )

        result = draw_result(
            last_frame,
            camera_corners,
            camera_points=used_camera if used_camera.size else None,
        )

        cv2.imshow("FINAL CALIBRATION", result)

        print("\nCalibration complete.")
        print("Press any key in an OpenCV window to finish.")

        cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")

    except Exception as error:
        print("\nCALIBRATION ERROR:")
        print(error)
        sys.exit(1)

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
