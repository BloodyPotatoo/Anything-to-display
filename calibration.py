import os
import platform
import sys
import time
import cv2
import numpy as np

# ============================================================
# SMARTIR - RGBW SCREEN CALIBRATION
#
# Logic:
#   RED capture  -> red screen mask
#   GREEN capture -> green screen mask
#   BLUE capture -> blue screen mask
#   WHITE capture -> white/bright screen mask
#
# The final screen region is the INTERSECTION:
#
#   RED ∩ GREEN ∩ BLUE ∩ WHITE
#
# This prevents unrelated objects/background areas from becoming
# part of the calibrated screen.
# ============================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
MAX_CAMERA_INDEX = 10

AVERAGE_FRAMES = 20
FRAME_DELAY = 0.03

# Basic color-detection floors used by the adaptive detector.
MIN_COLOR_SATURATION = 10
MIN_COLOR_VALUE = 25

# ============================================================
# CALIBRATION TOLERANCE SETTINGS
#
# IMPORTANT:
# We do NOT require the RGB masks to line up pixel-for-pixel.
# Projector/camera auto-exposure, auto-white-balance, noise and
# tiny camera movements can shift the detected boundaries.
#
# Each RGB mask is therefore:
#   1. detected using broad VALUE thresholds
#   2. cleaned
#   3. expanded by INTERSECTION_TOLERANCE_PIXELS
#   4. intersected with the other masks
#
# This gives a REAL intersection with spatial tolerance.
# ============================================================

# Minimum channel difference required to say that a pixel is
# meaningfully colored. Keep this low for webcams.
COLOR_DIFFERENCE_THRESHOLD = 3

# Minimum dominance of the expected RGB channel.
COLOR_DOMINANCE_THRESHOLD = 3

# White screen threshold.
MIN_WHITE_VALUE = 35
MAX_WHITE_SATURATION = 220

# Instead of requiring exact overlap, allow each mask to be
# expanded by this many pixels before intersection.
#
# 15-30 is a good range for a 1280x720 webcam image.
INTERSECTION_TOLERANCE_PIXELS = 100

# Final intersection threshold after confidence calculation.
INTERSECTION_THRESHOLD = 25

# The screen should occupy at least this much of the camera frame.
MIN_SCREEN_AREA_RATIO = 0.015

MORPH_KERNEL_SIZE = 7
APPROX_EPSILON_RATIO = 0.02

# OpenCV uses BGR here.
CALIBRATION_COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "WHITE": (255, 255, 255),
}


def get_camera_backends():
    system = platform.system()

    if system == "Windows":
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    if system == "Linux":
        return [cv2.CAP_V4L2, cv2.CAP_ANY]

    return [cv2.CAP_ANY]


def backend_name(backend):
    names = {
        cv2.CAP_ANY: "AUTO",
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_DSHOW: "DirectShow",
        cv2.CAP_AVFOUNDATION: "AVFoundation",
    }
    return names.get(backend, str(backend))


def try_camera(index, backend):
    print(f"Trying camera index {index} using {backend_name(backend)}...")

    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception:
        return None

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    try:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
    except Exception:
        pass

    time.sleep(0.2)

    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return cap
        time.sleep(0.05)

    cap.release()
    return None


def open_camera():
    print("\n" + "=" * 60)
    print("AUTOMATIC CAMERA DETECTION")
    print("=" * 60)

    # Check if a custom IP camera URL or specific index is provided via environment variable
    custom_cam = os.environ.get("SMARTIR_CAM")
    if custom_cam:
        print(f"Using custom camera source from SMARTIR_CAM: {custom_cam}")
        # If it's a digit, convert to int index, otherwise treat as network stream URL
        source = int(custom_cam) if custom_cam.isdigit() else custom_cam
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            return cap
        else:
            print("Warning: Could not open custom camera source. Falling back to auto-scan.")

    for index in range(MAX_CAMERA_INDEX):
        for backend in get_camera_backends():
            cap = try_camera(index, backend)
            if cap is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                print("\nCamera found!")
                print("Index:", index)
                print("Backend:", backend_name(backend))
                print(f"Resolution: {width} x {height}")
                print(f"FPS: {fps:.1f}")
                print("=" * 60)
                return cap

    raise RuntimeError("No usable camera was found.")


def create_projector_window():
    name = "PROJECTOR"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )
    return name


def show_color(window_name, color):
    image = np.zeros(
        (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8
    )
    image[:] = color
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


def wait_for_confirmation(color_name):
    print("\n" + "=" * 60)
    print(f"SHOW {color_name}")
    print("Make sure the projector covers the intended screen area.")
    print("Press ENTER in this terminal to capture.")
    print("Type Q and press ENTER to quit.")
    print("=" * 60)

    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer != "q"


def capture_average_frame(cap):
    frames = []

    print(f"Capturing {AVERAGE_FRAMES} frames...")

    for i in range(AVERAGE_FRAMES):
        ret, frame = cap.read()

        if ret and frame is not None and frame.size:
            frames.append(frame.astype(np.float32))

        cv2.waitKey(1)
        time.sleep(FRAME_DELAY)
        print(
            f"\rFrame {i + 1}/{AVERAGE_FRAMES}",
            end="",
            flush=True
        )

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


# ============================================================
# RGBW TEMPORAL SCREEN DETECTION SETTINGS
#
# The camera stays fixed while the projector changes:
# RED -> GREEN -> BLUE -> WHITE.
# We detect the region whose Lab colour changes strongly.
# ============================================================

# Higher percentile = stricter/smaller candidate region.
# Lower percentile = more permissive/larger candidate region.
RGBW_CHANGE_PERCENTILE = 92

# Minimum Lab colour-change score.
RGBW_MIN_CHANGE = 18

# Morphological closing kernel used to join gaps in the screen.
RGBW_CLOSE_KERNEL = 31

# Ignore very small disconnected components.
RGBW_MIN_COMPONENT_RATIO = 0.015

# Contour approximation epsilon.
RGBW_APPROX_EPSILON = 0.02

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
    brightness. This is much more reliable than asking four separate
    masks to overlap.
    """
    names = ["RED", "GREEN", "BLUE", "WHITE"]

    frames = [
        captures[name].astype(np.uint8)
        for name in names
    ]

    # All frames should already have the same size, but explicitly
    # resize just in case a backend changes resolution.
    reference_shape = frames[0].shape[:2]
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
    #
    # RGB projector changes produce a large range in a/b.
    a_stack = np.stack(
        [lab[:, :, 1] for lab in lab_frames],
        axis=0
    )

    b_stack = np.stack(
        [lab[:, :, 2] for lab in lab_frames],
        axis=0
    )

    l_stack = np.stack(
        [lab[:, :, 0] for lab in lab_frames],
        axis=0
    )

    a_range = np.ptp(a_stack, axis=0)
    b_range = np.ptp(b_stack, axis=0)
    l_range = np.ptp(l_stack, axis=0)

    # Chroma variation is the primary signal.
    chroma_range = np.sqrt(
        a_range * a_range +
        b_range * b_range
    )

    # Combine chroma and brightness change.
    change_score = (
        chroma_range +
        0.35 * l_range
    )

    # Ignore tiny numerical/camera noise.
    change_score = cv2.GaussianBlur(
        change_score.astype(np.float32),
        (5, 5),
        0
    )

    # Adaptive threshold:
    # the screen should be among the strongest-changing regions.
    #
    # We use a high percentile rather than a fixed RGB value.
    percentile_threshold = np.percentile(
        change_score,
        RGBW_CHANGE_PERCENTILE
    )

    threshold = max(
        RGBW_MIN_CHANGE,
        float(percentile_threshold)
    )

    mask = np.where(
        change_score >= threshold,
        255,
        0
    ).astype(np.uint8)

    # Clean isolated noise.
    mask = clean_mask(mask)

    # Close gaps across the projected screen.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            RGBW_CLOSE_KERNEL,
            RGBW_CLOSE_KERNEL
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # Fill small holes inside the screen.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask, change_score, threshold

    # Keep only sufficiently large candidates.
    frame_area = mask.shape[0] * mask.shape[1]

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= frame_area * RGBW_MIN_COMPONENT_RATIO:
            candidates.append(
                (area, contour)
            )

    if not candidates:
        return mask, change_score, threshold

    # Sort by area, but prefer a component that looks like a screen:
    # large enough and reasonably quadrilateral.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_contour = None
    best_score = -1

    for area, contour in candidates[:10]:
        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        polygon = cv2.approxPolyDP(
            contour,
            RGBW_APPROX_EPSILON * perimeter,
            True
        )

        # A 4-corner contour gets a strong preference.
        quad_bonus = 5.0 if len(polygon) == 4 else 0.0

        score = (
            area / frame_area +
            quad_bonus
        )

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
        -1
    )

    return final_mask, change_score, threshold


def clean_mask(mask):
    """Clean binary calibration masks."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def find_screen_corners(mask):
    """
    Find a quadrilateral screen from the final RGBW mask.

    We strongly prefer a true 4-point contour. If the contour is not
    exactly four points, we use a convex hull and try again. We do NOT
    silently use a giant minimum-area rectangle, because that was the
    reason the previous calibration produced a terrible rectangle
    around the room.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    frame_area = mask.shape[0] * mask.shape[1]

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours[:10]:
        area = cv2.contourArea(contour)

        if area < frame_area * MIN_SCREEN_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        for epsilon_ratio in [
            0.005,
            0.01,
            0.015,
            0.02,
            0.03,
            0.04
        ]:
            polygon = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True
            )

            if len(polygon) == 4:
                points = polygon.reshape(
                    4,
                    2
                ).astype(np.float32)

                # Reject a calibration rectangle whose corners are
                # outside the actual camera frame.
                h, w = mask.shape[:2]

                if np.any(points[:, 0] < 0):
                    continue
                if np.any(points[:, 0] >= w):
                    continue
                if np.any(points[:, 1] < 0):
                    continue
                if np.any(points[:, 1] >= h):
                    continue

                return order_corners(points)

        # Try convex hull if the raw contour was fragmented.
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(
            hull,
            True
        )

        for epsilon_ratio in [
            0.005,
            0.01,
            0.02,
            0.03,
            0.05
        ]:
            polygon = cv2.approxPolyDP(
                hull,
                epsilon_ratio * hull_perimeter,
                True
            )

            if len(polygon) == 4:
                points = polygon.reshape(
                    4,
                    2
                ).astype(np.float32)

                h, w = mask.shape[:2]

                if (
                    np.all(points[:, 0] >= 0) and
                    np.all(points[:, 0] < w) and
                    np.all(points[:, 1] >= 0) and
                    np.all(points[:, 1] < h)
                ):
                    return order_corners(points)

    return None


def order_corners(points):
    """Return TL, TR, BR, BL in that order."""
    points = np.asarray(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    sums = points.sum(axis=1)
    differences = points[:, 0] - points[:, 1]

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmax(differences)]
    ordered[3] = points[np.argmin(differences)]

    return ordered


def draw_result(frame, corners):
    output = frame.copy()

    labels = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    polygon = corners.astype(np.int32)

    cv2.polylines(
        output,
        [polygon],
        True,
        (0, 255, 0),
        3
    )

    for label, point in zip(labels, corners):
        x, y = map(int, point)

        cv2.circle(
            output,
            (x, y),
            9,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            output,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    return output




def save_calibration(corners):
    np.save("screen_corners.npy", corners)

    print("\nSaved:")
    print("  screen_corners.npy")


def main():
    cap = None

    try:
        print("\n" + "=" * 70)
        print("SMARTIR RGBW SCREEN CALIBRATION")
        print("RED ∩ GREEN ∩ BLUE ∩ WHITE")
        print("=" * 70)

        cap = open_camera()

        projector_window = create_projector_window()

        cv2.namedWindow(
            "Camera Preview",
            cv2.WINDOW_NORMAL
        )

        captures = {}

        for name, color in CALIBRATION_COLORS.items():
            show_color(projector_window, color)

            if not wait_for_confirmation(name):
                print("Calibration cancelled.")
                return

            captures[name] = capture_average_frame(cap)

            cv2.imshow(
                "Camera Preview",
                captures[name]
            )
            cv2.waitKey(300)

            print(f"{name} capture complete.")

        print("\nBuilding RGBW screen-change mask...")
        print(
            "Detecting pixels whose colour changes across "
            "RED -> GREEN -> BLUE -> WHITE."
        )

        intersection, change_score, change_threshold = (
            build_rgbw_screen_mask(captures)
        )

        print(
            f"Adaptive Lab-change threshold: "
            f"{change_threshold:.2f}"
        )

        cv2.imshow(
            "RGBW CHANGE MASK",
            intersection
        )

        # Normalize the change score for visual debugging.
        debug_score = cv2.normalize(
            change_score,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        cv2.imshow(
            "RGBW CHANGE SCORE",
            debug_score
        )

        cv2.imshow(
            "FINAL SCREEN INTERSECTION",
            intersection
        )
        cv2.waitKey(500)

        corners = find_screen_corners(intersection)

        if corners is None:
            raise RuntimeError(
                "No screen was found in the RGBW intersection."
            )

        print("\nDetected corners:")
        names = ["TOP LEFT", "TOP RIGHT", "BOTTOM RIGHT", "BOTTOM LEFT"]

        for name, point in zip(names, corners):
            print(
                f"{name:<14}: "
                f"({point[0]:.1f}, {point[1]:.1f})"
            )

        result = draw_result(
            captures["WHITE"],
            corners
        )

        cv2.imshow(
            "FINAL CALIBRATION",
            result
        )

        save_calibration(corners)

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
