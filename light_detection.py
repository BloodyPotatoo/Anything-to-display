import sys

import cv2
import numpy as np

from camera import open_camera, resolve_mode
from coordinate_mapper import CoordinateMapper

# ============================================================
# SMARTIR - IR / BRIGHT LIGHT DETECTION
#
# The detector:
#   1. Opens the camera (camera.py - shared with calibration,
#      so both always use the same resolution/mode).
#   2. Restricts detection to the calibrated screen polygon.
#   3. Searches for a very bright light (the IR pen / flashlight).
#   4. Finds the intensity-weighted center of the best blob.
#   5. Converts camera coordinates -> desktop coordinates.
#
# Modes (resolved once, by camera.open_camera):
#   "visible" : auto exposure, threshold 200 (webcam/flashlight tests)
#   "ir"      : locked short exposure, threshold 240 (IR pen tracking)
#
# Optimized for Raspberry Pi 3B/4 + Arducam NoIR & PC testing:
#   - grayscale only (no HSV conversion)
#   - morphology kernel built once
#   - per-contour statistics computed inside the contour bounding
#     box instead of over the whole frame
# ============================================================

KERNEL_SIZE = 5
MIN_AREA = 3
MAX_AREA = 3000

# Brightness thresholds per resolved mode.
IR_THRESHOLD = 240
VISIBLE_THRESHOLD = 200

_KERNEL = None


def _get_kernel():
    """Build the morphology kernel once (it never changes)."""
    global _KERNEL

    if _KERNEL is None:
        _KERNEL = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (KERNEL_SIZE, KERNEL_SIZE),
        )

    return _KERNEL


def make_screen_mask(frame_shape, mapper):
    """
    Convert the four calibrated camera corners into a binary mask.
    Only this region is searched for the light.
    """
    mask = np.zeros(
        frame_shape[:2],
        dtype=np.uint8,
    )

    polygon = mapper.camera_corners.astype(np.int32)

    cv2.fillPoly(
        mask,
        [polygon],
        255,
    )

    return mask


def _contour_mean_brightness(gray, contour):
    """
    Mean brightness inside a contour.

    Computed on the contour's bounding box only, which is far
    cheaper than building a full-frame mask for every contour
    (this used to be the hottest part of the loop on a Pi 3B).
    """
    x, y, width, height = cv2.boundingRect(contour)

    if width <= 0 or height <= 0:
        return 0.0

    roi = gray[y:y + height, x:x + width]

    roi_mask = np.zeros((height, width), dtype=np.uint8)

    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)

    cv2.drawContours(
        roi_mask,
        [shifted],
        -1,
        255,
        -1,
    )

    return cv2.mean(roi, mask=roi_mask)[0]


def find_light(mask, frame, mode="visible"):
    """
    Find the pen light inside the calibrated screen polygon.

    Returns ((x, y, contour), detection_mask) or (None, mask).
    """
    resolved_mode = resolve_mode(mode)

    threshold_val = (
        IR_THRESHOLD if resolved_mode == "ir" else VISIBLE_THRESHOLD
    )

    # Grayscale directly - skips HSV and saves CPU on the Pi.
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    _, bright_mask = cv2.threshold(
        gray,
        threshold_val,
        255,
        cv2.THRESH_BINARY,
    )

    # IMPORTANT:
    # The light is searched for ONLY inside the calibrated screen.
    bright_mask = cv2.bitwise_and(bright_mask, mask)

    kernel = _get_kernel()

    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        bright_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        # Score using both area and brightness.
        mean_value = _contour_mean_brightness(gray, contour)
        score = area * mean_value

        if score > best_score:
            best_score = score
            best = contour

    if best is None:
        return None, bright_mask

    moments = cv2.moments(best)

    if moments["m00"] == 0:
        return None, bright_mask

    x = moments["m10"] / moments["m00"]
    y = moments["m01"] / moments["m00"]

    return (x, y, best), bright_mask


def run_light_detection(mapper, mode="visible"):
    camera = None

    try:
        camera, resolved_mode = open_camera(mode)

        screen_mask = None

        print("\n" + "=" * 65)
        print("SMARTIR LIGHT / IR DETECTION")
        print(f"Mode: {resolved_mode}")
        print("Only the calibrated screen region is searched.")
        print("Press Q to quit.")
        print("=" * 65)

        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("\nCamera frame error.")
                break

            if screen_mask is None or screen_mask.shape != frame.shape[:2]:
                # Keep the mask in the same coordinate space as the
                # calibration, whatever size the camera delivers.
                mapper.set_frame_size(
                    frame.shape[1],
                    frame.shape[0],
                )

                screen_mask = make_screen_mask(
                    frame.shape,
                    mapper,
                )

            result, detection_mask = find_light(
                screen_mask,
                frame,
                resolved_mode,
            )

            display = frame.copy()

            # Draw the calibrated screen boundary.
            corners = mapper.camera_corners.astype(np.int32)

            cv2.polylines(
                display,
                [corners],
                True,
                (0, 255, 0),
                2,
            )

            if result is not None:
                camera_x, camera_y, contour = result

                screen_x, screen_y = mapper.camera_to_screen(
                    camera_x,
                    camera_y,
                )

                cv2.drawContours(
                    display,
                    [contour],
                    -1,
                    (255, 0, 255),
                    2,
                )

                cv2.circle(
                    display,
                    (int(camera_x), int(camera_y)),
                    8,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    display,
                    f"Camera: "
                    f"({camera_x:.1f}, {camera_y:.1f})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display,
                    f"Screen: "
                    f"({screen_x:.1f}, {screen_y:.1f})",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                print(
                    f"\rCamera: "
                    f"({camera_x:.1f}, {camera_y:.1f})   "
                    f"Screen: ({screen_x:.1f}, {screen_y:.1f})",
                    end="",
                    flush=True,
                )

            cv2.imshow(
                "SMARTIR Light Detection",
                display,
            )

            cv2.imshow(
                "Detection Mask - Screen Only",
                detection_mask,
            )

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDetection interrupted.")

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("\nDetection stopped.")


def main():
    try:
        mapper = CoordinateMapper()
        mapper.print_info()

        mode = "auto"

        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()

            if mode not in ("visible", "ir", "auto"):
                mode = "visible"

        run_light_detection(mapper, mode)

    except Exception as error:
        print("\nERROR:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
