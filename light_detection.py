import cv2
import numpy as np
import os
import sys
import platform
import time

from coordinate_mapper import CoordinateMapper

# ============================================================
# SMARTIR - IR / BRIGHT LIGHT DETECTION
#
# The detector:
#   1. Opens the camera.
#   2. Restricts detection to the calibrated screen polygon.
#   3. Searches for very bright light.
#   4. Finds the center of the brightest suitable blob.
#   5. Converts camera coordinates -> desktop coordinates.
#
# With a normal webcam this detects bright visible/IR light.
# With an IR-sensitive camera it can be used for the actual IR pen.
# ============================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
MAX_CAMERA_INDEX = 10

BRIGHTNESS_THRESHOLD = 200
KERNEL_SIZE = 5

MIN_AREA = 3
MAX_AREA = 3000

# Ignore points too close to the screen boundary.
SCREEN_MARGIN = 2


def get_backends():
    system = platform.system()

    if system == "Windows":
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    if system == "Linux":
        return [cv2.CAP_V4L2, cv2.CAP_ANY]

    return [cv2.CAP_ANY]


def open_camera():
    print("\nSearching for detection camera...")

    for index in range(MAX_CAMERA_INDEX):
        for backend in get_backends():

            try:
                camera = cv2.VideoCapture(
                    index,
                    backend
                )
            except Exception:
                continue

            if not camera.isOpened():
                camera.release()
                continue

            camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                CAMERA_WIDTH
            )

            camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                CAMERA_HEIGHT
            )

            camera.set(
                cv2.CAP_PROP_FPS,
                CAMERA_FPS
            )

            try:
                camera.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG")
                )
            except Exception:
                pass

            time.sleep(0.15)

            ret, frame = camera.read()

            if ret and frame is not None and frame.size:
                print(
                    f"Camera selected: index {index}"
                )

                return camera

            camera.release()

    raise RuntimeError(
        "Could not open a usable camera."
    )


def make_screen_mask(frame_shape, mapper):
    """
    Convert the four calibrated camera corners into a binary mask.
    Only this region is searched for the light.
    """
    mask = np.zeros(
        frame_shape[:2],
        dtype=np.uint8
    )

    polygon = mapper.camera_corners.astype(
        np.int32
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    return mask


def find_light(mask, frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    value = hsv[:, :, 2]

    _, gray_mask = cv2.threshold(
        gray,
        BRIGHTNESS_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    _, value_mask = cv2.threshold(
        value,
        BRIGHTNESS_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    bright_mask = cv2.bitwise_and(
        gray_mask,
        value_mask
    )

    # IMPORTANT:
    # The light is searched for ONLY inside the calibrated screen.
    bright_mask = cv2.bitwise_and(
        bright_mask,
        mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        bright_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        # Score using both area and brightness.
        contour_mask = np.zeros_like(gray)

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            -1
        )

        mean_value = cv2.mean(
            value,
            mask=contour_mask
        )[0]

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


def run_light_detection(mapper):
    camera = None

    try:
        camera = open_camera()

        screen_mask = None

        print("\n" + "=" * 65)
        print("SMARTIR LIGHT / IR DETECTION")
        print("Only the calibrated screen region is searched.")
        print("Press Q to quit.")
        print("=" * 65)

        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("\nCamera frame error.")
                break

            if screen_mask is None or screen_mask.shape != frame.shape[:2]:
                screen_mask = make_screen_mask(
                    frame.shape,
                    mapper
                )

            result, detection_mask = find_light(
                screen_mask,
                frame
            )

            display = frame.copy()

            # Draw the calibrated screen boundary.
            corners = mapper.camera_corners.astype(
                np.int32
            )

            cv2.polylines(
                display,
                [corners],
                True,
                (0, 255, 0),
                2
            )

            if result is not None:
                camera_x, camera_y, contour = result

                screen_x, screen_y = mapper.camera_to_screen(
                    camera_x,
                    camera_y
                )

                cv2.drawContours(
                    display,
                    [contour],
                    -1,
                    (255, 0, 255),
                    2
                )

                cv2.circle(
                    display,
                    (int(camera_x), int(camera_y)),
                    8,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    display,
                    f"Camera: "
                    f"({camera_x:.1f}, {camera_y:.1f})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"Screen: "
                    f"({screen_x:.1f}, {screen_y:.1f})",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                print(
                    f"\rCamera: "
                    f"({camera_x:.1f}, {camera_y:.1f})   "
                    f"Screen: "
                    f"({screen_x:.1f}, {screen_y:.1f})",
                    end="",
                    flush=True
                )

            cv2.imshow(
                "SMARTIR Light Detection",
                display
            )

            cv2.imshow(
                "Detection Mask - Screen Only",
                detection_mask
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

        run_light_detection(mapper)

    except Exception as error:
        print("\nERROR:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
