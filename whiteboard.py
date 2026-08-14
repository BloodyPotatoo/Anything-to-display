import os
import cv2
import numpy as np

from coordinate_mapper import CoordinateMapper
from light_detection import open_camera, make_screen_mask, find_light


# ============================================================
# SMARTIR - REAL-TIME DIGITAL WHITEBOARD
#
# This file is intentionally standalone.
#
# Existing working files are NOT modified:
#   calibration.py
#   coordinate_mapper.py
#   light_detection.py
#   main.py
#   screen_corners.npy
#
# Pipeline:
#
#   CAMERA
#      ↓
#   calibrated screen polygon
#      ↓
#   bright light detector
#      ↓
#   camera (x,y)
#      ↓
#   homography
#      ↓
#   desktop (x,y)
#      ↓
#   digital canvas
#
# No Tkinter is used.
# ============================================================


WINDOW_NAME = "SMARTIR DIGITAL WHITEBOARD"

# Drawing settings.
BRUSH_SIZE = 5
ERASER_SIZE = 35

# Point smoothing.
# 0.0 = no smoothing
# 1.0 = very slow/heavy smoothing
SMOOTHING = 0.35

# Do not connect two detections if the detected point suddenly jumps
# this far in screen coordinates.
MAX_JUMP = 180.0

# Ignore tiny movement caused by camera noise.
MIN_MOVEMENT = 1.0

# The detector can be left in visible mode while testing with a normal
# webcam. "ir" and "auto" are labels for the eventual IR detector.
DEFAULT_MODE = "visible"

SAVE_FILE = "whiteboard.png"


def create_canvas(width, height):
    """Create a white RGB drawing canvas."""
    return np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )


def smooth_point(previous, current):
    """Exponential smoothing for camera jitter."""
    if previous is None:
        return current

    x = (
        previous[0] * (1.0 - SMOOTHING)
        + current[0] * SMOOTHING
    )

    y = (
        previous[1] * (1.0 - SMOOTHING)
        + current[1] * SMOOTHING
    )

    return float(x), float(y)


def draw_segment(canvas, previous, current, eraser=False):
    """
    Draw one segment between two mapped screen coordinates.

    A missing detection is handled by the caller, so a new stroke
    begins whenever the light disappears.
    """
    if previous is None or current is None:
        return

    dx = current[0] - previous[0]
    dy = current[1] - previous[1]

    distance = float(np.hypot(dx, dy))

    # Ignore an accidental giant camera jump.
    if distance > MAX_JUMP:
        return

    if distance < MIN_MOVEMENT:
        return

    if eraser:
        color = (255, 255, 255)
        thickness = ERASER_SIZE
    else:
        color = (0, 0, 0)
        thickness = BRUSH_SIZE

    cv2.line(
        canvas,
        (
            int(round(previous[0])),
            int(round(previous[1])),
        ),
        (
            int(round(current[0])),
            int(round(current[1])),
        ),
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_canvas(canvas):
    """
    Save a clean copy without the on-screen control bar.
    """
    clean = canvas.copy()

    ok = cv2.imwrite(
        SAVE_FILE,
        clean,
    )

    if ok:
        print(
            f"\nWhiteboard saved to:\n"
            f"{os.path.abspath(SAVE_FILE)}"
        )
    else:
        print("\nERROR: Could not save whiteboard.")


def draw_status_bar(
    canvas,
    mode,
    detected,
    eraser,
):
    """
    Draw controls/status on a copy of the canvas.
    This copy is displayed but is NOT saved.
    """
    display = canvas.copy()

    # Semi-opaque-looking light gray bar.
    # We blend instead of permanently changing the canvas.
    bar = np.full(
        (50, display.shape[1], 3),
        235,
        dtype=np.uint8,
    )

    display[:50] = cv2.addWeighted(
        display[:50],
        0.25,
        bar,
        0.75,
        0,
    )

    status = "PEN" if not eraser else "ERASER"
    tracking = "TRACKING" if detected else "NO LIGHT"

    text = (
        f"SMARTIR  |  {status}  |  {tracking}  |  "
        f"MODE: {mode}  |  "
        "C Clear  E Eraser  S Save  D Debug  Q/ESC Quit"
    )

    cv2.putText(
        display,
        text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    # Tracking indicator.
    indicator_x = display.shape[1] - 25

    cv2.circle(
        display,
        (indicator_x, 25),
        7,
        (0, 0, 0) if detected else (160, 160, 160),
        -1,
    )

    return display


def make_debug_frame(
    frame,
    mapper,
    screen_mask,
    result,
    detection_mask,
):
    """
    Optional camera debug view.

    This is extremely useful during initial testing because it shows
    exactly what the detector thinks the pen is.
    """
    debug = frame.copy()

    # Calibrated projector boundary.
    corners = mapper.camera_corners.astype(np.int32)

    cv2.polylines(
        debug,
        [corners],
        True,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if result is not None:
        camera_x, camera_y, contour = result

        cv2.drawContours(
            debug,
            [contour],
            -1,
            (255, 0, 255),
            2,
        )

        cv2.circle(
            debug,
            (
                int(round(camera_x)),
                int(round(camera_y)),
            ),
            9,
            (0, 0, 255),
            -1,
        )

        screen_x, screen_y = mapper.camera_to_screen(
            camera_x,
            camera_y,
        )

        cv2.putText(
            debug,
            f"Camera: ({camera_x:.1f}, {camera_y:.1f})",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug,
            f"Screen: ({screen_x:.1f}, {screen_y:.1f})",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(
        "SMARTIR Camera Debug",
        debug,
    )

    cv2.imshow(
        "SMARTIR Detection Mask",
        detection_mask,
    )


def main():
    print("\n" + "=" * 70)
    print("SMARTIR - REAL-TIME DIGITAL WHITEBOARD")
    print("=" * 70)

    if not os.path.exists("screen_corners.npy"):
        raise FileNotFoundError(
            "screen_corners.npy was not found.\n"
            "Run your working calibration first."
        )

    # --------------------------------------------------------
    # EXISTING CALIBRATION
    # --------------------------------------------------------

    print("\nLoading existing coordinate mapper...")
    mapper = CoordinateMapper()
    mapper.print_info()

    width = int(mapper.screen_width)
    height = int(mapper.screen_height)

    print("\n" + "=" * 70)
    print("WHITEBOARD READY")
    print("=" * 70)
    print(f"Canvas: {width} x {height}")
    print("\nControls:")
    print("  C       Clear canvas")
    print("  E       Toggle pen / eraser")
    print("  S       Save whiteboard.png")
    print("  D       Toggle camera debug")
    print("  Q       Quit")
    print("  ESC     Quit")
    print("=" * 70)

    mode = input(
        f"\nDetection mode [visible/ir/auto] "
        f"(default {DEFAULT_MODE}): "
    ).strip().lower()

    if mode not in ("visible", "ir", "auto"):
        mode = DEFAULT_MODE

    print(f"\nDetection mode: {mode}")

    if mode != "visible":
        print(
            "NOTE: The current detector is still brightness-based. "
            "A real IR mode will be added when the IR-capable camera/"
            "pen hardware is available."
        )

    camera = None

    try:
        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        camera = open_camera()

        # Build the screen mask from the EXISTING calibration.
        screen_mask = make_screen_mask(
            (
                int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            ),
            mapper,
        )

        # If the camera backend reports invalid dimensions, rebuild
        # the mask after receiving the first frame.
        screen_mask = None

        canvas = create_canvas(
            width,
            height,
        )

        previous_point = None
        smoothed_point = None

        eraser = False
        debug = False

        # ----------------------------------------------------
        # OPEN-CV ONLY
        # ----------------------------------------------------

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

        print("\nDrawing app is running.")
        print("Point a bright light at the calibrated screen.")

        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("\nCamera frame error.")
                break

            # Rebuild screen mask if the actual camera frame size
            # differs from the previous frame.
            if (
                screen_mask is None
                or screen_mask.shape != frame.shape[:2]
            ):
                screen_mask = make_screen_mask(
                    frame.shape,
                    mapper,
                )

            # ------------------------------------------------
            # DETECT
            # ------------------------------------------------

            result, detection_mask = find_light(
                screen_mask,
                frame,
            )

            # ------------------------------------------------
            # MAP + DRAW
            # ------------------------------------------------

            if result is not None:
                camera_x, camera_y, _ = result

                # Extra safety check against points outside the
                # calibrated projector polygon.
                if mapper.inside_screen(
                    camera_x,
                    camera_y,
                ):
                    screen_x, screen_y = mapper.camera_to_screen(
                        camera_x,
                        camera_y,
                    )

                    current_point = (
                        float(screen_x),
                        float(screen_y),
                    )

                    smoothed_point = smooth_point(
                        smoothed_point,
                        current_point,
                    )

                    draw_segment(
                        canvas,
                        previous_point,
                        smoothed_point,
                        eraser,
                    )

                    previous_point = smoothed_point

                else:
                    previous_point = None
                    smoothed_point = None

            else:
                # IMPORTANT:
                # No detected light = pen lifted.
                # Start a new stroke on the next detection.
                previous_point = None
                smoothed_point = None

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display = draw_status_bar(
                canvas,
                mode,
                result is not None,
                eraser,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            if debug:
                make_debug_frame(
                    frame,
                    mapper,
                    screen_mask,
                    result,
                    detection_mask,
                )

            # waitKey(1) keeps the fullscreen app responsive.
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("c"):
                canvas = create_canvas(
                    width,
                    height,
                )

                previous_point = None
                smoothed_point = None

                print("\nCanvas cleared.")

            elif key == ord("e"):
                eraser = not eraser

                print(
                    "\nEraser:",
                    "ON" if eraser else "OFF",
                )

            elif key == ord("s"):
                save_canvas(canvas)

            elif key == ord("d"):
                debug = not debug

                if not debug:
                    cv2.destroyWindow(
                        "SMARTIR Camera Debug"
                    )
                    cv2.destroyWindow(
                        "SMARTIR Detection Mask"
                    )

                print(
                    "\nCamera debug:",
                    "ON" if debug else "OFF",
                )

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

    print("\nSMARTIR whiteboard stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSMARTIR whiteboard stopped.")
    except Exception as error:
        print("\nFATAL ERROR:")
        print(error)
        raise
