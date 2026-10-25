import os
import cv2
import numpy as np

from camera import (
    open_camera,
    require_display,
    preview_size,
    preview_window,
    cleanup_preview,
)
from coordinate_mapper import CoordinateMapper
from light_detection import make_screen_mask, find_light


# ============================================================
# SMARTIR - REAL-TIME DIGITAL WHITEBOARD
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
DEFAULT_MODE = "auto"

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
    Draw controls/status on a copy of the canvas and return the copy.
    The canvas content (strokes) IS included; the status bar overlays
    on top. The returned copy is what the windows show and what is
    saved, so the drawn strokes are always visible.
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
    # Clean early exit when no graphical session exists (the Qt
    # GUI backend would otherwise abort() the whole process).
    require_display()

    print("\n" + "=" * 70)
    print("SMARTIR - REAL-TIME DIGITAL WHITEBOARD")
    print("=" * 70)

    has_legacy = os.path.exists("screen_corners.npy")
    has_new = os.path.exists("calibration.npz")

    if not (has_legacy or has_new):
        raise FileNotFoundError(
            "No calibration data found (screen_corners.npy / "
            "calibration.npz).\n"
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

    camera = None

    try:
        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        # open_camera() also returns the RESOLVED mode, so the
        # status bar and the detector agree on visible/ir.
        camera, mode = open_camera(mode)

        # The screen mask is built after the first real frame
        # arrives (see inside the loop), so it always matches the
        # actual camera resolution.
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

            # ---- Coordinate space ----
            # set_frame_size() rescales the screen polygon and the
            # homography to the camera's actual frame size, so the
            # mask, the detected centroid, and inside_screen() all use
            # the same camera-pixel space. camera_to_screen() converts
            # the centroid to desktop/projector space for drawing.
            if screen_mask is None or screen_mask.shape != frame.shape[:2]:
                mapper.set_frame_size(
                    frame.shape[1],
                    frame.shape[0],
                )

                screen_mask = make_screen_mask(
                    frame.shape,
                    mapper,
                )

            # Detect in the camera's coordinate space. find_light() returns
            # (cx, cy, contour) in the coordinate space the mask was built
            # for, which is the camera space because we pass the mask
            # that matches the camera frame.
            result, detection_mask = find_light(
                screen_mask,
                frame,
                mode,
            )

            # ------------------------------------------------
            # MAP + DRAW
            # ------------------------------------------------

            if result is not None:
                camera_x, camera_y, _ = result

                # Safety check against points outside the calibrated
                # projector polygon (in projector space). Map the raw
                # camera centroid to projector space first, then test.
                screen_x, screen_y = mapper.camera_to_screen(
                    camera_x,
                    camera_y,
                )

                if mapper.inside_screen(
                    camera_x,
                    camera_y,
                ):
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

            # Live preview of what the camera sees (only if enabled).
            preview_name = preview_window()
            preview_dims = preview_size()

            if preview_name is not None and preview_dims is not None:
                try:
                    cv2.imshow(
                        preview_name,
                        cv2.resize(
                            frame,
                            preview_dims,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    )
                except Exception:
                    pass

            display = draw_status_bar(
                canvas,
                mode,
                result is not None,
                eraser,
            )

            # The canvas now contains the drawn strokes: show the full
            # canvas (strokes + status bar) in the main window.
            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            if debug:
                make_debug_frame(
                    frame,
                    mapper,
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
        cleanup_preview()

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
