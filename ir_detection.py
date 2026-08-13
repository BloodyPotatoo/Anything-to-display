import cv2
import numpy as np
import os
import sys

# ============================================================
# SMARTIR IR PEN DETECTION MODULE
# ============================================================
# This version is designed to be imported by main.py.
# The original IR detection algorithm is preserved.
# Coordinate transformation is now handled by CoordinateMapper.
# ============================================================

CAMERA_SOURCE = "AUTO"       # or "/dev/video1" / 0 / 1 / 2
WIDTH = 1280
HEIGHT = 720
BRIGHTNESS_THRESHOLD = 200
KERNEL_SIZE = 5
MIN_AREA = 3
MAX_AREA = 3000


def get_camera_candidates():
    if CAMERA_SOURCE != "AUTO":
        return [CAMERA_SOURCE]

    if sys.platform.startswith("linux"):
        return [p for p in ["/dev/video1", "/dev/video2", "/dev/video0"]
               if os.path.exists(p)]

    return [0, 1, 2, 3]


def open_camera():
    print("\nSearching for IR camera...")

    for source in get_camera_candidates():
        print(f"Trying camera: {source}")

        if isinstance(source, str) and source.startswith("/dev/"):
            camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            camera = cv2.VideoCapture(source)

        if not camera.isOpened():
            camera.release()
            continue

        # Same resolution used by calibration.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        # MJPEG is supported by the HP camera; ignored if unsupported.
        try:
            camera.set(cv2.CAP_PROP_FOURCC,
                       cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        ret, frame = camera.read()
        if not ret or frame is None:
            camera.release()
            continue

        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera selected: {source}")
        print(f"Camera resolution: {actual_width} x {actual_height}")

        if actual_width != WIDTH or actual_height != HEIGHT:
            print("WARNING: camera resolution differs from calibration.")
            print(f"Expected: {WIDTH} x {HEIGHT}")
            print(f"Actual:   {actual_width} x {actual_height}")

        return camera

    raise RuntimeError("Could not open any usable camera.")


def run_ir_detection(mapper):
    """Run IR detection using an already-created CoordinateMapper."""
    if mapper is None:
        raise ValueError("CoordinateMapper is required.")

    camera = None

    try:
        camera = open_camera()

        print("\n" + "=" * 60)
        print("IR PEN DETECTION STARTED")
        print("Press Q to quit.")
        print("=" * 60)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
        )

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("\nCamera frame error.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2]

            _, gray_mask = cv2.threshold(
                gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
            )
            _, hsv_mask = cv2.threshold(
                value, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
            )

            mask = cv2.bitwise_and(gray_mask, hsv_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            best_contour = None
            best_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_AREA or area > MAX_AREA:
                    continue
                if area > best_area:
                    best_area = area
                    best_contour = contour

            if best_contour is not None:
                moments = cv2.moments(best_contour)

                if moments["m00"] != 0:
                    camera_x = moments["m10"] / moments["m00"]
                    camera_y = moments["m01"] / moments["m00"]

                    # The mapper is now the single source of truth for
                    # camera -> desktop transformation.
                    screen_x, screen_y = mapper.camera_to_screen(
                        camera_x, camera_y
                    )

                    cv2.drawContours(
                        frame, [best_contour], -1, (0, 255, 0), 2
                    )
                    cv2.circle(
                        frame,
                        (int(camera_x), int(camera_y)),
                        8, (0, 0, 255), -1
                    )

                    cv2.putText(
                        frame,
                        f"Camera: ({camera_x:.1f}, {camera_y:.1f})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"Screen: ({screen_x:.1f}, {screen_y:.1f})",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )

                    print(
                        f"\rCamera: ({camera_x:.1f}, {camera_y:.1f})   "
                        f"Screen: ({screen_x:.1f}, {screen_y:.1f})",
                        end="", flush=True
                    )

            cv2.imshow("IR Pen Detection", frame)
            cv2.imshow("IR Detection Mask", mask)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nIR detection interrupted.")

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("\nIR detection stopped.")


def main():
    """Standalone test mode. Normal use should be through main.py."""
    try:
        from coordinate_mapper import CoordinateMapper
        mapper = CoordinateMapper()
        mapper.print_info()
        run_ir_detection(mapper)
    except Exception as error:
        print("\nERROR:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
