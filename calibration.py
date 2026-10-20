import sys
import time
import numpy as np
import cv2

from camera import open_camera, require_display, resolve_mode, CAMERA_WIDTH, CAMERA_HEIGHT
from coordinate_mapper import CoordinateMapper

# ============================================================
# SMARTIR - MANUAL 4-POINT CALIBRATION
# 
# A simpler calibration that replaces the 2-step RGBW + point-grid
# sweep with a single pass: 4 points are shown on the projected screen
# one at a time. The user shines an IR light at each point. The system
# detects the light position, records the camera coordinate, and after
# all 4 points are captured it computes a homography and saves the
# calibration data in the same format used by the rest of the codebase.
# ============================================================

CALIBRATION_FILE = "screen_corners.npy"
HOMOGRAPHY_FILE = "calibration.npz"

# How long to let the camera settle before measuring each point.
POINT_SETTLE_TIME = 0.30

# Frames averaged per measurement (noise reduction).
POINT_AVERAGE_FRAMES = 8

# Blob size limits for a reference point, in camera pixels.
POINT_MIN_AREA = 4
POINT_MAX_AREA_RATIO = 0.05

# Absolute brightness floor for a projected point. Prevents dim
# reflections and ambient IR from being accepted.
POINT_FLOOR = 120

# RANSAC threshold in DESKTOP pixels.
RANSAC_REPROJECTION_PX = 5.0

MIN_POINTS_FOR_HOMOGRAPHY = 4

# Warn when mean reprojection error exceeds this fraction of the
# desktop diagonal.
WARN_MEAN_ERROR_RATIO = 0.005

# How long to show each point on the projector before measuring.
POINT_DISPLAY_DURATION = 1.0  # seconds (minimum before capture starts)


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
        16,  # point radius in projector pixels
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


def wait_for_enter(timeout=10):
    """Wait for the user to press ENTER, with a timeout."""
    import select
    import tty
    import termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        print(
            "\nPress ENTER in this terminal to capture this point, "
            "or wait " + str(timeout) + " seconds to skip.",
            flush=True,
        )
        r, _, _ = select.select([fd], [], [], timeout)
        if r:
            ch = sys.stdin.read(1)
            return ch == "\n" or ch == ""
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def capture_point_frame(cap, quiet=False):
    """Capture a single averaged frame from the camera."""
    frames = []

    for _ in range(POINT_AVERAGE_FRAMES):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size:
            frames.append(frame.astype(np.float32))
        cv2.waitKey(1)
        time.sleep(0.02)  # small delay between frames

    if not frames:
        raise RuntimeError("Could not capture any camera frames.")

    average = np.mean(frames, axis=0)
    return np.clip(average, 0, 255).astype(np.uint8)


def find_light_in_frame(frame, roi=None):
    """Locate the brightest blob in a frame, returning (x, y) or None.

    Searches the full frame (or an optional ROI). Uses Otsu + absolute
    floor thresholding, then returns the intensity-weighted centroid.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    if roi is not None:
        x0, y0, x1, y1 = roi
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(gray.shape[1], x1)
        y1 = min(gray.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return None
        gray = gray[y0:y1, x0:x1]

    # Otsu threshold
    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = max(int(otsu_value), POINT_FLOOR)
    threshold_value = min(threshold_value, 255)

    binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)[1]

    # Small morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    region_area = float(gray.shape[0] * gray.shape[1])
    max_area = region_area * POINT_MAX_AREA_RATIO

    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < POINT_MIN_AREA or area > max_area:
            continue

        # Score = area * mean brightness (higher = better)
        mean_val = cv2.mean(gray, mask=cv2.drawContours(
            np.zeros_like(gray), [contour], -1, 255, -1
        )[1])[0]
        score = area * mean_val

        if score > best_score:
            best_score = score
            best = contour

    if best is None:
        return None

    # Weighted centroid (sub-pixel quality)
    moments = cv2.moments(best)
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # Convert back from ROI if we shifted
    if roi is not None:
        x0, y0, _, _ = roi
        cx += x0
        cy += y0

    return (float(cx), float(cy))


def compute_homography(camera_points, desktop_width, desktop_height):
    """Fit a camera -> desktop homography from exactly 4 point pairs.

    Returns (homography_matrix, mean_error_px) or (None, None) if RANSAC fails.
    """
    if len(camera_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, None

    camera_pts = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)
    # The four desktop corners in TL, TR, BR, BL order
    desktop_pts = np.array(
        [[0, 0], [desktop_width - 1, 0], [desktop_width - 1, desktop_height - 1], [0, desktop_height - 1]],
        dtype=np.float32,
    )

    if len(camera_pts) == 4:
        # Exact fit when we have exactly 4 points
        matrix, _ = cv2.findHomography(camera_pts, desktop_pts, 0)
    else:
        # RANSAC fit with more than 4 (shouldn't happen but be safe)
        matrix, inlier_mask = cv2.findHomography(
            camera_pts, desktop_pts, cv2.RANSAC, RANSAC_REPROJECTION_PX
        )
        if matrix is None or inlier_mask is None:
            return None, None
        inliers = inlier_mask.ravel().astype(bool)
        camera_pts = camera_pts[inliers]
        if len(camera_pts) < MIN_POINTS_FOR_HOMOGRAPHY:
            return None, None

    if matrix is None:
        return None, None

    matrix = matrix.astype(np.float32)

    # Compute per-point reprojection errors
    projected = cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - desktop_pts, axis=1)
    mean_error = float(errors.mean())

    # Report per-point residuals
    print("Per-point camera->desktop residuals (px):")
    for i, err in enumerate(errors):
        print(f"  point {i + 1}: {err:7.2f} px")

    print(f"Mean reprojection error: {mean_error:.2f} px")
    return matrix, mean_error


def save_calibration(camera_corners, homography, frame_width, frame_height,
                     desktop_width, desktop_height):
    """Save calibration data in the formats expected by the codebase.

    - screen_corners.npy : 4 camera-space corners (float32, 4x2)
    - calibration.npz    : homography, corners, frame/screen sizes
    """
    corners = np.asarray(camera_corners, dtype=np.float32)
    np.save(CALIBRATION_FILE, corners)

    np.savez(
        HOMOGRAPHY_FILE,
        homography=np.asarray(homography, dtype=np.float32),
        camera_corners=corners,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        desktop_width=int(desktop_width),
        desktop_height=int(desktop_height),
    )

    print("\nSaved:")
    print(f"  {CALIBRATION_FILE}")
    print(f"  {HOMOGRAPHY_FILE}")


def order_corners_tlbr(points):
    """Order 4 points as TL, TR, BR, BL."""
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum, bottom-right largest sum
    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]  # TL
    ordered[2] = points[np.argmax(sums)]  # BR

    # Top-right has smallest difference (x - y), bottom-left largest
    diffs = points[:, 0] - points[:, 1]
    ordered[1] = points[np.argmin(diffs)]  # TR
    ordered[3] = points[np.argmax(diffs)]  # BL

    return ordered


def main():
    cap = None
    projector_window = None

    try:
        # Pre-flight: ensure a display is available (instead of letting
        # the Qt/OpenCV backend abort with no Python traceback).
        require_display()

        print("\n" + "=" * 60)
        print("SMARTIR MANUAL 4-POINT CALIBRATION")
        print("=" * 60)
        print(
            "\nInstructions:"
        )
        print(
            "  1. Make sure the projector displays a full-screen image "
            "onto the screen."
        )
        print(
            "  2. A white point will appear on the screen one at a time."
        )
        print(
            "  3. Shine the IR pen/light at each point and press ENTER "
            "when it is centred."
        )
        print(
            "  4. After all 4 points are captured, a homography is "
            "computed and saved."
        )
        print("=" * 60)

        mode = "ir"  # Manual calibration always uses IR mode
        print(f"\nOpening camera in {mode} mode...")
        cap, resolved_mode = open_camera(mode)

        # Determine the real capture size
        ret, probe = cap.read()
        if ret and probe is not None and probe.size:
            frame_height, frame_width = probe.shape[:2]
        else:
            frame_height, frame_width = CAMERA_HEIGHT, CAMERA_WIDTH

        print(f"Capture size: {frame_width} x {frame_height}")

        # Determine the desktop/projector size.
        # First try to get it from the coordinate mapper if calibration
        # data already exists; otherwise ask the user or use a default.
        try:
            mapper = CoordinateMapper()
            screen_width = mapper.screen_width
            screen_height = mapper.screen_height
            print(
                f"Desktop size from existing calibration: "
                f"{screen_width} x {screen_height}"
            )
        except FileNotFoundError:
            # No existing calibration - ask user for projector resolution
            try:
                resp = input(
                    "\nEnter the projected screen width (px), "
                    "height (px) [e.g. 1280 720]: "
                ).strip()
                parts = resp.split()
                screen_width = int(parts[0])
                screen_height = int(parts[1]) if len(parts) > 1 else 720
            except (EOFError, KeyboardInterrupt):
                print("\nCalibration cancelled.")
                return

        projector_window = create_projector_window()

        # -------------------------------------------------------
        # COLLECT 4 POINTS
        # -------------------------------------------------------
        camera_points = []  # (cx, cy) in camera frame coordinates

        names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]

        for idx, name in enumerate(names):
            print(
                "\n" + "=" * 50,
                f"\nPoint {idx + 1}/{4}: {name}",
                "=" * 50,
            )

            # Show the point on the projector
            # Use 1/4 of the screen width/height as a reasonable size
            # for the point, placed at the corner position.
            px = screen_width // 4
            py = screen_height // 4

            # Map the corner to a projector position
            corner_positions = {
                "TOP-LEFT": (px, py),
                "TOP-RIGHT": (screen_width - px, py),
                "BOTTOM-RIGHT": (screen_width - px, screen_height - py),
                "BOTTOM-LEFT": (px, screen_height - py),
            }
            dx, dy = corner_positions[name]

            show_projector_point(projector_window, dx, dy, screen_width, screen_height)

            print(f"  Point position on screen: ({dx}, {dy})")
            print("  Shine the IR light at the point...")
            time.sleep(POINT_SETTLE_TIME)

            # Wait for user confirmation
            print("  Press ENTER when the light is centred on the point.")
            if not wait_for_enter(timeout=15):
                print("  Timed out - using whatever position we get.")
            # Dismiss the projector window between points
            cv2.destroyWindow(projector_window)

            # Capture a frame and find the light
            print("  Capturing frame...")
            frame = capture_point_frame(cap)

            # Find the brightest blob - could be the IR light
            point = find_light_in_frame(frame)

            if point is not None:
                cx, cy = point
                print(f"  Detected light at camera coordinates: ({cx:.1f}, {cy:.1f})")
                camera_points.append((cx, cy))
            else:
                print("  !!! No light detected - trying again...")
                # Try one more frame
                frame2 = capture_point_frame(cap)
                point2 = find_light_in_frame(frame2)
                if point2 is not None:
                    cx, cy = point2
                    print(f"  Detected on retry: ({cx:.1f}, {cy:.1f})")
                    camera_points.append((cx, cy))
                else:
                    print("  !!! Still no light detected. Calibration may be inaccurate.")
                    # Store a placeholder so we still have 4 entries
                    camera_points.append((frame_width // 2, frame_height // 2))

            # Re-create the projector window for the next point
            projector_window = create_projector_window()

        # -------------------------------------------------------
        # COMPUTE HOMOGRAPHY
        # -------------------------------------------------------
        print("\n" + "=" * 50)
        print("Computing homography from 4 point correspondences...")
        print("=" * 50)

        homography, mean_error = compute_homography(
            camera_points, screen_width, screen_height
        )

        if homography is None:
            raise RuntimeError(
                "Could not compute a homography from the 4 point "
                "correspondences. Make sure the IR light was detected "
                "at all 4 points and that the camera didn't move."
            )

        # Derive the screen corners by inverse-mapping the desktop rectangle
        inverse = np.linalg.inv(homography.astype(np.float64)).astype(np.float32)
        rect = np.array(
            [[0, 0], [screen_width - 1, 0], [screen_width - 1, screen_height - 1], [0, screen_height - 1]],
            dtype=np.float32,
        )
        camera_corners_4pt = cv2.perspectiveTransform(
            rect.reshape(-1, 1, 2), inverse
        ).reshape(-1, 2)
        camera_corners = order_corners_tlbr(camera_corners_4pt)

        # -------------------------------------------------------
        # SAVE CALIBRATION
        # -------------------------------------------------------
        save_calibration(
            camera_corners,
            homography,
            frame_width,
            frame_height,
            screen_width,
            screen_height,
        )

        # -------------------------------------------------------
        # SHOW RESULT
        # -------------------------------------------------------
        # Draw the calibrated screen polygon on the last frame
        if last_frame is None:
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


if __name__ == "__main__":
    main()