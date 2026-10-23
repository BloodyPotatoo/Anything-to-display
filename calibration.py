import sys
import time
import numpy as np
import cv2

from camera import open_camera, require_display, resolve_mode, CAMERA_WIDTH, CAMERA_HEIGHT, IR_EXPOSURE_US, IR_ANALOGUE_GAIN
from coordinate_mapper import CoordinateMapper

# ============================================================
# SMARTIR - MANUAL 9-POINT CALIBRATION WITH AUTO-DETECTION
# 
# A calibration that displays a 3x3 grid of points covering the
# full projected screen. The system auto-detects the IR light when
# it is present for at least IR_CAPTURE_DURATION seconds, then
# captures the position. After all 9 points are captured, a homography
# is computed and saved.
# ============================================================

CALIBRATION_FILE = "screen_corners.npy"
HOMOGRAPHY_FILE = "calibration.npz"

# --- Grid configuration ---
GRID_COLS = 3
GRID_ROWS = 3
NUM_POINTS = GRID_COLS * GRID_ROWS  # 9 points

# Keep points away from screen edge by this fraction (0 = full edge-to-edge)
GRID_INSET = 0.0  # Changed from previous 0.07 to cover full screen

# Point size in projector pixels
POINT_RADIUS = 20

# --- Auto-detection timing ---
# Wait this many seconds the light is present before auto-capturing.
IR_CAPTURE_DURATION = 2.0  # seconds

# How long to show each point before starting the auto-detect timer.
POINT_DISPLAY_TIME = 0.5  # seconds

# How long to wait after capture before moving to next point.
POINT_POST_CAPTURE_DELAY = 0.3  # seconds

# --- Blob / detection limits ---
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


def wait_for_ir_light(cap, duration=IR_CAPTURE_DURATION, timeout=15.0):
    """Wait for IR light to be present for at least `duration` seconds.
    Returns the camera coordinates (x, y) when light is detected, or None if timeout.
    
    Auto-detection: continuously captures frames and looks for a bright blob.
    If the light stays present for >= duration seconds, we capture and return.
    """
    start_wait = time.time()
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

        # Find light in this frame
        gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold - use high value for IR mode
        _, binary = cv2.threshold(
            gray, 240, 255, cv2.THRESH_BINARY
        )

        # Morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        light_detected = False
        best_point = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < POINT_MIN_AREA or area > (gray.shape[0] * gray.shape[1] * POINT_MAX_AREA_RATIO):
                continue

            mean_val = cv2.mean(gray, mask=cv2.drawContours(
                np.zeros_like(gray), [contour], -1, 255, -1
            )[1])[0]

            score = area * mean_val

            if score > 1000:  # Minimum score to count as light
                light_detected = True
                moments = cv2.moments(contour)
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    best_point = (float(cx), float(cy))

        if light_detected and best_point is not None:
            elapsed = time.time() - start_wait
            print(
                f"  IR light detected! Elapsed: {elapsed:.1f}s "
                f"(need {duration:.1f}s)",
                end="",
                flush=True,
            )
            if elapsed >= duration:
                print(f" -> CAPTURE at {best_point[0]:.1f}, {best_point[1]:.1f}")
                return best_point
            else:
                print(f" ... {elapsed:.1f}s", end="", flush=True)
        else:
            # Print a dot to show we're still waiting
            print(".", end="", flush=True)

        # Check timeout
        if time.time() - start_wait > timeout:
            print(" -> Timeout, no light sustained.")
            return None

        cv2.waitKey(1)
        time.sleep(0.1)


def capture_point_frame(cap, quiet=False):
    """Capture a single frame from the camera."""
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Could not capture camera frame.")
    return frame.astype(np.uint8)


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


def compute_homography(camera_points, desktop_width, desktop_height):
    """Fit a camera -> desktop homography from exactly NUM_POINTS point pairs.

    Returns (homography_matrix, mean_error_px) or (None, None) if RANSAC fails.
    """
    if len(camera_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, None

    camera_pts = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)

    # Desktop grid points in TL, TR, BR, BL order for the 3x3 grid
    # We use the 4 corners of the full desktop for the homography fit
    desktop_pts = np.array(
        [[0, 0], [desktop_width - 1, 0], [desktop_width - 1, desktop_height - 1], [0, desktop_height - 1]],
        dtype=np.float32,
    )

    if len(camera_pts) >= 4:
        # Use RANSAC with all available points
        matrix, inlier_mask = cv2.findHomography(
            camera_pts, desktop_pts, cv2.RANSAC, RANSAC_REPROJECTION_PX
        )

        if matrix is None:
            return None, None

        matrix = matrix.astype(np.float32)

        # Compute per-point reprojection errors
        inliers = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.ones(len(camera_pts), dtype=bool)
        camera_pts_in = camera_pts[inliers]

        if len(camera_pts_in) < MIN_POINTS_FOR_HOMOGRAPHY:
            return None, None

        projected = cv2.perspectiveTransform(
            camera_pts_in.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)

        errors = np.linalg.norm(projected - desktop_pts[:len(camera_pts_in)], axis=1)
        mean_error = float(errors.mean())

        # Report per-point residuals
        print("Per-point camera->desktop residuals (px):")
        all_errors = np.linalg.norm(
            cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2), matrix).reshape(-1, 2) - desktop_pts, axis=1
        )
        for i, err in enumerate(all_errors):
            marker = "" if inlier_mask is None or inlier_mask.ravel()[i] else " (rejected)"
            print(f"  point {i + 1}: {err:7.2f}px{marker}")

        print(f"Mean reprojection error: {mean_error:.2f} px")
        return matrix, mean_error
    else:
        # Exact fit with exactly 4 points
        matrix, _ = cv2.findHomography(camera_pts, desktop_pts, 0)
        if matrix is None:
            return None, None
        matrix = matrix.astype(np.float32)
        # Compute error
        projected = cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - desktop_pts, axis=1)
        mean_error = float(errors.mean())
        return matrix, mean_error


def save_calibration(camera_corners, homography, frame_width, frame_height,
                     screen_width, screen_height):
    """Save calibration data in the formats expected by the codebase.
    
    Uses screen_width/screen_height keys (not desktop_width/height) so
    CoordinateMapper can load the calibration.npz directly.
    """
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


def main():
    cap = None
    projector_window = None

    try:
        # Pre-flight: ensure a display is available.
        require_display()

        print("\n" + "=" * 60)
        print("SMARTIR MANUAL 9-POINT CALIBRATION (AUTO-DETECT)")
        print("=" * 60)
        print(
            "\nInstructions:"
        )
        print(
            "  1. Make sure the projector displays a full-screen image "
            "onto the screen."
        )
        print(
            "  2. A 3x3 grid of points will appear on the screen one at a time."
        )
        print(
            "  3. Shine the IR pen/light at each point. The system will "
            f"auto-detect when the light is present for >= {IR_CAPTURE_DURATION:.1f}s."
        )
        print(
            "  4. After all 9 points are captured, a homography is computed "
            "and saved."
        )
        print("=" * 60)

        mode = "ir"
        print(f"\nOpening camera in {mode} mode (IR exposure {IR_EXPOSURE_US}us, gain {IR_ANALOGUE_GAIN}x)...")
        cap, resolved_mode = open_camera(mode)

        # Determine the real capture size
        ret, probe = cap.read()
        if ret and probe is not None and probe.size:
            frame_height, frame_width = probe.shape[:2]
        else:
            frame_height, frame_width = CAMERA_HEIGHT, CAMERA_WIDTH

        print(f"Capture size: {frame_width} x {frame_height}")

        # Determine the desktop/projector size.
        try:
            mapper = CoordinateMapper()
            screen_width = mapper.screen_width
            screen_height = mapper.screen_height
            print(
                f"Desktop size from existing calibration: "
                f"{screen_width} x {screen_height}"
            )
        except FileNotFoundError:
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
        # COLLECT 9 POINTS (3x3 grid, full screen coverage)
        # -------------------------------------------------------
        camera_points = []  # (cx, cy) in camera frame coordinates

        grid_points = build_grid_points(screen_width, screen_height)

        print(f"\nCalibrating {NUM_POINTS} points (3x3 grid, full {screen_width}x{screen_height})...")

        for idx, (dx, dy) in enumerate(grid_points):
            point_idx = idx + 1
            print(
                "\n" + "=" * 50,
                f"\nPoint {point_idx}/{NUM_POINTS}: ({dx:.0f}, {dy:.0f})",
                "=" * 50,
            )

            # Show the point on the projector
            show_projector_point(projector_window, dx, dy, screen_width, screen_height)
            time.sleep(POINT_DISPLAY_TIME)

            # Auto-detect IR light when present for IR_CAPTURE_DURATION seconds
            print("  Shine IR light at the point...")
            point = wait_for_ir_light(cap, duration=IR_CAPTURE_DURATION, timeout=15.0)

            if point is not None:
                cx, cy = point
                print(f"  Detected light at camera coordinates: ({cx:.1f}, {cy:.1f})")
                camera_points.append((cx, cy))
            else:
                print("  !!! No light detected sustained - using center as placeholder")
                camera_points.append((frame_width // 2, frame_height // 2))

            # Small pause before next point
            time.sleep(POINT_POST_CAPTURE_DELAY)

            # Dismiss and recreate projector window
            cv2.destroyWindow(projector_window)
            projector_window = create_projector_window()

        # -------------------------------------------------------
        # COMPUTE HOMOGRAPHY
        # -------------------------------------------------------
        print("\n" + "=" * 50)
        print("Computing homography from 9 point correspondences...")
        print("=" * 50)

        homography, mean_error = compute_homography(
            camera_points, screen_width, screen_height
        )

        if homography is None:
            raise RuntimeError(
                "Could not compute a homography from the 9 point "
                "correspondences. Make sure the IR light was detected "
                "at enough points and that the camera didn't move."
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