import cv2
import numpy as np
import time
import sys
import platform


# ============================================================
# SMARTIR DIGITAL NOTEBOOK
# UNIVERSAL PROJECTOR SCREEN CALIBRATION
#
# Camera handling is designed to work across:
#   - Linux / V4L2
#   - Windows / Media Foundation or DirectShow
#   - macOS / AVFoundation
#
# The program automatically searches for a usable camera instead
# of assuming /dev/video1 or camera index 1.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Camera indices to search.
MAX_CAMERA_INDEX = 10

AVERAGE_FRAMES = 30
FRAME_DELAY = 0.03

CR_THRESHOLD = 20
MORPH_KERNEL_SIZE = 7
MIN_SCREEN_AREA_RATIO = 0.10
APPROX_EPSILON_RATIO = 0.02

CALIBRATION_COLORS = {
    "RED": (0, 0, 255),
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "WHITE": (255, 255, 255),
}


# ============================================================
# CAMERA BACKEND SELECTION
# ============================================================

def get_camera_backends():
    """
    Return sensible OpenCV backends for the current OS.

    CAP_ANY is included as the final fallback so OpenCV can choose
    whatever backend is available on the machine.
    """
    system = platform.system()

    if system == "Windows":
        return [
            cv2.CAP_MSMF,
            cv2.CAP_DSHOW,
            cv2.CAP_ANY,
        ]

    if system == "Darwin":
        return [
            cv2.CAP_AVFOUNDATION,
            cv2.CAP_ANY,
        ]

    if system == "Linux":
        return [
            cv2.CAP_V4L2,
            cv2.CAP_ANY,
        ]

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


# ============================================================
# TEST A CAMERA
# ============================================================

def try_camera(index, backend):
    """
    Try opening one camera index using one backend.

    Returns:
        cap, frame
    or:
        None, None
    """

    print(
        f"Trying camera index {index} "
        f"using {backend_name(backend)}..."
    )

    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception:
        return None, None

    if not cap.isOpened():
        cap.release()
        return None, None

    # Try the requested resolution.
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    # MJPEG is especially useful for USB webcams on Linux.
    # On other systems OpenCV may ignore this setting, which is fine.
    try:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
    except Exception:
        pass

    # Give the camera a moment to start.
    time.sleep(0.2)

    # Some cameras return empty frames immediately after opening.
    frame = None

    for _ in range(10):
        ret, candidate = cap.read()

        if ret and candidate is not None and candidate.size > 0:
            frame = candidate
            break

        time.sleep(0.05)

    if frame is None:
        cap.release()
        return None, None

    # Basic sanity check.
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        cap.release()
        return None, None

    return cap, frame


# ============================================================
# AUTOMATIC CAMERA DETECTION
# ============================================================

def open_camera():
    """
    Automatically find a working camera.

    No fixed /dev/videoX path is required.
    """

    print()
    print("=" * 60)
    print("AUTOMATIC CAMERA DETECTION")
    print("=" * 60)

    backends = get_camera_backends()

    for index in range(MAX_CAMERA_INDEX):

        for backend in backends:

            cap, frame = try_camera(
                index,
                backend
            )

            if cap is not None:

                actual_width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                actual_height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                actual_fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                print()
                print("Camera found!")
                print("Index:", index)
                print("Backend:", backend_name(backend))
                print(
                    f"Resolution: "
                    f"{actual_width} x {actual_height}"
                )
                print(
                    f"FPS: {actual_fps:.1f}"
                )
                print(
                    "Frame shape:",
                    frame.shape
                )
                print("=" * 60)

                # Disable auto white balance where supported.
                try:
                    cap.set(
                        cv2.CAP_PROP_AUTO_WB,
                        0
                    )
                except Exception:
                    pass

                return cap

    print()
    print("ERROR: No working camera was found.")
    print()
    print("Make sure:")
    print("  1. A webcam is connected.")
    print("  2. No other application is exclusively using it.")
    print("  3. Camera permissions are enabled.")
    print()
    sys.exit(1)


# ============================================================
# PROJECTOR WINDOW
# ============================================================

def create_projector_window():
    window_name = "PROJECTOR"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    return window_name


# ============================================================
# SHOW CALIBRATION COLOR
# ============================================================

def show_color(window_name, color):
    image = np.zeros(
        (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8
    )

    image[:] = color

    cv2.imshow(
        window_name,
        image
    )

    cv2.waitKey(1)


# ============================================================
# WAIT FOR USER CONFIRMATION
# ============================================================

def wait_for_confirmation(color_name):
    print()
    print("=" * 60)
    print(f"SHOW {color_name}")
    print("Position the projector correctly.")
    print("Press ENTER in this terminal to capture.")
    print("Type Q and press ENTER to quit.")
    print("=" * 60)

    try:
        user_input = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if user_input == "q":
        return False

    return True


# ============================================================
# FRAME AVERAGING
# ============================================================

def capture_average_frame(cap, number_of_frames):
    print(
        f"\nCapturing {number_of_frames} frames..."
    )

    frames = []

    for i in range(number_of_frames):

        ret, frame = cap.read()

        if not ret or frame is None:
            print(
                f"\nWARNING: Failed to read "
                f"frame {i + 1}."
            )
            continue

        frames.append(
            frame.astype(np.float32)
        )

        cv2.imshow(
            "Camera Preview",
            frame
        )

        cv2.waitKey(1)

        time.sleep(FRAME_DELAY)

        print(
            f"\rFrame {i + 1}/{number_of_frames}",
            end="",
            flush=True
        )

    print()

    if len(frames) == 0:
        raise RuntimeError(
            "No camera frames captured."
        )

    average = np.mean(
        frames,
        axis=0
    )

    average = np.clip(
        average,
        0,
        255
    ).astype(np.uint8)

    print(
        f"Successfully averaged "
        f"{len(frames)} frames."
    )

    return average


# ============================================================
# BGR -> YCrCb
# ============================================================

def convert_to_ycrcb(frame):
    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2YCrCb
    )


# ============================================================
# CR CHANNEL
# ============================================================

def extract_cr(frame_ycrcb):
    return frame_ycrcb[:, :, 1]


# ============================================================
# COLOR DIFFERENCE
# ============================================================

def calculate_difference(
    color_cr,
    white_cr
):
    return cv2.absdiff(
        color_cr,
        white_cr
    )


# ============================================================
# THRESHOLDING
# ============================================================

def threshold_difference(difference):
    _, mask = cv2.threshold(
        difference,
        CR_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    return mask


# ============================================================
# MORPHOLOGICAL PROCESSING
# ============================================================

def morphological_processing(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed


# ============================================================
# COMBINE RGB MASKS
# ============================================================

def combine_masks(
    red_mask,
    green_mask,
    blue_mask
):
    combined = cv2.bitwise_or(
        red_mask,
        green_mask
    )

    combined = cv2.bitwise_or(
        combined,
        blue_mask
    )

    return combined


# ============================================================
# FIND LARGEST CONTOUR
# ============================================================

def find_largest_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea
    )


# ============================================================
# CONTOUR APPROXIMATION
# ============================================================

def approximate_contour(contour):
    perimeter = cv2.arcLength(
        contour,
        True
    )

    epsilon = (
        APPROX_EPSILON_RATIO *
        perimeter
    )

    return cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )


# ============================================================
# ORDER FOUR CORNERS
#
# 0 = Top Left
# 1 = Top Right
# 2 = Bottom Right
# 3 = Bottom Left
# ============================================================

def order_corners(points):
    points = np.array(
        points,
        dtype=np.float32
    )

    if points.shape[0] != 4:
        raise ValueError(
            "Exactly four points are required."
        )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    sums = points.sum(axis=1)

    ordered[0] = points[
        np.argmin(sums)
    ]

    ordered[2] = points[
        np.argmax(sums)
    ]

    differences = (
        points[:, 0] -
        points[:, 1]
    )

    ordered[1] = points[
        np.argmax(differences)
    ]

    ordered[3] = points[
        np.argmin(differences)
    ]

    return ordered


# ============================================================
# HOMOGRAPHY
# ============================================================

def calculate_homography(corners):
    width = CAMERA_WIDTH
    height = CAMERA_HEIGHT

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    homography, _ = cv2.findHomography(
        corners,
        destination,
        cv2.RANSAC
    )

    return homography


# ============================================================
# APPLY HOMOGRAPHY
# ============================================================

def transform_point(
    point,
    homography
):
    point = np.array(
        [[point]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        homography
    )

    return transformed[0][0]


# ============================================================
# DRAW DETECTED SCREEN
# ============================================================

def draw_corners(
    frame,
    corners
):
    output = frame.copy()

    labels = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    for i, point in enumerate(corners):

        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            output,
            (x, y),
            10,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            output,
            labels[i],
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    polygon = corners.astype(
        np.int32
    )

    cv2.polylines(
        output,
        [polygon],
        True,
        (0, 255, 0),
        3
    )

    return output


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_calibration(
    corners,
    homography
):
    np.save(
        "screen_corners.npy",
        corners
    )

    np.save(
        "homography.npy",
        homography
    )

    print()
    print("Calibration saved:")
    print("  screen_corners.npy")
    print("  homography.npy")


# ============================================================
# MAIN CALIBRATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SMARTIR DIGITAL NOTEBOOK")
    print("UNIVERSAL PROJECTOR SCREEN CALIBRATION")
    print("=" * 70)

    cap = None

    try:

        # ----------------------------------------------------
        # AUTOMATIC CAMERA DETECTION
        # ----------------------------------------------------

        cap = open_camera()

        # ----------------------------------------------------
        # WINDOWS
        # ----------------------------------------------------

        projector_window = (
            create_projector_window()
        )

        cv2.namedWindow(
            "Camera Preview",
            cv2.WINDOW_NORMAL
        )

        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        captured_frames = {}
        cr_channels = {}
        masks = {}

        # ----------------------------------------------------
        # CALIBRATION SEQUENCE
        # ----------------------------------------------------

        for color_name, color in (
            CALIBRATION_COLORS.items()
        ):

            show_color(
                projector_window,
                color
            )

            confirmed = (
                wait_for_confirmation(
                    color_name
                )
            )

            if not confirmed:
                print(
                    "Calibration cancelled."
                )
                return

            frame = (
                capture_average_frame(
                    cap,
                    AVERAGE_FRAMES
                )
            )

            captured_frames[
                color_name
            ] = frame

            ycrcb = (
                convert_to_ycrcb(
                    frame
                )
            )

            cr = extract_cr(
                ycrcb
            )

            cr_channels[
                color_name
            ] = cr

            print(
                f"{color_name} "
                "calibration captured."
            )

        # ----------------------------------------------------
        # WHITE REFERENCE
        # ----------------------------------------------------

        white_cr = (
            cr_channels["WHITE"]
        )

        # ----------------------------------------------------
        # COLOR DIFFERENCES
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("CALCULATING COLOR DIFFERENCES")
        print("=" * 60)

        for color_name in [
            "RED",
            "GREEN",
            "BLUE"
        ]:

            difference = (
                calculate_difference(
                    cr_channels[color_name],
                    white_cr
                )
            )

            mask = (
                threshold_difference(
                    difference
                )
            )

            mask = (
                morphological_processing(
                    mask
                )
            )

            masks[color_name] = mask

            cv2.imshow(
                f"{color_name} Difference",
                difference
            )

            cv2.imshow(
                f"{color_name} Threshold",
                mask
            )

            cv2.waitKey(500)

        # ----------------------------------------------------
        # COMBINE MASKS
        # ----------------------------------------------------

        print(
            "Combining RGB masks..."
        )

        combined_mask = (
            combine_masks(
                masks["RED"],
                masks["GREEN"],
                masks["BLUE"]
            )
        )

        cv2.imshow(
            "Combined RGB Mask",
            combined_mask
        )

        cv2.waitKey(500)

        # ----------------------------------------------------
        # MORPHOLOGICAL PROCESSING
        # ----------------------------------------------------

        print(
            "Applying morphological processing..."
        )

        processed_mask = (
            morphological_processing(
                combined_mask
            )
        )

        cv2.imshow(
            "Morphological Mask",
            processed_mask
        )

        cv2.waitKey(500)

        # ----------------------------------------------------
        # FIND CONTOUR
        # ----------------------------------------------------

        print(
            "Finding contours..."
        )

        contour = (
            find_largest_contour(
                processed_mask
            )
        )

        if contour is None:
            print()
            print(
                "ERROR: No screen contour detected."
            )
            print(
                "Try reducing CR_THRESHOLD."
            )
            return

        # ----------------------------------------------------
        # AREA CHECK
        # ----------------------------------------------------

        area = cv2.contourArea(
            contour
        )

        image_area = (
            CAMERA_WIDTH *
            CAMERA_HEIGHT
        )

        area_ratio = (
            area / image_area
        )

        print(
            f"Detected contour area: "
            f"{area_ratio * 100:.2f}%"
        )

        if (
            area_ratio <
            MIN_SCREEN_AREA_RATIO
        ):
            print()
            print(
                "ERROR: Detected region "
                "is too small."
            )
            print(
                "Lower CR_THRESHOLD or "
                "improve projector/camera "
                "conditions."
            )
            return

        # ----------------------------------------------------
        # APPROXIMATE CONTOUR
        # ----------------------------------------------------

        print(
            "Approximating contour..."
        )

        polygon = (
            approximate_contour(
                contour
            )
        )

        print(
            f"Polygon vertices detected: "
            f"{len(polygon)}"
        )

        # ----------------------------------------------------
        # FOUR CORNERS
        # ----------------------------------------------------

        if len(polygon) == 4:

            corners = (
                polygon
                .reshape(4, 2)
                .astype(np.float32)
            )

        else:

            print(
                "Contour did not produce "
                "exactly four corners."
            )

            print(
                "Using minimum-area rectangle..."
            )

            rectangle = (
                cv2.minAreaRect(
                    contour
                )
            )

            corners = (
                cv2.boxPoints(
                    rectangle
                ).astype(np.float32)
            )

        # ----------------------------------------------------
        # ORDER CORNERS
        # ----------------------------------------------------

        corners = order_corners(
            corners
        )

        # ----------------------------------------------------
        # PRINT CORNERS
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("DETECTED SCREEN CORNERS")
        print("=" * 60)

        print(
            f"TOP LEFT     : "
            f"({corners[0][0]:.2f}, "
            f"{corners[0][1]:.2f})"
        )

        print(
            f"TOP RIGHT    : "
            f"({corners[1][0]:.2f}, "
            f"{corners[1][1]:.2f})"
        )

        print(
            f"BOTTOM RIGHT : "
            f"({corners[2][0]:.2f}, "
            f"{corners[2][1]:.2f})"
        )

        print(
            f"BOTTOM LEFT  : "
            f"({corners[3][0]:.2f}, "
            f"{corners[3][1]:.2f})"
        )

        # ----------------------------------------------------
        # HOMOGRAPHY
        # ----------------------------------------------------

        print()
        print(
            "Calculating homography..."
        )

        homography = (
            calculate_homography(
                corners
            )
        )

        if homography is None:
            print(
                "ERROR: Homography "
                "calculation failed."
            )
            return

        print()
        print(
            "HOMOGRAPHY MATRIX:"
        )
        print(homography)

        # ----------------------------------------------------
        # DRAW RESULT
        # ----------------------------------------------------

        white_frame = (
            captured_frames["WHITE"]
        )

        result = draw_corners(
            white_frame,
            corners
        )

        cv2.imshow(
            "FINAL SCREEN CALIBRATION",
            result
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_calibration(
            corners,
            homography
        )

        # ----------------------------------------------------
        # TEST HOMOGRAPHY
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("HOMOGRAPHY TEST")
        print("=" * 60)

        test_points = [
            ("TOP LEFT", corners[0]),
            ("TOP RIGHT", corners[1]),
            ("BOTTOM RIGHT", corners[2]),
            ("BOTTOM LEFT", corners[3])
        ]

        for name, point in test_points:

            transformed = (
                transform_point(
                    point,
                    homography
                )
            )

            print(
                f"{name}: "
                f"camera=({point[0]:.1f}, "
                f"{point[1]:.1f}) "
                f"-> "
                f"screen=({transformed[0]:.1f}, "
                f"{transformed[1]:.1f})"
            )

        print()
        print("=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)

        print()
        print(
            "Press any key in an OpenCV "
            "window to exit."
        )

        cv2.waitKey(0)

    except KeyboardInterrupt:
        print(
            "\nCalibration interrupted."
        )

    except Exception as e:
        print()
        print(
            "ERROR:",
            repr(e)
        )

    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()
