import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

WIDTH = 1280
HEIGHT = 720

# Increase if too many things are detected.
# Decrease if the IR LED is not detected.
BRIGHTNESS_THRESHOLD = 200

# Small noise removal
KERNEL_SIZE = 5

# Valid IR blob size
MIN_AREA = 3
MAX_AREA = 3000


# ============================================================
# LOAD HOMOGRAPHY
# ============================================================

try:

    homography = np.load("homography.npy")
    screen_corners = np.load("screen_corners.npy")

    print("Calibration loaded.")

except Exception as e:

    print("ERROR: Calibration files not found.")
    print("Run your screen calibration program first.")
    print()
    print("Required files:")
    print("  homography.npy")
    print("  screen_corners.npy")
    print()
    print("Error:", e)

    exit()


# ============================================================
# OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

if not camera.isOpened():

    print("ERROR: Could not open camera.")

    exit()


print()
print("IR PEN DETECTION STARTED")
print("Press Q to quit.")
print()


# ============================================================
# MORPHOLOGICAL KERNEL
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (KERNEL_SIZE, KERNEL_SIZE)
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # GET CAMERA FRAME
    # --------------------------------------------------------

    ret, frame = camera.read()

    if not ret:

        print("Camera frame error.")
        break


    # --------------------------------------------------------
    # 1. GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # 2. HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    value = hsv[:, :, 2]


    # --------------------------------------------------------
    # 3. GRAYSCALE BINARY MASK
    # --------------------------------------------------------

    _, gray_mask = cv2.threshold(
        gray,
        BRIGHTNESS_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )


    # --------------------------------------------------------
    # 4. HSV BRIGHTNESS MASK
    # --------------------------------------------------------

    _, hsv_mask = cv2.threshold(
        value,
        BRIGHTNESS_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )


    # --------------------------------------------------------
    # 5. INTERSECTION
    # --------------------------------------------------------

    mask = cv2.bitwise_and(
        gray_mask,
        hsv_mask
    )


    # --------------------------------------------------------
    # 6. MORPHOLOGICAL OPENING
    # --------------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    # --------------------------------------------------------
    # 7. MORPHOLOGICAL CLOSING
    # --------------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # 8. FIND CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    best_contour = None
    best_area = 0


    # --------------------------------------------------------
    # 9. FIND VALID IR BLOB
    # --------------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue


        if area > best_area:

            best_area = area
            best_contour = contour


    # --------------------------------------------------------
    # 10. FIND IR PEN CENTER
    # --------------------------------------------------------

    if best_contour is not None:

        moments = cv2.moments(
            best_contour
        )


        if moments["m00"] != 0:

            camera_x = (
                moments["m10"]
                /
                moments["m00"]
            )

            camera_y = (
                moments["m01"]
                /
                moments["m00"]
            )


            # ------------------------------------------------
            # 11. CAMERA → PROJECTOR
            # ------------------------------------------------

            point = np.array(
                [
                    [
                        [
                            camera_x,
                            camera_y
                        ]
                    ]
                ],
                dtype=np.float32
            )


            transformed = cv2.perspectiveTransform(
                point,
                homography
            )


            screen_x = (
                transformed[0][0][0]
            )

            screen_y = (
                transformed[0][0][1]
            )


            # ------------------------------------------------
            # 12. DRAW DETECTION
            # ------------------------------------------------

            cv2.drawContours(
                frame,
                [best_contour],
                -1,
                (0, 255, 0),
                2
            )


            cv2.circle(
                frame,
                (
                    int(camera_x),
                    int(camera_y)
                ),
                8,
                (0, 0, 255),
                -1
            )


            # ------------------------------------------------
            # 13. DISPLAY COORDINATES
            # ------------------------------------------------

            text1 = (
                f"Camera: "
                f"({camera_x:.1f}, "
                f"{camera_y:.1f})"
            )

            text2 = (
                f"Screen: "
                f"({screen_x:.1f}, "
                f"{screen_y:.1f})"
            )


            cv2.putText(
                frame,
                text1,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                text2,
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
                end=""
            )


    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "IR Pen Detection",
        frame
    )


    # --------------------------------------------------------
    # SHOW MASK
    # --------------------------------------------------------

    cv2.imshow(
        "IR Detection Mask",
        mask
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("IR detection stopped.")