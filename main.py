import os
import sys


CALIBRATION_FILE = "screen_corners.npy"
HOMOGRAPHY_FILE = "calibration.npz"


def calibration_exists():
    return os.path.exists(CALIBRATION_FILE) or os.path.exists(HOMOGRAPHY_FILE)


def run_calibration(mode="auto"):
    """Launch the calibration application."""
    from calibration import main as calibration_main

    calibration_main(mode)


def run_whiteboard():
    from whiteboard import main as whiteboard_main

    whiteboard_main()


def main():
    print("\n" + "=" * 70)
    print("SMARTIR - WII REMOTE STYLE DIGITAL WHITEBOARD")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Make sure screen calibration exists
    # --------------------------------------------------------

    if not calibration_exists():
        print("\nNo calibration data found.")
        print("Screen calibration is required (RGBW + point grid).")

        choice = input(
            "Run calibration now? [Y/N]: "
        ).strip().lower()

        if choice != "y":
            print("Program stopped.")
            return

        run_calibration()

    else:
        print(
            "\nExisting calibration found:"
            f" {HOMOGRAPHY_FILE if os.path.exists(HOMOGRAPHY_FILE) else CALIBRATION_FILE}"
        )

        choice = input(
            "Recalibrate the screen? [Y/N]: "
        ).strip().lower()

        if choice == "y":
            run_calibration()

    # --------------------------------------------------------
    # STEP 2: Verify calibration
    # --------------------------------------------------------

    if not calibration_exists():
        print(
            "\nERROR: Calibration did not produce "
            f"{CALIBRATION_FILE} / {HOMOGRAPHY_FILE}."
        )
        return

    # --------------------------------------------------------
    # STEP 3: START THE ACTUAL DRAWING APPLICATION
    #
    # IMPORTANT:
    # Do NOT call run_light_detection() here.
    #
    # light_detection.py is the detector/test component.
    # whiteboard.py is the actual application that:
    #
    #   camera -> detect light -> homography -> draw
    # --------------------------------------------------------

    print("\nLoading coordinate mapper and starting whiteboard...")

    run_whiteboard()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nSMARTIR stopped.")

    except Exception as error:
        print("\nFATAL ERROR:")
        print(error)
        sys.exit(1)
