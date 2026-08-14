import os
import sys

from coordinate_mapper import CoordinateMapper
from light_detection import run_light_detection


CALIBRATION_FILE = "screen_corners.npy"


def run_calibration():
    from calibration import main as calibration_main

    calibration_main()


def main():
    print("\n" + "=" * 70)
    print("SMARTIR - WII REMOTE STYLE DIGITAL WHITEBOARD")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Calibration
    # --------------------------------------------------------

    if not os.path.exists(CALIBRATION_FILE):
        print("\nNo calibration data found.")
        print("RGBW screen calibration is required.")

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
            f" {CALIBRATION_FILE}"
        )

        choice = input(
            "Recalibrate the screen? [Y/N]: "
        ).strip().lower()

        if choice == "y":
            run_calibration()

    # --------------------------------------------------------
    # STEP 2: Verify calibration
    # --------------------------------------------------------

    if not os.path.exists(CALIBRATION_FILE):
        print(
            "\nERROR: Calibration did not produce "
            f"{CALIBRATION_FILE}."
        )
        return

    # --------------------------------------------------------
    # STEP 3: Load coordinate transformation
    # --------------------------------------------------------

    print("\nLoading coordinate mapper...")

    mapper = CoordinateMapper()

    mapper.print_info()

    # --------------------------------------------------------
    # STEP 4: Detect IR / bright light
    # --------------------------------------------------------

    print("\nStarting light/IR detection...")

    run_light_detection(mapper)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nSMARTIR stopped.")

    except Exception as error:
        print("\nFATAL ERROR:")
        print(error)
        sys.exit(1)
