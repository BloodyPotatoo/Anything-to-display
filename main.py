"""
SMARTIR DIGITAL NOTEBOOK - MAIN CONTROLLER

Run this file only:
    python main.py

Project structure:
    main.py
    calibration.py
    coordinate_mapper.py
    ir_detection.py
    screen_corners.npy       (created by calibration)
    homography.npy           (created by calibration; kept for compatibility)
"""

import os
import sys

from coordinate_mapper import CoordinateMapper
from ir_detection import run_ir_detection


def run_calibration():
    """Run the calibration module, then return to this controller."""
    from calibration import main as calibration_main
    calibration_main()


def main():
    print("\n" + "=" * 70)
    print("SMARTIR DIGITAL NOTEBOOK")
    print("INTEGRATED CONTROLLER")
    print("=" * 70)

    if not os.path.exists("screen_corners.npy"):
        print("\nNo calibration data was found.")
        print("Calibration is required before IR detection.\n")
        choice = input("Run calibration now? [Y/N]: ").strip().lower()
        if choice != "y":
            print("Program stopped.")
            return
        run_calibration()
    else:
        print("\nExisting calibration found: screen_corners.npy")
        choice = input("Recalibrate screen? [Y/N]: ").strip().lower()
        if choice == "y":
            run_calibration()

    # Calibration may have been cancelled or failed.
    if not os.path.exists("screen_corners.npy"):
        print("\nERROR: screen_corners.npy is still missing.")
        print("Cannot start IR detection.")
        return

    print("\nLoading coordinate mapper...")
    mapper = CoordinateMapper()
    mapper.print_info()

    print("\nStarting IR pen detection...")
    run_ir_detection(mapper)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSMARTIR stopped.")
    except Exception as error:
        print("\nFATAL ERROR:")
        print(error)
        sys.exit(1)
