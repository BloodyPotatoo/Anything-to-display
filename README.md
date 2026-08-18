# 🖥️ SMARTIR - Interactive Digital Whiteboard System

An advanced, hardware-agnostic digital whiteboard system that emulates optical tracking solutions using standard camera and light-emitting hardware.

The software processes input through a multi-stage pipeline to map physical light coordinates to desktop coordinates:

```text
RGBW PROJECTOR CALIBRATION
        |
        v
RED ∩ GREEN ∩ BLUE ∩ WHITE
        |
        v
SCREEN CORNERS
        |
        v
HOMOGRAPHY / COORDINATE MAPPING
        |
        v
IR / BRIGHT LIGHT DETECTION
        |
        v
CAMERA COORDINATE
        |
        v
DESKTOP COORDINATE
```

---

## 📦 System Components

*   `main.py` — Main entry point to run the complete system.
*   `calibration.py` — Detects and isolates the projected screen area using RGBW intersection.
*   `coordinate_mapper.py` — Computes homography to map camera coordinates to desktop coordinates.
*   `light_detection.py` — Detects infrared or bright light sources strictly within the calibrated screen boundary.
*   `screen_corners.npy` — Calibration output file containing the detected screen corner coordinates.

---

## 🚀 Installation

Install the required dependencies using `pip`:

```bash
pip install opencv-python numpy
```

*Note: On certain Linux distributions, `tkinter` may also be required for desktop resolution detection.*

---

## 💻 Usage

To launch the application, execute the main script:

```bash
python main.py
```

The system will prompt you to initiate the calibration sequence.

---

## 📐 Calibration Procedure

The calibration process maps the projection area by sequentially displaying four solid colors:

1.  🔴 **RED**
2.  🟢 **GREEN**
3.  🔵 **BLUE**
4.  ⚪ **WHITE**

### Step-by-Step Instructions:
1. Project each color onto your screen.
2. Press **ENTER** in the terminal once the color is fully visible to capture the frame.
3. The system will automatically compute a tolerant intersection of the four masks:

```text
RED confidence
      AND
GREEN confidence
      AND
BLUE confidence
      AND
WHITE confidence
```

Each pixel is assigned a confidence value from `0` to `255`. The weakest of the four confidence values is evaluated. A pixel is retained if its weakest confidence exceeds the `INTERSECTION_THRESHOLD`.

This robust approach ensures that exact RGB values are not required, making the system highly resilient to varying camera and projector characteristics.

The detected corners are saved to:
```text
screen_corners.npy
```

---

## 🔍 Light Detection

Following successful calibration, the camera continuously scans for a bright point source. 

To prevent false positives, the search area is strictly restricted to the calibrated screen polygon:

```text
+--------------------------------------+
|                                      |
|        +--------------------+        |
|        |                    |        |
|        |        IR ●        |        |
|        |                    |        |
|        +--------------------+        |
|                                      |
+--------------------------------------+
```

*   **Standard Webcams:** Operates as a bright-light detector (e.g., phone flashlight).
*   **IR-Sensitive Cameras:** Operates as a dedicated infrared pen tracker.

---

## 🔄 Coordinate Conversion

When the camera detects a coordinate:
```text
camera = (734, 421)
```
The homography matrix transforms it into the corresponding desktop coordinate:
```text
screen = (x, y)
```
This transformation is calculated dynamically using the four calibrated screen corners.

---

## ⚙️ Configuration & Tuning

Key parameters can be adjusted in `calibration.py` to optimize performance:

*   `COLOR_DIFFERENCE_THRESHOLD` (Default: `3`): Minimum channel difference for color detection.
*   `COLOR_DOMINANCE_THRESHOLD` (Default: `3`): Minimum dominance of the expected RGB channel.
*   `INTERSECTION_THRESHOLD` (Default: `25`): Lower this value (e.g., `15` or `10`) if calibration is too strict, or increase it if background noise is detected.
*   `INTERSECTION_TOLERANCE_PIXELS` (Default: `100`): Spatial expansion applied to each mask before intersection to account for camera movement or exposure changes.
