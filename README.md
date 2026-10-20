# SMARTIR - Wii Remote Style Digital Whiteboard

This project imitates Johnny Chung Lee's Wii Remote interactive
whiteboard: a camera tracks a bright light pen on a projected screen,
a homography converts camera coordinates into desktop coordinates,
and the desktop coordinates are drawn onto a canvas.

It runs on a PC (webcam testing) and on a Raspberry Pi with a camera
on the default ribbon (CSI) port.

```text
CALIBRATION
  STEP 1: RED -> GREEN -> BLUE -> WHITE color-change
          -> screen mask + approximate corners
  STEP 2: 9 reference points projected one at a time
          -> measured in camera space
          -> RANSAC homography + error report
          -> screen_corners.npy + calibration.npz
DETECTION
  camera frame -> screen polygon mask -> bright blob -> centroid
          -> homography -> desktop coordinate -> drawn segment
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | entry point: calibrate (if needed) then start the whiteboard |
| `camera.py` | shared camera layer: CSI/picamera2, GStreamer, V4L2, exposure modes |
| `calibration.py` | screen detection (RGBW) + point-grid homography calibration |
| `coordinate_mapper.py` | camera -> desktop homography, screen-size detection |
| `light_detection.py` | bright/IR light detector (standalone test view) |
| `whiteboard.py` | the drawing application |
| `screen_corners.npy` | generated: screen polygon in camera space |
| `calibration.npz` | generated: homography, points, frame size, screen size |

## Install

PC:

```bash
pip install opencv-python numpy
```

Raspberry Pi (use the distro packages - they match the camera stack):

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy python3-tk
sudo raspi-config   # Interface Options -> Camera -> enable
```

On a Pi the camera on the ribbon port is opened through
**picamera2 (libcamera)** automatically. If `python3-picamera2` is
missing, SMARTIR falls back to a `libcamerasrc` GStreamer pipeline,
then to scanning `/dev/video*`.

## Run

```bash
python3 main.py
```

The program asks whether to (re)calibrate, then asks for the
detection mode (`visible` / `ir` / `auto`) and starts the whiteboard.

Other entry points:

```bash
python3 calibration.py            # calibration only (asks for mode)
python3 calibration.py visible    # calibration in visible-light mode
python3 light_detection.py auto   # detector test view only
python3 run_whiteboard.py         # whiteboard only (needs calibration)
```

## Calibration procedure

Calibration has two steps and needs the projector running.

**STEP 1 - RGBW screen detection (where is the screen?)**

The projector shows RED, GREEN, BLUE, WHITE. Press ENTER in the
terminal after each color. Pixels whose Lab color changes strongly
across those four captures belong to the projected screen:

```text
screen = pixels with large color change across RED/GREEN/BLUE/WHITE
```

This produces the search mask and four approximate corners. It is
skipped automatically in `ir` mode (an IR camera cannot see colors).

**STEP 2 - point-grid sweep (how does camera space map to desktop?)**

Press ENTER to start. The projector then shows 9 reference points,
one at a time, at known desktop coordinates, and each is measured
automatically (no keys needed). The 9 camera/desktop pairs are
fitted with `cv2.findHomography(..., RANSAC)` and the program prints
a real accuracy report:

```text
Calibration accuracy (desktop pixels):
  points measured : 9 / 9
  mean error      : 1.83 px
  max  error      : 3.41 px
  mean / diagonal : 0.098 %
```

This is the same idea as Johnny Chung Lee's calibration: real point
correspondences instead of guessing a rectangle from a blob. If fewer
than 4 points can be measured, the program falls back to the 4-corner
RGBW estimate and says so.

Saved files:

```text
screen_corners.npy   screen polygon in camera space (detection mask)
calibration.npz       homography + points + capture size + desktop size
```

Do not move the camera, projector, or screen between step 1 and the
end of step 2.

## Detection

The detector searches for a bright blob **only inside the calibrated
screen polygon**:

```text
+--------------------------------------+
|                                      |
|        +--------------------+        |
|        |                    |        |
|        |        IR pen ●    |        |
|        |                    |        |
|        +--------------------+        |
|                                      |
+--------------------------------------+
```

Modes:

| Mode | Exposure | Threshold | Use |
|------|----------|-----------|-----|
| `visible` | auto | 200 | webcam + flashlight testing |
| `ir` | locked short exposure | 240 | IR pen tracking |
| `auto` | resolved by platform | - | `ir` on Raspberry Pi / ARM, `visible` elsewhere |

`auto` mode is resolved **once**, when the camera opens.

With a normal laptop/Pi webcam (IR-cut filter still fitted) this is a
bright-light detector: it cannot guarantee the light is infrared.
For a real IR pen the camera must be IR sensitive:

- Raspberry Pi **NoIR** camera, or
- a webcam with its IR-cut filter removed, plus an
  **IR-pass / visible-block filter** (720 / 850 / 950 nm)

## Coordinate conversion

```text
camera pixel (x, y)
    --3x3 homography-->  desktop pixel (X, Y)
```

The homography is fitted from the measured reference points. It is
valid at the capture size it was made at; `coordinate_mapper.py`
rescales it automatically if the camera later delivers another frame
size (same aspect ratio). If the desktop resolution changes, run the
calibration again.

## Raspberry Pi 3B notes

- Default capture size on ARM is **640x480** (both for calibration
  and detection - they must match). Override with `SMARTIR_RES`.
- The detection loop is grayscale + threshold + morphology only, with
  the morphology kernel built once and per-contour statistics
  computed inside the contour bounding box.
- `python3-picamera2` is strongly recommended: it is the only path
  that can lock exposure and gain for IR tracking.
- If only the projector is connected as a display, calibration still
  works: the point sweep prints its progress in the terminal and
  draws nothing but the reference point on the projector.

## Environment variables

```bash
SMARTIR_CAM=0               # camera index or stream URL
SMARTIR_RES=640x480         # capture size (calibration AND detection)
SMARTIR_FPS=30              # capture framerate
SMARTIR_EXPOSURE_US=8000    # IR exposure in microseconds (picamera2)
SMARTIR_GAIN=2.0            # IR analogue gain (picamera2)
SMARTIR_V4L2_EXPOSURE=10    # IR exposure value for V4L2 drivers
```

Example (IR pen on a Pi with the camera on the ribbon port):

```bash
SMARTIR_RES=640x480 python3 main.py
```

## Whiteboard controls

```text
C       clear canvas
E       toggle pen / eraser
S       save whiteboard.png
D       toggle camera debug view
Q / ESC quit
```

No detected light = pen lifted (a new stroke starts on the next
detection).

## Troubleshooting calibration

- `Detected screen area: x % of the frame` (step 1) should be a few
  percent to ~50 %. Near 0 % or ~100 % means the color phase failed -
  check that the projector is actually visible to the camera.
- Low point-sweep hit rate: darken the room, check that the camera
  really sees the projector, and verify you are not in `ir` mode with
  an IR-filtered camera.
- High reprojection error: something moved between captures, or the
  projector and desktop resolutions do not match.
- Detection too noisy: raise the threshold in
  `light_detection.py` (`IR_THRESHOLD` / `VISIBLE_THRESHOLD`).
