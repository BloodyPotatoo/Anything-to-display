# SMARTIR - Wii Remote Style Digital Whiteboard

This project imitates Johnny Chung Lee's Wii Remote interactive
whiteboard: a camera tracks a bright light pen on a projected screen,
a homography converts camera coordinates into desktop coordinates,
and the desktop coordinates are drawn onto a canvas.

It runs on a PC (webcam testing) and on a Raspberry Pi with a camera
on the default ribbon (CSI) port.

```text
CALIBRATION
  STEP 1 (auto): RED / GREEN / BLUE / WHITE color phase -> screen mask
                 + 4 calibrated screen corners
  STEP 2 (auto): 9 reference points projected one at a time
                 -> pen measured automatically -> RANSAC homography
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
| `calibration.py` | screen detection (RGBW) + point-grid homography calibration (fully automatic) |
| `coordinate_mapper.py` | camera -> desktop homography, screen-size detection |
| `light_detection.py` | bright/IR light detector (standalone test view) |
| `whiteboard.py` | the drawing application (canvas content is always shown) |
| `screen_corners.npy` | generated: screen polygon in camera space |
| `calibration.npz` | generated: homography, points, frame size, screen size |

## Install

PC:

```bash
pip install opencv-python numpy picamera2
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
python3 calibration.py            # calibration only (fully automatic)
python3 calibration.py visible    # calibration in visible-light mode
python3 light_detection.py auto   # detector test view only
python3 run_whiteboard.py         # whiteboard only (needs calibration)
```

## Calibration procedure

Calibrate has two steps and runs **completely automatically** - no key
presses, no ENTER prompts, no manual colour changes.

**STEP 1 - RGBW screen detection (automatic)**

The projector cycles RED → GREEN → BLUE → WHITE. The detector captures
each phase and keeps only the pixels whose Lab colour changes strongly
across the four captures. Those pixels are the projected screen, so
they produce the search mask and the four calibrated screen corners.

**STEP 2 - point-grid sweep (automatic)**

The projector shows 9 reference points, one at a time, at known
desktop coordinates. On each point the pen is auto-detected (the
system waits for the light to be present for `IR_CAPTURE_DURATION`
seconds). The 9 camera/desktop pairs are fitted with
`cv2.findHomography(..., RANSAC)` and the program prints a real
accuracy report:

```text
Calibration accuracy (desktop pixels):
  points measured : 9 / 9
  mean error      : 1.83 px
  max  error      : 3.41 px
  mean / diagonal : 0.098 %
```

A point that the pen never reaches is simply skipped - it is never
written into the homography, so a missed point cannot corrupt the
calibration.

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
| `visible` | auto exposure | adaptive | webcam + flashlight testing |
| `ir` | locked exposure (shutter) | adaptive | IR pen tracking |
| `auto` | resolved by platform | - | `ir` on Raspberry Pi / ARM, `visible` elsewhere |

### Why IR light is not detected on a low-power sensor

IR light is invisible to the human eye, so the camera can only see it
because the sensor is fed *light* to collect. The exposure (shutter
speed) is what catches IR photons:

- **Short shutter** = the sensor gets very little light, so a dim IR
  pen is invisible (frame looks black).
- **Long shutter** = the sensor accumulates more light, the pen blob
  is brighter, and the detector can find it.

SMARTIR therefore runs the IR mode with a **long exposure** at startup:

```text
SMARTIR_EXPOSURE_US (picamera2)  -> default 40000 us (40 ms)
SMARTIR_V4L2_EXPOSURE (V4L2)     -> default 16   (manual exposure value)
SMARTIR_GAIN                     -> default 2.0  (analogue gain)
```

You can override them: `SMARTIR_EXPOSURE_US=80000 python main.py` for a
very long shutter. The detection threshold in `calibration.py` is also
**adaptive**: it is a fraction of the frame's own brightness, so a
dim pen at a high exposure is matched instead of a hard 95/240 floor.

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
SMARTIR_EXPOSURE_US=40000   # IR exposure in microseconds (picamera2)
SMARTIR_GAIN=2.0            # IR analogue gain (picamera2)
SMARTIR_V4L2_EXPOSURE=16    # IR exposure value for V4L2 drivers
```

These are the **high (long) values** that catch IR light on a low-power
sensor. The defaults are already the good values; override only if the
pen is still too dim (e.g. `SMARTIR_EXPOSURE_US=80000` for an even
longer shutter).

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

The canvas content is always drawn to the main window - strokes are
never lost in the status bar. No detected light = pen lifted (a new
stroke starts on the next detection).

## Troubleshooting calibration

- STEP 1 (RGBW) should report a screen area of a few percent to ~50 %.
  Near 0 % or ~100 % means the colour phase failed - check that the
  projector is actually visible to the camera.
- Low point-sweep hit rate: darken the room, check that the camera
  really sees the projector, and verify you are not in `ir` mode with
  an IR-filtered camera.
- High reprojection error: something moved between captures, or the
  projector and desktop resolutions do not match.
- Detection too noisy: raise the threshold in
  `light_detection.py` (`IR_THRESHOLD` / `VISIBLE_THRESHOLD`).

The PiCameraStream class in `camera.py` calls `self._camera.capture_array("main")`
inside `read()` (not `self._camera.close()`, which would stop the
camera). `release()` is correct and stops the camera properly.
