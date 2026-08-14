# SMARTIR - Wii Remote Style Digital Whiteboard

This project imitates the basic idea behind Johnny Chung Lee's Wii Remote
interactive whiteboard, but uses your own camera/light hardware.

The current software has three main stages:

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

## Files

- `main.py` - runs the complete system.
- `calibration.py` - finds the projected screen using RGBW intersection.
- `coordinate_mapper.py` - converts camera coordinates to desktop coordinates.
- `light_detection.py` - detects IR/bright light only inside the calibrated screen.
- `screen_corners.npy` - generated after successful calibration.

## Install

```bash
pip install opencv-python numpy
```

On some Linux systems, tkinter may also be needed for desktop-size detection.

## Run

Run only:

```bash
python main.py
```

The program will ask whether to calibrate.

## Calibration procedure

The projector displays:

1. RED
2. GREEN
3. BLUE
4. WHITE

Press ENTER in the terminal after each color is correctly projected.

The program creates four masks.

The important operation is a TOLERANT intersection:

```text
RED confidence
      AND
GREEN confidence
      AND
BLUE confidence
      AND
WHITE confidence
```

Each pixel gets a confidence value from 0..255. The weakest of the
four confidence values is used. A pixel is kept when that weakest
confidence is above `INTERSECTION_THRESHOLD`.

This means the camera/projector does NOT need to produce exact RGB
values. It only needs to be sufficiently similar to the expected
color.

That intersection is used to find the four screen corners.

The corners are saved as:

```text
screen_corners.npy
```

## Detection

After calibration, the camera continuously searches for a bright point.

The search is restricted to the calibrated screen polygon.

For example:

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

Light outside the calibrated screen is ignored.

With a normal laptop webcam, the detector behaves as a bright-light detector.
It cannot guarantee that a bright pixel is infrared.

With an IR-sensitive camera, the same pipeline can be used for the real IR pen.

## Coordinate conversion

Suppose the camera sees:

```text
camera = (734, 421)
```

The homography transforms it into:

```text
screen = (x, y)
```

The transformation is calculated from the four detected screen corners.

## Current hardware-testing mode

You can test the complete software using:

- laptop webcam
- projected/displayed RGB colors
- phone flashlight or other bright light
- IR LED later, when the real hardware is available

The light detector is deliberately not tied to an IR-only camera yet.


## Calibration tolerance

The main tuning values are in `calibration.py`:

```python
COLOR_DIFFERENCE_THRESHOLD = 12
COLOR_DOMINANCE_THRESHOLD = 8
INTERSECTION_THRESHOLD = 100
```

If calibration is still too strict, try:

```python
INTERSECTION_THRESHOLD = 70
```

or:

```python
INTERSECTION_THRESHOLD = 50
```

If the screen mask starts detecting too much background, increase the
threshold again.

The system still requires the RGBW intersection; it is simply using
thresholded confidence values instead of exact pixel matches.


## Important: spatial tolerance

The RGBW calibration does NOT require the four masks to line up
pixel-for-pixel.

Real cameras can change exposure/white balance between captures, and
the projected screen boundary can move by a few pixels. Each mask is
therefore expanded before the intersection.

Main setting:

```python
INTERSECTION_TOLERANCE_PIXELS = 25
```

For a difficult webcam/projector setup, try:

```python
INTERSECTION_TOLERANCE_PIXELS = 40
```

If the intersection becomes too large/noisy, reduce it to 15 or 10.

The program also prints the percentage of the frame detected by each
individual RED/GREEN/BLUE/WHITE mask. This makes it possible to tell
which stage is failing.


## Latest calibration method

The calibration now uses a more robust approach for laptop webcams:

1. Detect the dominant RGB channel instead of matching exact RGB values.
2. Use adaptive thresholds based on the captured frame.
3. Keep the largest coherent candidate region for each color.
4. Expand each candidate region by the spatial tolerance.
5. Intersect RED, GREEN, BLUE and WHITE.

For the current webcam-testing setup the default spatial tolerance is
100 pixels. Once the actual hardware/camera position is fixed, this can
be reduced.
