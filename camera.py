import glob
import os
import platform
import sys
import threading
import time

import cv2

# ============================================================
# SMARTIR - SHARED CAMERA LAYER
#
# Everything that OPENS the camera lives here so calibration.py
# and light_detection.py behave identically:
#   - same resolution (required, otherwise the saved calibration
#     coordinates would not match the detection coordinates)
#   - same "visible" / "ir" / "auto" mode handling
#   - same exposure control
#
# Camera sources tried, in order:
#   1. SMARTIR_CAM environment variable (index or stream URL)
#   2. Raspberry Pi CSI ribbon camera via picamera2 (libcamera)
#   3. libcamerasrc GStreamer pipeline
#   4. OpenCV backend scan (V4L2 / DirectShow / AVFoundation)
#
# Environment variables:
#   SMARTIR_CAM           camera index or URL, e.g. "0" or rtsp://...
#   SMARTIR_RES           capture size, e.g. "640x480"
#   SMARTIR_FPS           capture framerate, e.g. "30"
#   SMARTIR_EXPOSURE_US   IR mode exposure in microseconds (picamera2)
#   SMARTIR_GAIN          IR mode analogue gain (picamera2)
#   SMARTIR_V4L2_EXPOSURE IR mode exposure value (V4L2 drivers)
# ============================================================

MAX_CAMERA_INDEX = 10

DEFAULT_PC_WIDTH = 1280
DEFAULT_PC_HEIGHT = 720

# Pi 3B has 4x A53 cores and 1 GB RAM. 640x480 keeps the whole
# detection loop (grayscale + threshold + morphology + contours)
# comfortably inside a 30 fps budget.
DEFAULT_PI_WIDTH = 640
DEFAULT_PI_HEIGHT = 480

DEFAULT_FPS = 30

# ------------------------------------------------------------
# Camera preview window
#
# Show what the camera actually sees while calibrating / tracking.
# The preview is only a view - detection uses the full-resolution
# frame, never the downscaled preview.
#
# Environment variables:
#   SMARTIR_PREVIEW        1 = show the preview, 0 = hide (default: 1)
#   SMARTIR_PREVIEW_RES    WIDTHxHEIGHT of the preview window
#                          (default: 480x360, capped to the capture size)


# ------------------------------------------------------------
# IR-mode settings (Raspberry Pi NoIR / IR-sensitive cameras).
#
# EXPOSURE (shutter speed):
#   The camera's shutter (exposure) is what CATCHES IR light.
#   - A short exposure only works when the sensor is a real NoIR /
#     IR-sensitive one. With a normal webcam (IR-cut filter still
#     fitted) a short exposure makes the frame black and the pen
#     disappears, so on a normal camera we rely on auto exposure.
#   - SMARTIR_EXPOSURE_US: IR exposure in microseconds (picamera2).
#     HIGH = longer shutter = more light gathered = the pen shows
#     up on a weak / low-power sensor or a far-away pen.
#   - SMARTIR_V4L2_EXPOSURE: IR exposure value (V4L2 drivers)
#     HIGH = longer shutter for the same reason.
#
#   Defaults are HIGH now so the pen is always visible:
#     picamera2  -> 40000 us  (40 ms)
#     V4L2       -> 16         (manual exposure value)
#
#   For a real NoIR camera this is stable and catches the IR pen
#   even in a normal room, without any IR-pass filter.
# ------------------------------------------------------------
IR_EXPOSURE_US = int(os.environ.get("SMARTIR_EXPOSURE_US", "40000"))
IR_ANALOGUE_GAIN = float(os.environ.get("SMARTIR_GAIN", "2.0"))
IR_V4L2_EXPOSURE = int(os.environ.get("SMARTIR_V4L2_EXPOSURE", "16"))

# ------------------------------------------------------------
# Startup attempt timeouts (seconds).
#
# Opening a camera source can block indefinitely: on a desktop
# Linux box with GStreamer present, the libcamerasrc pipeline
# stalls for 30+ seconds; some /dev/video nodes block on read()
# forever. Every candidate therefore runs with a wall-clock
# budget so one bad source cannot hang program startup.
# ------------------------------------------------------------
PICAMERA2_ATTEMPT_TIMEOUT = 20
GSTREAMER_ATTEMPT_TIMEOUT = 12
V4L2_ATTEMPT_TIMEOUT = 12


def attempt_with_timeout(description, factory, timeout_seconds):
    """
    Run a camera-source attempt in a worker thread with a time
    budget.

    Returns the factory's value, or None if it failed or took
    longer than timeout_seconds. A timed-out attempt is abandoned
    (its daemon thread may stay blocked in the driver; nothing
    else uses it).
    """
    result = {}

    def worker():
        try:
            result["value"] = factory()
        except Exception as error:
            result["error"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print(
            f"TIMED OUT after {timeout_seconds}s: {description}\n"
            "        Skipping this source and trying the next one."
        )
        return None

    if "error" in result:
        print(f"{description} failed: {result['error']}")
        return None

    return result.get("value")


def display_available():
    """
    Best-effort check that a graphical session exists.

    OpenCV's Qt backend calls abort() when it cannot connect to
    the display, which kills the process with no Python
    traceback - so we detect the situation first and print real
    instructions instead.
    """
    # A headless Qt platform (offscreen/minimal) needs no session
    # at all - only those are trusted blindly. A fallback list
    # like "wayland;xcb" still needs a real session, so it falls
    # through to the checks below.
    qt_platform = os.environ.get("QT_QPA_PLATFORM", "")

    headless_platforms = {"offscreen", "minimal", "minimalegl"}

    if any(
        part.strip() in headless_platforms
        for part in qt_platform.split(";")
        if part.strip()
    ):
        return True

    if platform.system() != "Linux":
        return True

    # Wayland (Raspberry Pi OS Bookworm and most modern desktops).
    wayland = os.environ.get("WAYLAND_DISPLAY")

    if wayland:
        runtime = os.environ.get(
            "XDG_RUNTIME_DIR",
            f"/run/user/{os.getuid()}",
        )

        if os.path.exists(os.path.join(runtime, wayland)):
            return True

    # X11.
    display = os.environ.get("DISPLAY")

    if display:
        if display.startswith(":"):
            # Local display such as ":0" / ":0.0" needs its socket.
            # A stale DISPLAY with no X server = Qt abort.
            number = display[1:].split(".")[0]

            if os.path.exists(f"/tmp/.X11-unix/X{number}"):
                return True
        else:
            # Remote/forwarded display (ssh -X): trust it.
            return True

    return False


def require_display():
    """
    Exit cleanly with setup instructions when no graphical
    session is available. Called before any window is created.
    """
    if display_available():
        return

    print("\n" + "=" * 66)
    print("NO GRAPHICAL DISPLAY - CANNOT OPEN THE PROJECTOR WINDOW")
    print("=" * 66)
    print(
        "DISPLAY / WAYLAND_DISPLAY is not set or does not point to\n"
        "a running session. SMARTIR needs a graphical session for:\n"
        "  - projecting the calibration colors and points\n"
        "  - the whiteboard / detector windows\n"
        "\n"
        "Fix (pick one):\n"
        "  1. Run from a terminal ON the Pi's desktop (simplest).\n"
        "  2. X session, running over SSH:\n"
        "         export DISPLAY=:0\n"
        "  3. Wayland session (Raspberry Pi OS Bookworm), over SSH:\n"
        "         export XDG_RUNTIME_DIR=/run/user/$(id -u)\n"
        "         export WAYLAND_DISPLAY=wayland-0\n"
        "  4. SSH with X forwarding:  ssh -X pi@<pi-address>\n"
        "\n"
        "Tip: add the export lines to ~/.bashrc so SSH sessions get\n"
        "them automatically."
    )
    print("=" * 66)
    sys.exit(1)


def is_raspberry_pi():
    """True on Raspberry Pi boards (and other small ARM Linux boards)."""
    machine = platform.machine().lower()
    is_arm = machine.startswith("arm") or machine.startswith("aarch64")

    try:
        with open("/proc/device-tree/model", "rb") as handle:
            model = handle.read().decode("utf-8", "ignore").lower()

        if "raspberry pi" in model:
            return True
    except OSError:
        pass

    return is_arm and platform.system() == "Linux"


def _resolution_from_env():
    raw = os.environ.get("SMARTIR_RES", "").strip().lower()

    if not raw:
        return None

    try:
        width_text, height_text = raw.split("x")
        width = int(width_text)
        height = int(height_text)

        if width >= 160 and height >= 120:
            return width, height

        raise ValueError("resolution too small")

    except ValueError:
        print(
            f"WARN: ignoring SMARTIR_RES={raw} "
            "(expected WIDTHxHEIGHT, e.g. 640x480)"
        )
        return None


def _default_resolution():
    custom = _resolution_from_env()

    if custom is not None:
        return custom

    if is_raspberry_pi():
        return DEFAULT_PI_WIDTH, DEFAULT_PI_HEIGHT

    return DEFAULT_PC_WIDTH, DEFAULT_PC_HEIGHT


CAMERA_WIDTH, CAMERA_HEIGHT = _default_resolution()

try:
    CAMERA_FPS = int(os.environ.get("SMARTIR_FPS", str(DEFAULT_FPS)))
except ValueError:
    CAMERA_FPS = DEFAULT_FPS


def _resolution_from_env(name, default):
    """Parse a WIDTHxHEIGHT env var, returning the default if unset/invalid."""
    raw = os.environ.get(name, "").strip().lower()

    if not raw:
        return default

    try:
        width_text, height_text = raw.split("x")
        width = int(width_text)
        height = int(height_text)

        if width >= 160 and height >= 120:
            return width, height

        raise ValueError("resolution too small")
    except ValueError:
        print(
            f"WARN: ignoring {name}={raw} "
            f"(expected WIDTHxHEIGHT, e.g. 640x480); using {default[0]}x{default[1]}"
        )
        return default


_PREVIEW_ENABLED = os.environ.get("SMARTIR_PREVIEW", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

if _PREVIEW_ENABLED:
    _PREVIEW_RES = _resolution_from_env("SMARTIR_PREVIEW_RES", (480, 360))
    _PREVIEW_WIDTH, _PREVIEW_HEIGHT = _PREVIEW_RES
else:
    _PREVIEW_WIDTH = _PREVIEW_HEIGHT = None


def preview_enabled():
    """True when the camera preview window is active."""
    return _PREVIEW_ENABLED


def preview_size():
    """The preview window size (width, height), or None when disabled."""
    if not _PREVIEW_ENABLED:
        return None
    return (_PREVIEW_WIDTH, _PREVIEW_HEIGHT)


def resolve_mode(mode="auto"):
    """
    Map a requested mode onto a concrete one.

    This function is PURE (it never prints), because it is called
    once per frame by the detector.

        auto -> "ir"      on Raspberry Pi / ARM Linux
        auto -> "visible" everywhere else
        anything unknown -> "visible"
    """
    if mode == "auto":
        if is_raspberry_pi():
            return "ir"

        return "visible"

    if mode in ("visible", "ir"):
        return mode

    return "visible"


def get_backends():
    system = platform.system()

    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]

    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION]

    if system == "Linux":
        return [cv2.CAP_V4L2, cv2.CAP_ANY]

    return [cv2.CAP_ANY]


def backend_name(backend):
    names = {
        cv2.CAP_ANY: "AUTO",
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_DSHOW: "DirectShow",
        cv2.CAP_AVFOUNDATION: "AVFoundation",
        cv2.CAP_GSTREAMER: "GStreamer",
    }

    return names.get(backend, str(backend))


def get_available_indices():
    """Find plausible camera indices based on OS devices."""
    if platform.system() == "Linux":
        devices = sorted(glob.glob("/dev/video*"))
        indices = []

        for device in devices:
            try:
                indices.append(
                    int(device.replace("/dev/video", ""))
                )
            except ValueError:
                pass

        return indices if indices else [0]

    return list(range(MAX_CAMERA_INDEX))


# ============================================================
# Raspberry Pi CSI camera (picamera2 / libcamera)
# ============================================================

class PiCameraStream:
    """
    cv2.VideoCapture-compatible wrapper around picamera2.

    The rest of SMARTIR only uses read(), isOpened(), release()
    and get(), so this class can be dropped in wherever a
    VideoCapture is expected.

    This is the path used for a camera on the default ribbon
    port (CSI) on Raspberry Pi OS.
    """

    def __init__(self, width, height, fps, mode):
        from picamera2 import Picamera2

        self._camera = Picamera2()

        config = self._camera.create_video_configuration(
            main={
                "size": (int(width), int(height)),
                "format": "RGB888",
            },
            buffer_count=2,
            controls={"FrameRate": float(fps)},
        )

        self._camera.configure(config)

        if mode == "ir":
            controls = {
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": int(IR_EXPOSURE_US),
                "AnalogueGain": float(IR_ANALOGUE_GAIN),
            }
        else:
            controls = {
                "AeEnable": True,
                "AwbEnable": True,
            }

        try:
            self._camera.set_controls(controls)
        except Exception as error:
            print(f"WARN: could not apply camera controls: {error}")

        self._camera.start()

        # Let the sensor settle so the first frame is not black.
        time.sleep(0.4)

        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)

        # The preview window only downscales what the camera sees;
        # the detection loop works on the full-resolution frame.
        self._opened = True
        self._preview_window = None

    def isOpened(self):
        return self._opened

    def read(self):
        if not self.isOpened():
            return False, None

        try:
            frame = self._camera.capture_array("main")
        except Exception as error:
            print(f"Pi camera frame error: {error}")
            return False, None

        if frame is None or frame.size == 0:
            return False, None

        if frame.ndim == 2:
            return True, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Requested format is RGB888, OpenCV wants BGR.
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        if not self._opened:
            return

        self._opened = False

        try:
            self._camera.stop()
        except Exception:
            pass

        try:
            self._camera.close()
        except Exception:
            pass

    def set(self, prop, value):
        # The pipeline is fixed at configure time (size/fps/exposure
        # are applied in __init__), mirroring what SMARTIR needs.
        return False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)

        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)

        if prop == cv2.CAP_PROP_FPS:
            return self._fps

        return 0.0


def create_preview_window(window_name, width, height):
    """
    Create a small live preview window.

    The preview is only a view of the camera; detection works on the
    full-resolution frame, so the preview never limits detection.
    """
    if width <= 0 or height <= 0:
        return None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    return window_name


def draw_preview(window_name, frame, width, height):
    """
    Draw the full-resolution frame into a downscaled preview window.

    Only runs between frames; the preview is not part of the
    detection pipeline.
    """
    if not _PREVIEW_ENABLED:
        return

    try:
        preview = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, preview)
    except Exception:
        pass


def _try_open_picamera2(resolved_mode):
    """Open the CSI ribbon camera through picamera2, if available."""
    try:
        from picamera2 import Picamera2  # noqa: F401
    except Exception:
        return None

    try:
        stream = PiCameraStream(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            CAMERA_FPS,
            resolved_mode,
        )

        print(
            "Camera selected: Raspberry Pi CSI (picamera2/libcamera)\n"
            f"Resolution: {CAMERA_WIDTH} x {CAMERA_HEIGHT}  "
            f"FPS: {CAMERA_FPS}  Mode: {resolved_mode}"
        )

        return stream

    except Exception as error:
        print(f"picamera2 could not open the CSI camera: {error}")
        return None


# ============================================================
# libcamera GStreamer fallback (Pi without python3-picamera2)
# ============================================================

def _try_libcamera_gstreamer(resolved_mode):
    if platform.system() != "Linux":
        return None

    if resolved_mode == "ir":
        print(
            "NOTE: IR exposure cannot be locked through the "
            "libcamerasrc pipeline. Install python3-picamera2 "
            "for correct IR behaviour."
        )

    # The libcamerasrc (libcamera) stack on Raspberry Pi reports
    # resolutions differently from OpenCV: asking for the full
    # CAMERA_WIDTH/CAMERA_HEIGHT often fails, and a successful read
    # arrives at an unsupported aspect ratio. Try the configured
    # resolution first, then a smaller safe value. The camera is
    # re-validated after opening, so a fallback is reported instead
    # of silently passing a mismatched frame.
    pipelines = [
        f"libcamerasrc ! video/x-raw, width={CAMERA_WIDTH}, "
        f"height={CAMERA_HEIGHT}, framerate={CAMERA_FPS}/1 ! "
        "videoconvert ! appsink drop=true",
        "libcamerasrc ! video/x-raw, width=640, height=480 ! "
        "videoconvert ! appsink drop=true",
    ]

    for pipeline in pipelines:
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if cap.isOpened():
                time.sleep(0.2)
                ret, frame = cap.read()

                if ret and frame is not None and frame.size:
                    print("Camera selected: libcamerasrc GStreamer pipeline")
                    return cap

                cap.release()

        except Exception:
            pass

    return None


# ============================================================
# OpenCV / V4L2 paths
# ============================================================

def _apply_exposure(cap, resolved_mode):
    """Lock (ir) or enable (visible) exposure on a VideoCapture."""
    if resolved_mode == "ir":
        try:
            # V4L2 convention: 1 = manual mode.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, IR_V4L2_EXPOSURE)
            print(
                "Applied manual low exposure "
                f"(value {IR_V4L2_EXPOSURE}) for IR tracking."
            )
        except Exception as error:
            print(f"WARN: could not lock exposure: {error}")
    else:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        except Exception:
            pass


def _configure_and_validate(cap, resolved_mode):
    """Set size/fps/exposure and confirm real frames arrive."""
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    except Exception:
        pass

    _apply_exposure(cap, resolved_mode)

    time.sleep(0.15)

    for _ in range(5):
        ret, frame = cap.read()

        if ret and frame is not None and frame.size:
            if frame.ndim == 3 and frame.shape[2] == 3:
                return True

        time.sleep(0.05)

    # Retry with a lower resolution (common on Pi / UVC cameras).
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.1)

        _apply_exposure(cap, resolved_mode)

        ret, frame = cap.read()

        if ret and frame is not None and frame.size:
            print(
                "WARN: requested resolution unavailable, "
                "fell back to 640x480. Recalibrate after this."
            )
            return True

    except Exception:
        pass

    return False


def _try_open_v4l2(index_or_source, resolved_mode, backend=None):
    if backend is None:
        backends = get_backends()
    else:
        backends = [backend]

    for current_backend in backends:
        print(
            f"Trying camera {index_or_source} using "
            f"{backend_name(current_backend)}..."
        )

        try:
            cap = cv2.VideoCapture(index_or_source, current_backend)
        except Exception:
            continue

        if not cap.isOpened():
            cap.release()
            continue

        if _configure_and_validate(cap, resolved_mode):
            print(f"Camera selected: index {index_or_source}")
            return cap

        cap.release()

    return None


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def open_camera(mode="visible"):
    """
    Open the best available camera for SMARTIR.

    Returns (capture, resolved_mode).

    capture is either a cv2.VideoCapture or a PiCameraStream;
    both provide read()/isOpened()/release()/get().
    """
    resolved_mode = resolve_mode(mode)

    print("\n" + "=" * 60)
    print("SMARTIR CAMERA SETUP")
    print("=" * 60)
    print(f"Mode        : {resolved_mode}")
    print(f"Resolution  : {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    print(f"Framerate   : {CAMERA_FPS}")
    print(f"Board       : {'Raspberry Pi / ARM' if is_raspberry_pi() else platform.system()}")

    if resolved_mode == "ir":
        print(
            f"IR settings : exposure {IR_EXPOSURE_US} us, "
            f"gain {IR_ANALOGUE_GAIN} (picamera2) / "
            f"V4L2 exposure {IR_V4L2_EXPOSURE}"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # 1. Explicit override
    # --------------------------------------------------------
    custom_cam = os.environ.get("SMARTIR_CAM")

    if custom_cam:
        print(f"Using custom camera source from SMARTIR_CAM: {custom_cam}")

        source = int(custom_cam) if custom_cam.isdigit() else custom_cam
        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY

        cap = attempt_with_timeout(
            f"SMARTIR_CAM source {custom_cam}",
            lambda: _try_open_v4l2(source, resolved_mode, backend),
            V4L2_ATTEMPT_TIMEOUT,
        )

        if cap is not None:
            return cap, resolved_mode

        print("WARN: could not open SMARTIR_CAM source. Falling back.")

    # --------------------------------------------------------
    # 2. Raspberry Pi CSI ribbon camera
    # --------------------------------------------------------
    if platform.system() == "Linux":
        stream = attempt_with_timeout(
            "picamera2 CSI camera",
            lambda: _try_open_picamera2(resolved_mode),
            PICAMERA2_ATTEMPT_TIMEOUT,
        )

        if stream is not None:
            # Start a small live preview of what the camera sees.
            if _PREVIEW_ENABLED:
                _PREVIEW_WINDOW = create_preview_window(
                    "SMARTIR Camera Preview",
                    _PREVIEW_WIDTH,
                    _PREVIEW_HEIGHT,
                )

            return stream, resolved_mode

        if is_raspberry_pi():
            print(
                "\nHINT: the ribbon-port camera was not opened through "
                "picamera2.\n"
                "      Install it with:  sudo apt install python3-picamera2\n"
                "      and make sure the camera is enabled with "
                "sudo raspi-config.\n"
            )

            # libcamerasrc is only meaningful on a board that has a
            # libcamera stack; on desktop Linux it just stalls the
            # startup for no benefit.
            print("Trying Raspberry Pi libcamera GStreamer pipeline...")

            stream = attempt_with_timeout(
                "libcamerasrc GStreamer pipeline",
                lambda: _try_libcamera_gstreamer(resolved_mode),
                GSTREAMER_ATTEMPT_TIMEOUT,
            )

            if stream is not None:
                return stream, resolved_mode

    # --------------------------------------------------------
    # 3. Generic device scan
    # --------------------------------------------------------
    for index in get_available_indices():
        cap = attempt_with_timeout(
            f"camera index {index}",
            lambda index=index: _try_open_v4l2(index, resolved_mode),
            V4L2_ATTEMPT_TIMEOUT,
        )

        if cap is not None:
            # Start a small live preview of what the camera sees.
            if _PREVIEW_ENABLED:
                _PREVIEW_WINDOW = create_preview_window(
                    "SMARTIR Camera Preview",
                    _PREVIEW_WIDTH,
                    _PREVIEW_HEIGHT,
                )

            return cap, resolved_mode

    # --------------------------------------------------------
    # 4. Show the preview alongside the camera
    # --------------------------------------------------------
    if _PREVIEW_ENABLED:
        # Only create the preview when the camera actually opened.
        # The del above is a no-op when the camera failed: there was
        # no _PREVIEW_WINDOW to delete, so skip the delete entirely.
        if "PREVIEW_WINDOW" in globals():
            del _PREVIEW_WINDOW
        _PREVIEW_WINDOW = create_preview_window(
            "SMARTIR Camera Preview",
            _PREVIEW_WIDTH,
            _PREVIEW_HEIGHT,
        )

    raise RuntimeError(
        "No usable camera found.\n"
        "  - Ribbon (CSI) camera: sudo apt install python3-picamera2, "
        "then enable the camera in sudo raspi-config.\n"
        "  - USB camera: check that /dev/video* exists (ls /dev/video*).\n"
        "  - Or set the source explicitly: SMARTIR_CAM=0 python main.py"
    )


# ============================================================
# LIVE PREVIEW
#
# The preview window shows what the camera actually sees. It is only
# a view: detection always works on the full-resolution frame, so
# the preview never slows down or lowers detection quality.
# ============================================================

_PREVIEW_WINDOW = None


def preview_window():
    """Return the preview window name, or None when hidden."""
    return _PREVIEW_WINDOW


def cleanup_preview():
    """Close the preview window when the app exits."""
    global _PREVIEW_WINDOW

    if _PREVIEW_WINDOW is not None:
        cv2.destroyWindow(_PREVIEW_WINDOW)
        _PREVIEW_WINDOW = None
