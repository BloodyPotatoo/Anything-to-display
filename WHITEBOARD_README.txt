SMARTIR WHITEBOARD ADD-ON

New files:
  whiteboard.py
  run_whiteboard.py

These files use the existing:
  screen_corners.npy
  coordinate_mapper.py
  light_detection.py

No existing project file is required to be replaced.

Run:
  python run_whiteboard.py

Controls:
  C = clear
  S = save
  E = pen/eraser
  Q or ESC = quit

Important:
The current light_detection.py is still brightness-based. The
"ir" selection here does not magically make a normal webcam see IR.
For a true IR pen, the camera must expose IR or use an IR-capable
camera/module. Once the hardware camera is available, the detector
can be changed without changing the drawing/coordinate pipeline.
