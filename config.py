"""Global configuration for the 3D black-hole ray tracer.

All geometric values are expressed in geometrised units (G = c = M = 1).
One Schwarzschild radius therefore equals ``RS_GEO = 2`` in code units.
Physical values shown in the UI are computed in :mod:`src.physics`.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Window / preview
# ---------------------------------------------------------------------------
PREVIEW_WIDTH: int = 320
PREVIEW_HEIGHT: int = 180
WINDOW_SCALE: int = 3
TARGET_FPS: int = 60

# ---------------------------------------------------------------------------
# High-resolution still render
# ---------------------------------------------------------------------------
RENDER_WIDTH: int = 1920
RENDER_HEIGHT: int = 1080

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
DEFAULT_DISTANCE_RS: float = 60.0
DEFAULT_INCLINATION_DEG: float = 18.0
DEFAULT_AZIMUTH_DEG: float = 25.0
DEFAULT_FOV_DEG: float = 72.0
MIN_DISTANCE_RS: float = 8.0
MAX_DISTANCE_RS: float = 250.0
MIN_INCLINATION_DEG: float = 1.0
MAX_INCLINATION_DEG: float = 89.0

# ---------------------------------------------------------------------------
# Black hole and accretion disk
# ---------------------------------------------------------------------------
DEFAULT_MASS_SOLAR: float = 10.0
DEFAULT_SPIN: float = 0.60
RS_GEO: float = 2.0                 # rs = 2GM/c^2 with M = 1
MASS_GEO: float = 1.0               # G = c = M = 1
DISK_INNER_RS: float = 3.0          # ISCO = 3 rs = 6 M
DISK_OUTER_RS: float = 20.0         # 20 rs
DISK_PEAK_TEMPERATURE_K: float = 5200.0   # visual colour temperature scale
DISK_GAIN: float = 5.0
DISK_TEXTURE_STRENGTH: float = 0.28

# ---------------------------------------------------------------------------
# Geodesic integration
# ---------------------------------------------------------------------------
MAX_STEPS: int = 900
STEP_FACTOR: float = 0.02           # dlambda ~ STEP_FACTOR * r
STEP_MIN: float = 0.01
STEP_MAX: float = 0.80
ESCAPE_RADIUS_WORLD: float = 90.0   # stop once an outgoing ray is this far out

# ---------------------------------------------------------------------------
# Starfield
# ---------------------------------------------------------------------------
STAR_COUNT: int = 2200
STAR_SEED: int = 20240607
STAR_RADIUS_PIXELS: float = 0.9

# ---------------------------------------------------------------------------
# Colours (RGB)
# ---------------------------------------------------------------------------
PANEL_BG: tuple[int, int, int, int] = (8, 10, 18, 210)
PANEL_BORDER: tuple[int, int, int] = (70, 90, 120)
TEXT_COLOR: tuple[int, int, int] = (232, 238, 248)
MUTED_COLOR: tuple[int, int, int] = (148, 164, 190)
ACCENT_COLOR: tuple[int, int, int] = (255, 174, 92)
ERROR_COLOR: tuple[int, int, int] = (255, 110, 110)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
IMAGE_DIR: Path = ROOT_DIR / "images"
