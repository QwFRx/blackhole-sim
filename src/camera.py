"""Orbiting 3D camera for the black-hole renderer.

The camera looks at the origin and is parametrised by its distance in units of
the Schwarzschild radius, an elevation above the equatorial (disk) plane, and
an azimuth.  Pixel rays are generated from a standard perspective basis with
the vertical field of view supplied by the user/configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class Camera:
    """A perspective camera orbiting a Schwarzschild black hole."""

    distance_rs: float = config.DEFAULT_DISTANCE_RS
    inclination_deg: float = config.DEFAULT_INCLINATION_DEG
    azimuth_deg: float = config.DEFAULT_AZIMUTH_DEG
    fov_deg: float = config.DEFAULT_FOV_DEG

    def clamp(self) -> None:
        """Keep all camera parameters inside their physically useful ranges."""

        self.distance_rs = min(
            max(self.distance_rs, config.MIN_DISTANCE_RS),
            config.MAX_DISTANCE_RS,
        )
        self.inclination_deg = min(
            max(self.inclination_deg, config.MIN_INCLINATION_DEG),
            config.MAX_INCLINATION_DEG,
        )
        self.azimuth_deg %= 360.0

    def position(self, rs: float = config.RS_GEO) -> np.ndarray:
        """Return the Cartesian camera position in code units.

        ``x = d cos(i) cos(phi)``, ``y = d cos(i) sin(phi)``, ``z = d sin(i)``.
        """

        distance = self.distance_rs * rs
        inc = math.radians(self.inclination_deg)
        az = math.radians(self.azimuth_deg)
        return np.array(
            [
                distance * math.cos(inc) * math.cos(az),
                distance * math.cos(inc) * math.sin(az),
                distance * math.sin(inc),
            ],
            dtype=np.float64,
        )

    def view_basis(
        self, rs: float = config.RS_GEO
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the orthonormal ``(forward, right, up)`` screen basis."""

        pos = self.position(rs)
        forward = -pos / np.linalg.norm(pos)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1.0e-9:
            # Looking almost exactly along the polar axis: choose a stable
            # reference direction to avoid a degenerate screen basis.
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return forward, right, up

    def view_parameters(
        self, rs: float = config.RS_GEO
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Return ``(forward, right, up, tan(fov/2), aspect)`` for the kernel."""

        forward, right, up = self.view_basis(rs)
        tan_half = math.tan(math.radians(self.fov_deg) * 0.5)
        return forward, right, up, tan_half, 1.0

    def ray_directions(self, width: int, height: int, rs: float = config.RS_GEO) -> np.ndarray:
        """Generate one normalised direction per pixel, shape ``(width*height, 3)``.

        Row zero is the top of the image and column zero is the left edge.
        """

        forward, right, up, tan_half, _ = self.view_parameters(rs)
        aspect = width / float(height)
        u = 2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0
        v = 1.0 - 2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height

        dirs = (
            forward[None, None, :]
            + right[None, None, :] * (u[None, :, None] * tan_half * aspect)
            + up[None, None, :] * (v[:, None, None] * tan_half)
        )
        dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
        return dirs.reshape(-1, 3)

    def rotate(self, dx_pixels: float, dy_pixels: float) -> None:
        """Rotate the camera azimuth/elevation from a mouse drag."""

        self.azimuth_deg -= dx_pixels * 0.25
        self.inclination_deg += dy_pixels * 0.20
        self.clamp()

    def adjust_inclination(self, delta_deg: float) -> None:
        """Change the elevation above the disk plane."""

        self.inclination_deg += delta_deg
        self.clamp()

    def zoom(self, wheel_steps: float) -> None:
        """Change the orbital distance exponentially with wheel input."""

        self.distance_rs *= math.exp(-wheel_steps * 0.12)
        self.clamp()
