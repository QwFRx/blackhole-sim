"""Relativistic backward ray tracer for a Schwarzschild black hole.

The renderer integrates the Cartesian null-geodesic equation

    d^2 x / dlambda^2 = -(3/2) rs h^2 x / |x|^5,   h = |x cross v|,

with an adaptive RK4 step.  Rays start at the camera, are traced backwards in
time, and are either captured by the horizon, cross the thin equatorial disk,
or escape to the procedural starfield.

Kerr spin is deliberately *not* integrated here: the exact Kerr metric is
axisymmetric but not spherically symmetric, so the simple Cartesian form above
does not apply.  Spin changes only the analytic values in the information
panel.  This is stated again in the README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config
from .camera import Camera
from .disk import disk_emission_scalar, disk_emission_vector
from .physics import blackbody_rgb_array

try:  # pragma: no cover - the fallback branch is exercised only without numba
    from numba import njit, prange

    NUMBA_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args: object, **kwargs: object):  # type: ignore[no-redef]
        """No-op numba decorator used by the pure-numpy fallback."""

        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range  # type: ignore[assignment]


@dataclass(frozen=True)
class StarField:
    """Deterministic procedural sky used behind the black hole."""

    directions: np.ndarray
    colours: np.ndarray
    band_normal: np.ndarray
    grid_size: int
    grid_offsets: np.ndarray
    grid_indices: np.ndarray


def make_starfield(count: int = config.STAR_COUNT, seed: int = config.STAR_SEED) -> StarField:
    """Create a seeded, isotropic starfield with blackbody colours.

    Star directions are drawn from a normal distribution and normalised, which
    is uniform on the sphere.  Brightness follows a steep power law so that a
    few stars dominate while the majority remain faint.
    """

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    temperatures = rng.uniform(2600.0, 13000.0, size=count)
    colours = blackbody_rgb_array(temperatures)
    brightness = 0.22 + 1.25 * rng.random(count) ** 3
    colours = np.clip(colours * brightness[:, None], 0.0, 1.5)

    # A faint great-circle band mimics a Milky-Way-like plane and makes the
    # gravitational distortion of the background easy to see.
    band_normal = np.array([0.36, 0.78, 0.51], dtype=np.float64)
    band_normal /= np.linalg.norm(band_normal)

    # Build a coarse 3D direction grid so the JIT kernel only tests stars in
    # the 27 cells surrounding an escaped ray.  This keeps full-HD renders
    # fast even with a few thousand stars.
    grid_size = 28
    cells = np.floor((directions + 1.0) * 0.5 * grid_size).astype(np.int64)
    np.clip(cells, 0, grid_size - 1, out=cells)
    flat = (cells[:, 0] * grid_size + cells[:, 1]) * grid_size + cells[:, 2]
    counts = np.bincount(flat, minlength=grid_size**3)
    grid_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    grid_indices = np.argsort(flat, kind="stable").astype(np.int64)

    return StarField(
        directions=np.ascontiguousarray(directions, dtype=np.float64),
        colours=np.ascontiguousarray(colours, dtype=np.float64),
        band_normal=band_normal,
        grid_size=grid_size,
        grid_offsets=grid_offsets,
        grid_indices=grid_indices,
    )


@dataclass(frozen=True)
class RenderSettings:
    """Geometric and visual parameters shared by both renderer backends."""

    rs: float = config.RS_GEO
    disk_inner_rs: float = config.DISK_INNER_RS
    disk_outer_rs: float = config.DISK_OUTER_RS
    disk_peak_temperature: float = config.DISK_PEAK_TEMPERATURE_K
    disk_gain: float = config.DISK_GAIN
    disk_texture_strength: float = config.DISK_TEXTURE_STRENGTH
    max_steps: int = config.MAX_STEPS

    @property
    def disk_inner(self) -> float:
        """Disk inner edge in code units (ISCO = 3 rs)."""

        return self.disk_inner_rs * self.rs

    @property
    def disk_outer(self) -> float:
        """Disk outer edge in code units (20 rs)."""

        return self.disk_outer_rs * self.rs

    @property
    def escape_radius(self) -> float:
        """Radius beyond which an outgoing ray is treated as escaped."""

        return max(config.ESCAPE_RADIUS_WORLD, self.disk_outer + 20.0)


@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def _trace_kernel(
    ox: float,
    oy: float,
    oz: float,
    fx: float,
    fy: float,
    fz: float,
    rx: float,
    ry: float,
    rz: float,
    ux: float,
    uy: float,
    uz: float,
    tan_half: float,
    width: int,
    height: int,
    rs: float,
    disk_inner: float,
    disk_outer: float,
    disk_tpeak: float,
    disk_gain: float,
    disk_texture_strength: float,
    phase: float,
    star_dirs: np.ndarray,
    star_cols: np.ndarray,
    star_grid_size: int,
    star_grid_offsets: np.ndarray,
    star_grid_indices: np.ndarray,
    star_radius: float,
    max_steps: int,
    escape_radius: float,
    band_x: float,
    band_y: float,
    band_z: float,
    out: np.ndarray,
) -> None:
    """Render all pixels with numba's parallel scalar loop.

    Every pixel owns its own geodesic state; there are no shared mutable
    quantities except the output image, so ``prange`` is race-free.
    """

    aspect = width / float(height)
    cos_star_radius = math.cos(star_radius)

    for iy in prange(height):
        v_coord = 1.0 - 2.0 * (iy + 0.5) / height
        for ix in range(width):
            u_coord = (2.0 * (ix + 0.5) / width - 1.0) * aspect

            dx = fx + rx * u_coord * tan_half + ux * v_coord * tan_half
            dy = fy + ry * u_coord * tan_half + uy * v_coord * tan_half
            dz = fz + rz * u_coord * tan_half + uz * v_coord * tan_half
            inv_dir = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            vx = dx * inv_dir
            vy = dy * inv_dir
            vz = dz * inv_dir

            px = ox
            py = oy
            pz = oz

            # Conserved angular momentum squared h^2 = |x cross v|^2.
            hx = py * vz - pz * vy
            hy = pz * vx - px * vz
            hz = px * vy - py * vx
            h2 = hx * hx + hy * hy + hz * hz

            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            escaped = False

            for _ in range(max_steps):
                r2 = px * px + py * py + pz * pz
                r = math.sqrt(r2)
                if r <= rs:
                    break
                radial_dot = px * vx + py * vy + pz * vz
                if r > escape_radius and radial_dot > 0.0:
                    escaped = True
                    break

                # Adaptive affine step: smaller near the horizon and photon
                # sphere, larger in the asymptotically flat region.
                dl = 0.02 * r + 0.005
                if dl > 0.80:
                    dl = 0.80
                elif dl < 0.01:
                    dl = 0.01

                old_x = px
                old_y = py
                old_z = pz

                # ---- RK4 stage 1 ----
                fac = -1.5 * rs * h2 / (r2 * r2 * r)
                a1x = fac * px
                a1y = fac * py
                a1z = fac * pz

                # ---- RK4 stage 2 ----
                x2 = px + 0.5 * dl * vx
                y2 = py + 0.5 * dl * vy
                z2 = pz + 0.5 * dl * vz
                vx2 = vx + 0.5 * dl * a1x
                vy2 = vy + 0.5 * dl * a1y
                vz2 = vz + 0.5 * dl * a1z
                r2_2 = x2 * x2 + y2 * y2 + z2 * z2
                r_2 = math.sqrt(r2_2)
                fac2 = -1.5 * rs * h2 / (r2_2 * r2_2 * r_2)
                a2x = fac2 * x2
                a2y = fac2 * y2
                a2z = fac2 * z2

                # ---- RK4 stage 3 ----
                x3 = px + 0.5 * dl * vx2
                y3 = py + 0.5 * dl * vy2
                z3 = pz + 0.5 * dl * vz2
                vx3 = vx + 0.5 * dl * a2x
                vy3 = vy + 0.5 * dl * a2y
                vz3 = vz + 0.5 * dl * a2z
                r2_3 = x3 * x3 + y3 * y3 + z3 * z3
                r_3 = math.sqrt(r2_3)
                fac3 = -1.5 * rs * h2 / (r2_3 * r2_3 * r_3)
                a3x = fac3 * x3
                a3y = fac3 * y3
                a3z = fac3 * z3

                # ---- RK4 stage 4 ----
                x4 = px + dl * vx3
                y4 = py + dl * vy3
                z4 = pz + dl * vz3
                vx4 = vx + dl * a3x
                vy4 = vy + dl * a3y
                vz4 = vz + dl * a3z
                r2_4 = x4 * x4 + y4 * y4 + z4 * z4
                r_4 = math.sqrt(r2_4)
                fac4 = -1.5 * rs * h2 / (r2_4 * r2_4 * r_4)
                a4x = fac4 * x4
                a4y = fac4 * y4
                a4z = fac4 * z4

                # ---- RK4 combination ----
                nx = px + dl * (vx + 2.0 * vx2 + 2.0 * vx3 + vx4) / 6.0
                ny = py + dl * (vy + 2.0 * vy2 + 2.0 * vy3 + vy4) / 6.0
                nz = pz + dl * (vz + 2.0 * vz2 + 2.0 * vz3 + vz4) / 6.0
                nvx = vx + dl * (a1x + 2.0 * a2x + 2.0 * a3x + a4x) / 6.0
                nvy = vy + dl * (a1y + 2.0 * a2y + 2.0 * a3y + a4y) / 6.0
                nvz = vz + dl * (a1z + 2.0 * a2z + 2.0 * a3z + a4z) / 6.0

                # Detect an equatorial (z = 0) crossing.  Linearly interpolate
                # the crossing point and velocity for the disk sample.
                if old_z * nz < 0.0:
                    t_cross = old_z / (old_z - nz)
                    cx = old_x + t_cross * (nx - old_x)
                    cy = old_y + t_cross * (ny - old_y)
                    cvx = vx + t_cross * (nvx - vx)
                    cvy = vy + t_cross * (nvy - vy)
                    cvz = vz + t_cross * (nvz - vz)
                    rc = math.sqrt(cx * cx + cy * cy)
                    hit, red, green, blue = disk_emission_scalar(
                        cx,
                        cy,
                        cvx,
                        cvy,
                        cvz,
                        rc,
                        disk_inner,
                        disk_outer,
                        disk_tpeak,
                        disk_gain,
                        disk_texture_strength,
                        phase,
                        rs,
                    )
                    if hit:
                        acc_r += red
                        acc_g += green
                        acc_b += blue

                px = nx
                py = ny
                pz = nz
                vx = nvx
                vy = nvy
                vz = nvz

            # Stars are visible only along rays that actually escaped.  A ray
            # that fell through the horizon contributes no background light,
            # but any disk emission collected in front of the hole is kept.
            if escaped:
                v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
                if v_norm > 0.0:
                    vx /= v_norm
                    vy /= v_norm
                    vz /= v_norm

                # Restrict the nearest-star search to the 3x3x3 grid cells
                # around the ray's sky direction.
                gx = int((vx + 1.0) * 0.5 * star_grid_size)
                gy = int((vy + 1.0) * 0.5 * star_grid_size)
                gz = int((vz + 1.0) * 0.5 * star_grid_size)
                if gx < 0:
                    gx = 0
                elif gx >= star_grid_size:
                    gx = star_grid_size - 1
                if gy < 0:
                    gy = 0
                elif gy >= star_grid_size:
                    gy = star_grid_size - 1
                if gz < 0:
                    gz = 0
                elif gz >= star_grid_size:
                    gz = star_grid_size - 1

                best_dot = -2.0
                best_idx = 0
                for dgx in range(-1, 2):
                    nx = gx + dgx
                    if nx < 0 or nx >= star_grid_size:
                        continue
                    for dgy in range(-1, 2):
                        ny = gy + dgy
                        if ny < 0 or ny >= star_grid_size:
                            continue
                        for dgz in range(-1, 2):
                            nz = gz + dgz
                            if nz < 0 or nz >= star_grid_size:
                                continue
                            cell = (nx * star_grid_size + ny) * star_grid_size + nz
                            start = star_grid_offsets[cell]
                            end = star_grid_offsets[cell + 1]
                            for k in range(start, end):
                                s = star_grid_indices[k]
                                dot = (
                                    vx * star_dirs[s, 0]
                                    + vy * star_dirs[s, 1]
                                    + vz * star_dirs[s, 2]
                                )
                                if dot > best_dot:
                                    best_dot = dot
                                    best_idx = s
                if best_dot > cos_star_radius:
                    acc_r += star_cols[best_idx, 0]
                    acc_g += star_cols[best_idx, 1]
                    acc_b += star_cols[best_idx, 2]

                # Faint Milky-Way-like band, analytic and deterministic.
                band_dot = vx * band_x + vy * band_y + vz * band_z
                band = math.exp(-0.5 * (band_dot / 0.17) ** 2)
                acc_r += 0.030 * band
                acc_g += 0.022 * band
                acc_b += 0.050 * band

            # Luminance-preserving filmic tone mapping.  Applying the same
            # compression factor to all three channels keeps the Doppler
            # colour shift saturated instead of washing it out to white.
            luminance = 0.2126 * acc_r + 0.7152 * acc_g + 0.0722 * acc_b
            if luminance > 1.0e-12:
                scale = (1.0 - math.exp(-luminance)) / luminance
            else:
                scale = 1.0
            out[iy, ix, 0] = int(255.0 * min(max(acc_r * scale, 0.0), 1.0) + 0.5)
            out[iy, ix, 1] = int(255.0 * min(max(acc_g * scale, 0.0), 1.0) + 0.5)
            out[iy, ix, 2] = int(255.0 * min(max(acc_b * scale, 0.0), 1.0) + 0.5)


def _acceleration(
    positions: np.ndarray, h2: np.ndarray, rs: float
) -> np.ndarray:
    """Vectorised Cartesian geodesic acceleration.

    ``a = -(3/2) rs h^2 x / |x|^5``.
    """

    r2 = np.einsum("ij,ij->i", positions, positions)
    r = np.sqrt(r2)
    factor = -1.5 * rs * h2 / (r2 * r2 * r)
    return factor[:, None] * positions


class RayTracer:
    """High-level renderer that selects numba or numpy automatically."""

    def __init__(self, stars: StarField, settings: RenderSettings) -> None:
        self.stars = stars
        self.settings = settings
        self.backend: str = "numba" if NUMBA_AVAILABLE else "numpy"

    def render(
        self, camera: Camera, width: int, height: int, phase: float = 0.0
    ) -> np.ndarray:
        """Render one frame and return an ``(height, width, 3)`` uint8 image."""

        if self.backend == "numba":
            try:
                return self._render_numba(camera, width, height, phase)
            except Exception as exc:  # pragma: no cover - environment dependent
                print(
                    "numba backend failed; falling back to numpy "
                    f"({type(exc).__name__}: {exc})"
                )
                self.backend = "numpy"
        return self._render_numpy(camera, width, height, phase)

    def _render_numba(
        self, camera: Camera, width: int, height: int, phase: float
    ) -> np.ndarray:
        """Run the JIT kernel and return the image."""

        settings = self.settings
        forward, right, up, tan_half, _ = camera.view_parameters(settings.rs)
        origin = camera.position(settings.rs)
        star_radius = (
            config.STAR_RADIUS_PIXELS * math.radians(camera.fov_deg) / height
        )

        out = np.empty((height, width, 3), dtype=np.uint8)
        _trace_kernel(
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
            float(forward[0]),
            float(forward[1]),
            float(forward[2]),
            float(right[0]),
            float(right[1]),
            float(right[2]),
            float(up[0]),
            float(up[1]),
            float(up[2]),
            float(tan_half),
            int(width),
            int(height),
            float(settings.rs),
            float(settings.disk_inner),
            float(settings.disk_outer),
            float(settings.disk_peak_temperature),
            float(settings.disk_gain),
            float(settings.disk_texture_strength),
            float(phase),
            self.stars.directions,
            self.stars.colours,
            int(self.stars.grid_size),
            self.stars.grid_offsets,
            self.stars.grid_indices,
            float(star_radius),
            int(settings.max_steps),
            float(settings.escape_radius),
            float(self.stars.band_normal[0]),
            float(self.stars.band_normal[1]),
            float(self.stars.band_normal[2]),
            out,
        )
        return out

    def _render_numpy(
        self, camera: Camera, width: int, height: int, phase: float
    ) -> np.ndarray:
        """Pure-numpy vectorised fallback renderer.

        All live rays advance together.  The loop is slower than the numba
        kernel but still usable at preview resolution on a modern CPU.
        """

        s = self.settings
        count = width * height
        origin = camera.position(s.rs)
        dirs = camera.ray_directions(width, height, s.rs)

        x = np.broadcast_to(origin, (count, 3)).copy()
        v = dirs.copy()
        h_vec = np.cross(x, v)
        h2 = np.einsum("ij,ij->i", h_vec, h_vec)

        alive = np.ones(count, dtype=bool)
        escaped = np.zeros(count, dtype=bool)
        accum = np.zeros((count, 3), dtype=np.float64)
        escape_radius = s.escape_radius

        for _ in range(s.max_steps):
            active = np.flatnonzero(alive)
            if active.size == 0:
                break

            xa = x[active]
            va = v[active]
            r = np.linalg.norm(xa, axis=1)
            radial_dot = np.einsum("ij,ij->i", xa, va)
            cap = r <= s.rs
            esc = (r > escape_radius) & (radial_dot > 0.0)
            remove = cap | esc
            if np.any(esc):
                escaped[active[esc]] = True
            if np.any(remove):
                alive[active[remove]] = False
                keep = ~remove
                active = active[keep]
                xa = xa[keep]
                va = va[keep]
                r = r[keep]
            if active.size == 0:
                continue

            h2a = h2[active]
            dl = np.clip(0.02 * r + 0.005, 0.01, 0.80)
            old_z = xa[:, 2]

            # RK4 stages, vectorised over all active rays.
            a1 = _acceleration(xa, h2a, s.rs)
            x2 = xa + 0.5 * dl[:, None] * va
            v2 = va + 0.5 * dl[:, None] * a1
            a2 = _acceleration(x2, h2a, s.rs)

            x3 = xa + 0.5 * dl[:, None] * v2
            v3 = va + 0.5 * dl[:, None] * a2
            a3 = _acceleration(x3, h2a, s.rs)

            x4 = xa + dl[:, None] * v3
            v4 = va + dl[:, None] * a3
            a4 = _acceleration(x4, h2a, s.rs)

            nx = xa + dl[:, None] * (va + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
            nv = va + dl[:, None] * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0

            crossing = old_z * nx[:, 2] < 0.0
            if np.any(crossing):
                ci = np.flatnonzero(crossing)
                t_cross = old_z[ci] / (old_z[ci] - nx[ci, 2])
                cx = xa[ci, 0] + t_cross * (nx[ci, 0] - xa[ci, 0])
                cy = xa[ci, 1] + t_cross * (nx[ci, 1] - xa[ci, 1])
                cvx = va[ci, 0] + t_cross * (nv[ci, 0] - va[ci, 0])
                cvy = va[ci, 1] + t_cross * (nv[ci, 1] - va[ci, 1])
                cvz = va[ci, 2] + t_cross * (nv[ci, 2] - va[ci, 2])
                rc = np.sqrt(cx * cx + cy * cy)
                colours, hit = disk_emission_vector(
                    cx,
                    cy,
                    cvx,
                    cvy,
                    cvz,
                    rc,
                    s.disk_inner,
                    s.disk_outer,
                    s.disk_peak_temperature,
                    s.disk_gain,
                    s.disk_texture_strength,
                    phase,
                    s.rs,
                )
                if np.any(hit):
                    accum[active[ci[hit]]] += colours[hit]

            x[active] = nx
            v[active] = nv

        # Background stars for escaped rays, evaluated in chunks to keep the
        # temporary dot-product matrix small.
        esc_idx = np.flatnonzero(escaped)
        if esc_idx.size:
            directions = v[esc_idx]
            directions /= np.maximum(
                np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12
            )
            star_radius = (
                config.STAR_RADIUS_PIXELS * math.radians(camera.fov_deg) / height
            )
            cos_radius = math.cos(star_radius)
            star_t = self.stars.directions.T
            chunk = 4096
            for start in range(0, esc_idx.size, chunk):
                stop = min(start + chunk, esc_idx.size)
                dots = directions[start:stop] @ star_t
                best = np.argmax(dots, axis=1)
                best_values = dots[np.arange(stop - start), best]
                visible = best_values > cos_radius
                if np.any(visible):
                    target = esc_idx[start:stop][visible]
                    accum[target] += self.stars.colours[best[visible]]

            band_dot = directions @ self.stars.band_normal
            band = np.exp(-0.5 * (band_dot / 0.17) ** 2)
            accum[esc_idx, 0] += 0.030 * band
            accum[esc_idx, 1] += 0.022 * band
            accum[esc_idx, 2] += 0.050 * band

        luminance = (
            0.2126 * accum[:, 0] + 0.7152 * accum[:, 1] + 0.0722 * accum[:, 2]
        )
        scale = np.ones_like(luminance)
        bright = luminance > 1.0e-12
        scale[bright] = (1.0 - np.exp(-luminance[bright])) / luminance[bright]
        image = np.clip(accum * scale[:, None], 0.0, 1.0) * 255.0
        image = image.astype(np.uint8)
        return image.reshape(height, width, 3)
