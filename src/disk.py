"""Thin equatorial accretion-disk model.

The disk lies in the plane ``z = 0``.  Its inner edge is the Schwarzschild
ISCO (``3 rs``) and its outer edge is ``20 rs``.  Emission is computed at every
equatorial plane crossing, so a single camera ray can collect the primary
image, the secondary image and higher-order "photon-ring" images.

Scalar helpers are JIT-compiled with numba; the vectorised helpers are used by
the pure-numpy fallback renderer.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - fallback only without numba
    from numba import njit
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args: object, **kwargs: object):  # type: ignore[no-redef]
        """No-op numba decorator used by the pure-Python fallback."""

        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator
else:
    NUMBA_AVAILABLE = True

from .physics import blackbody_rgb_array, blackbody_rgb_scalar

# Maximum of f(r) = (rin/r)^(3/4) (1 - sqrt(rin/r))^(1/4).
# With u = sqrt(rin/r), f ~ u^(3/2) (1-u)^(1/4), maximal at u = 6/7.
_PROFILE_NORM: float = (6.0 / 7.0) ** 1.5 * (1.0 / 7.0) ** 0.25


@njit(cache=True, fastmath=True)
def disk_emission_scalar(
    x: float,
    y: float,
    vx: float,
    vy: float,
    vz: float,
    rc: float,
    inner: float,
    outer: float,
    t_peak: float,
    gain: float,
    texture_strength: float,
    phase: float,
    rs: float,
) -> tuple[bool, float, float, float]:
    """Sample the disk at one equatorial crossing (scalar, numba-friendly).

    Parameters
    ----------
    x, y:
        Cartesian coordinates of the crossing in the disk plane.
    vx, vy, vz:
        Ray velocity at the crossing (the backward integration direction).
    rc:
        Cylindrical radius ``sqrt(x^2 + y^2)``.
    inner, outer:
        Disk inner and outer radius in code units.
    t_peak:
        Colour-temperature scale of the disk in kelvin.
    gain:
        Artistic/visual exposure multiplier.
    texture_strength:
        Amplitude of the procedural spiral texture in ``[0, 1]``.
    phase:
        Animation phase in seconds; freeze it to stop disk rotation.
    rs:
        Schwarzschild radius in code units.

    Returns
    -------
    tuple[bool, float, float, float]
        ``(hit, red, green, blue)`` with linear RGB values.
    """

    if rc < inner or rc > outer:
        return False, 0.0, 0.0, 0.0

    # Temperature profile:
    # T(r) = T_peak * (rin/r)^(3/4) * (1 - sqrt(rin/r))^(1/4) / norm.
    u = math.sqrt(inner / rc)
    profile = u**1.5 * max(1.0 - u, 0.0) ** 0.25
    temp = t_peak * profile / _PROFILE_NORM
    if temp <= 0.0:
        return False, 0.0, 0.0, 0.0

    # Keplerian velocity:
    # beta = sqrt(GM/r), v_orbit = Omega x r = beta * (-sin(phi), cos(phi)).
    mass_geo = 0.5 * rs
    beta = math.sqrt(mass_geo / rc)
    beta2 = min(beta * beta, 0.999)
    inv_rc = 1.0 / rc
    bvx = -y * inv_rc * beta
    bvy = x * inv_rc * beta
    gamma = 1.0 / math.sqrt(1.0 - beta2)

    # Photon propagation direction toward the observer is n = -v / |v|.
    v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if v_norm <= 0.0:
        return False, 0.0, 0.0, 0.0
    n_dot_beta = -(vx * bvx + vy * bvy) / v_norm

    # Doppler + gravitational shift:
    # g = sqrt(1 - rs/r) / (gamma (1 - n . beta)).
    grav = math.sqrt(max(0.0, 1.0 - rs / rc))
    g_shift = grav / (gamma * max(1.0 - n_dot_beta, 1.0e-3))
    g_shift = min(max(g_shift, 0.05), 6.0)

    # Observed temperature and bolometric intensity:
    # T_obs = g T_emit, I_obs = g^4 I_emit.
    t_obs = g_shift * temp
    red, green, blue = blackbody_rgb_scalar(t_obs)

    # Procedural differential-rotation texture (deterministic, no noise table).
    phi = math.atan2(y, x)
    omega_vis = 1.7 * (inner / rc) ** 1.5
    texture = 1.0 + texture_strength * math.sin(
        3.0 * phi - 7.0 * math.log(rc / inner) + 2.5 * phase * omega_vis
    )
    texture = max(texture, 0.05)

    intensity = gain * (temp / t_peak) ** 4 * g_shift**4 * texture
    return True, red * intensity, green * intensity, blue * intensity


def disk_emission_vector(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rc: np.ndarray,
    inner: float,
    outer: float,
    t_peak: float,
    gain: float,
    texture_strength: float,
    phase: float,
    rs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised disk sampling used by the numpy fallback renderer.

    Returns an ``(N, 3)`` array of linear RGB values and a boolean hit mask.
    """

    count = rc.shape[0]
    colours = np.zeros((count, 3), dtype=np.float64)
    hit = (rc >= inner) & (rc <= outer)
    if not np.any(hit):
        return colours, hit

    idx = np.flatnonzero(hit)
    xh = x[idx]
    yh = y[idx]
    vxh = vx[idx]
    vyh = vy[idx]
    vzh = vz[idx]
    rh = rc[idx]

    # Temperature profile: T ~ (rin/r)^(3/4) (1 - sqrt(rin/r))^(1/4).
    u = np.sqrt(inner / rh)
    profile = np.power(u, 1.5) * np.power(np.maximum(1.0 - u, 0.0), 0.25)
    temp = t_peak * profile / _PROFILE_NORM

    # Keplerian circular velocity, beta = sqrt(GM/r).
    mass_geo = 0.5 * rs
    beta = np.sqrt(mass_geo / rh)
    inv_rc = 1.0 / rh
    bvx = -yh * inv_rc * beta
    bvy = xh * inv_rc * beta
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta * beta, 1.0e-6))

    # n = -v/|v|; n.beta controls Doppler beaming.
    v_norm = np.sqrt(vxh * vxh + vyh * vyh + vzh * vzh)
    n_dot_beta = -(vxh * bvx + vyh * bvy) / np.maximum(v_norm, 1.0e-12)

    # g = sqrt(1 - rs/r) / (gamma (1 - n.beta)).
    grav = np.sqrt(np.maximum(1.0 - rs / rh, 0.0))
    g_shift = grav / (gamma * np.maximum(1.0 - n_dot_beta, 1.0e-3))
    g_shift = np.clip(g_shift, 0.05, 6.0)

    # T_obs = g T, I_obs = g^4 I.
    t_obs = g_shift * temp
    rgb = blackbody_rgb_array(t_obs)

    phi = np.arctan2(yh, xh)
    omega_vis = 1.7 * np.power(inner / rh, 1.5)
    texture = 1.0 + texture_strength * np.sin(
        3.0 * phi - 7.0 * np.log(rh / inner) + 2.5 * phase * omega_vis
    )
    texture = np.maximum(texture, 0.05)

    intensity = gain * np.power(temp / t_peak, 4.0) * g_shift**4 * texture
    colours[idx] = rgb * intensity[:, None]
    return colours, hit
