"""Schwarzschild / Kerr formulae and blackbody colours.

The ray tracer works in geometrised units with ``G = c = M = 1`` and a
Schwarzschild radius of ``rs = 2``.  The functions in this module also expose
the same quantities in SI units for the live information panel and for the
unit tests.  The Kerr helpers are *analytic only*; the renderer itself remains
Schwarzschild (see the comment in :mod:`src.raytracer`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - the fallback branch is exercised only without numba
    from numba import njit

    NUMBA_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args: object, **kwargs: object):  # type: ignore[no-redef]
        """Return a no-op decorator when numba is unavailable."""

        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G: float = 6.67430e-11                 # gravitational constant [m^3 kg^-1 s^-2]
C: float = 299792458.0                 # speed of light [m s^-1]
H_PLANCK: float = 6.62607015e-34       # Planck constant [J s]
HBAR: float = H_PLANCK / (2.0 * math.pi)
K_B: float = 1.380649e-23              # Boltzmann constant [J K^-1]
M_SUN: float = 1.98847e30              # solar mass [kg]
YEAR: float = 3.15576e7                # Julian year [s]


# ---------------------------------------------------------------------------
# Schwarzschild quantities in SI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemReport:
    """All quantities displayed in the live information panel."""

    mass_solar: float
    spin: float
    rs_m: float
    photon_sphere_m: float
    isco_schwarzschild_m: float
    shadow_b_m: float
    camera_r_rs: float
    camera_r_m: float
    time_dilation: float
    hawking_temperature_k: float
    evaporation_time_years: float
    entropy_j_per_k: float
    tidal_acceleration_m_s2: float
    kerr_r_plus_m: float
    kerr_ergosphere_equator_m: float
    kerr_isco_prograde_m: float
    kerr_isco_retrograde_m: float


def mass_from_solar(mass_solar: float) -> float:
    """Convert a mass in solar masses to kilograms."""

    return float(mass_solar) * M_SUN


def schwarzschild_radius(mass_kg: float) -> float:
    """Schwarzschild radius ``rs = 2 G M / c^2`` in metres."""

    return 2.0 * G * mass_kg / (C * C)


def schwarzschild_radius_solar(mass_solar: float = 1.0) -> float:
    """Schwarzschild radius of a ``mass_solar`` black hole in metres."""

    return schwarzschild_radius(mass_from_solar(mass_solar))


def photon_sphere_radius(mass_kg: float) -> float:
    """Photon-sphere radius ``r_ph = 3 G M / c^2 = 1.5 rs`` in metres."""

    return 3.0 * G * mass_kg / (C * C)


def isco_radius(mass_kg: float) -> float:
    """Schwarzschild ISCO radius ``r_isco = 6 G M / c^2 = 3 rs`` in metres."""

    return 6.0 * G * mass_kg / (C * C)


def shadow_impact_parameter(mass_kg: float) -> float:
    """Critical impact parameter ``b_c = 3 sqrt(3) G M / c^2`` in metres."""

    return 3.0 * math.sqrt(3.0) * G * mass_kg / (C * C)


def hawking_temperature(mass_kg: float) -> float:
    """Hawking temperature ``T_H = hbar c^3 / (8 pi G M k_B)`` in kelvin."""

    return HBAR * C**3 / (8.0 * math.pi * G * mass_kg * K_B)


def evaporation_time(mass_kg: float) -> float:
    """Blackbody evaporation time ``t_ev = 5120 pi G^2 M^3 / (hbar c^4)`` [s]."""

    return 5120.0 * math.pi * G * G * mass_kg**3 / (HBAR * C**4)


def bekenstein_hawking_entropy(mass_kg: float) -> float:
    """Bekenstein-Hawking entropy ``S = k_B c^3 A / (4 G hbar)`` [J/K]."""

    # Horizon area of a Schwarzschild black hole: A = 4 pi rs^2.
    area = 4.0 * math.pi * schwarzschild_radius(mass_kg) ** 2
    return K_B * C**3 * area / (4.0 * G * HBAR)


def time_dilation(camera_r_rs: float) -> float:
    """Gravitational time dilation ``sqrt(1 - rs/r)`` at ``r`` in units of rs."""

    x = max(float(camera_r_rs), 1.0 + 1.0e-12)
    return math.sqrt(max(0.0, 1.0 - 1.0 / x))


def tidal_acceleration_at_horizon(
    mass_kg: float, proper_length_m: float = 1.0
) -> float:
    """Tidal acceleration ``2 G M l / r^3`` at the horizon [m/s^2].

    ``l`` is the proper separation between two freely falling test masses.
    At ``r = rs`` this reduces to ``c^6 l / (4 G^2 M^2)``.
    """

    rs = schwarzschild_radius(mass_kg)
    return 2.0 * G * mass_kg * proper_length_m / (rs**3)


# ---------------------------------------------------------------------------
# Kerr quantities (analytic diagnostics only)
# ---------------------------------------------------------------------------
def kerr_horizon_radius(mass_kg: float, spin: float) -> float:
    """Outer Kerr horizon ``r+ = M + sqrt(M^2 - a^2)`` in metres.

    ``spin`` is the dimensionless ``a/M`` in ``[0, 0.99]``.
    """

    a = min(max(float(spin), 0.0), 0.999999)
    length_scale = G * mass_kg / (C * C)
    return length_scale * (1.0 + math.sqrt(max(0.0, 1.0 - a * a)))


def kerr_ergosphere_equator(mass_kg: float, spin: float = 0.0) -> float:
    """Ergosphere radius at the equator ``r_E = 2 M`` in metres.

    The result is independent of spin at ``theta = pi/2``; ``spin`` is accepted
    so callers can use a uniform signature.
    """

    _ = spin
    return 2.0 * G * mass_kg / (C * C)


def kerr_isco_radius(mass_kg: float, spin: float, prograde: bool = True) -> float:
    """Kerr ISCO radius from the Bardeen-Press-Teukolsky formula.

    ``Z1 = 1 + (1-a^2)^(1/3)[(1+a)^(1/3) + (1-a)^(1/3)]``,
    ``Z2 = sqrt(3a^2 + Z1^2)`` and
    ``r_isco = M [3 + Z2 -/+ sqrt((3-Z1)(3+Z1+2Z2))]``.
    """

    a = min(max(float(spin), 0.0), 0.999999)
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    sign = -1.0 if prograde else 1.0
    radial_term = max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    r_over_m = 3.0 + z2 + sign * math.sqrt(radial_term)
    return r_over_m * G * mass_kg / (C * C)


# ---------------------------------------------------------------------------
# Disk kinematics in geometric units
# ---------------------------------------------------------------------------
def keplerian_angular_velocity(radius: float, mass: float = 1.0) -> float:
    """Keplerian angular velocity ``Omega = sqrt(G M / r^3)`` (G = 1)."""

    return math.sqrt(mass / (radius**3))


def keplerian_orbital_speed(radius: float, mass: float = 1.0) -> float:
    """Circular-orbit speed ``v = sqrt(G M / r)`` (G = c = 1)."""

    return math.sqrt(mass / radius)


# ---------------------------------------------------------------------------
# Blackbody colour
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def blackbody_rgb_scalar(temperature_k: float) -> tuple[float, float, float]:
    """Map a blackbody temperature to linear RGB in ``[0, 1]``.

    Uses the standard Planckian-locus approximation (Tanner Helland):
    ``T <= 6600 K: R=255, G=99.47 ln(T/100)-161.12,
    B=138.52 ln(T/100-10)-305.04``; above 6600 K the red and green channels
    are power laws and blue is 255.  The input is clamped to 1000-40000 K.
    """

    t = min(max(float(temperature_k), 1000.0), 40000.0) / 100.0

    if t <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(t) - 161.1195681661
        if t <= 19.0:
            blue = 0.0
        else:
            blue = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * (t - 60.0) ** -0.1332047592
        green = 288.1221695283 * (t - 60.0) ** -0.0755148492
        blue = 255.0

    return (
        min(max(red / 255.0, 0.0), 1.0),
        min(max(green / 255.0, 0.0), 1.0),
        min(max(blue / 255.0, 0.0), 1.0),
    )


def blackbody_rgb_array(temperature_k: np.ndarray) -> np.ndarray:
    """Vectorised version of :func:`blackbody_rgb_scalar`.

    Parameters
    ----------
    temperature_k:
        Array of temperatures in kelvin.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(..., 3)`` with RGB values in ``[0, 1]``.
    """

    t = np.clip(np.asarray(temperature_k, dtype=np.float64), 1000.0, 40000.0) / 100.0

    red = np.where(
        t <= 66.0,
        255.0,
        329.698727446 * np.power(np.maximum(t - 60.0, 1.0e-6), -0.1332047592),
    )
    green = np.where(
        t <= 66.0,
        99.4708025861 * np.log(t) - 161.1195681661,
        288.1221695283 * np.power(np.maximum(t - 60.0, 1.0e-6), -0.0755148492),
    )
    blue = np.where(
        t >= 66.0,
        255.0,
        np.where(
            t <= 19.0,
            0.0,
            138.5177312231 * np.log(np.maximum(t - 10.0, 1.0e-9))
            - 305.0447927307,
        ),
    )

    rgb = np.stack((red, green, blue), axis=-1) / 255.0
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------
def build_system_report(
    mass_solar: float, spin: float, camera_r_rs: float
) -> SystemReport:
    """Collect every physical diagnostic used by the UI.

    Parameters
    ----------
    mass_solar:
        Black-hole mass in solar masses.
    spin:
        Dimensionless Kerr spin ``a/M`` in ``[0, 0.99]``.
    camera_r_rs:
        Camera radial coordinate in units of the Schwarzschild radius.
    """

    mass_kg = mass_from_solar(mass_solar)
    rs = schwarzschild_radius(mass_kg)
    return SystemReport(
        mass_solar=float(mass_solar),
        spin=float(spin),
        rs_m=rs,
        photon_sphere_m=photon_sphere_radius(mass_kg),
        isco_schwarzschild_m=isco_radius(mass_kg),
        shadow_b_m=shadow_impact_parameter(mass_kg),
        camera_r_rs=float(camera_r_rs),
        camera_r_m=float(camera_r_rs) * rs,
        time_dilation=time_dilation(camera_r_rs),
        hawking_temperature_k=hawking_temperature(mass_kg),
        evaporation_time_years=evaporation_time(mass_kg) / YEAR,
        entropy_j_per_k=bekenstein_hawking_entropy(mass_kg),
        tidal_acceleration_m_s2=tidal_acceleration_at_horizon(mass_kg),
        kerr_r_plus_m=kerr_horizon_radius(mass_kg, spin),
        kerr_ergosphere_equator_m=kerr_ergosphere_equator(mass_kg, spin),
        kerr_isco_prograde_m=kerr_isco_radius(mass_kg, spin, prograde=True),
        kerr_isco_retrograde_m=kerr_isco_radius(mass_kg, spin, prograde=False),
    )
