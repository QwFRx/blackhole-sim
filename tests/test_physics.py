"""Unit tests for the analytic Schwarzschild/Kerr formulae."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import physics  # noqa: E402


def test_schwarzschild_radius_of_sun() -> None:
    """The solar Schwarzschild radius is approximately 2953 m."""

    assert physics.schwarzschild_radius_solar(1.0) == pytest.approx(
        2953.0, rel=2.0e-3
    )


def test_photon_sphere_is_one_and_a_half_rs() -> None:
    """r_ph = 3 GM/c^2 = 1.5 rs."""

    rs = physics.schwarzschild_radius_solar(1.0)
    assert physics.photon_sphere_radius(physics.M_SUN) == pytest.approx(
        1.5 * rs, rel=1.0e-12
    )


def test_shadow_critical_impact_parameter() -> None:
    """b_c = 3 sqrt(3) GM/c^2, i.e. 3 sqrt(3)/2 times rs."""

    rs = physics.schwarzschild_radius_solar(1.0)
    b = physics.shadow_impact_parameter(physics.M_SUN)
    assert b / rs == pytest.approx(1.5 * math.sqrt(3.0), rel=1.0e-12)
    assert b == pytest.approx(7673.0, rel=2.0e-3)


def test_hawking_temperature_of_sun() -> None:
    """T_H(Msun) is approximately 6.17e-8 K."""

    assert physics.hawking_temperature(physics.M_SUN) == pytest.approx(
        6.17e-8, rel=2.0e-3
    )


def test_schwarzschild_isco_is_three_rs() -> None:
    """The Schwarzschild ISCO is 6 GM/c^2 = 3 rs."""

    rs = physics.schwarzschild_radius_solar(1.0)
    assert physics.isco_radius(physics.M_SUN) == pytest.approx(
        3.0 * rs, rel=1.0e-12
    )


def test_kerr_isco_reduces_to_schwarzschild_at_zero_spin() -> None:
    """Prograde and retrograde Kerr ISCOs both equal 3 rs when a = 0."""

    expected = 3.0 * physics.schwarzschild_radius_solar(1.0)
    assert physics.kerr_isco_radius(physics.M_SUN, 0.0, True) == pytest.approx(
        expected, rel=1.0e-12
    )
    assert physics.kerr_isco_radius(physics.M_SUN, 0.0, False) == pytest.approx(
        expected, rel=1.0e-12
    )


def test_kerr_prograde_isco_is_smaller_than_retrograde() -> None:
    """Frame dragging moves the prograde ISCO inward."""

    pro = physics.kerr_isco_radius(physics.M_SUN, 0.9, True)
    retro = physics.kerr_isco_radius(physics.M_SUN, 0.9, False)
    assert pro < retro


def test_horizon_and_ergosphere() -> None:
    """r+ <= rs, and the equatorial ergosphere is exactly 2 GM/c^2 = rs."""

    rs = physics.schwarzschild_radius_solar(1.0)
    r_plus = physics.kerr_horizon_radius(physics.M_SUN, 0.9)
    ergo = physics.kerr_ergosphere_equator(physics.M_SUN, 0.9)
    assert 0.0 < r_plus < rs
    assert ergo == pytest.approx(rs, rel=1.0e-12)


def test_time_dilation() -> None:
    """sqrt(1 - rs/r) at r = 60 rs."""

    assert physics.time_dilation(60.0) == pytest.approx(
        math.sqrt(59.0 / 60.0), rel=1.0e-12
    )


def test_thermodynamic_quantities_are_positive() -> None:
    """Evaporation time, entropy and horizon tides are finite and positive."""

    assert physics.evaporation_time(physics.M_SUN) > 0.0
    assert physics.bekenstein_hawking_entropy(physics.M_SUN) > 0.0
    assert physics.tidal_acceleration_at_horizon(physics.M_SUN) > 0.0


def test_blackbody_colours_are_normalised() -> None:
    """Blackbody RGB channels stay inside [0, 1] and red dominates when cool."""

    r_cool, g_cool, b_cool = physics.blackbody_rgb_scalar(2500.0)
    assert r_cool == pytest.approx(1.0)
    assert b_cool < g_cool < r_cool

    r_hot, g_hot, b_hot = physics.blackbody_rgb_scalar(25000.0)
    for channel in (r_hot, g_hot, b_hot):
        assert 0.0 <= channel <= 1.0
    assert b_hot > r_hot


def test_system_report_consistency() -> None:
    """The aggregate report must expose all UI quantities."""

    report = physics.build_system_report(10.0, 0.6, 60.0)
    assert report.rs_m == pytest.approx(
        physics.schwarzschild_radius_solar(10.0), rel=1.0e-12
    )
    assert report.time_dilation == pytest.approx(
        physics.time_dilation(60.0), rel=1.0e-12
    )
    assert report.kerr_isco_prograde_m < report.kerr_isco_retrograde_m
