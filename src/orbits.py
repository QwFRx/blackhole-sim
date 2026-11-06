"""Bonus: 3D relativistic test-star orbit with perihelion precession.

The orbit is integrated numerically from the exact Schwarzschild massive-
particle equation in the orbital plane::

    d^2u/dphi^2 + u = G M / h^2 + 3 G M u^2 / c^2,

with ``u = 1/r`` and ``h = sqrt(G M a (1 - e^2))``.  Setting ``G = M = c = 1``
gives the form used below.  The analytic weak-field precession per orbit is

    Delta phi = 6 pi G M / (c^2 a (1 - e^2)).

Run with ``python -m src.orbits`` (saves ``images/orbits.png`` by default).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402


def analytic_precession(
    semi_major: float, eccentricity: float, mass: float = 1.0
) -> float:
    """Weak-field precession ``6 pi G M / (c^2 a (1-e^2))`` in radians/orbit.

    In geometric units ``G = c = 1`` and ``mass = 1``.
    """

    return 6.0 * math.pi * mass / (
        semi_major * (1.0 - eccentricity * eccentricity)
    )


def integrate_orbit(
    semi_major: float = 100.0,
    eccentricity: float = 0.50,
    periods: float = 4.0,
    dphi: float = 2.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate the relativistic orbit equation with classical RK4.

    Parameters
    ----------
    semi_major:
        Semi-major axis in units of ``G M / c^2``.
    eccentricity:
        Orbital eccentricity in ``[0, 1)``.
    periods:
        Number of orbital periods to integrate.
    dphi:
        RK4 step in azimuth (radians).

    Returns
    -------
    tuple
        ``(x, y, z, perihelion_angles, numerical_precession)``.  The first
        three arrays are the 3D orbit, the fourth contains detected
        perihelion azimuths, and the last is the measured precession in
        radians per orbit.
    """

    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must be in [0, 1)")
    if semi_major <= 1.0:
        raise ValueError("semi_major must be greater than 1 M")

    # Specific angular momentum h = sqrt(G M a (1 - e^2)).
    h = math.sqrt(semi_major * (1.0 - eccentricity * eccentricity))
    h2 = h * h

    u = 1.0 / (semi_major * (1.0 - eccentricity))
    du = 0.0
    phi = 0.0
    phi_max = 2.0 * math.pi * periods + 0.75

    phis: list[float] = [0.0]
    us: list[float] = [u]
    perihelion: list[float] = [0.0]
    previous_du = du

    while phi < phi_max:
        # --- RK4 stage 1 -------------------------------------------------
        k1_u = du
        k1_d = -u + 1.0 / h2 + 3.0 * u * u

        # --- RK4 stage 2 -------------------------------------------------
        u2 = u + 0.5 * dphi * k1_u
        d2 = du + 0.5 * dphi * k1_d
        k2_u = d2
        k2_d = -u2 + 1.0 / h2 + 3.0 * u2 * u2

        # --- RK4 stage 3 -------------------------------------------------
        u3 = u + 0.5 * dphi * k2_u
        d3 = du + 0.5 * dphi * k2_d
        k3_u = d3
        k3_d = -u3 + 1.0 / h2 + 3.0 * u3 * u3

        # --- RK4 stage 4 -------------------------------------------------
        u4 = u + dphi * k3_u
        d4 = du + dphi * k3_d
        k4_u = d4
        k4_d = -u4 + 1.0 / h2 + 3.0 * u4 * u4

        # --- RK4 combination ---------------------------------------------
        u += dphi * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u) / 6.0
        du += dphi * (k1_d + 2.0 * k2_d + 2.0 * k3_d + k4_d) / 6.0
        phi += dphi

        phis.append(phi)
        us.append(u)

        # A perihelion is a maximum of u (minimum of r): du changes from
        # positive to negative.
        if previous_du > 0.0 >= du:
            perihelion.append(phi)
        previous_du = du

    phi_array = np.asarray(phis)
    r = 1.0 / np.asarray(us)

    # Tilt the orbital plane by 55 degrees so the 3D rosette is visible.
    inclination = math.radians(55.0)
    x = r * np.cos(phi_array)
    y = r * np.sin(phi_array) * math.cos(inclination)
    z = r * np.sin(phi_array) * math.sin(inclination)

    peri = np.asarray(perihelion, dtype=np.float64)
    if peri.size >= 2:
        numerical = float(np.mean(np.diff(peri) - 2.0 * math.pi))
    else:  # pragma: no cover - requires an extremely short integration
        numerical = float("nan")
    return x, y, z, peri, numerical


def make_plot(
    semi_major: float = 100.0,
    eccentricity: float = 0.50,
    periods: float = 4.0,
    output: Path = config.IMAGE_DIR / "orbits.png",
    show: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Integrate one orbit, save a 3D matplotlib figure and return diagnostics."""

    x, y, z, peri, numerical = integrate_orbit(
        semi_major=semi_major, eccentricity=eccentricity, periods=periods
    )
    analytic = analytic_precession(semi_major, eccentricity)

    fig = plt.figure(figsize=(8.0, 7.0), facecolor="#05070d")
    ax = fig.add_subplot(111, projection="3d", facecolor="#05070d")

    # Orbital trajectory, coloured by azimuth to show the rosette advance.
    ax.plot(x, y, z, color="#ffb15c", linewidth=1.4, alpha=0.95)
    ax.plot(
        [0.0],
        [0.0],
        [0.0],
        marker="o",
        markersize=9,
        color="black",
        markeredgecolor="#ff7a1a",
    )

    if peri.size:
        # Approximate perihelion positions using r_peri = a(1-e).
        r_peri = semi_major * (1.0 - eccentricity)
        px = r_peri * np.cos(peri)
        py = r_peri * np.sin(peri) * math.cos(math.radians(55.0))
        pz = r_peri * np.sin(peri) * math.sin(math.radians(55.0))
        ax.scatter(px, py, pz, s=14, color="#7fd7ff", depthshade=False)

    limit = semi_major * (1.0 + eccentricity) * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x [GM/c²]", color="#a9b6cc")
    ax.set_ylabel("y [GM/c²]", color="#a9b6cc")
    ax.set_zlabel("z [GM/c²]", color="#a9b6cc")
    ax.tick_params(colors="#a9b6cc", labelsize=8)
    ax.grid(True, color="#1a2233", linewidth=0.5)
    for spine in (ax.xaxis, ax.yaxis, ax.zaxis):
        spine.pane.set_facecolor((0.02, 0.03, 0.05, 1.0))
        spine.pane.set_edgecolor("#223047")
    ax.view_init(elev=24.0, azim=-58.0)

    degrees_num = math.degrees(numerical)
    degrees_ana = math.degrees(analytic)
    ax.set_title(
        "Relativistic test-star orbit\n"
        f"Δφ numeric = {degrees_num:.2f}° / orbit, "
        f"analytic = {degrees_ana:.2f}° / orbit",
        color="#f4f7ff",
        fontsize=11,
        pad=14,
    )
    fig.tight_layout()

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return x, y, z, numerical, analytic


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse orbit-plot options."""

    parser = argparse.ArgumentParser(description="3D relativistic orbit with precession.")
    parser.add_argument("--semi-major", type=float, default=100.0)
    parser.add_argument("--eccentricity", type=float, default=0.50)
    parser.add_argument("--periods", type=float, default=4.0)
    parser.add_argument(
        "--output", type=Path, default=config.IMAGE_DIR / "orbits.png"
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the orbit integration and create the diagnostic plot."""

    args = parse_args(argv)
    _, _, _, numerical, analytic = make_plot(
        semi_major=args.semi_major,
        eccentricity=args.eccentricity,
        periods=args.periods,
        output=args.output,
        show=args.show,
    )
    print(f"Saved {args.output}")
    print(
        "Perihelion precession: "
        f"numeric = {math.degrees(numerical):.4f} deg/orbit, "
        f"analytic = {math.degrees(analytic):.4f} deg/orbit"
    )


if __name__ == "__main__":
    main()
