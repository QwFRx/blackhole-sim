"""Entry point for the 3D relativistic black-hole simulation.

Examples
--------
Interactive preview::

    python main.py

High-resolution still::

    python main.py --render images/blackhole.png --inclination 18

The renderer is a Schwarzschild ray tracer.  The Kerr spin parameter changes
only the analytic values in the live panel; the spacetime integration remains
Schwarzschild (the required Cartesian null-geodesic form is spherically
symmetric).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

import config
from src.camera import Camera
from src.physics import build_system_report
from src.raytracer import (
    NUMBA_AVAILABLE,
    RayTracer,
    RenderSettings,
    make_starfield,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="3D Schwarzschild black-hole ray tracer (G = c = M = 1)."
    )
    parser.add_argument("--width", type=int, default=None, help="preview width")
    parser.add_argument("--height", type=int, default=None, help="preview height")
    parser.add_argument(
        "--mass", type=float, default=config.DEFAULT_MASS_SOLAR, help="mass in Msun"
    )
    parser.add_argument(
        "--spin", type=float, default=config.DEFAULT_SPIN, help="Kerr a/M in [0, 0.99]"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=config.DEFAULT_DISTANCE_RS,
        help="camera distance in rs",
    )
    parser.add_argument(
        "--inclination",
        type=float,
        default=config.DEFAULT_INCLINATION_DEG,
        help="camera elevation above the disk plane in degrees",
    )
    parser.add_argument(
        "--render", type=Path, default=None, help="render a still PNG and exit"
    )
    parser.add_argument(
        "--seed", type=int, default=config.STAR_SEED, help="starfield seed"
    )
    args = parser.parse_args(argv)

    if args.mass <= 0.0:
        parser.error("--mass must be positive")
    if args.width is not None and args.width < 16:
        parser.error("--width must be at least 16")
    if args.height is not None and args.height < 16:
        parser.error("--height must be at least 16")
    return args


def build_tracer(seed: int) -> RayTracer:
    """Create the starfield and the ray tracer."""

    return RayTracer(make_starfield(config.STAR_COUNT, seed), RenderSettings())


def build_camera(args: argparse.Namespace) -> Camera:
    """Create and clamp the camera from CLI arguments."""

    camera = Camera(
        distance_rs=float(args.distance),
        inclination_deg=float(args.inclination),
        azimuth_deg=config.DEFAULT_AZIMUTH_DEG,
        fov_deg=config.DEFAULT_FOV_DEG,
    )
    camera.clamp()
    return camera


def save_png(
    path: Path,
    tracer: RayTracer,
    camera: Camera,
    width: int,
    height: int,
    phase: float,
) -> np.ndarray:
    """Render a frame and write it to ``path`` with Pillow."""

    from PIL import Image

    path = path.expanduser()
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)

    image = tracer.render(camera, width, height, phase)
    Image.fromarray(image).save(path)
    return image


def run_render(args: argparse.Namespace) -> None:
    """Headless high-resolution (or user-sized) still render."""

    width = int(args.width) if args.width is not None else config.RENDER_WIDTH
    height = int(args.height) if args.height is not None else config.RENDER_HEIGHT
    camera = build_camera(args)
    tracer = build_tracer(args.seed)

    if not NUMBA_AVAILABLE and width * height > 500_000:
        print(
            "numba is not installed: the pure-numpy fallback may take a long time "
            "for a full-HD still.  Install numba for the JIT backend."
        )

    started = time.perf_counter()
    image = save_png(args.render, tracer, camera, width, height, phase=0.5)
    elapsed = time.perf_counter() - started
    path = args.render
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    print(
        f"Saved {path}  {width}x{height}  "
        f"backend={tracer.backend}  {elapsed:.1f} s  "
        f"mean={float(image.mean()):.2f}"
    )


def _window_size(width: int, height: int) -> tuple[int, int]:
    """Choose an integer-scaled window that fits a 1600x900 desktop."""

    scale = min(
        float(config.WINDOW_SCALE),
        1600.0 / max(1, width),
        900.0 / max(1, height),
    )
    scale = max(scale, 0.25)
    return int(width * scale), int(height * scale)


def run_preview(args: argparse.Namespace) -> None:
    """Interactive pygame preview with a live physics panel."""

    try:
        import pygame
        from src.ui import InfoPanel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pygame is required for the preview.  Run: pip install -r requirements.txt"
        ) from exc

    width = int(args.width) if args.width is not None else config.PREVIEW_WIDTH
    height = int(args.height) if args.height is not None else config.PREVIEW_HEIGHT
    window_width, window_height = _window_size(width, height)

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption(
        "Black Hole Simulation | Schwarzschild ray tracer | H = help"
    )
    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    preview_surface = pygame.Surface((width, height))

    camera = build_camera(args)
    tracer = build_tracer(args.seed)
    panel = InfoPanel(font_size=max(9, window_height // 34))

    mass_solar = float(args.mass)
    spin = min(max(float(args.spin), 0.0), 0.99)
    phase = 0.0
    paused = False
    show_help = False
    dragging = False
    message = ""
    message_colour = config.MUTED_COLOR
    message_until = 0.0
    running = True

    if not NUMBA_AVAILABLE:
        message = "numba missing: using the slower pure-numpy backend"
        message_until = time.monotonic() + 8.0

    while running:
        dt = clock.tick(config.TARGET_FPS) / 1000.0
        now = time.monotonic()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    camera.adjust_inclination(2.0)
                elif event.key == pygame.K_DOWN:
                    camera.adjust_inclination(-2.0)
                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):
                    mass_solar = min(mass_solar * 1.15, 1.0e9)
                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):
                    mass_solar = max(mass_solar / 1.15, 0.1)
                elif event.key == pygame.K_LEFTBRACKET:
                    spin = max(0.0, spin - 0.02)
                elif event.key == pygame.K_RIGHTBRACKET:
                    spin = min(0.99, spin + 0.02)
                elif event.key == pygame.K_p:
                    pygame.display.set_caption("Black Hole Simulation | rendering...")
                    pygame.display.flip()
                    save_path = (
                        config.IMAGE_DIR
                        / f"blackhole_{datetime.now():%Y%m%d_%H%M%S}.png"
                    )
                    save_png(
                        save_path,
                        tracer,
                        camera,
                        config.RENDER_WIDTH,
                        config.RENDER_HEIGHT,
                        phase,
                    )
                    message = f"Saved {save_path.name} (1920x1080)"
                    message_colour = config.ACCENT_COLOR
                    message_until = time.monotonic() + 8.0
                    pygame.display.set_caption(
                        "Black Hole Simulation | Schwarzschild ray tracer | H = help"
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                camera.rotate(float(event.rel[0]), float(event.rel[1]))
            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom(float(event.y))

        if not paused:
            phase += dt

        frame = tracer.render(camera, width, height, phase)
        rgb_surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        preview_surface = pygame.transform.smoothscale(
            rgb_surface, (width, height)
        )
        pygame.transform.smoothscale(
            preview_surface, (window_width, window_height), screen
        )

        report = build_system_report(mass_solar, spin, camera.distance_rs)
        panel.draw(screen, report, clock.get_fps(), paused, tracer.backend)

        if message and now < message_until:
            panel.draw_status(screen, message, message_colour)
        if show_help:
            panel.draw_help(screen)

        pygame.display.flip()

    pygame.quit()


def main(argv: Sequence[str] | None = None) -> None:
    """Program entry point."""

    args = parse_args(argv)
    if args.render is not None:
        run_render(args)
    else:
        run_preview(args)


if __name__ == "__main__":
    main()
