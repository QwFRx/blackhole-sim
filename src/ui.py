"""Pygame overlay: live physics panel, FPS counter and help screen."""

from __future__ import annotations

import pygame

import config
from .physics import SystemReport


def _scientific(value: float, unit: str, digits: int = 3) -> str:
    """Format a number in scientific notation with a unit suffix."""

    return f"{value:.{digits}e} {unit}"


class InfoPanel:
    """Draw the live black-hole diagnostics on top of the preview."""

    def __init__(self, font_size: int = 13) -> None:
        """Create fonts sized for the current window."""

        self.font_size = font_size
        self.font = pygame.font.SysFont("monospace", font_size)
        self.title_font = pygame.font.SysFont("monospace", font_size + 3, bold=True)

    def draw(
        self,
        screen: pygame.Surface,
        report: SystemReport,
        fps: float,
        paused: bool,
        backend: str,
    ) -> None:
        """Render the information panel in the top-left corner."""

        rs_km = report.rs_m / 1000.0
        shadow_km = report.shadow_b_m / 1000.0
        camera_km = report.camera_r_m / 1000.0
        r_plus_km = report.kerr_r_plus_m / 1000.0
        ergo_km = report.kerr_ergosphere_equator_m / 1000.0
        isco_pro_km = report.kerr_isco_prograde_m / 1000.0
        isco_retro_km = report.kerr_isco_retrograde_m / 1000.0

        lines: list[tuple[str, tuple[int, int, int]]] = [
            ("BLACK HOLE SIMULATION", config.ACCENT_COLOR),
            ("Schwarzschild ray tracer", config.MUTED_COLOR),
            (f"Mass        {report.mass_solar:10.2f} Msun", config.TEXT_COLOR),
            (f"Spin a/M    {report.spin:10.2f}", config.TEXT_COLOR),
            (f"r_s         {rs_km:10.2f} km", config.TEXT_COLOR),
            (f"Shadow b    {shadow_km:10.2f} km", config.TEXT_COLOR),
            (f"Camera      {report.camera_r_rs:10.1f} r_s", config.TEXT_COLOR),
            (f"            {camera_km:10.0f} km", config.MUTED_COLOR),
            (f"Time dil.   {report.time_dilation:10.6f}", config.TEXT_COLOR),
            (
                f"T_Hawking   {_scientific(report.hawking_temperature_k, 'K')}",
                config.TEXT_COLOR,
            ),
            (
                f"Evaporation {_scientific(report.evaporation_time_years, 'yr')}",
                config.TEXT_COLOR,
            ),
            (
                f"Entropy     {_scientific(report.entropy_j_per_k, 'J/K')}",
                config.TEXT_COLOR,
            ),
            (
                f"Tidal @1 m  {_scientific(report.tidal_acceleration_m_s2, 'm/s2')}",
                config.TEXT_COLOR,
            ),
            (f"Kerr r+     {r_plus_km:10.2f} km", config.TEXT_COLOR),
            (f"Ergosphere  {ergo_km:10.2f} km", config.TEXT_COLOR),
            (f"ISCO pro    {isco_pro_km:10.2f} km", config.TEXT_COLOR),
            (f"ISCO retro  {isco_retro_km:10.2f} km", config.TEXT_COLOR),
            (
                f"FPS {fps:5.1f}  {backend:5s}  "
                f"{'PAUSED' if paused else 'RUNNING'}",
                config.ACCENT_COLOR if paused else config.MUTED_COLOR,
            ),
        ]

        line_height = self.font_size + 4
        width = 360
        height = line_height * len(lines) + 20
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(config.PANEL_BG)
        pygame.draw.rect(panel, config.PANEL_BORDER, panel.get_rect(), 1)

        y = 9
        for text, colour in lines:
            if text.startswith("BLACK HOLE"):
                rendered = self.title_font.render(text, True, colour)
            else:
                rendered = self.font.render(text, True, colour)
            panel.blit(rendered, (11, y))
            y += line_height

        screen.blit(panel, (10, 10))

    def draw_help(self, screen: pygame.Surface) -> None:
        """Render the full-screen keyboard/mouse help overlay."""

        width, height = screen.get_size()
        box_width = min(660, width - 40)
        box_height = min(440, height - 40)
        box = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        box.fill((6, 8, 14, 238))
        pygame.draw.rect(box, config.ACCENT_COLOR, box.get_rect(), 1)

        title = self.title_font.render("CONTROLS", True, config.ACCENT_COLOR)
        box.blit(title, (22, 16))

        rows = [
            ("Mouse drag", "rotate camera azimuth / elevation"),
            ("Mouse wheel", "zoom (change orbital distance)"),
            ("Up / Down", "increase / decrease inclination"),
            ("+ / -", "increase / decrease black-hole mass"),
            ("[ / ]", "decrease / increase spin a/M"),
            ("SPACE", "pause / resume disk rotation"),
            ("P", "save a 1920x1080 PNG to images/"),
            ("H", "toggle this help overlay"),
            ("ESC", "quit"),
        ]
        y = 54
        for key, description in rows:
            key_text = self.font.render(key, True, config.ACCENT_COLOR)
            desc_text = self.font.render(description, True, config.TEXT_COLOR)
            box.blit(key_text, (22, y))
            box.blit(desc_text, (170, y))
            y += self.font_size + 10

        note = self.font.render(
            "Ray tracer: Schwarzschild.  Kerr values in the panel are analytic.",
            True,
            config.MUTED_COLOR,
        )
        box.blit(note, (22, box_height - 36))

        screen.blit(
            box,
            ((width - box_width) // 2, (height - box_height) // 2),
        )

    def draw_status(
        self, screen: pygame.Surface, message: str, colour: tuple[int, int, int]
    ) -> None:
        """Draw a transient status message at the bottom-left of the window."""

        if not message:
            return
        rendered = self.font.render(message, True, colour)
        width, height = screen.get_size()
        screen.blit(rendered, (12, height - rendered.get_height() - 10))
