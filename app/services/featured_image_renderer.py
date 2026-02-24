"""Deterministic HTML/CSS renderer for featured blog images."""

from __future__ import annotations

import asyncio
import html
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.agents.featured_image_template import FeaturedImageTemplateSpec, ShapeSpec
from app.config import settings


@dataclass(slots=True)
class RenderedFeaturedImage:
    """Rendered image bytes and metadata."""

    payload: bytes
    mime_type: str
    width: int
    height: int


class FeaturedImageRendererError(RuntimeError):
    """Raised when deterministic rendering fails."""


class FeaturedImageRenderer:
    """Render template specs into PNG assets using Playwright CLI screenshots."""

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.width = int(width or settings.content_image_width)
        self.height = int(height or settings.content_image_height)
        self.timeout_ms = int(timeout_ms or settings.content_image_render_timeout_ms)

    async def render_png(
        self,
        *,
        template: FeaturedImageTemplateSpec,
        title_text: str,
        logo_data_url: str | None,
    ) -> RenderedFeaturedImage:
        """Render a featured image PNG from structured template specs."""
        with tempfile.TemporaryDirectory(prefix="featured-image-") as tmp_dir:
            base_path = Path(tmp_dir)
            html_path = base_path / "canvas.html"
            png_path = base_path / "image.png"

            html_path.write_text(
                self._build_html(template=template, title_text=title_text, logo_data_url=logo_data_url),
                encoding="utf-8",
            )

            cmd = [
                "playwright",
                "screenshot",
                "--browser",
                "chromium",
                "--viewport-size",
                f"{self.width},{self.height}",
                "--timeout",
                str(self.timeout_ms),
                html_path.as_uri(),
                str(png_path),
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise FeaturedImageRendererError(
                    "Playwright screenshot failed: "
                    + (stderr.decode("utf-8", errors="ignore").strip() or "unknown error")
                )

            if not png_path.exists():
                raise FeaturedImageRendererError("Rendered PNG file was not produced")

            payload = png_path.read_bytes()
            width, height = self._detect_size(payload)
            return RenderedFeaturedImage(
                payload=payload,
                mime_type="image/png",
                width=width,
                height=height,
            )

    def _build_html(
        self,
        *,
        template: FeaturedImageTemplateSpec,
        title_text: str,
        logo_data_url: str | None,
    ) -> str:
        safe_title = html.escape(title_text)
        title = template.title_zone
        logo = template.logo_zone

        title_css = (
            f"left:{title.x * 100:.2f}%;"
            f"top:{title.y * 100:.2f}%;"
            f"width:{title.width * 100:.2f}%;"
            f"height:{title.height * 100:.2f}%;"
            f"padding:{title.padding_px}px;"
            f"color:{self._safe_color(title.typography.color, fallback='#111111')};"
            f"font-family:{title.typography.font_family};"
            f"font-size:{title.typography.font_size_px}px;"
            f"font-weight:{title.typography.font_weight};"
            f"line-height:{title.typography.line_height};"
            f"letter-spacing:{title.typography.letter_spacing_em}em;"
            f"text-align:{title.typography.align};"
        )

        logo_html = ""
        if logo_data_url and logo.enabled and logo.include_if_logo_available:
            logo_css = (
                f"left:{logo.x * 100:.2f}%;"
                f"top:{logo.y * 100:.2f}%;"
                f"width:{logo.width * 100:.2f}%;"
                f"height:{logo.height * 100:.2f}%;"
                f"opacity:{logo.opacity};"
            )
            logo_html = (
                f'<div class="logo-zone" style="{logo_css}">'
                f'<img src="{logo_data_url}" alt="Brand logo" />'
                "</div>"
            )

        shapes_html = "".join(self._render_shape(shape) for shape in template.shapes)
        gradient_css = self._gradient_css(template)

        return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body {{
        margin: 0;
        width: {self.width}px;
        height: {self.height}px;
        overflow: hidden;
      }}
      .canvas {{
        position: relative;
        width: {self.width}px;
        height: {self.height}px;
        background: {gradient_css};
      }}
      .title-zone {{
        position: absolute;
        box-sizing: border-box;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: {title.typography.max_lines};
        overflow: hidden;
        word-break: break-word;
        z-index: 3;
      }}
      .shape {{
        position: absolute;
        z-index: 1;
      }}
      .logo-zone {{
        position: absolute;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 4;
      }}
      .logo-zone img {{
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
      }}
    </style>
  </head>
  <body>
    <div class="canvas">
      {shapes_html}
      <div class="title-zone" style="{title_css}">{safe_title}</div>
      {logo_html}
    </div>
  </body>
</html>
"""

    def _gradient_css(self, template: FeaturedImageTemplateSpec) -> str:
        base = self._safe_color(template.background_color, fallback="#F7F7F7")
        gradient = template.gradient
        if gradient is None or not gradient.stops:
            return base

        stops = []
        for stop in gradient.stops:
            stops.append(f"{self._safe_color(stop.color, fallback=base)} {stop.position}%")

        return f"linear-gradient({gradient.angle}deg, {', '.join(stops)})"

    def _render_shape(self, shape: ShapeSpec) -> str:
        border_radius = (
            "999px"
            if shape.shape_type == "circle"
            else f"{shape.border_radius_px}px"
        )
        if shape.shape_type == "line":
            border_radius = "999px"

        style = (
            f"left:{shape.x * 100:.2f}%;"
            f"top:{shape.y * 100:.2f}%;"
            f"width:{shape.width * 100:.2f}%;"
            f"height:{shape.height * 100:.2f}%;"
            f"background:{self._safe_color(shape.color, fallback='#CCCCCC')};"
            f"opacity:{shape.opacity};"
            f"border-radius:{border_radius};"
            f"transform:rotate({shape.rotation_deg}deg);"
            f"filter:blur({shape.blur_px}px);"
        )
        return f'<div class="shape" style="{style}"></div>'

    @staticmethod
    def _safe_color(value: str, *, fallback: str) -> str:
        candidate = str(value or "").strip()
        if not candidate:
            return fallback
        return candidate

    @staticmethod
    def _detect_size(payload: bytes) -> tuple[int, int]:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
        return int(width), int(height)
