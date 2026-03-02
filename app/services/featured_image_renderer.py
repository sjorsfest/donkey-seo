"""Deterministic HTML/CSS renderer for featured blog images."""

from __future__ import annotations

import asyncio
import base64
import html
import logging
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from app.agents.featured_image_template import FeaturedImageTemplateSpec, ShapeSpec
from app.config import settings

logger = logging.getLogger(__name__)
FontType = ImageFont.ImageFont | ImageFont.FreeTypeFont


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
    """Render template specs into PNG assets with Playwright and a Pillow fallback."""

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
        playwright_errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="featured-image-") as tmp_dir:
            base_path = Path(tmp_dir)
            html_path = base_path / "canvas.html"
            png_path = base_path / "image.png"

            html_path.write_text(
                self._build_html(template=template, title_text=title_text, logo_data_url=logo_data_url),
                encoding="utf-8",
            )

            playwright_errors = await self._run_playwright_render(html_path=html_path, png_path=png_path)
            if png_path.exists():
                payload = png_path.read_bytes()
                width, height = self._detect_size(payload)
                return RenderedFeaturedImage(
                    payload=payload,
                    mime_type="image/png",
                    width=width,
                    height=height,
                )

        if playwright_errors:
            logger.warning(
                "Playwright unavailable or failed; using Pillow featured-image fallback: %s",
                "; ".join(playwright_errors),
            )

        try:
            payload = self._render_with_pillow(
                template=template,
                title_text=title_text,
                logo_data_url=logo_data_url,
            )
        except Exception as exc:
            details = "; ".join(playwright_errors) if playwright_errors else "no Playwright attempts"
            raise FeaturedImageRendererError(
                f"Playwright screenshot failed ({details}) and Pillow fallback failed: {exc}"
            ) from exc

        width, height = self._detect_size(payload)
        return RenderedFeaturedImage(
            payload=payload,
            mime_type="image/png",
            width=width,
            height=height,
        )

    async def _run_playwright_render(self, *, html_path: Path, png_path: Path) -> list[str]:
        args = [
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
        errors: list[str] = []
        for cmd in self._playwright_commands(args):
            if png_path.exists():
                png_path.unlink()
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                errors.append(f"{cmd[0]} not found: {exc}")
                continue

            stdout, stderr = await process.communicate()
            if process.returncode == 0 and png_path.exists():
                return []

            stderr_text = stderr.decode("utf-8", errors="ignore").strip()
            stdout_text = stdout.decode("utf-8", errors="ignore").strip()
            message = stderr_text or stdout_text or f"exit code {process.returncode}"
            errors.append(f"{' '.join(cmd[:3])}: {message}")

        return errors

    @staticmethod
    def _playwright_commands(args: list[str]) -> list[list[str]]:
        return [
            ["playwright", *args],
            [sys.executable, "-m", "playwright", *args],
            ["uv", "run", "playwright", *args],
        ]

    def _render_with_pillow(
        self,
        *,
        template: FeaturedImageTemplateSpec,
        title_text: str,
        logo_data_url: str | None,
    ) -> bytes:
        canvas = self._build_background(template)
        self._draw_shapes(canvas=canvas, template=template)
        self._draw_title(canvas=canvas, template=template, title_text=title_text)
        self._draw_logo(canvas=canvas, template=template, logo_data_url=logo_data_url)

        payload = BytesIO()
        canvas.save(payload, format="PNG")
        return payload.getvalue()

    def _build_background(self, template: FeaturedImageTemplateSpec) -> Image.Image:
        fallback = self._safe_color(template.background_color, fallback="#F7F7F7")
        gradient = template.gradient
        if gradient is None or not gradient.stops:
            return Image.new("RGBA", (self.width, self.height), self._rgba(fallback, alpha=1.0))

        stops: list[tuple[int, tuple[int, int, int]]] = []
        for stop in gradient.stops:
            color = self._safe_color(stop.color, fallback=fallback)
            stops.append((stop.position, self._rgb(color, fallback=fallback)))
        stops.sort(key=lambda item: item[0])
        if not stops:
            return Image.new("RGBA", (self.width, self.height), self._rgba(fallback, alpha=1.0))

        if stops[0][0] > 0:
            stops.insert(0, (0, stops[0][1]))
        if stops[-1][0] < 100:
            stops.append((100, stops[-1][1]))

        horizontal = 45 <= (gradient.angle % 360) < 135 or 225 <= (gradient.angle % 360) < 315
        length = self.width if horizontal else self.height
        draw_image = Image.new("RGBA", (self.width, self.height))
        draw = ImageDraw.Draw(draw_image, "RGBA")
        for index in range(length):
            position = int((index / max(1, length - 1)) * 100)
            color = self._interpolated_color(stops=stops, position=position)
            if horizontal:
                draw.line([(index, 0), (index, self.height)], fill=(*color, 255))
            else:
                draw.line([(0, index), (self.width, index)], fill=(*color, 255))
        return draw_image

    def _draw_shapes(self, *, canvas: Image.Image, template: FeaturedImageTemplateSpec) -> None:
        for shape in template.shapes:
            self._draw_shape(canvas=canvas, shape=shape)

    def _draw_shape(self, *, canvas: Image.Image, shape: ShapeSpec) -> None:
        left = int(shape.x * self.width)
        top = int(shape.y * self.height)
        shape_width = max(1, int(shape.width * self.width))
        shape_height = max(1, int(shape.height * self.height))
        right = left + shape_width
        bottom = top + shape_height
        color = (*self._rgb(shape.color, fallback="#CCCCCC"), int(max(0.0, min(1.0, shape.opacity)) * 255))

        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        bbox = (left, top, right, bottom)

        if shape.shape_type == "circle":
            draw.ellipse(bbox, fill=color)
        else:
            radius = shape.border_radius_px
            if shape.shape_type in {"blob", "line"}:
                radius = max(radius, min(shape_width, shape_height) // 2)
            draw.rounded_rectangle(bbox, radius=radius, fill=color)

        if shape.rotation_deg:
            layer = layer.rotate(
                shape.rotation_deg,
                resample=Image.Resampling.BICUBIC,
                center=(left + shape_width // 2, top + shape_height // 2),
            )
        if shape.blur_px:
            layer = layer.filter(ImageFilter.GaussianBlur(shape.blur_px))

        canvas.alpha_composite(layer)

    def _draw_title(
        self,
        *,
        canvas: Image.Image,
        template: FeaturedImageTemplateSpec,
        title_text: str,
    ) -> None:
        title = template.title_zone
        typography = title.typography
        draw = ImageDraw.Draw(canvas, "RGBA")

        zone_left = int(title.x * self.width) + title.padding_px
        zone_top = int(title.y * self.height) + title.padding_px
        zone_width = max(1, int(title.width * self.width) - (2 * title.padding_px))
        zone_height = max(1, int(title.height * self.height) - (2 * title.padding_px))

        font = self._load_font(font_size_px=typography.font_size_px, font_weight=typography.font_weight)
        letter_spacing = int(round(typography.letter_spacing_em * typography.font_size_px))
        all_lines = self._wrap_lines(
            draw=draw,
            text=title_text,
            font=font,
            max_width=zone_width,
            max_lines=typography.max_lines,
            letter_spacing=letter_spacing,
        )
        if not all_lines:
            return

        line_height = max(1, int(round(typography.font_size_px * typography.line_height)))
        max_visible_lines = max(1, min(len(all_lines), zone_height // line_height))
        lines = all_lines[:max_visible_lines]
        if len(all_lines) > max_visible_lines:
            lines[-1] = self._truncate_with_ellipsis(
                draw=draw,
                text=lines[-1],
                font=font,
                max_width=zone_width,
                letter_spacing=letter_spacing,
            )

        text_color = self._rgba(typography.color, alpha=1.0, fallback="#111111")
        for index, line in enumerate(lines):
            y = zone_top + index * line_height
            line_width = self._measure_text(
                draw=draw,
                text=line,
                font=font,
                letter_spacing=letter_spacing,
            )
            x = zone_left
            if typography.align == "center":
                x += max(0, int((zone_width - line_width) / 2))
            self._draw_text(
                draw=draw,
                text=line,
                x=x,
                y=y,
                font=font,
                fill=text_color,
                letter_spacing=letter_spacing,
            )

    def _draw_logo(
        self,
        *,
        canvas: Image.Image,
        template: FeaturedImageTemplateSpec,
        logo_data_url: str | None,
    ) -> None:
        logo = template.logo_zone
        if not (logo_data_url and logo.enabled and logo.include_if_logo_available):
            return

        logo_image = self._decode_logo_data_url(logo_data_url)
        if logo_image is None:
            return

        box_width = max(1, int(logo.width * self.width))
        box_height = max(1, int(logo.height * self.height))
        logo_image.thumbnail((box_width, box_height), Image.Resampling.LANCZOS)

        if logo.opacity < 1.0:
            alpha_channel = logo_image.getchannel("A")
            alpha_channel = alpha_channel.point(
                lambda value: int(value * max(0.0, min(1.0, logo.opacity)))
            )
            logo_image.putalpha(alpha_channel)

        x = int(logo.x * self.width) + max(0, (box_width - logo_image.width) // 2)
        y = int(logo.y * self.height) + max(0, (box_height - logo_image.height) // 2)
        canvas.alpha_composite(logo_image, dest=(x, y))

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

    def _rgb(self, value: str, *, fallback: str) -> tuple[int, int, int]:
        candidate = self._safe_color(value, fallback=fallback)
        try:
            rgb = ImageColor.getrgb(candidate)
        except ValueError:
            rgb = ImageColor.getrgb(fallback)
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    def _rgba(self, value: str, *, alpha: float, fallback: str = "#000000") -> tuple[int, int, int, int]:
        rgb = self._rgb(value, fallback=fallback)
        clamped_alpha = int(max(0.0, min(1.0, alpha)) * 255)
        return rgb[0], rgb[1], rgb[2], clamped_alpha

    @staticmethod
    def _interpolated_color(
        *,
        stops: list[tuple[int, tuple[int, int, int]]],
        position: int,
    ) -> tuple[int, int, int]:
        for index in range(1, len(stops)):
            left_pos, left_color = stops[index - 1]
            right_pos, right_color = stops[index]
            if position > right_pos:
                continue
            span = max(1, right_pos - left_pos)
            ratio = (position - left_pos) / span
            return (
                int(left_color[0] + (right_color[0] - left_color[0]) * ratio),
                int(left_color[1] + (right_color[1] - left_color[1]) * ratio),
                int(left_color[2] + (right_color[2] - left_color[2]) * ratio),
            )
        return stops[-1][1]

    @staticmethod
    def _load_font(*, font_size_px: int, font_weight: int) -> FontType:
        preferred = ["DejaVuSans-Bold.ttf"] if font_weight >= 600 else ["DejaVuSans.ttf"]
        preferred.extend(["Arial.ttf", "Helvetica.ttf"])
        for font_name in preferred:
            try:
                return ImageFont.truetype(font_name, size=max(12, font_size_px))
            except OSError:
                continue
        return ImageFont.load_default()

    def _wrap_lines(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: FontType,
        max_width: int,
        max_lines: int,
        letter_spacing: int,
    ) -> list[str]:
        words = " ".join(text.split()).split(" ")
        if not words or words == [""]:
            return []

        lines: list[str] = []
        current = ""
        word_index = 0
        while word_index < len(words):
            word = words[word_index]
            candidate = word if not current else f"{current} {word}"
            if self._measure_text(
                draw=draw,
                text=candidate,
                font=font,
                letter_spacing=letter_spacing,
            ) <= max_width:
                current = candidate
                word_index += 1
                continue

            if current:
                lines.append(current)
                current = ""
                if len(lines) >= max_lines:
                    lines[-1] = self._truncate_with_ellipsis(
                        draw=draw,
                        text=lines[-1],
                        font=font,
                        max_width=max_width,
                        letter_spacing=letter_spacing,
                    )
                    return lines
                continue

            truncated = self._truncate_text_to_width(
                draw=draw,
                text=word,
                font=font,
                max_width=max_width,
                letter_spacing=letter_spacing,
            )
            if not truncated:
                break
            lines.append(truncated)
            remainder = word[len(truncated) :].strip()
            if remainder:
                words[word_index] = remainder
            else:
                word_index += 1
            if len(lines) >= max_lines:
                lines[-1] = self._truncate_with_ellipsis(
                    draw=draw,
                    text=lines[-1],
                    font=font,
                    max_width=max_width,
                    letter_spacing=letter_spacing,
                )
                return lines

        if current:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = self._truncate_with_ellipsis(
                draw=draw,
                text=lines[-1],
                font=font,
                max_width=max_width,
                letter_spacing=letter_spacing,
            )
        return lines

    def _truncate_with_ellipsis(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: FontType,
        max_width: int,
        letter_spacing: int,
    ) -> str:
        suffix = "..."
        base = self._truncate_text_to_width(
            draw=draw,
            text=text,
            font=font,
            max_width=max_width,
            letter_spacing=letter_spacing,
            suffix=suffix,
        )
        if not base:
            return ""
        return f"{base}{suffix}"

    def _truncate_text_to_width(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: FontType,
        max_width: int,
        letter_spacing: int,
        suffix: str = "",
    ) -> str:
        if self._measure_text(
            draw=draw,
            text=text + suffix,
            font=font,
            letter_spacing=letter_spacing,
        ) <= max_width:
            return text

        low, high = 0, len(text)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = text[:mid].rstrip()
            if self._measure_text(
                draw=draw,
                text=candidate + suffix,
                font=font,
                letter_spacing=letter_spacing,
            ) <= max_width:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best

    @staticmethod
    def _measure_text(
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: FontType,
        letter_spacing: int,
    ) -> float:
        if not text:
            return 0.0
        width = float(draw.textlength(text, font=font))
        return width + max(0, len(text) - 1) * max(0, letter_spacing)

    @staticmethod
    def _draw_text(
        *,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        font: FontType,
        fill: tuple[int, int, int, int],
        letter_spacing: int,
    ) -> None:
        if letter_spacing <= 0:
            draw.text((x, y), text, font=font, fill=fill)
            return

        cursor = x
        for char in text:
            draw.text((cursor, y), char, font=font, fill=fill)
            cursor += int(draw.textlength(char, font=font)) + letter_spacing

    @staticmethod
    def _decode_logo_data_url(data_url: str) -> Image.Image | None:
        if not data_url.startswith("data:"):
            return None
        header, _, payload = data_url.partition(",")
        if not payload:
            return None
        if ";base64" not in header:
            return None

        try:
            raw = base64.b64decode(payload, validate=False)
        except (ValueError, TypeError):
            return None

        try:
            image = Image.open(BytesIO(raw))
            return image.convert("RGBA")
        except OSError:
            return None

    @staticmethod
    def _detect_size(payload: bytes) -> tuple[int, int]:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
        return int(width), int(height)
