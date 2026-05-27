#!/usr/bin/env python3
"""Render MakeMeAHanzi glyph paths to PNG images.

Examples:
    python3 makemeahanzi/hanzi_to_png.py 西浦
    python3 makemeahanzi/hanzi_to_png.py 西 浦 -o makemeahanzi/output_png --size 256
    python3 makemeahanzi/hanzi_to_png.py 西 -o xi.png --transparent
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw


COMMAND_RE = re.compile(r"[MmLlHhVvQqCcZz]|-?\d+(?:\.\d+)?")
DEFAULT_GRAPHICS = Path(__file__).with_name("graphics.txt")


def load_glyphs(graphics_path: Path, characters: set[str]) -> dict[str, dict]:
    glyphs: dict[str, dict] = {}
    with graphics_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            char = item.get("character")
            if char in characters:
                glyphs[char] = item
                if len(glyphs) == len(characters):
                    break
    return glyphs


def is_command(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def to_canvas(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return x, 900 - y


def quadratic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        points.append((x, y))
    return points


def cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        x = (
            mt * mt * mt * p0[0]
            + 3 * mt * mt * t * p1[0]
            + 3 * mt * t * t * p2[0]
            + t * t * t * p3[0]
        )
        y = (
            mt * mt * mt * p0[1]
            + 3 * mt * mt * t * p1[1]
            + 3 * mt * t * t * p2[1]
            + t * t * t * p3[1]
        )
        points.append((x, y))
    return points


def parse_path(path_data: str, curve_steps: int) -> list[list[tuple[float, float]]]:
    tokens = COMMAND_RE.findall(path_data)
    paths: list[list[tuple[float, float]]] = []
    current_path: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    command = ""
    i = 0

    def read_number() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    def read_point(relative: bool) -> tuple[float, float]:
        x = read_number()
        y = read_number()
        if relative:
            return current[0] + x, current[1] + y
        return x, y

    while i < len(tokens):
        if is_command(tokens[i]):
            command = tokens[i]
            i += 1

        relative = command.islower()
        op = command.upper()

        if op == "M":
            current = read_point(relative)
            start = current
            current_path = [to_canvas(current)]
            paths.append(current_path)
            command = "l" if relative else "L"
        elif op == "L":
            current = read_point(relative)
            current_path.append(to_canvas(current))
        elif op == "H":
            x = read_number()
            current = (current[0] + x, current[1]) if relative else (x, current[1])
            current_path.append(to_canvas(current))
        elif op == "V":
            y = read_number()
            current = (current[0], current[1] + y) if relative else (current[0], y)
            current_path.append(to_canvas(current))
        elif op == "Q":
            control = read_point(relative)
            end = read_point(relative)
            current_path.extend(to_canvas(p) for p in quadratic(current, control, end, curve_steps))
            current = end
        elif op == "C":
            control1 = read_point(relative)
            control2 = read_point(relative)
            end = read_point(relative)
            current_path.extend(
                to_canvas(p) for p in cubic(current, control1, control2, end, curve_steps)
            )
            current = end
        elif op == "Z":
            if current_path and current_path[0] != current_path[-1]:
                current_path.append(current_path[0])
            current = start
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")

    return [path for path in paths if len(path) >= 3]


def scale_polygon(
    polygon: list[tuple[float, float]],
    size: int,
    supersample: int,
) -> list[tuple[int, int]]:
    factor = size / 1024 * supersample
    return [(round(x * factor), round(y * factor)) for x, y in polygon]


def parse_color(value: str) -> tuple[int, int, int, int]:
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6:
        r, g, b = value[0:2], value[2:4], value[4:6]
        return int(r, 16), int(g, 16), int(b, 16), 255
    if len(value) == 8:
        r, g, b, a = value[0:2], value[2:4], value[4:6], value[6:8]
        return int(r, 16), int(g, 16), int(b, 16), int(a, 16)
    raise ValueError("Colors must be #RRGGBB or #RRGGBBAA")


def render_glyph(
    glyph: dict,
    output_path: Path,
    size: int,
    stroke_color: str,
    background: str,
    transparent: bool,
    curve_steps: int,
    supersample: int,
) -> None:
    canvas_size = size * supersample
    bg = (255, 255, 255, 0) if transparent else parse_color(background)
    image = Image.new("RGBA", (canvas_size, canvas_size), bg)
    draw = ImageDraw.Draw(image)
    fill = parse_color(stroke_color)

    for stroke in glyph["strokes"]:
        for polygon in parse_path(stroke, curve_steps):
            draw.polygon(scale_polygon(polygon, size, supersample), fill=fill)

    image = image.resize((size, size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def collect_characters(values: list[str]) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    for value in values:
        for char in value:
            if char.strip() and char not in seen:
                chars.append(char)
                seen.add(char)
    return chars


def output_path_for(char: str, output: Path, total: int) -> Path:
    if total == 1 and output.suffix.lower() == ".png":
        return output
    return output / f"{char}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MakeMeAHanzi characters to PNG.")
    parser.add_argument("characters", nargs="+", help="Chinese characters, e.g. 西浦 or 西 浦")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hanzi_png"),
        help="Output directory, or a .png file when rendering one character.",
    )
    parser.add_argument("--graphics", type=Path, default=DEFAULT_GRAPHICS, help="Path to graphics.txt")
    parser.add_argument("--size", type=int, default=512, help="Output image size in pixels")
    parser.add_argument("--color", default="#000000", help="Stroke color, #RRGGBB or #RRGGBBAA")
    parser.add_argument("--background", default="#ffffff", help="Background color")
    parser.add_argument("--transparent", action="store_true", help="Use transparent background")
    parser.add_argument("--curve-steps", type=int, default=24, help="Samples per Bezier curve")
    parser.add_argument("--supersample", type=int, default=3, help="Antialiasing scale factor")
    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if args.curve_steps <= 0:
        raise SystemExit("--curve-steps must be positive")
    if args.supersample <= 0:
        raise SystemExit("--supersample must be positive")

    chars = collect_characters(args.characters)
    glyphs = load_glyphs(args.graphics, set(chars))
    missing = [char for char in chars if char not in glyphs]
    if missing:
        raise SystemExit(f"Characters not found in {args.graphics}: {''.join(missing)}")

    for char in chars:
        output_path = output_path_for(char, args.output, len(chars))
        render_glyph(
            glyphs[char],
            output_path,
            args.size,
            args.color,
            args.background,
            args.transparent,
            args.curve_steps,
            args.supersample,
        )
        print(output_path)


if __name__ == "__main__":
    main()
