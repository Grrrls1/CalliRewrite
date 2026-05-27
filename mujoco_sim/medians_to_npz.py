#!/usr/bin/env python3
"""
Convert MakeMeAHanzi (makemeahanzi/graphics.txt) stroke medians into a 3D
trajectory .npz compatible with mujoco_simulator.py.

This script uses the "medians" field (per-stroke polyline) as the 2D path.
It then generates pen-up/pen-down segments and maps a (radius -> z) function
similar to convert_and_visualize.py.

e.g python medians_to_npz.py --char 日 --out rl_npy/ri.npz --alpha 0.1 --beta 0.7 --max-step-mm 1 --pen-down-depth-mm 8

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def densify_polyline_3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    max_step_m: Optional[float],
) -> Tuple[List[float], List[float], List[float]]:
    if max_step_m is None:
        return list(map(float, x)), list(map(float, y)), list(map(float, z))
    if max_step_m <= 0:
        raise ValueError(f"max_step_m must be > 0, got {max_step_m}")
    if len(x) != len(y) or len(y) != len(z):
        raise ValueError("x/y/z length mismatch")
    if len(x) < 2:
        return list(map(float, x)), list(map(float, y)), list(map(float, z))

    x2, y2, z2 = [float(x[0])], [float(y[0])], [float(z[0])]
    for i in range(1, len(x)):
        x0, y0, z0 = float(x[i - 1]), float(y[i - 1]), float(z[i - 1])
        x1, y1, z1 = float(x[i]), float(y[i]), float(z[i])
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        n_seg = int(np.ceil(dist / max_step_m)) if dist > 0 else 1
        for k in range(1, n_seg + 1):
            t = k / n_seg
            x2.append(x0 + dx * t)
            y2.append(y0 + dy * t)
            z2.append(z0 + dz * t)
    return x2, y2, z2

def iter_graphics_lines(graphics_path: Path) -> Iterable[dict]:
    with graphics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def find_char_graphics(graphics_path: Path, character: str) -> dict:
    for obj in iter_graphics_lines(graphics_path):
        if obj.get("character") == character:
            return obj
    raise KeyError(f"character {character!r} not found in {graphics_path}")


def median_point_to_unit_xy(x: float, y: float) -> Tuple[float, float]:
    """
    MakeMeAHanzi coordinate system for medians matches strokes:
      upper-left  (0, 900)
      lower-right (1024, -124)
    so y decreases as you move down.

    Map to unit square with y increasing down:
      x01 = x / 1024
      y01 = (900 - y) / 1024
    """
    x01 = float(x) / 1024.0
    y01 = (900.0 - float(y)) / 1024.0
    return x01, y01


def build_trajectory_from_medians(
    medians: Sequence[Sequence[Sequence[float]]],
    *,
    alpha_m: float,
    beta: float,
    radius_px: float,
    pen_up_z: float,
    pen_down_depth_m: float,
    pre_hover_points: int,
    desk_offset_m: float,
    max_step_m: Optional[float],
) -> Tuple[List[float], List[float], List[float]]:
    if alpha_m <= 0:
        raise ValueError(f"alpha_m must be > 0, got {alpha_m}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if radius_px <= 0:
        raise ValueError(f"radius_px must be > 0, got {radius_px}")
    if pre_hover_points < 0:
        raise ValueError(f"pre_hover_points must be >= 0, got {pre_hover_points}")
    if pen_down_depth_m < 0:
        raise ValueError(f"pen_down_depth_m must be >= 0, got {pen_down_depth_m}")

    # Convert a nominal "radius in 0..1024 space" into meters in the same way
    # the RL converter does: scale by alpha then apply beta.
    r01 = float(radius_px) / 1024.0
    r_m = r01 * alpha_m * beta
    z_down = -float(pen_down_depth_m)

    record_x: List[float] = []
    record_y: List[float] = []
    record_z: List[float] = []

    def append_point(x_m: float, y_m: float, z_m: float) -> None:
        record_x.append(float(x_m))
        record_y.append(float(y_m))
        record_z.append(float(z_m))

    is_first_stroke = True
    for stroke in medians:
        if not stroke:
            continue

        # Add start point at pen-up height
        x0_01, y0_01 = median_point_to_unit_xy(stroke[0][0], stroke[0][1])
        x0_m = x0_01 * alpha_m
        y0_m = y0_01 * alpha_m
        append_point(x0_m, y0_m, pen_up_z)

        # Add pre-hover stabilization segment before the first stroke.
        if is_first_stroke and pre_hover_points > 0:
            for _ in range(pre_hover_points):
                append_point(x0_m, y0_m, pen_up_z)
        is_first_stroke = False

        # Collect stroke points for densification
        stroke_x: List[float] = []
        stroke_y: List[float] = []
        stroke_z: List[float] = []
        for pt in stroke:
            x01, y01 = median_point_to_unit_xy(pt[0], pt[1])
            stroke_x.append(x01 * alpha_m)
            stroke_y.append(y01 * alpha_m)
            stroke_z.append(z_down)

        # Densify only the stroke points if max_step_m is set
        if max_step_m is not None:
            stroke_x, stroke_y, stroke_z = densify_polyline_3d(stroke_x, stroke_y, stroke_z, max_step_m)

        # Append densified stroke points
        for sx, sy, sz in zip(stroke_x, stroke_y, stroke_z):
            append_point(sx, sy, sz)

        # Add end point at pen-up height
        append_point(record_x[-1], record_y[-1], pen_up_z)

    # Remove the global densification since we densify per stroke
    # if max_step_m is not None and record_x:
    #     before = len(record_x)
    #     record_x, record_y, record_z = densify_polyline_3d(record_x, record_y, record_z, max_step_m)
    #     print(f"🔧 densify: {before} → {len(record_x)} (max_step={max_step_m*1000:.1f}mm)")

    return record_x, record_y, record_z


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MakeMeAHanzi medians to mujoco .npz trajectory")
    parser.add_argument("--graphics", type=str, default="../makemeahanzi/graphics.txt", help="Path to graphics.txt")
    parser.add_argument("--char", type=str, required=True, help="Single Chinese character, e.g. 永")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--alpha", type=float, default=0.04, help="Character size in meters (default 0.04)")
    parser.add_argument("--beta", type=float, default=0.5, help="Stroke width scale (default 0.5)")
    parser.add_argument("--radius-px", type=float, default=18.0, help="Nominal stroke radius in 0..1024 space")
    parser.add_argument("--pen-up-z", type=float, default=0.01, help="Pen-up Z height (meters)")
    parser.add_argument(
        "--pen-down-depth-mm",
        type=float,
        default=8.0,
        help="Pen-down depth relative to paper plane (millimeters, default 8)",
    )
    parser.add_argument(
        "--pre-hover-points",
        type=int,
        default=20,
        help="Extra pen-up points inserted before first stroke for stabilization",
    )
    parser.add_argument("--desk-offset", type=float, default=0.01, help="Subtract from calibrated Z (meters)")
    parser.add_argument("--max-step-mm", type=float, default=None, help="Optional densify max step length (mm)")
    args = parser.parse_args()

    if len(args.char) != 1:
        raise ValueError("--char must be a single character")

    graphics_path = Path(args.graphics)
    obj = find_char_graphics(graphics_path, args.char)
    medians = obj.get("medians")
    if not medians:
        raise KeyError(f"character {args.char!r} has no 'medians' in {graphics_path}")

    max_step_m = None if args.max_step_mm is None else float(args.max_step_mm) / 1000.0
    pen_down_depth_m = float(args.pen_down_depth_mm) / 1000.0
    x, y, z = build_trajectory_from_medians(
        medians,
        alpha_m=float(args.alpha),
        beta=float(args.beta),
        radius_px=float(args.radius_px),
        pen_up_z=float(args.pen_up_z),
        pen_down_depth_m=pen_down_depth_m,
        pre_hover_points=int(args.pre_hover_points),
        desk_offset_m=float(args.desk_offset),
        max_step_m=max_step_m,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, pos_3d_x=x, pos_3d_y=y, pos_3d_z=z)
    print(f"✅ saved {len(x)} points to: {out_path}")
    print(
        f"   x(mm)=[{np.min(x)*1000:.1f},{np.max(x)*1000:.1f}] "
        f"y(mm)=[{np.min(y)*1000:.1f},{np.max(y)*1000:.1f}] "
        f"z(mm)=[{np.min(z)*1000:.1f},{np.max(z)*1000:.1f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


'''
python medians_to_npz.py --char 永 --out rl_npz/yong.npz --alpha 0.1 --beta 0.5 --max-step-mm 1

'''
