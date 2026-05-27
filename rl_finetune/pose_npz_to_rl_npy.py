#!/usr/bin/env python3
"""
Convert pose trajectory .npz files into the .npy skeleton format used by CalliEnv.

Input .npz format:
    pos_3d_x: (N,) x positions in meters
    pos_3d_y: (N,) y positions in meters
    pos_3d_z: (N,) z positions in meters

Output .npy format:
    (M, 7) float32 array of quadratic Bezier rows:
    [p_t, x0, y0, x1, y1, x2, y2]

CalliEnv/skel_utils.py interprets p_t == 1 as a stroke separator and p_t == 0
as a drawable Bezier segment in image pixel coordinates.

Useful direction fixes:
    --transform flip-y      mirror vertically
    --transform flip-x      mirror horizontally
    --transform rot180      rotate 180 degrees
    --transform swap-xy     exchange x/y axes
    --transform swap-xy flip-y
                            apply multiple transforms in order
    --reverse-point-order   reverse drawing direction within each stroke
    --reverse-stroke-order  draw strokes in reverse order
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


TRANSFORM_CHOICES = ("identity", "flip-x", "flip-y", "swap-xy", "rot90-cw", "rot90-ccw", "rot180")


def load_pose_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    required = ("pos_3d_x", "pos_3d_y", "pos_3d_z")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} missing keys: {', '.join(missing)}")

    x = np.asarray(data["pos_3d_x"], dtype=np.float64).reshape(-1)
    y = np.asarray(data["pos_3d_y"], dtype=np.float64).reshape(-1)
    z = np.asarray(data["pos_3d_z"], dtype=np.float64).reshape(-1)
    if not (len(x) == len(y) == len(z)):
        raise ValueError(f"{path} has length mismatch: x={len(x)}, y={len(y)}, z={len(z)}")
    if len(x) < 2:
        raise ValueError(f"{path} needs at least 2 points, got {len(x)}")
    return x, y, z


def split_pen_down_strokes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pen_down_z: float,
) -> List[np.ndarray]:
    pen_down = z <= pen_down_z
    strokes: List[np.ndarray] = []
    current: List[Tuple[float, float]] = []

    for xi, yi, is_down in zip(x, y, pen_down):
        if is_down:
            current.append((float(xi), float(yi)))
        elif current:
            if len(current) >= 2:
                strokes.append(np.asarray(current, dtype=np.float64))
            current = []

    if len(current) >= 2:
        strokes.append(np.asarray(current, dtype=np.float64))

    return strokes


def meter_to_pixel_strokes(
    strokes: Sequence[np.ndarray],
    image_size: int,
    alpha: float,
    normalize_bbox: bool,
    padding: float,
) -> List[np.ndarray]:
    if image_size <= 0:
        raise ValueError(f"image_size must be > 0, got {image_size}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if padding < 0 or padding >= image_size / 2:
        raise ValueError(f"padding must be in [0, image_size/2), got {padding}")

    converted = [stroke.copy() for stroke in strokes]
    if not converted:
        return []

    if normalize_bbox:
        all_points = np.vstack(converted)
        xy_min = all_points.min(axis=0)
        xy_max = all_points.max(axis=0)
        span = float(np.max(xy_max - xy_min))
        if span <= 0:
            raise ValueError("Cannot normalize a zero-size trajectory bbox.")
        scale = (image_size - 2.0 * padding) / span
        for stroke in converted:
            stroke -= xy_min
            stroke *= scale
            stroke += padding
    else:
        scale = image_size / alpha
        for stroke in converted:
            stroke *= scale

    return converted


def transform_pixel_strokes(
    strokes: Sequence[np.ndarray],
    image_size: int,
    transforms: Sequence[str],
    reverse_point_order: bool,
    reverse_stroke_order: bool,
) -> List[np.ndarray]:
    converted = [stroke.copy() for stroke in strokes]
    if not converted:
        return []

    for transform in transforms:
        if transform == "identity":
            continue
        for stroke in converted:
            x = stroke[:, 0].copy()
            y = stroke[:, 1].copy()
            if transform == "flip-x":
                stroke[:, 0] = image_size - x
            elif transform == "flip-y":
                stroke[:, 1] = image_size - y
            elif transform == "swap-xy":
                stroke[:, 0] = y
                stroke[:, 1] = x
            elif transform == "rot90-cw":
                stroke[:, 0] = image_size - y
                stroke[:, 1] = x
            elif transform == "rot90-ccw":
                stroke[:, 0] = y
                stroke[:, 1] = image_size - x
            elif transform == "rot180":
                stroke[:, 0] = image_size - x
                stroke[:, 1] = image_size - y
            else:
                raise ValueError(f"Unsupported transform: {transform}")

    if reverse_point_order:
        converted = [stroke[::-1].copy() for stroke in converted]
    if reverse_stroke_order:
        converted = list(reversed(converted))

    return converted


def remove_too_close_points(stroke: np.ndarray, min_step_px: float) -> np.ndarray:
    if len(stroke) <= 1:
        return stroke

    kept = [stroke[0]]
    for point in stroke[1:]:
        if np.linalg.norm(point - kept[-1]) >= min_step_px:
            kept.append(point)

    if len(kept) == 1 and len(stroke) > 1:
        kept.append(stroke[-1])
    elif np.linalg.norm(stroke[-1] - kept[-1]) > 1e-6:
        kept.append(stroke[-1])

    return np.asarray(kept, dtype=np.float64)


def stroke_to_bezier_rows(stroke: np.ndarray, min_segment_px: float) -> List[List[float]]:
    stroke = remove_too_close_points(stroke, min_segment_px)
    if len(stroke) < 2:
        return []

    rows: List[List[float]] = []
    start = stroke[0]
    rows.append([1.0, start[0], start[1], start[0], start[1], start[0], start[1]])

    for prev, curr in zip(stroke[:-1], stroke[1:]):
        if np.linalg.norm(curr - prev) < min_segment_px:
            continue
        ctrl = (prev + curr) * 0.5
        rows.append([0.0, prev[0], prev[1], ctrl[0], ctrl[1], curr[0], curr[1]])

    return rows if len(rows) > 1 else []


def convert_pose_npz_to_rl_npy(
    src: Path,
    dst: Path,
    image_size: int,
    alpha: float,
    pen_down_z: float,
    min_segment_px: float,
    normalize_bbox: bool,
    padding: float,
    transforms: Sequence[str],
    reverse_point_order: bool,
    reverse_stroke_order: bool,
    overwrite: bool,
) -> Tuple[int, int]:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")

    x, y, z = load_pose_npz(src)
    strokes_m = split_pen_down_strokes(x, y, z, pen_down_z)
    if not strokes_m:
        raise ValueError(f"{src} contains no pen-down strokes with pos_3d_z <= {pen_down_z}")

    strokes_px = meter_to_pixel_strokes(strokes_m, image_size, alpha, normalize_bbox, padding)
    strokes_px = transform_pixel_strokes(
        strokes_px,
        image_size,
        transforms,
        reverse_point_order,
        reverse_stroke_order,
    )
    rows: List[List[float]] = []
    for stroke in strokes_px:
        rows.extend(stroke_to_bezier_rows(stroke, min_segment_px))

    if not rows:
        raise ValueError(f"{src} produced no drawable Bezier segments.")

    arr = np.asarray(rows, dtype=np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)
    return len(strokes_m), arr.shape[0]


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Input file must be .npz: {input_path}")
        yield input_path
    elif input_path.is_dir():
        yield from sorted(input_path.glob("*.npz"))
    else:
        raise FileNotFoundError(input_path)


def output_path_for(src: Path, input_path: Path, output: Path | None) -> Path:
    if output is None:
        return src.with_suffix(".npy")
    if input_path.is_file():
        if output.suffix == ".npy":
            return output
        return output / src.with_suffix(".npy").name
    return output / src.with_suffix(".npy").name


def flatten_transforms(values: Sequence[Sequence[str]]) -> List[str]:
    return [transform for group in values for transform in group]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input pose .npz file or directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .npy file or directory")
    parser.add_argument("--image-size", type=int, default=256, help="RL image coordinate size")
    parser.add_argument("--alpha", type=float, default=0.1, help="Character size in meters used for meter->pixel scaling")
    parser.add_argument("--pen-down-z", type=float, default=0.0, help="Points with z <= this value are treated as pen-down")
    parser.add_argument("--min-segment-px", type=float, default=2.0, help="Skip Bezier segments shorter than this many pixels")
    parser.add_argument("--normalize-bbox", action="store_true", help="Fit the trajectory bbox into image coordinates")
    parser.add_argument("--padding", type=float, default=8.0, help="Pixel padding used with --normalize-bbox")
    parser.add_argument(
        "--transform",
        action="append",
        nargs="+",
        default=[],
        choices=TRANSFORM_CHOICES,
        metavar="TRANSFORM",
        help=(
            "Pixel-space transform(s) to apply after scaling, in order. "
            "Can be used as '--transform swap-xy flip-y' or repeated as "
            "'--transform swap-xy --transform flip-y'."
        ),
    )
    parser.add_argument(
        "--reverse-point-order",
        action="store_true",
        help="Reverse drawing direction inside each pen-down stroke.",
    )
    parser.add_argument(
        "--reverse-stroke-order",
        action="store_true",
        help="Reverse the order of pen-down strokes.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    args = parser.parse_args()
    args.transform = flatten_transforms(args.transform)
    return args


def main() -> int:
    args = parse_args()
    input_path = args.input
    files = list(iter_input_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {input_path}")

    for src in files:
        dst = output_path_for(src, input_path, args.output)
        stroke_count, row_count = convert_pose_npz_to_rl_npy(
            src=src,
            dst=dst,
            image_size=args.image_size,
            alpha=args.alpha,
            pen_down_z=args.pen_down_z,
            min_segment_px=args.min_segment_px,
            normalize_bbox=args.normalize_bbox,
            padding=args.padding,
            transforms=args.transform,
            reverse_point_order=args.reverse_point_order,
            reverse_stroke_order=args.reverse_stroke_order,
            overwrite=args.overwrite,
        )
        print(f"{src} -> {dst} ({stroke_count} strokes, {row_count} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


'''
python3 pose_npz_to_rl_npy.py  data/train_data/qin.npz \
-o data/train_data/7.npy \
--image-size 256 \
--alpha 0.1 \
--overwrite \
--transform rot90-cw \
--transform flip-x
'''
