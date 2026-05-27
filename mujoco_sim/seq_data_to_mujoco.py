#!/usr/bin/env python3
"""
直接从 seq_extract/seq_data/*.npz 转换到 MuJoCo 轨迹 NPZ
使用 LSTM 输出的 bezier 参数直接重建全局坐标，不依赖图像处理

原理（来自 gif_making.py）：
- 每个 round 的 cursor 位置固定（来自 init_cursors）
- window_size 用每个笔段的 scaling 参数追踪
- 笔段全局坐标 = cursor_pos + x2y2 * (window_size/2)  （像素坐标）
"""

import numpy as np
import os


def bezier_sample(start, ctrl, end, n_points):
    """采样二次贝塞尔曲线"""
    t = np.linspace(0, 1, n_points)
    x = (1-t)**2 * start[0] + 2*(1-t)*t * ctrl[0] + t**2 * end[0]
    y = (1-t)**2 * start[1] + 2*(1-t)*t * ctrl[1] + t**2 * end[1]
    return np.stack([x, y], axis=1)

def densify_polyline_3d(x, y, z, max_step_m):
    """
    Insert linear-interpolated points so that consecutive 3D distance <= max_step_m.

    This increases "control points" count while preserving the path shape.
    """
    if max_step_m is None:
        return x, y, z
    if max_step_m <= 0:
        raise ValueError(f"max_step_m must be > 0, got {max_step_m}")
    if len(x) < 2:
        return x, y, z

    x2 = [float(x[0])]
    y2 = [float(y[0])]
    z2 = [float(z[0])]
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
    return np.array(x2, dtype=np.float64), np.array(y2, dtype=np.float64), np.array(z2, dtype=np.float64)


def seq_data_to_mujoco(seq_npz_path, output_npz_path,
                       char_size=0.12,
                       pen_depth=-0.002,
                       lift_height=0.05,
                       raster_size=128,
                       min_window_size=32,
                       bezier_samples=8,
                       max_step_mm=None):
    """
    将 seq_data NPZ 转换为 MuJoCo 轨迹 NPZ

    Args:
        seq_npz_path: seq_data 输入文件路径
        output_npz_path: 输出 MuJoCo NPZ 路径
        char_size: 字符在真实世界中的大小（米），默认 12cm
        pen_depth: 笔接触纸面的 Z 坐标（米，负值），默认 -2mm
        lift_height: 抬笔高度（米），默认 5cm
        raster_size: 模型的 raster_size，默认 128
        min_window_size: 模型的 min_window_size，默认 32
        bezier_samples: 每条贝塞尔曲线的采样点数，默认 8
        max_step_mm: 可选，控制点增密：限制相邻点最大3D距离（mm）
    """
    d = np.load(seq_npz_path, allow_pickle=True)
    strokes_data = d['strokes_data']  # (N, 7): pen_state, x1, y1, x2, y2, width, scaling
    init_cursors = d['init_cursors']  # (num_rounds, 2), in [0,1] normalized image coords
    image_size = int(d['image_size'])
    round_lengths = d['round_length']  # (num_rounds,)

    print(f"  image_size: {image_size}")
    print(f"  rounds: {len(round_lengths)}, lengths: {round_lengths}")
    print(f"  total strokes: {len(strokes_data)}")

    all_x, all_y, all_z = [], [], []

    # 收集所有坐标用于全局归一化
    all_points_px = []  # 用于计算字符边界框

    # 第一遍：计算所有点的像素坐标（用于归一化）
    cursor_idx = 0
    stroke_idx = 0
    for round_idx in range(len(round_lengths)):
        round_length = int(round_lengths[round_idx])
        cursor_pos = init_cursors[cursor_idx]  # (2), in [0,1]
        cursor_idx += 1

        prev_scaling = 1.0
        prev_window_size = float(raster_size)

        for rel_i in range(round_length):
            # 计算当前 window_size
            curr_window_size = prev_scaling * prev_window_size
            curr_window_size = max(min_window_size, min(image_size, curr_window_size))

            pen_state = int(strokes_data[stroke_idx, 0])
            x1, y1 = strokes_data[stroke_idx, 1], strokes_data[stroke_idx, 2]
            x2, y2 = strokes_data[stroke_idx, 3], strokes_data[stroke_idx, 4]
            scaling = strokes_data[stroke_idx, 6]

            # 全局像素坐标
            cursor_px = cursor_pos * image_size  # (2)
            end_px = cursor_px + np.array([x2, y2]) * (curr_window_size / 2)

            if pen_state == 0:  # 绘制笔段
                ctrl_px = cursor_px + np.array([x1, y1]) * (curr_window_size / 2)
                pts = bezier_sample(cursor_px, ctrl_px, end_px, bezier_samples)
                all_points_px.extend(pts.tolist())

            prev_scaling = scaling
            prev_window_size = curr_window_size
            stroke_idx += 1

    if not all_points_px:
        raise ValueError("未检测到任何绘制笔段")

    all_points_px = np.array(all_points_px)
    x_min_px = all_points_px[:, 0].min()
    x_max_px = all_points_px[:, 0].max()
    y_min_px = all_points_px[:, 1].min()
    y_max_px = all_points_px[:, 1].max()
    scale_px = max(x_max_px - x_min_px, y_max_px - y_min_px)

    print(f"  字符像素范围: X [{x_min_px:.1f}, {x_max_px:.1f}]  Y [{y_min_px:.1f}, {y_max_px:.1f}]")
    print(f"  像素 scale: {scale_px:.1f}")

    def px_to_3d(px_x, px_y):
        """像素坐标 → 3D 世界坐标"""
        x = (px_x - x_min_px) / scale_px * char_size
        y = (px_y - y_min_px) / scale_px * char_size
        return x, y

    # 第二遍：生成轨迹
    cursor_idx = 0
    stroke_idx = 0
    pen_is_down = False

    for round_idx in range(len(round_lengths)):
        round_length = int(round_lengths[round_idx])
        cursor_pos = init_cursors[cursor_idx]
        cursor_idx += 1

        prev_scaling = 1.0
        prev_window_size = float(raster_size)

        for rel_i in range(round_length):
            curr_window_size = prev_scaling * prev_window_size
            curr_window_size = max(min_window_size, min(image_size, curr_window_size))

            pen_state = int(strokes_data[stroke_idx, 0])
            x1, y1 = strokes_data[stroke_idx, 1], strokes_data[stroke_idx, 2]
            x2, y2 = strokes_data[stroke_idx, 3], strokes_data[stroke_idx, 4]
            scaling = strokes_data[stroke_idx, 6]

            cursor_px = cursor_pos * image_size
            end_px = cursor_px + np.array([x2, y2]) * (curr_window_size / 2)
            ctrl_px = cursor_px + np.array([x1, y1]) * (curr_window_size / 2)

            end_3d = px_to_3d(end_px[0], end_px[1])

            if pen_state == 0:  # 绘制笔段
                if pen_is_down:
                    # 已在纸上，直接画贝塞尔
                    pts = bezier_sample(cursor_px, ctrl_px, end_px, bezier_samples)
                    for p in pts[1:]:  # 跳过第一个（已在该位置）
                        bx, by = px_to_3d(p[0], p[1])
                        all_x.append(bx)
                        all_y.append(by)
                        all_z.append(pen_depth)
                else:
                    # 笔抬起，需要移动到起点然后落笔
                    start_3d = px_to_3d(cursor_px[0], cursor_px[1])

                    # 悬停到起点上方
                    if len(all_x) > 0:
                        all_x.append(all_x[-1])
                        all_y.append(all_y[-1])
                        all_z.append(lift_height)
                    all_x.append(start_3d[0])
                    all_y.append(start_3d[1])
                    all_z.append(lift_height)
                    # 落笔
                    all_x.append(start_3d[0])
                    all_y.append(start_3d[1])
                    all_z.append(pen_depth)
                    pen_is_down = True

                    # 画贝塞尔
                    pts = bezier_sample(cursor_px, ctrl_px, end_px, bezier_samples)
                    for p in pts[1:]:
                        bx, by = px_to_3d(p[0], p[1])
                        all_x.append(bx)
                        all_y.append(by)
                        all_z.append(pen_depth)

            else:  # pen_state == 1，抬笔移动
                if pen_is_down:
                    # 抬笔
                    all_x.append(all_x[-1])
                    all_y.append(all_y[-1])
                    all_z.append(lift_height)
                    pen_is_down = False

                # 悬停移动到目标位置
                all_x.append(end_3d[0])
                all_y.append(end_3d[1])
                all_z.append(lift_height)

            prev_scaling = scaling
            prev_window_size = curr_window_size
            stroke_idx += 1

    # 最后抬笔
    if pen_is_down and len(all_x) > 0:
        all_x.append(all_x[-1])
        all_y.append(all_y[-1])
        all_z.append(lift_height)

    pos_3d_x = np.array(all_x, dtype=np.float64)
    pos_3d_y = np.array(all_y, dtype=np.float64)
    pos_3d_z = np.array(all_z, dtype=np.float64)

    if max_step_mm is not None:
        before = len(pos_3d_x)
        max_step_m = float(max_step_mm) / 1000.0
        pos_3d_x, pos_3d_y, pos_3d_z = densify_polyline_3d(pos_3d_x, pos_3d_y, pos_3d_z, max_step_m)
        print(f"  🔧 增密: {before} → {len(pos_3d_x)} (max_step={max_step_mm}mm)")

    print(f"  总控制点数: {len(pos_3d_x)}")
    print(f"  X 范围: [{pos_3d_x.min():.4f}, {pos_3d_x.max():.4f}] m")
    print(f"  Y 范围: [{pos_3d_y.min():.4f}, {pos_3d_y.max():.4f}] m")

    os.makedirs(os.path.dirname(output_npz_path) or '.', exist_ok=True)
    np.savez(output_npz_path,
             pos_3d_x=pos_3d_x,
             pos_3d_y=pos_3d_y,
             pos_3d_z=pos_3d_z)
    print(f"  ✅ 已保存: {output_npz_path}")
    return output_npz_path


if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="从 seq_data/*.npz 转换为 MuJoCo 轨迹 NPZ")
    parser.add_argument("--input", "-i", type=str, default=None, help="输入 seq_data npz 文件（不填则使用默认批处理目录）")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 MuJoCo npz 文件路径（配合 --input 使用）")
    parser.add_argument("--char-size", type=float, default=0.12, help="字符大小（米）")
    parser.add_argument("--pen-depth", type=float, default=-0.002, help="落笔深度（米，负值）")
    parser.add_argument("--lift-height", type=float, default=0.05, help="抬笔高度（米）")
    parser.add_argument("--raster-size", type=int, default=128, help="模型 raster_size")
    parser.add_argument("--min-window-size", type=int, default=32, help="模型 min_window_size")
    parser.add_argument("--bezier-samples", type=int, default=8, help="每段贝塞尔采样点数（越大控制点越多）")
    parser.add_argument("--max-step-mm", type=float, default=None, help="可选增密：限制相邻点最大3D距离（mm）")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            raise SystemExit("使用 --input 时必须同时指定 --output")
        seq_data_to_mujoco(
            args.input,
            args.output,
            char_size=args.char_size,
            pen_depth=args.pen_depth,
            lift_height=args.lift_height,
            raster_size=args.raster_size,
            min_window_size=args.min_window_size,
            bezier_samples=args.bezier_samples,
            max_step_mm=args.max_step_mm,
        )
    else:
        # Backward-compatible default batch conversion
        seq_dir = '/home/lkh/vscode/CalliRewrite/seq_extract/outputs/sampling/__new_train_phase_2/seq_data'
        out_dir = '/home/lkh/vscode/CalliRewrite/mujoco_sim/demo_outputs/mujoco_seqdata'
        os.makedirs(out_dir, exist_ok=True)

        npz_files = sorted(glob.glob(f'{seq_dir}/*.npz'))
        print(f"找到 {len(npz_files)} 个 seq_data 文件")

        for path in npz_files:
            name = os.path.basename(path)
            out_path = os.path.join(out_dir, name)
            print(f"\n处理: {name}")
            seq_data_to_mujoco(
                path,
                out_path,
                char_size=args.char_size,
                pen_depth=args.pen_depth,
                lift_height=args.lift_height,
                raster_size=args.raster_size,
                min_window_size=args.min_window_size,
                bezier_samples=args.bezier_samples,
                max_step_mm=args.max_step_mm,
            )
