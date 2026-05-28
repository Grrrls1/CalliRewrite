#!/usr/bin/env python3
"""
通用轨迹转换和可视化工具
支持从RL输出(.npy)转换为机器人控制点(.npz)并生成轨迹可视化
"""

import numpy as np
import os
import sys
import argparse

# ============================================================================
# Trajectory densification
# ============================================================================

def densify_polyline_3d(x, y, z, max_step_m, densify_flags=None):
    """
    Insert linear-interpolated points so that consecutive 3D distance <= max_step_m.

    Args:
        x, y, z: 1D sequences with same length
        max_step_m: float meters, must be > 0
        densify_flags: optional per-point bool sequence. A segment is densified
            only when both adjacent points are True.

    Returns:
        (x2, y2, z2): lists
    """
    if max_step_m is None:
        return list(x), list(y), list(z)
    if max_step_m <= 0:
        raise ValueError(f"max_step_m must be > 0, got {max_step_m}")
    if len(x) != len(y) or len(y) != len(z):
        raise ValueError("x/y/z length mismatch")
    if densify_flags is None:
        densify_flags = [True] * len(x)
    if len(densify_flags) != len(x):
        raise ValueError("densify_flags length mismatch")
    if len(x) < 2:
        return list(x), list(y), list(z)

    x2, y2, z2 = [float(x[0])], [float(y[0])], [float(z[0])]
    for i in range(1, len(x)):
        x0, y0, z0 = float(x[i - 1]), float(y[i - 1]), float(z[i - 1])
        x1, y1, z1 = float(x[i]), float(y[i]), float(z[i])
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        should_densify = bool(densify_flags[i - 1]) and bool(densify_flags[i])
        n_seg = int(np.ceil(dist / max_step_m)) if should_densify and dist > 0 else 1
        # Add intermediate points (exclude start, include end)
        for k in range(1, n_seg + 1):
            t = k / n_seg
            x2.append(x0 + dx * t)
            y2.append(y0 + dy * t)
            z2.append(z0 + dz * t)
    return x2, y2, z2


def smooth_contact_z(z, pen_down, window):
    """Moving-average Z only while the pen is down; keep pen-up heights intact."""
    if window is None or window <= 1:
        return list(z)
    if window % 2 == 0:
        raise ValueError("z_smooth_window must be odd so smoothing is centered")

    z_arr = np.asarray(z, dtype=float)
    down = np.asarray(pen_down, dtype=bool)
    smoothed = z_arr.copy()
    half = window // 2
    for i in range(len(z_arr)):
        if not down[i]:
            continue
        lo = max(0, i - half)
        hi = min(len(z_arr), i + half + 1)
        mask = down[lo:hi]
        if np.any(mask):
            smoothed[i] = float(np.mean(z_arr[lo:hi][mask]))
    return smoothed.tolist()


def simplify_polyline_for_robot(x, y, z, pen_down, min_step_m):
    """
    Drop points closer than min_step_m while preserving endpoints and pen state changes.

    This is intended for blocking robot controllers where thousands of tiny
    segments cause start-stop jitter.
    """
    if min_step_m is None:
        return list(x), list(y), list(z), list(pen_down)
    if min_step_m <= 0:
        raise ValueError(f"min_step_m must be > 0, got {min_step_m}")
    if len(x) != len(y) or len(y) != len(z) or len(z) != len(pen_down):
        raise ValueError("x/y/z/pen_down length mismatch")
    if len(x) < 3:
        return list(x), list(y), list(z), list(pen_down)

    keep = [0]
    last_kept = np.asarray([x[0], y[0], z[0]], dtype=float)
    for i in range(1, len(x) - 1):
        state_changed = bool(pen_down[i]) != bool(pen_down[i - 1])
        next_state_changes = bool(pen_down[i]) != bool(pen_down[i + 1])
        point = np.asarray([x[i], y[i], z[i]], dtype=float)
        far_enough = float(np.linalg.norm(point - last_kept)) >= min_step_m
        if state_changed or next_state_changes or far_enough:
            keep.append(i)
            last_kept = point
    keep.append(len(x) - 1)

    return (
        [float(x[i]) for i in keep],
        [float(y[i]) for i in keep],
        [float(z[i]) for i in keep],
        [bool(pen_down[i]) for i in keep],
    )


TRANSFORM_CHOICES = (
    "identity",
    "rotate-cw",
    "rotate-ccw",
    "rotate-180",
    "flip-x",
    "flip-y",
    "swap-xy",
)


def apply_transform_op(x, y, op):
    """Apply one normalized unit-square coordinate transform."""
    if op == "identity":
        return x, y
    if op == "rotate-cw":
        return y, 1.0 - x
    if op == "rotate-ccw":
        return 1.0 - y, x
    if op == "rotate-180":
        return 1.0 - x, 1.0 - y
    if op == "flip-x":
        return 1.0 - x, y
    if op == "flip-y":
        return x, 1.0 - y
    if op == "swap-xy":
        return y, x
    raise ValueError(f"Unknown transform op: {op}")


def transform_unit_xy(x, y, transform_ops=None):
    """
    Apply a configurable coordinate transform in the normalized unit square.

    With image-style normalized coordinates (x right, y down), the combined
    The default transform is identity; pass --transform to rotate or flip.
    """
    if transform_ops is None:
        transform_ops = ("identity",)
    x = float(x)
    y = float(y)
    for op in transform_ops:
        x, y = apply_transform_op(x, y, op)
    return x, y


# ============================================================================
# 校准函数（从calibrate.py复制）
# ============================================================================

def func_brush_precalibrated(radii):
    """
    预校准的毛笔函数
    如需重新校准，请使用calibrate.py
    """
    if radii >= 0 and radii <= 7.72667536e-04:
        z = -5.9701493 * (radii - 7.72667536e-04) + 4.34974603e-03
    elif radii > 7.72667536e-04 and radii <= 1.78125854e-03:
        z = -1.538473 * (radii - 7.72667536e-04) + 4.34974603e-03
    elif radii > 1.78125854e-03 and radii <= 2.45866277e-03:
        z = -6.028019 * (radii - 1.78125854e-03) + 2.79805600e-03
    elif radii > 2.45866277e-03 and radii < 0.0045:
        z = -2.37843574 * (radii - 2.45866277e-03) - 1.28534957e-03
    else:
        return -0.006
    return z


# ============================================================================
# RL数据转NPZ
# ============================================================================

def convert_rl_to_npz(npy_path, output_npz_path,
                     calibration_func=func_brush_precalibrated,
                     alpha=0.04, beta=0.5, style_type=1,
                     max_step_mm=None,
                     min_step_mm=None,
                     z_smooth_window=1,
                     transform_ops=None,
                     z_shallow=-0.002, z_deep=-0.004):
    """
    将RL输出的.npy文件转换为机器人控制点.npz文件

    Args:
        npy_path: RL输出的.npy文件路径
        output_npz_path: 输出的.npz文件路径
        calibration_func: 校准函数（r -> z），保留兼容；当前默认使用r范围线性映射到z_shallow/z_deep
        alpha: 字符尺寸（米），例如0.04 = 4cm宽
        beta: 笔画宽度对比度调整（0.5 = 正常）
        style_type: 0=隶书（clerical），1=楷书（regular）
        z_shallow: 最小笔宽对应的Z值，默认-0.002m（纸面下2mm）
        z_deep: 最大笔宽对应的Z值，默认-0.004m（纸面下4mm）

    Returns:
        x, y, z: 控制点列表（米）
    """
    # 加载RL状态
    data = np.load(npy_path)
    print(f"加载RL状态: {npy_path}")
    print(f"  数据形状: {data.shape}")
    print(f"  字符尺寸: {alpha*100} cm")
    print(f"  笔画宽度调整: {beta}")
    if transform_ops is None:
        transform_ops = ("identity",)
    print(f"  坐标变换: {' -> '.join(transform_ops)}")

    # 读取当前RL输出里的r范围，并映射到固定下压区间。
    # r越大，笔压越深：min(r)->z_shallow，max(r)->z_deep。
    r_values = data[:, 3].astype(float) * alpha * beta
    r_min = float(np.min(r_values))
    r_max = float(np.max(r_values))
    print(f"  r范围: [{r_min:.6f}, {r_max:.6f}] m")
    print(f"  Z映射: min(r)->{z_shallow*1000:.1f}mm, max(r)->{z_deep*1000:.1f}mm")

    def radius_to_z(radius):
        if abs(r_max - r_min) < 1e-12:
            return 0.5 * (z_shallow + z_deep)
        t = (radius - r_min) / (r_max - r_min)
        t = np.clip(t, 0.0, 1.0)
        return z_shallow + t * (z_deep - z_shallow)

    record_x = []
    record_y = []
    record_z = []
    record_pen_down = []

    for i in range(data.shape[0]):
        # 只使用前4列：p_t, x, y, r（忽略后面的颜色等信息）
        p_t, x, y, r = data[i, :4]

        # RL输出已经是归一化坐标/半径：x,y约在[0,1]，r约在[0,0.0576]
        x = float(x)
        y = float(y)
        r = float(r)
        x, y = transform_unit_xy(x, y, transform_ops)

        # 缩放到实际尺寸
        x_ = x * alpha
        y_ = y * alpha
        r_ = r * alpha * beta

        # 将当前文件内的r线性映射为纸面下2mm到4mm的书写深度。
        h = radius_to_z(r_)

        if p_t == 0:
            # 正常笔画点
            record_x.append(x_)
            record_y.append(y_)
            record_z.append(h)
            record_pen_down.append(True)

        elif p_t == 1:
            # 新笔画开始
            if i == data.shape[0] - 1:
                continue  # 如果是最后一个点则跳过

            if style_type == 1:
                # 楷书：从左上方进入
                record_x.append(x_ - 2 * r_)
                record_y.append(y_ - 2 * r_)
                record_z.append(0.05)  # 抬笔
                record_pen_down.append(False)
            else:
                # 隶书：从上方进入
                record_x.append(x_)
                record_y.append(y_)
                record_z.append(0.05)
                record_pen_down.append(False)

            # 添加当前点（落笔）
            record_x.append(x_)
            record_y.append(y_)
            record_z.append(h)
            record_pen_down.append(True)

            if style_type == 0:
                # 隶书：笔画起始处的回锋
                nxt_x, nxt_y = data[i + 1, 1:3].astype(float)
                nxt_x, nxt_y = transform_unit_xy(nxt_x, nxt_y, transform_ops)
                nxt_vec = np.array([nxt_x, nxt_y]) - np.array([x, y])  # 归一化后计算方向
                vec_norm = np.linalg.norm(nxt_vec)
                if vec_norm > 0:
                    nxt_vec = nxt_vec / vec_norm
                    record_x.append(x_ - 2 * r_ * nxt_vec[0])
                    record_y.append(y_ - 2 * r_ * nxt_vec[1])
                    record_z.append(h)
                    record_pen_down.append(True)
                record_x.append(x_)
                record_y.append(y_)
                record_z.append(h)
                record_pen_down.append(True)

    # 结束时逐渐抬笔
    for _ in range(5):
        record_x.append(record_x[-1])
        record_y.append(record_y[-1])
        record_z.append(record_z[-1] + 0.015)
        record_pen_down.append(False)

    # 可选：控制点增密（限制每段最大步长）。这更适合仿真，真机通常不要开太小。
    if max_step_mm is not None:
        max_step_m = float(max_step_mm) / 1000.0
        before = len(record_x)
        record_x, record_y, record_z = densify_polyline_3d(
            record_x, record_y, record_z, max_step_m, record_pen_down
        )
        record_pen_down = [z < 0.0 for z in record_z]
        print(f"🔧 增密: {before} → {len(record_x)} (max_step={max_step_mm}mm)")

    # 真机后处理：平滑Z并减少过密点，避免逐点阻塞控制造成抖动和极慢执行。
    if z_smooth_window and z_smooth_window > 1:
        record_z = smooth_contact_z(record_z, record_pen_down, z_smooth_window)
        print(f"🔧 Z平滑: contact-only moving average window={z_smooth_window}")

    if min_step_mm is not None:
        min_step_m = float(min_step_mm) / 1000.0
        before = len(record_x)
        record_x, record_y, record_z, record_pen_down = simplify_polyline_for_robot(
            record_x, record_y, record_z, record_pen_down, min_step_m
        )
        print(f"🔧 真机简化: {before} → {len(record_x)} (min_step={min_step_mm}mm)")

    # 保存为npz
    np.savez(output_npz_path,
             pos_3d_x=record_x,
             pos_3d_y=record_y,
             pos_3d_z=record_z)

    print(f"✅ 已转换 {len(record_x)} 个控制点")
    print(f"  X范围: [{np.min(record_x)*1000:.1f}, {np.max(record_x)*1000:.1f}] mm")
    print(f"  Y范围: [{np.min(record_y)*1000:.1f}, {np.max(record_y)*1000:.1f}] mm")
    print(f"  Z范围: [{np.min(record_z)*1000:.1f}, {np.max(record_z)*1000:.1f}] mm")
    print(f"✅ 已保存到: {output_npz_path}")

    return record_x, record_y, record_z


# ============================================================================
# NPZ可视化
# ============================================================================

def visualize_npz_trajectory(npz_file, output_image):
    """可视化NPZ文件中的轨迹"""

    # Lazy import: allow conversion-only usage even if matplotlib is unavailable
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 加载数据
    data = np.load(npz_file)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    print(f"\n可视化轨迹: {npz_file}")
    print(f"  控制点数: {len(x)}")
    print(f"  X范围: [{x.min()*1000:.1f}, {x.max()*1000:.1f}] mm")
    print(f"  Y范围: [{y.min()*1000:.1f}, {y.max()*1000:.1f}] mm")
    print(f"  Z范围: [{z.min()*1000:.1f}, {z.max()*1000:.1f}] mm")

    # 创建图形
    fig = plt.figure(figsize=(12, 10))

    # 1. XY平面轨迹（俯视图）
    ax1 = plt.subplot(2, 2, 1)
    # 根据Z值判断是否接触（Z < 0表示接触纸面）
    contact_mask = z < 0

    # 画非接触部分（虚线）
    if np.any(~contact_mask):
        ax1.plot(x[~contact_mask]*1000, y[~contact_mask]*1000, 'gray',
                linestyle='--', linewidth=1, alpha=0.5, label='Pen Up')

    # 画接触部分（实线）
    if np.any(contact_mask):
        ax1.plot(x[contact_mask]*1000, y[contact_mask]*1000, 'black',
                linewidth=2, label='Writing')

    # 标记起点和终点
    ax1.plot(x[0]*1000, y[0]*1000, 'go', markersize=8, label='Start')
    ax1.plot(x[-1]*1000, y[-1]*1000, 'ro', markersize=8, label='End')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('XY Plane (Top View)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    ax1.invert_yaxis()  # Y轴反转，使其与纸面方向一致

    # 2. XZ平面轨迹（侧视图）
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x*1000, z*1000, 'b-', linewidth=1.5)
    ax2.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Paper')
    ax2.plot(x[0]*1000, z[0]*1000, 'go', markersize=8, label='Start')
    ax2.plot(x[-1]*1000, z[-1]*1000, 'ro', markersize=8, label='End')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('XZ Plane (Side View)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. YZ平面轨迹（前视图）
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(y*1000, z*1000, 'r-', linewidth=1.5)
    ax3.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Paper')
    ax3.plot(y[0]*1000, z[0]*1000, 'go', markersize=8, label='Start')
    ax3.plot(y[-1]*1000, z[-1]*1000, 'ro', markersize=8, label='End')
    ax3.set_xlabel('Y (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('YZ Plane (Front View)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Z坐标随时间变化
    ax4 = plt.subplot(2, 2, 4)
    time_points = np.arange(len(z))
    ax4.plot(time_points, z*1000, 'purple', linewidth=1.5)
    ax4.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Paper')
    ax4.fill_between(time_points, z*1000, 0, where=(z<0), alpha=0.3, color='blue', label='Contact')
    ax4.set_xlabel('Control Point Index')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('Pen Height Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"✅ 轨迹可视化已保存: {output_image}")
    plt.close()

    # 统计信息
    contact_points = np.sum(contact_mask)
    print(f"\n统计信息:")
    print(f"  接触点数: {contact_points}/{len(z)} ({contact_points/len(z)*100:.1f}%)")
    print(f"  轨迹长度: {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))*1000:.1f} mm")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='将RL输出转换为NPZ并生成轨迹可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从npy文件转换并可视化
  python convert_and_visualize.py input.npy

  # 指定输出路径
  python convert_and_visualize.py input.npy -o output.npz --viz output_viz.png

  # 自定义参数
  python convert_and_visualize.py input.npy --alpha 0.05 --beta 0.7

  # 调试轨迹方向：可重复传入，按顺序执行
  python convert_and_visualize.py input.npy -o out.npz --transform rotate-cw --transform flip-x
  python convert_and_visualize.py input.npy -o out.npz --transform identity
  python convert_and_visualize.py input.npy -o out.npz --transform rotate-180

  # 只可视化已有的npz文件
  python convert_and_visualize.py --npz existing.npz --viz output.png
        """
    )

    parser.add_argument('input', nargs='?', help='输入的.npy文件路径')
    parser.add_argument('-o', '--output', help='输出的.npz文件路径（默认：与输入同名）')
    parser.add_argument('--viz', '--visualize', dest='viz_output',
                       help='可视化输出图片路径（默认：npz文件名_trajectory.png）')
    parser.add_argument('--npz', help='直接可视化已有的npz文件（跳过转换步骤）')
    parser.add_argument('--alpha', type=float, default=0.04,
                       help='字符尺寸（米），默认0.04（4cm）')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='笔画宽度调整，默认0.5')
    parser.add_argument('--style', type=int, default=0, choices=[0, 1],
                       help='书法风格：0=隶书（默认），1=楷书')
    parser.add_argument('--max-step-mm', type=float, default=None,
                       help='控制点增密：限制相邻点最大3D距离（mm）。例如 0.3 会显著增加控制点数量')
    parser.add_argument('--min-step-mm', type=float, default=None,
                       help='真机简化：丢弃相邻距离小于该值的控制点（mm），保留抬笔/落笔边界。建议 0.6~1.2')
    parser.add_argument('--z-smooth-window', type=int, default=1,
                       help='真机Z平滑窗口，必须为奇数；只平滑落笔点。建议 3 或 5')
    parser.add_argument('--z-shallow', type=float, default=-0.002,
                       help='最小r对应的Z值（米），默认-0.002，即纸面下2mm')
    parser.add_argument('--z-deep', type=float, default=-0.004,
                       help='最大r对应的Z值（米），默认-0.004，即纸面下4mm')
    parser.add_argument('--transform', action='append', choices=TRANSFORM_CHOICES,
                       help=('坐标变换操作，可重复传入并按顺序执行。'
                             '默认: identity。'
                             '可选: ' + ', '.join(TRANSFORM_CHOICES)))
    parser.add_argument('--no-viz', action='store_true',
                       help='不生成可视化图片')

    args = parser.parse_args()

    # 模式1：只可视化已有的npz文件
    if args.npz:
        npz_path = args.npz
        viz_path = args.viz_output or npz_path.replace('.npz', '_trajectory.png')

        print("=" * 70)
        print("可视化NPZ轨迹")
        print("=" * 70)
        visualize_npz_trajectory(npz_path, viz_path)
        print("=" * 70)
        return

    # 模式2：从npy转换
    if not args.input:
        parser.print_help()
        return

    npy_path = args.input

    if not os.path.exists(npy_path):
        print(f"❌ 错误：找不到输入文件 {npy_path}")
        return

    # 确定输出路径
    if args.output:
        npz_path = args.output
    else:
        # 默认：替换扩展名为.npz
        base_name = os.path.splitext(npy_path)[0]
        npz_path = base_name + '.npz'

    # 确定可视化输出路径
    if args.viz_output:
        viz_path = args.viz_output
    else:
        viz_path = npz_path.replace('.npz', '_trajectory.png')

    print("=" * 70)
    print("RL轨迹转换和可视化")
    print("=" * 70)
    print(f"输入文件: {npy_path}")
    print(f"输出NPZ: {npz_path}")
    transform_ops = tuple(args.transform) if args.transform else ("identity",)
    print(f"坐标变换: {' -> '.join(transform_ops)}")
    if not args.no_viz:
        print(f"可视化图: {viz_path}")
    print("=" * 70)
    print()

    # 转换
    try:
        convert_rl_to_npz(
            npy_path,
            npz_path,
            alpha=args.alpha,
            beta=args.beta,
            style_type=args.style,
            max_step_mm=args.max_step_mm,
            min_step_mm=args.min_step_mm,
            z_smooth_window=args.z_smooth_window,
            transform_ops=transform_ops,
            z_shallow=args.z_shallow,
            z_deep=args.z_deep
        )
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 可视化
    if not args.no_viz:
        print()
        try:
            visualize_npz_trajectory(npz_path, viz_path)
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return

    print()
    print("=" * 70)
    print("✅ 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()


'''
使用示例，将RL输出的.npy文件转换为.npz
python convert_and_visualize.py /home/lkh/vscode/CalliRewrite/rl_finetune/result/demo/arrays/3_250.npy \
  -o rl_npz/3_250.npz \
  --alpha 10 \
  --beta 0.5 \
  --no-viz
'''
