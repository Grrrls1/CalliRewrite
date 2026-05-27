#!/usr/bin/env python3
"""
从书法图像提取密集轨迹 - 生成用于MuJoCo仿真的NPZ文件（支持可变深度）
"""

import os

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label


def _compute_depth_from_radius(radius_px, depth_min_m, depth_max_m):
    """将局部半径映射到深度范围（返回负值 z）。"""
    if len(radius_px) == 0:
        return np.array([], dtype=np.float64)

    r_lo = float(np.percentile(radius_px, 5))
    r_hi = float(np.percentile(radius_px, 95))
    if r_hi <= r_lo + 1e-6:
        return np.full_like(radius_px, -depth_max_m, dtype=np.float64)

    t = (radius_px - r_lo) / (r_hi - r_lo)
    t = np.clip(t, 0.0, 1.0)
    depth_m = depth_min_m + t * (depth_max_m - depth_min_m)
    return -depth_m.astype(np.float64)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    """8邻域计数（不含自身）。"""
    sk = skel.astype(np.uint8)
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=np.uint8,
    )
    return cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)


def _prune_short_spurs(skel: np.ndarray, max_spur_len: int = 10, rounds: int = 2) -> np.ndarray:
    """
    删除骨架短毛刺分支：
    从端点出发走到第一个分叉点/端点，若长度<=max_spur_len则删除该段。
    """
    sk = skel.copy().astype(bool)
    h, w = sk.shape

    def nbrs(px, py):
        for ny in range(max(0, py - 1), min(h, py + 2)):
            for nx in range(max(0, px - 1), min(w, px + 2)):
                if nx == px and ny == py:
                    continue
                if sk[ny, nx]:
                    yield nx, ny

    for _ in range(max(1, rounds)):
        changed = False
        deg = _neighbor_count(sk)
        endpoints = np.column_stack(np.where(sk & (deg == 1)))  # (y, x)

        for ey, ex in endpoints:
            if not sk[ey, ex]:
                continue

            path = [(ex, ey)]
            prev = None
            cx, cy = ex, ey

            while True:
                ns = list(nbrs(cx, cy))
                if prev is not None:
                    ns = [p for p in ns if p != prev]
                if len(ns) == 0:
                    break
                nx, ny = ns[0]
                path.append((nx, ny))
                prev = (cx, cy)
                cx, cy = nx, ny

                d = int(_neighbor_count(sk)[cy, cx])
                if d != 2:
                    break

                if len(path) > max_spur_len:
                    break

            end_deg = int(_neighbor_count(sk)[cy, cx])
            if len(path) <= max_spur_len and end_deg >= 3:
                for px, py in path[:-1]:
                    sk[py, px] = False
                changed = True

        if not changed:
            break

    return sk


def extract_skeleton_trajectory(
    image_path,
    output_npz,
    char_size=0.12,
    depth_min=-0.0015,
    depth_max=-0.0040,
    threshold=127,
    lift_height=0.05,
    morph_kernel=3,
    min_component_area=16,
    spur_max_len=10,
    spur_rounds=2,
):
    """
    从书法图像提取骨架轨迹（按局部粗细生成可变深度）

    Args:
        image_path: 输入图像路径
        output_npz: 输出NPZ路径
        char_size: 字符实际大小（米）
        depth_min: 最浅落笔深度（米，负值）
        depth_max: 最深落笔深度（米，负值）
        threshold: 二值化阈值
        lift_height: 抬笔高度（米）
        morph_kernel: 形态学核尺寸（奇数，>=1）
        min_component_area: 最小连通域面积（像素）
        spur_max_len: 骨架短毛刺最大长度（像素）
        spur_rounds: 毛刺剪枝迭代轮数
    """
    if depth_min >= 0 or depth_max >= 0:
        raise ValueError("depth_min/depth_max 必须为负值")

    depth_min_m = min(abs(depth_min), abs(depth_max))
    depth_max_m = max(abs(depth_min), abs(depth_max))

    print("=" * 70)
    print("从书法图像提取密集轨迹（可变深度）")
    print("=" * 70)
    print(f"输入图像: {image_path}")
    print(f"字符大小: {char_size*100:.1f} cm")
    print(f"落笔深度范围: {depth_min_m*1000:.2f} ~ {depth_max_m*1000:.2f} mm\n")

    # 1. 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    h, w = img.shape
    print(f"✅ 图像尺寸: {w}x{h}")

    # 2. 二值化（反转：黑字->白字）
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    if morph_kernel >= 2:
        k = int(morph_kernel)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 去掉小噪声连通域
    n_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= int(min_component_area):
            clean[cc_labels == i] = 255
    binary = clean

    # 3. 骨架提取
    print("骨架提取中...")
    skeleton = skeletonize(binary > 0)
    skeleton = _prune_short_spurs(skeleton, max_spur_len=int(spur_max_len), rounds=int(spur_rounds))

    # 4. 找到所有骨架点
    y_coords, x_coords = np.where(skeleton)
    print(f"✅ 骨架点数: {len(x_coords)}")

    if len(x_coords) == 0:
        raise ValueError("未检测到骨架点，请检查图像")

    # 5. 距离变换用于估计骨架处局部粗细
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 6. 使用连通组件分析分离笔画
    labeled = label(skeleton, connectivity=2)
    num_strokes = labeled.max()
    print(f"✅ 检测到笔画数: {num_strokes}")

    # 7. 按笔画提取轨迹
    all_x, all_y, all_z = [], [], []

    for stroke_id in range(1, num_strokes + 1):
        # 提取当前笔画的点
        stroke_mask = (labeled == stroke_id)
        sy, sx = np.where(stroke_mask)

        if len(sx) < 3:  # 跳过太短的笔画（噪点）
            continue

        # 排序：从一端到另一端（简化：使用最近邻排序）
        points = np.column_stack([sx, sy])
        sorted_points = sort_stroke_points(points)

        stroke_x = sorted_points[:, 0]
        stroke_y = sorted_points[:, 1]

        # 全局归一化（使用图像边界）
        x_norm = (stroke_x / w) * char_size
        y_norm = (stroke_y / h) * char_size
        stroke_radius = dist_map[stroke_y, stroke_x].astype(np.float64)
        stroke_z = _compute_depth_from_radius(stroke_radius, depth_min_m, depth_max_m)

        # 移动到起点上方
        all_x.append(x_norm[0])
        all_y.append(y_norm[0])
        all_z.append(lift_height)

        # 下笔
        all_x.append(x_norm[0])
        all_y.append(y_norm[0])
        all_z.append(stroke_z[0])

        # 书写笔画
        for i in range(len(stroke_x)):
            all_x.append(x_norm[i])
            all_y.append(y_norm[i])
            all_z.append(stroke_z[i])
            
        all_x.append(x_norm[-1])
        all_y.append(y_norm[-1])
        all_z.append(lift_height)



        # 笔画结束
        print(
            f"  笔画 {stroke_id}: {len(stroke_x)} 个点, "
            f"半径px=[{stroke_radius.min():.2f}, {stroke_radius.max():.2f}]"
        )

    # 8. 兜底：确保轨迹最后是抬笔状态
    if all_z and all_z[-1] != lift_height:
        all_x.append(all_x[-1])
        all_y.append(all_y[-1])
        all_z.append(lift_height)

    # 9. 转为numpy数组
    pos_3d_x = np.array(all_x)
    pos_3d_y = np.array(all_y)
    pos_3d_z = np.array(all_z)

    print(f"\n✅ 总控制点数: {len(pos_3d_x)}")
    print(f"   X 范围: [{pos_3d_x.min():.4f}, {pos_3d_x.max():.4f}] m")
    print(f"   Y 范围: [{pos_3d_y.min():.4f}, {pos_3d_y.max():.4f}] m")
    print(f"   Z 范围: [{pos_3d_z.min():.4f}, {pos_3d_z.max():.4f}] m")

    # 10. 保存NPZ
    os.makedirs(os.path.dirname(output_npz) or '.', exist_ok=True)
    np.savez(output_npz,
             pos_3d_x=pos_3d_x,
             pos_3d_y=pos_3d_y,
             pos_3d_z=pos_3d_z)

    print(f"\n✅ NPZ文件已保存: {output_npz}")
    print("=" * 70)

    return output_npz


def sort_stroke_points(points):
    """
    使用最近邻排序笔画点（从一端到另一端）

    Args:
        points: (N, 2) 数组

    Returns:
        sorted_points: (N, 2) 排序后的数组
    """
    if len(points) == 0:
        return points

    # 找起点：选择最左上角的点
    start_idx = np.argmin(points[:, 0] + points[:, 1])

    sorted_points = [points[start_idx]]
    remaining = list(range(len(points)))
    remaining.remove(start_idx)

    current = points[start_idx]

    # 贪心最近邻
    while remaining:
        # 计算到所有剩余点的距离
        dists = np.linalg.norm(points[remaining] - current, axis=1)
        nearest_idx = np.argmin(dists)

        actual_idx = remaining[nearest_idx]
        sorted_points.append(points[actual_idx])
        current = points[actual_idx]
        remaining.pop(nearest_idx)

    return np.array(sorted_points)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从书法图像提取密集轨迹（可变深度）")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str,
                       default="demo_outputs/dense_trajectory.npz",
                       help="输出NPZ路径")
    parser.add_argument("--size", type=float, default=0.12,
                       help="字符大小（米），默认12cm")
    parser.add_argument("--depth-min", type=float, default=-0.002,
                       help="最浅落笔深度（米，负值）")
    parser.add_argument("--depth-max", type=float, default=-0.004,
                       help="最深落笔深度（米，负值）")
    parser.add_argument("--threshold", type=int, default=127,
                       help="二值化阈值")
    parser.add_argument("--lift-height", type=float, default=0.05,
                       help="抬笔高度（米）")
    parser.add_argument("--morph-kernel", type=int, default=3,
                       help="形态学核尺寸（默认3）")
    parser.add_argument("--min-component-area", type=int, default=16,
                       help="最小连通域面积（像素，默认16）")
    parser.add_argument("--spur-max-len", type=int, default=10,
                       help="短毛刺最大长度（像素，默认10）")
    parser.add_argument("--spur-rounds", type=int, default=2,
                       help="毛刺剪枝轮数（默认2）")

    args = parser.parse_args()

    extract_skeleton_trajectory(
        args.image,
        args.output,
        args.size,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        threshold=args.threshold,
        lift_height=args.lift_height,
        morph_kernel=args.morph_kernel,
        min_component_area=args.min_component_area,
        spur_max_len=args.spur_max_len,
        spur_rounds=args.spur_rounds,
    )
