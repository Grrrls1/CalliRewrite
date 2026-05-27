#!/usr/bin/env python3
"""
从 order/ PNG + seq_data/ NPZ 精确提取笔画轨迹

原理：
- seq_data 里有每个笔段的 pen_state 和总笔段数
- order PNG 里的颜色用 get_colors(total_strokes) 生成
- 每个像素的颜色 = 最后一个覆盖它的笔段的颜色
- 通过找最近颜色 → 每个骨架像素对应哪一笔段 → 正确分离笔画
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label as sk_label
import os
import glob


def get_colors_py(color_num):
    """复现 seq_extract/utils.py 的 get_colors 函数"""
    split_num = (color_num // 8 + 1) * 8

    r_break = [0, 0, 0, 0, 128, 255, 255, 255, 128]
    g_break = [0, 0, 128, 255, 255, 255, 128, 0, 0]
    b_break = [128, 255, 255, 255, 128, 0, 0, 0, 0]

    def rgb_trans(n, bv):
        num_splits = len(bv) - 1  # = 8
        sps = n // num_splits      # slices per split
        result = []
        for i in range(num_splits):
            gap = float(bv[i+1] - bv[i]) / float(sps)
            for s in range(sps):
                result.append(int(round(bv[i] + gap * s)))
        return result

    r_list = rgb_trans(split_num, r_break)
    g_list = rgb_trans(split_num, g_break)
    b_list = rgb_trans(split_num, b_break)
    return np.array([(r_list[i], g_list[i], b_list[i]) for i in range(split_num)],
                    dtype=np.float32)


def sort_stroke_points(points):
    """最近邻排序笔画点（从一端到另一端）"""
    if len(points) <= 1:
        return points
    # 起点选最左上角的点
    start_idx = np.argmin(points[:, 0] + points[:, 1])
    sorted_pts = [points[start_idx]]
    remaining = list(range(len(points)))
    remaining.remove(start_idx)
    current = points[start_idx]
    while remaining:
        dists = np.linalg.norm(points[remaining] - current, axis=1)
        nearest = np.argmin(dists)
        actual = remaining[nearest]
        sorted_pts.append(points[actual])
        current = points[actual]
        remaining.pop(nearest)
    return np.array(sorted_pts)


def extract_ordered_strokes_v2(order_img_path, seq_data_path, output_npz,
                                char_size=0.12, pen_depth=-0.002, lift_height=0.05,
                                downsample=2, min_stroke_pts=3):
    """
    从 order PNG + seq_data NPZ 精确提取笔画轨迹

    Args:
        order_img_path: order/X.png 路径
        seq_data_path:  seq_data/X.npz 路径
        output_npz:     输出 MuJoCo NPZ 路径
    """
    # 读取 seq_data 获取笔段信息
    d = np.load(seq_data_path, allow_pickle=True)
    strokes_data = d['strokes_data']
    total_strokes = len(strokes_data)
    pen_states = strokes_data[:, 0].astype(int)
    draw_indices = np.where(pen_states == 0)[0]  # pen=0 的索引（对应 color_idx）

    print(f"  总笔段: {total_strokes}, pen=0 绘制笔段: {len(draw_indices)}")

    # 计算完整颜色表（包含所有笔段，含 pen=1）
    color_table = get_colors_py(total_strokes)  # (total_strokes, 3) in [0, 255]

    # 读取 order 图像
    img = cv2.imread(order_img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {order_img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = img.shape[:2]

    # 提取非白色像素（笔画区域）
    white = (img_rgb[:,:,0] > 200) & (img_rgb[:,:,1] > 200) & (img_rgb[:,:,2] > 200)
    stroke_mask = ~white

    # 骨架化
    skeleton = skeletonize(stroke_mask)

    # 获取所有骨架像素坐标
    skel_pts = np.argwhere(skeleton)  # (N, 2): [row, col]
    if len(skel_pts) == 0:
        raise ValueError("骨架为空")

    # 对每个骨架像素，找最近的 pen=0 笔段颜色 → 分配 stroke_idx
    skel_colors = img_rgb[skel_pts[:, 0], skel_pts[:, 1]]  # (N, 3)

    # 只考虑 pen=0 的颜色（pen=1 的笔段不在 order 图里）
    draw_colors = color_table[draw_indices]  # (num_draw, 3)

    # 向量化计算每个骨架像素到每个 pen=0 颜色的距离
    # skel_colors: (N, 3), draw_colors: (M, 3)
    dists = np.sqrt(((skel_colors[:, None, :] - draw_colors[None, :, :]) ** 2).sum(axis=2))
    # dists shape: (N, M)

    nearest_draw_idx = np.argmin(dists, axis=1)  # (N,): 最近的 pen=0 笔段索引（0~len(draw_indices)-1）
    nearest_color_idx = draw_indices[nearest_draw_idx]  # 对应的 color_idx（在 strokes_data 中的绝对索引）

    print(f"  唯一颜色分配: {len(np.unique(nearest_color_idx))} 个笔段")

    # 按笔段分组骨架像素（按 color_idx 排序 = 绘制顺序）
    stroke_groups = {}
    for i, cidx in enumerate(nearest_color_idx):
        if cidx not in stroke_groups:
            stroke_groups[cidx] = []
        stroke_groups[cidx].append(skel_pts[i])

    # 按 color_idx 升序 = 绘制顺序
    sorted_cidx = sorted(stroke_groups.keys())
    print(f"  分组后笔段数: {len(sorted_cidx)}")

    # 收集所有点用于归一化
    all_skel_pts_for_norm = skel_pts.copy()
    y_min, x_min = all_skel_pts_for_norm.min(axis=0)
    y_max, x_max = all_skel_pts_for_norm.max(axis=0)
    scale = max(x_max - x_min, y_max - y_min)
    if scale == 0:
        scale = 1

    def to_3d(px_col, px_row):
        x = (px_col - x_min) / scale * char_size
        y = (px_row - y_min) / scale * char_size
        return x, y

    all_x, all_y, all_z = [], [], []

    for stroke_i, cidx in enumerate(sorted_cidx):
        pts = np.array(stroke_groups[cidx])  # (N, 2): [row, col]
        if len(pts) < min_stroke_pts:
            continue

        # 按列坐标（x）排序（近似从左到右）
        # 更好：最近邻排序
        pts_downsample = pts[::downsample] if len(pts) > 10 else pts
        sorted_pts = sort_stroke_points(pts_downsample)

        # 笔画开始：抬笔移到起点上方
        if len(all_x) > 0:
            all_x.append(all_x[-1])
            all_y.append(all_y[-1])
            all_z.append(lift_height)

        sx, sy = to_3d(sorted_pts[0, 1], sorted_pts[0, 0])
        all_x.append(sx)
        all_y.append(sy)
        all_z.append(lift_height)
        # 落笔
        all_x.append(sx)
        all_y.append(sy)
        all_z.append(pen_depth)

        # 沿笔画写
        for p in sorted_pts[1:]:
            px, py = to_3d(p[1], p[0])
            all_x.append(px)
            all_y.append(py)
            all_z.append(pen_depth)

    # 最终抬笔
    if all_x:
        all_x.append(all_x[-1])
        all_y.append(all_y[-1])
        all_z.append(lift_height)

    pos_3d_x = np.array(all_x, dtype=np.float64)
    pos_3d_y = np.array(all_y, dtype=np.float64)
    pos_3d_z = np.array(all_z, dtype=np.float64)

    print(f"  总控制点: {len(pos_3d_x)}")

    os.makedirs(os.path.dirname(output_npz) or '.', exist_ok=True)
    np.savez(output_npz, pos_3d_x=pos_3d_x, pos_3d_y=pos_3d_y, pos_3d_z=pos_3d_z)
    print(f"  ✅ 已保存: {output_npz}")
    return len(sorted_cidx)


if __name__ == '__main__':
    order_dir = '../seq_extract/outputs/__new_train_phase_2/order'
    seq_dir = '../seq_extract/outputs/__new_train_phase_2/seq_data'
    out_dir = '../seq_extract/outputs/mujoco_ordered_v2'
    os.makedirs(out_dir, exist_ok=True)

    order_files = sorted(glob.glob(f'{order_dir}/*.png'))
    print(f"找到 {len(order_files)} 个 order 图像\n")

    for path in order_files:
        name = os.path.splitext(os.path.basename(path))[0]
        seq_path = f'{seq_dir}/{name}.npz'
        out_path = f'{out_dir}/{name}.npz'
        print(f"处理 {name}:")
        n = extract_ordered_strokes_v2(path, seq_path, out_path)
        print()
