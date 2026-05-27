#!/usr/bin/env python3
"""
阶段0 · Step 2 ── NPZ 轨迹文件检查 + 可视化（无需机器人）

对所有将要执行的 NPZ 文件逐一检查：
- 坐标范围是否合理
- Z 轴分布（抬笔/接触比例）
- 可视化俯视图和 Z 轴曲线
- 标记潜在风险点

用法:
    python 02_check_npz.py                        # 检查 examples/ 下所有 NPZ
    python 02_check_npz.py path/to/your.npz       # 检查指定文件
    python 02_check_npz.py path/to/your.npz --plot  # 同时弹出可视化
"""

import sys
import os
import argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def check_npz(path, plot=False):
    print(f'\n{"=" * 60}')
    print(f'文件: {path}')
    print('=' * 60)

    if not os.path.exists(path):
        print(f'  ❌ 文件不存在')
        return False

    d = np.load(path)
    x = d['pos_3d_x']
    y = d['pos_3d_y']
    z = d['pos_3d_z']
    n = len(x)

    print(f'  控制点数: {n}')
    print(f'  X 范围: [{x.min():.4f}, {x.max():.4f}] m  跨度={x.max()-x.min():.4f} m')
    print(f'  Y 范围: [{y.min():.4f}, {y.max():.4f}] m  跨度={y.max()-y.min():.4f} m')
    print(f'  Z 范围: [{z.min():.4f}, {z.max():.4f}] m')

    # Z 轴分布
    lifted   = np.sum(z > 0.01)      # 抬笔
    touching = np.sum((z >= -0.01) & (z <= 0.01))  # 轻触
    pressing = np.sum(z < -0.01)     # 按压
    print(f'\n  Z 轴分布:')
    print(f'    抬笔  (z>0.01m):            {lifted:4d} 点 ({lifted/n*100:.1f}%)')
    print(f'    轻触  (-0.01≤z≤0.01m):     {touching:4d} 点 ({touching/n*100:.1f}%)')
    print(f'    按压  (z<-0.01m):           {pressing:4d} 点 ({pressing/n*100:.1f}%)')

    # 风险检查
    issues = []
    if z.min() < -0.15:
        issues.append(f'⚠️ Z最小值={z.min():.4f}m 过深，可能损坏笔或纸面（建议>-0.12m）')
    if z.max() > 0.15:
        issues.append(f'⚠️ Z最大值={z.max():.4f}m 过高（建议<0.1m）')
    if x.max() - x.min() > 0.15:
        issues.append(f'⚠️ X跨度={x.max()-x.min():.4f}m 超过15cm，写字面积偏大')
    if y.max() - y.min() > 0.15:
        issues.append(f'⚠️ Y跨度={y.max()-y.min():.4f}m 超过15cm，写字面积偏大')
    if n < 5:
        issues.append(f'⚠️ 控制点过少({n})，轨迹可能不完整')
    if np.any(np.diff(x)**2 + np.diff(y)**2 > 0.04):  # 单步>20cm
        issues.append(f'⚠️ 存在单步跳跃 >20cm，请检查轨迹连续性')

    if issues:
        print(f'\n  风险提示:')
        for iss in issues:
            print(f'    {iss}')
    else:
        print(f'\n  ✅ 坐标范围正常，未发现风险点')

    # 可视化
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(os.path.basename(path), fontsize=13)

            # 俯视图
            ax = axes[0]
            contact = z < 0.01
            ax.plot(x, y, 'lightgray', linewidth=0.8, zorder=1)
            ax.scatter(x[contact], y[contact], c='blue', s=2, zorder=2, label='接触')
            ax.scatter(x[~contact], y[~contact], c='orange', s=2, zorder=2, label='抬笔')
            ax.scatter(x[0], y[0], c='green', s=80, marker='o', zorder=3, label='起点')
            ax.scatter(x[-1], y[-1], c='red', s=80, marker='x', zorder=3, label='终点')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('俯视图（蓝=接触 橙=抬笔）')
            ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(markerscale=3)

            # Z 轴变化
            ax = axes[1]
            ax.plot(z, 'b-', linewidth=0.8)
            ax.axhline(0, color='red', linestyle='--', linewidth=1, label='纸面 z=0')
            ax.axhline(0.01, color='orange', linestyle=':', linewidth=1, label='抬笔阈值')
            ax.fill_between(range(n), z, 0, where=z < 0, alpha=0.2, color='blue', label='按压')
            ax.set_xlabel('控制点序号'); ax.set_ylabel('Z (m)')
            ax.set_title('Z 轴高度变化'); ax.legend(); ax.grid(True, alpha=0.3)

            # 3D 轨迹
            ax3d = fig.add_subplot(1, 3, 3, projection='3d')
            ax3d.plot(x, y, z, 'b-', linewidth=0.5, alpha=0.7)
            ax3d.scatter(x[0], y[0], z[0], c='green', s=60)
            ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
            ax3d.set_title('3D 轨迹')

            plt.tight_layout()
            save_path = path.replace('.npz', '_check.png')
            plt.savefig(save_path, dpi=120)
            print(f'\n  📊 可视化已保存: {save_path}')
            plt.show()
        except Exception as e:
            print(f'\n  ⚠️ 可视化失败: {e}')

    return len(issues) == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NPZ 轨迹文件检查工具')
    parser.add_argument('files', nargs='*', help='NPZ 文件路径（默认检查 examples/ 下所有文件）')
    parser.add_argument('--plot', action='store_true', help='显示可视化图像')
    args = parser.parse_args()

    if args.files:
        targets = args.files
    else:
        # 默认检查所有 examples
        ex_dir = os.path.join(ROOT, 'examples')
        targets = sorted([
            os.path.join(ex_dir, f)
            for f in os.listdir(ex_dir) if f.endswith('.npz')
        ]) if os.path.exists(ex_dir) else []
        # 也检查 mujoco_ordered_v2（如果存在）
        v2_dir = os.path.join(ROOT, '..', 'seq_extract', 'outputs', 'mujoco_ordered_v2')
        if os.path.exists(v2_dir):
            targets += sorted([
                os.path.join(v2_dir, f)
                for f in os.listdir(v2_dir) if f.endswith('.npz')
            ])[:3]  # 最多检查3个

    if not targets:
        print('未找到 NPZ 文件，请指定路径或先运行 generate_example_npz.py')
        sys.exit(1)

    all_ok = True
    for t in targets:
        ok = check_npz(t, plot=args.plot)
        all_ok = all_ok and ok

    print(f'\n{"=" * 60}')
    if all_ok:
        print('✅ 所有 NPZ 文件检查通过')
    else:
        print('⚠️  部分文件存在风险，请处理后再继续')
    print('=' * 60)
