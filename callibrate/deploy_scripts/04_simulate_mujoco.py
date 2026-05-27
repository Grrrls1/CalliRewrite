#!/usr/bin/env python3
"""
阶段0 · Step 4 ── MuJoCo 仿真验证（无需机器人）

在 MuJoCo 中跑一遍轨迹，保存画布图像，肉眼确认字形正确。
通过后才上真机。

用法:
    python 04_simulate_mujoco.py                      # 仿真 examples/example_永.npz
    python 04_simulate_mujoco.py path/to/file.npz     # 仿真指定文件
    python 04_simulate_mujoco.py --all                # 仿真所有 v2 NPZ
"""

import sys
import os
import argparse
import numpy as np

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SIM_DIR   = os.path.join(ROOT, '..', 'mujoco_sim')
OUT_DIR   = os.path.join(ROOT, 'deploy_scripts', 'sim_outputs')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SIM_DIR)


def simulate_npz(npz_path, speed=0.05):
    label = os.path.splitext(os.path.basename(npz_path))[0]
    print(f'\n[仿真] {npz_path}')

    try:
        import cv2
        from mujoco_simulator import FrankaCalligraphySimulator
    except ImportError as e:
        print(f'  ❌ 导入失败: {e}')
        print(f'     请先安装 mujoco: pip install mujoco')
        return False

    sim = FrankaCalligraphySimulator(render_mode='rgb_array')

    # 降低接触力阈值（与 generate_composite_video.py 一致）
    def patched_update():
        brush_pos, contact_force = sim.get_brush_contact()
        is_contact = contact_force > 0.001
        sim.ink_traces.append((*brush_pos, is_contact))
        if is_contact:
            sim._draw_on_canvas(brush_pos)
        else:
            if hasattr(sim, '_last_canvas_pos'):
                delattr(sim, '_last_canvas_pos')
    sim._update_ink_trace = patched_update

    d = np.load(npz_path)
    x_pts, y_pts, z_pts = d['pos_3d_x'], d['pos_3d_y'], d['pos_3d_z']
    n = len(x_pts)
    print(f'  控制点数: {n}')

    for i in range(n):
        z_world  = z_pts[i] + 0.011
        z_clamped = max(z_world, 0.008)
        target = np.array([
            x_pts[i] + sim.paper_offset[0],
            y_pts[i] + sim.paper_offset[1],
            z_clamped,
        ])
        sim.move_to_position(target, speed=speed, wait_time=0.0)
        if i % 50 == 0:
            print(f'  进度: {i}/{n} ({i/n*100:.0f}%)')

    canvas_path = os.path.join(OUT_DIR, f'{label}_canvas.png')
    cv2.imwrite(canvas_path, sim.paper_canvas)
    sim.close()

    contact_pts = sum(1 for _, _, _, c in sim.ink_traces if c)
    print(f'  接触点: {contact_pts}/{len(sim.ink_traces)}')
    print(f'  ✅ 画布已保存: {canvas_path}')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuJoCo 仿真验证')
    parser.add_argument('file', nargs='?', default=None, help='NPZ 文件路径')
    parser.add_argument('--all', action='store_true', help='仿真所有 v2 NPZ')
    parser.add_argument('--speed', type=float, default=0.05)
    args = parser.parse_args()

    print('=' * 60)
    print('MuJoCo 仿真验证 · 阶段0 Step 4')
    print('=' * 60)

    if args.all:
        v2_dir = os.path.join(ROOT, '..', 'seq_extract', 'outputs', 'mujoco_ordered_v2')
        if not os.path.exists(v2_dir):
            print(f'❌ v2 目录不存在: {v2_dir}')
            sys.exit(1)
        targets = sorted([
            os.path.join(v2_dir, f)
            for f in os.listdir(v2_dir) if f.endswith('.npz')
        ])
    elif args.file:
        targets = [args.file]
    else:
        default = os.path.join(ROOT, 'examples', 'example_永.npz')
        targets = [default]

    results = []
    for t in targets:
        ok = simulate_npz(t, speed=args.speed)
        results.append((t, ok))

    print(f'\n{"=" * 60}')
    print(f'仿真完成: {sum(1 for _, ok in results if ok)}/{len(results)} 成功')
    print(f'输出目录: {OUT_DIR}')
    print('\n请打开 sim_outputs/ 目录，肉眼检查每个字的画布图像是否正确。')
    print('确认无误后，才可进入机器人测试阶段（Step 5+）。')
    print('=' * 60)
