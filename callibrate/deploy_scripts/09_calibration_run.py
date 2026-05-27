#!/usr/bin/env python3
"""
阶段5 · Step 9 ── 标定线段执行（验证完整坐标映射）

在纸上画几条横线，验证:
  1. X/Y 坐标映射正确（线的方向和间距）
  2. 沿纸面法向的接触深度合适（线粗细均匀）
  3. 坐标系方向与预期一致

执行的是 calibrate.py 生成的标定轨迹（多条不同压力的横线）。

⚠️ 运行前请确认 Step 8 接触测试已通过，workspace_center 和纸面法向已校准。
"""

import sys
import os
import time

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')

try:
    import yaml
    import numpy as np
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    sys.exit(1)

sys.path.insert(0, ROOT)
from RoboControl2 import FrankaCalligraphyController


def load_config():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    wc = cfg['workspace_center']
    return cfg, (wc['x'], wc['y'], wc['z'])


def generate_line_npz():
    """生成简单标定轨迹：5 条横线，不同 Z 压力"""
    import numpy as np

    z_levels = [-0.001, -0.0015, -0.002, -0.003, -0.004]  # 轻到重
    x_arr, y_arr, z_arr = [], [], []

    for i, zl in enumerate(z_levels):
        y_offset = i * 0.008   # 线间距 8mm
        x_start, x_end = 0.0, 0.04   # 4cm 横线

        # 抬笔到起点上方
        x_arr.append(x_start);  y_arr.append(y_offset);  z_arr.append(0.04)
        # 下笔
        x_arr.append(x_start);  y_arr.append(y_offset);  z_arr.append(zl)
        # 画横线（每 2mm 一个点）
        for xi in np.arange(x_start, x_end + 0.001, 0.002):
            x_arr.append(xi);  y_arr.append(y_offset);  z_arr.append(zl)
        # 抬笔
        x_arr.append(x_end);    y_arr.append(y_offset);  z_arr.append(0.04)

    # 末尾抬笔
    x_arr.append(x_arr[-1]); y_arr.append(y_arr[-1]); z_arr.append(0.05)

    out = os.path.join(ROOT, 'deploy_scripts', 'calib_lines.npz')
    np.savez(out,
             pos_3d_x=np.array(x_arr),
             pos_3d_y=np.array(y_arr),
             pos_3d_z=np.array(z_arr))
    return out, np.array(x_arr), np.array(y_arr), np.array(z_arr)


def map_to_robot(x_npz, y_npz, z_npz, controller):
    """
    将 NPZ 纸面相对坐标映射到 Franka 基坐标。
    x/y: 沿纸面坐标轴，中心对齐到 workspace_center
    z:   沿纸面法向（z_npz=0 -> 纸面）
    """
    x_c = (x_npz.min() + x_npz.max()) / 2
    y_c = (y_npz.min() + y_npz.max()) / 2
    points = np.asarray([
        controller.paper_to_robot(x - x_c, y - y_c, z)
        for x, y, z in zip(x_npz, y_npz, z_npz)
    ])
    return points[:, 0], points[:, 1], points[:, 2]


def main():
    print('=' * 60)
    print('标定线段执行 · 阶段5 Step 9')
    print('⚠️  机器人将在纸上写 5 条横线')
    print('=' * 60)

    cfg, (wx, wy, wz) = load_config()
    robot_ip = cfg.get('robot_ip', '172.16.0.2')
    speed = 0.02
    ctrl = FrankaCalligraphyController(
        robot_ip=robot_ip,
        default_speed=speed,
        workspace_center=(wx, wy, wz),
    )

    print(f'\n生成标定轨迹...')
    npz_path, x_npz, y_npz, z_npz = generate_line_npz()
    print(f'  轨迹文件: {npz_path}')
    print(f'  控制点数: {len(x_npz)}')

    x_r, y_r, z_r = map_to_robot(x_npz, y_npz, z_npz, ctrl)
    print(f'\n坐标映射后:')
    print(f'  X: [{x_r.min():.4f}, {x_r.max():.4f}] m')
    print(f'  Y: [{y_r.min():.4f}, {y_r.max():.4f}] m')
    print(f'  Z: [{z_r.min():.4f}, {z_r.max():.4f}] m')
    print(f'  纸面法向: {np.round(ctrl.paper_normal, 6).tolist()}')
    print('  Z 压力已沿纸面法向映射，末端在全部点保持垂直纸面')

    print(f'\n机器人 IP: {robot_ip}  速度: {speed}')
    confirm = input('\n确认后输入 YES 开始执行（其他任意键退出）: ')
    if confirm.strip() != 'YES':
        print('已取消。')
        sys.exit(0)

    if not ctrl.connect():
        print('❌ 连接失败')
        sys.exit(1)

    completed = False
    try:
        print('\n开始执行标定线段...')
        n = len(x_r)
        completed = True
        for i in range(n):
            ok = ctrl.move_cartesian(x_r[i], y_r[i], z_r[i], speed=speed)
            if not ok:
                print(f'❌ 第 {i} 个点运动失败，已停止。')
                completed = False
                break
            if i % 20 == 0:
                print(f'  进度: {i}/{n} ({i/n*100:.0f}%)')

        if completed:
            print('\n✅ 标定线段执行完成！')
            print('\n请检查纸上的 5 条横线:')
            print('  □ 线段方向是否沿纸面 X 轴')
            print('  □ 5 条线是否均匀排列（间距约 8mm）')
            print('  □ 线的粗细是否从细到粗（法向压入深度递增）')
            print('  □ 线迹是否清晰连续（无断墨）')
            print('\n如方向/位置不对 → 重新运行 Step 6b 校准纸面坐标轴')
            print('如粗细不对       → 调整本脚本中的 z_levels 参数')

    except KeyboardInterrupt:
        print('\n⚠️ 用户中断')
        completed = False
    finally:
        if completed:
            ctrl.move_to_home()
        else:
            print('未自动返回 Home；请先在 Franka Desk 中确认机器人状态和安全路径。')
        ctrl.disconnect()

    print(f'\n{"=" * 60}')
    if completed:
        print('Step 9 完成！标定线通过后，可进行完整字符执行。')
        print('运行: python3 ../RoboControl2.py <your_npz> ' + robot_ip + ' 0.03')
    else:
        print('Step 9 未通过，请排查运动失败或中断原因后重试。')
    print('=' * 60)


if __name__ == '__main__':
    main()
