#!/usr/bin/env python3
"""
阶段4 · Step 8 ── 单点接触测试（轻触纸面中心一次）

验证:
  1. workspace_center 与 paper_orientation 对应实际纸面
  2. 笔尖能正确接触纸面（轻微印记）
  3. 接触力合理，不会戳穿纸或打滑

路径: Home → 沿纸面法向上方5cm → 纸面轻触 → 沿法向抬起 → Home

⚠️ 运行前请确认 Step 7 已通过。
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


def main():
    print('=' * 60)
    print('单点接触测试 · 阶段4 Step 8')
    print('⚠️  机器人将轻触纸面，请确认安全')
    print('=' * 60)

    cfg, (wx, wy, wz) = load_config()
    robot_ip = cfg.get('robot_ip', '172.16.0.2')
    speed = 0.015  # 极慢
    ctrl = FrankaCalligraphyController(
        robot_ip=robot_ip,
        default_speed=speed,
        workspace_center=(wx, wy, wz),
    )
    above_point = ctrl.paper_to_robot(0.0, 0.0, 0.05)
    touch_point = ctrl.paper_to_robot(0.0, 0.0, -0.001)

    print(f'\n配置:')
    print(f'  机器人 IP:  {robot_ip}')
    print(f'  纸面中心:   ({wx:.4f}, {wy:.4f}, {wz:.4f})')
    print(f'  纸面法向:   {np.round(ctrl.paper_normal, 6).tolist()}')
    print(f'  接触目标:   ({touch_point[0]:.4f}, {touch_point[1]:.4f}, {touch_point[2]:.4f})')
    print('                ← 沿纸面法向压入 1mm，末端保持垂直纸面')
    print(f'  速度:       {speed}')
    print('\n执行路径:')
    print(f'  1. 沿法向移到纸面上方 5cm  ({above_point[0]:.3f}, {above_point[1]:.3f}, {above_point[2]:.3f})')
    print(f'  2. 沿法向缓慢轻触纸面       ({touch_point[0]:.3f}, {touch_point[1]:.3f}, {touch_point[2]:.3f})')
    print(f'  3. 停留 1 秒，观察接触情况')
    print(f'  4. 抬起到上方 5cm')
    print(f'  5. 返回 Home')

    confirm = input('\n确认后输入 YES 继续: ')
    if confirm.strip() != 'YES':
        print('已取消。')
        sys.exit(0)

    if not ctrl.connect():
        print('❌ 连接失败')
        sys.exit(1)

    try:
        # 1. 移到上方
        print('\n[1/5] 移到纸面上方 5cm...')
        if not ctrl.move_surface(0.0, 0.0, 0.05, speed=speed):
            print('❌ 无法到达纸面上方安全位置，停止测试。')
            return
        time.sleep(0.5)

        # 2. 轻触
        print('[2/5] 轻触纸面（极慢下降）...')
        print('  → 随时准备按急停！')
        if not ctrl.move_surface(0.0, 0.0, -0.001, speed=0.01):
            print('❌ 轻触运动失败，停止测试。')
            return

        # 3. 停留观察
        print('[3/5] 停留 2 秒，请观察接触情况...')
        time.sleep(2)
        print('  问题1: 笔尖是否碰到纸面？（应该是）')
        print('  问题2: 接触力是否太大/太小？')
        obs = input('  接触情况正常吗？(y/N): ')

        # 4. 抬起
        print('[4/5] 抬起...')
        if not ctrl.move_surface(0.0, 0.0, 0.05, speed=speed):
            print('❌ 抬笔运动失败，请在 Franka Desk 中确认机械臂状态。')
            return
        time.sleep(0.5)

        # 5. Home
        print('[5/5] 返回 Home...')
        ctrl.move_to_home()

        if obs.strip().lower() == 'y':
            print('\n✅ 单点接触测试通过！')
            print('\n如果接触深度不合适，请重新运行 Step 6b 修正纸面中心，')
            print('或调整本脚本中的轻触深度 -0.001 m 后复测。')
        else:
            print('\n⚠️  接触不正常。请重新示教纸面中心/法向或减小压入深度后重试。')

    finally:
        ctrl.disconnect()

    print(f'\n{"=" * 60}')
    print('下一步: Step 9 —— 标定线段（验证完整坐标映射）')
    print('=' * 60)


if __name__ == '__main__':
    main()
