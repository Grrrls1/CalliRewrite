#!/usr/bin/env python3
"""
阶段3 · Step 7 ── 首次运动测试：仅移动到 Home（纸面上方）

只做一件事：把机器人沿纸面法向移到 workspace_center 上方 10cm 处，然后停下。
不接触纸面，不执行任何轨迹。

⚠️ 运行前请确认:
  - 手放在急停按钮上
  - 机器人周围无障碍物
  - franka_config.yaml 已由 Step 6b 写入 workspace_center 与 paper_orientation

通过标准: 机器人缓慢移到纸面上方 10cm，位置看起来正确，无碰撞报错。
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
    print('首次运动测试（仅 Home 位置） · 阶段3 Step 7')
    print('⚠️  机器人将运动！请确认周围安全，手放急停按钮')
    print('=' * 60)

    cfg, (wx, wy, wz) = load_config()
    robot_ip = cfg.get('robot_ip', '172.16.0.2')
    speed    = 0.02   # 极慢，最大速度的 2%
    ctrl = FrankaCalligraphyController(
        robot_ip=robot_ip,
        default_speed=speed,
        workspace_center=(wx, wy, wz),
    )
    home_target = ctrl.paper_to_robot(0.0, 0.0, 0.10)

    print(f'\n配置:')
    print(f'  机器人 IP:       {robot_ip}')
    print(f'  workspace_center: ({wx}, {wy}, {wz})')
    print(f'  纸面法向:         {np.round(ctrl.paper_normal, 6).tolist()}')
    print(f'  目标位置:         ({home_target[0]:.4f}, {home_target[1]:.4f}, {home_target[2]:.4f})')
    print('                    ← 沿纸面法向上方 10cm，末端保持垂直纸面')
    print(f'  速度:             {speed}（极慢）')

    print('\n⚠️  请确认:')
    print('  1. 手放在急停按钮上')
    print('  2. 机器人周围 50cm 内无人无物')
    print('  3. 机器人已解锁（Franka Desk 显示 Ready）')
    confirm = input('\n确认以上条件后输入 YES 继续（其他任意键退出）: ')
    if confirm.strip() != 'YES':
        print('已取消。')
        sys.exit(0)

    if not ctrl.connect():
        print('❌ 连接失败，请检查机器人状态。')
        sys.exit(1)

    ok = False
    try:
        print('\n移动到 Home 位置（纸面上方 10cm）...')
        print('  → 如有异常立即按急停！')
        ok = ctrl.move_surface(0.0, 0.0, 0.10, speed=speed)
        if ok:
            print(f'\n✅ 成功到达 Home 位置!')
            print(f'   请观察末端位置是否在纸面正上方约 10cm 处。')
            time.sleep(2)
        else:
            print('\n❌ 运动失败，请检查 Franka Desk 错误信息。')
    finally:
        ctrl.disconnect()

    print(f'\n{"=" * 60}')
    if ok:
        print('下一步: Step 8 —— 单点接触测试（轻触纸面）')
    else:
        print('Step 7 未通过，请检查配置和机器人状态后重新执行。')
    print('=' * 60)


if __name__ == '__main__':
    main()
