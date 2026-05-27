#!/usr/bin/env python3
"""
阶段2 · Step 6 ── 工作台坐标教点（需要机器人 + 手动导引）

引导你用 Franka Desk 手动把末端移到纸面中心，然后读取坐标，
自动写入 franka_config.yaml。

操作步骤:
  1. 打开 Franka Desk（浏览器 → https://<robot_ip>）
  2. 解锁机器人，切换到「引导模式」（按住末端导引按钮）
  3. 把笔尖引导到纸面中心
  4. 回到此脚本，按 Enter 读取坐标
  5. 脚本自动更新 franka_config.yaml

⚠️ 此步骤机器人需要处于「手动导引」状态，不会自动运动。
"""

import sys
import os
import argparse

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')

try:
    import yaml
except ImportError:
    print('❌ pyyaml 未安装 → pip install pyyaml')
    sys.exit(1)


def get_robot_ip():
    if os.path.exists(CFG_PATH):
        cfg = yaml.safe_load(open(CFG_PATH))
        return cfg.get('robot_ip', '172.16.0.2')
    return '172.16.0.2'


def read_robot_pose(ip):
    """读取机器人当前末端笛卡尔位置"""
    sys.path.insert(0, ROOT)
    try:
        import franky
        robot = franky.Robot(ip)
        state = robot.state
        pose  = state.O_T_EE  # Affine 或 4x4 矩阵，取决于 franky 版本
        # 新版 franky: pose 是 Affine 对象，有 .translation 属性
        # 旧版 franky: pose 是 4x4 列表/数组，用 pose[0][3] 取 x
        if hasattr(pose, 'translation'):
            t = pose.translation
            x, y, z = float(t[0]), float(t[1]), float(t[2])
        elif hasattr(pose, 'x'):
            x, y, z = float(pose.x), float(pose.y), float(pose.z)
        else:
            x = pose[0][3]
            y = pose[1][3]
            z = pose[2][3]
        del robot
        return x, y, z, 'franky'
    except ImportError:
        pass
    except Exception as e:
        print(f'  franky 读取失败: {e}')

    try:
        import frankx
        robot = frankx.Robot(ip)
        robot.set_default_behavior()
        state = robot.read_once()
        # frankx state has O_T_EE (column-major 4x4)
        t = state.O_T_EE
        x, y, z = t[12], t[13], t[14]
        del robot
        return x, y, z, 'frankx'
    except ImportError:
        raise RuntimeError('无法导入 franky 或 frankx')
    except Exception as e:
        raise RuntimeError(f'读取机器人位置失败: {e}')


def save_config(x, y, z):
    """更新 franka_config.yaml 中的 workspace_center"""
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    cfg.setdefault('workspace_center', {})
    cfg['workspace_center']['x'] = round(float(x), 4)
    cfg['workspace_center']['y'] = round(float(y), 4)
    cfg['workspace_center']['z'] = round(float(z), 4)

    with open(CFG_PATH, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f'\n  ✅ 已更新 franka_config.yaml:')
    print(f'     workspace_center: x={x:.4f}, y={y:.4f}, z={z:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='工作台坐标教点')
    parser.add_argument('--ip', type=str, default=get_robot_ip())
    parser.add_argument('--dry-run', action='store_true',
                        help='不写入配置，只打印读取到的坐标')
    args = parser.parse_args()

    print('=' * 60)
    print('工作台坐标教点 · 阶段2 Step 6')
    print('=' * 60)
    print(f'\n机器人 IP: {args.ip}')
    print('\n操作步骤:')
    print('  1. 打开 Franka Desk（浏览器 → https://' + args.ip + '）')
    print('  2. 解锁机器人 → 切换到引导模式（按住末端黑色按钮）')
    print('  3. 把笔尖引导到纸面中心（纸张中央）')
    print('  4. 保持末端静止，回到此终端按 Enter')

    input('\n>>> 准备好后按 Enter 读取坐标...')

    print('\n正在读取末端位置...')
    try:
        x, y, z, lib = read_robot_pose(args.ip)
        print(f'\n  读取成功（使用 {lib}）:')
        print(f'    x = {x:.4f} m')
        print(f'    y = {y:.4f} m')
        print(f'    z = {z:.4f} m  ← 这就是纸面高度')
    except Exception as e:
        print(f'\n  ❌ 读取失败: {e}')
        print('\n  备用方案: 手动在 Franka Desk 查看「当前位置」，手动填写到 franka_config.yaml')
        sys.exit(1)

    if not args.dry_run:
        confirm = input(f'\n是否将此坐标写入 franka_config.yaml？(y/N): ')
        if confirm.strip().lower() == 'y':
            save_config(x, y, z)
        else:
            print('  跳过写入。请手动更新 franka_config.yaml。')
    else:
        print('\n  （--dry-run 模式，不写入配置）')

    print(f'\n{"=" * 60}')
    print('下一步: 若使用 RoboControl2，请运行 06b_teach_paper_orientation.py 标定纸面法向')
    print('        然后运行 03_check_config.py 复验配置，再进行 Step 7 首次运动测试')
    print('=' * 60)
