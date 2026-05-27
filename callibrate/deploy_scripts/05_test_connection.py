#!/usr/bin/env python3
"""
阶段1 · Step 5 ── 网络连通 + 连接测试（需要机器人通电联网）

仅测试 TCP 连接，不发送任何运动指令，机器人不会动。

前置条件:
  - 机器人通电并解锁（Franka Desk 中显示 Ready）
  - 电脑与机器人在同一网络
  - franka_config.yaml 中 robot_ip 已填写

用法:
    python 05_test_connection.py
    python 05_test_connection.py --ip 172.16.0.2
"""

import sys
import os
import argparse
import subprocess

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')

try:
    import yaml
    cfg = yaml.safe_load(open(CFG_PATH))
    DEFAULT_IP = cfg.get('robot_ip', '172.16.0.2')
except Exception:
    DEFAULT_IP = '172.16.0.2'


def ping_test(ip):
    print(f'\n[1] Ping 测试: {ip}')
    result = subprocess.run(
        ['ping', '-c', '3', '-W', '1', ip],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        # 提取 RTT
        for line in result.stdout.splitlines():
            if 'avg' in line or 'rtt' in line:
                print(f'  ✅ 可达  {line.strip()}')
                break
        return True
    else:
        print(f'  ❌ 无法 ping 通 {ip}')
        print(f'     请检查:')
        print(f'       1. 机器人是否通电并联网')
        print(f'       2. 网线/WiFi 是否连接正确')
        print(f'       3. IP 地址是否正确（Franka Desk → Settings → Network）')
        return False


def _franky_version_hint(sv):
    """根据 FCI server version 返回对应的 libfranka wheel 版本"""
    table = {'9': '0-16-1', '8': '0-14-1', '7': '0-13-3',
             '6': '0-10-0', '5': '0-9-2',  '4': '0-8-0'}
    return table.get(str(sv))


def franka_connect_test(ip):
    import re
    print(f'\n[2] Franka 库连接测试: {ip}')
    sys.path.insert(0, ROOT)

    # ── 尝试 franky（pip install franky-control） ──────────────────
    try:
        import franky
        print('  使用 franky...')

        # 兼容不同版本：Robot / Franka / FrankaRobot
        RobotCls = None
        for name in ('Robot', 'Franka', 'FrankaRobot'):
            RobotCls = getattr(franky, name, None)
            if RobotCls is not None:
                break

        if RobotCls is None:
            avail = [x for x in dir(franky) if not x.startswith('_')]
            print(f'  ❌ franky 模块中找不到 Robot 类')
            print(f'     当前可用: {avail}')
            print(f'     → 请升级: pip install -U franky-control')
            return False

        robot = RobotCls(ip)
        print(f'  ✅ franky 连接成功！')
        del robot
        return True

    except ImportError:
        print('  ⚠️  franky 未安装 → pip install franky-control')

    except Exception as e:
        err = str(e)
        m = re.search(r'server version\s*(\d+).*?library version\s*(\d+)', err)
        if m:
            sv, lv = m.group(1), m.group(2)
            hint = _franky_version_hint(sv)
            print(f'  ❌ FCI 版本不匹配: 机器人 v{sv}，库 v{lv}')
            print(f'\n  修复（安装兼容 FCI v{sv} 的 wheel）:')
            print(f'    pip uninstall franky-control -y')
            if hint:
                print(f'    VERSION={hint}')
                print(f'    wget https://github.com/TimSchneider42/franky/releases/latest'
                      f'/download/libfranka_${{VERSION}}_wheels.zip')
                print(f'    unzip libfranka_${{VERSION}}_wheels.zip')
                print(f'    pip install --no-index --find-links=./dist franky-control')
            else:
                print(f'    # 到 https://github.com/TimSchneider42/franky/releases 选对应版本')
        elif 'not ready' in err.lower() or 'locked' in err.lower():
            print(f'  ❌ 机器人未解锁')
            print(f'     → 打开 Franka Desk，点击解锁按钮，确认状态为 Ready')
        else:
            print(f'  ❌ franky 连接失败: {e}')
            print(f'     常见原因:')
            print(f'       1. Franka Desk 浏览器标签页未关闭（占用连接）')
            print(f'       2. 机器人未解锁')
            print(f'       3. IP 地址错误')
        return False

    # ── 尝试 frankx（旧版备用） ────────────────────────────────────
    try:
        import frankx
        print('  使用 frankx...')
        robot = frankx.Robot(ip)
        robot.set_default_behavior()
        print(f'  ✅ frankx 连接成功！')
        del robot
        return True
    except ImportError:
        print('  ❌ franky 和 frankx 均未安装')
        print('     → pip install franky-control')
        return False
    except Exception as e:
        print(f'  ❌ frankx 连接失败: {e}')
        print(f'     常见原因:')
        print(f'       1. 机器人在 Franka Desk 中未解锁')
        print(f'       2. Franka Desk 浏览器标签页仍打开（请关闭）')
        print(f'       3. IP 地址错误')
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Franka 连接测试（仅网络，不运动）')
    parser.add_argument('--ip', type=str, default=DEFAULT_IP, help=f'机器人 IP（默认: {DEFAULT_IP}）')
    args = parser.parse_args()

    print('=' * 60)
    print('连接测试 · 阶段1 Step 5')
    print('⚠️  机器人不会运动，仅测试网络连接')
    print('=' * 60)

    p = ping_test(args.ip)
    if not p:
        print('\n❌ 网络不通，请先解决网络问题再继续。')
        sys.exit(1)

    c = franka_connect_test(args.ip)

    print(f'\n{"=" * 60}')
    if c:
        print('✅ 连接测试通过！')
        print('\n下一步: 阶段2 —— 用 Franka Desk 手动教点，测量纸面中心和姿态')
        print('        运行 06b_teach_paper_orientation.py 更新 franka_config.yaml')
    else:
        print('❌ 连接失败，请排查后重试。')
    print('=' * 60)
