#!/usr/bin/env python3
"""
阶段2 · Step 6b -- 纸面姿态三点示教（需要机器人 + 手动导引）

通过纸面上的三个不共线点测量实际纸面法向和书写 X 方向，并写入
franka_config.yaml，供 RoboControl2 保持笔轴垂直纸面。

测量过程中请尽量保持夹笔器姿态不变，只让同一笔尖轻触三个点。
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')
POSITION_SCRIPT = os.path.join(os.path.dirname(__file__), '06_teach_workspace.py')

try:
    import yaml
except ImportError:
    print('pyyaml 未安装 -> pip install pyyaml')
    sys.exit(1)


def get_robot_ip():
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as handle:
            cfg = yaml.safe_load(handle) or {}
        return cfg.get('robot_ip', '172.16.0.2')
    return '172.16.0.2'


def load_position_reader():
    """Reuse the Franka pose reading compatibility logic from Step 6."""
    spec = importlib.util.spec_from_file_location('teach_workspace', POSITION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_robot_pose


def read_point(read_robot_pose, ip, label, instruction):
    print(f'\n{label}: {instruction}')
    input('  保持笔尖静止后按 Enter 读取此点...')
    x, y, z, library = read_robot_pose(ip)
    point = np.array([x, y, z], dtype=float)
    print(f'  {label} = ({x:.6f}, {y:.6f}, {z:.6f}) m  [{library}]')
    return point


def calculate_surface_frame(center, x_point, y_point):
    x_vector = x_point - center
    y_vector = y_point - center
    x_span = float(np.linalg.norm(x_vector))
    y_span = float(np.linalg.norm(y_vector))
    if x_span < 0.01 or y_span < 0.01:
        raise ValueError('示教点距离中心过近；X/Y 参考点应距中心至少 10 mm')

    normal_raw = np.cross(x_vector, y_vector)
    area = float(np.linalg.norm(normal_raw))
    if area < 1e-5:
        raise ValueError('三个点接近共线；请重新选择纸面 X/Y 方向参考点')

    normal = normal_raw / area
    if normal[2] < 0.0:
        normal = -normal
        print('  提示: 点序得到向下法向，已自动翻转为朝上的纸面法向。')

    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_axis = x_axis / np.linalg.norm(x_axis)
    return normal, x_axis, x_span, y_span


def save_config(center, normal, x_axis):
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as handle:
            cfg = yaml.safe_load(handle) or {}
    else:
        cfg = {}

    cfg['workspace_center'] = {
        'x': round(float(center[0]), 6),
        'y': round(float(center[1]), 6),
        'z': round(float(center[2]), 6),
    }
    cfg['paper_orientation'] = {
        'normal': {
            'x': round(float(normal[0]), 8),
            'y': round(float(normal[1]), 8),
            'z': round(float(normal[2]), 8),
        },
        'x_axis': {
            'x': round(float(x_axis[0]), 8),
            'y': round(float(x_axis[1]), 8),
            'z': round(float(x_axis[2]), 8),
        },
        'tool_z_points_toward_paper': True,
    }

    with open(CFG_PATH, 'w') as handle:
        yaml.safe_dump(cfg, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description='纸面姿态三点示教')
    parser.add_argument('--ip', type=str, default=get_robot_ip())
    parser.add_argument('--dry-run', action='store_true', help='只计算并打印，不写入配置')
    args = parser.parse_args()

    print('=' * 64)
    print('纸面姿态三点示教 · 阶段2 Step 6b')
    print('=' * 64)
    print(f'\n机器人 IP: {args.ip}')
    print('此脚本不会自动运动。请在 Franka Desk 中启用手动导引。')
    print('使用同一笔尖轻触三个点，并尽量保持夹笔器朝向不变。')
    print('P0 建议为纸张中心；Px/Py 各离 P0 至少 3 cm。')

    try:
        read_robot_pose = load_position_reader()
        center = read_point(read_robot_pose, args.ip, 'P0', '将笔尖移动到纸面中心')
        x_point = read_point(read_robot_pose, args.ip, 'Px', '沿期望书写 +X 方向移动笔尖')
        y_point = read_point(read_robot_pose, args.ip, 'Py', '从中心沿期望书写 +Y 方向移动笔尖')
        normal, x_axis, x_span, y_span = calculate_surface_frame(center, x_point, y_point)
    except Exception as exc:
        print(f'\n读取或计算失败: {exc}')
        return 1

    print('\n计算结果:')
    print(f'  P0/workspace_center = {np.round(center, 6).tolist()} m')
    print(f'  示教跨度 X/Y       = {x_span * 1000:.1f} / {y_span * 1000:.1f} mm')
    print(f'  paper normal       = {np.round(normal, 8).tolist()}')
    print(f'  paper x_axis       = {np.round(x_axis, 8).tolist()}')
    print(f'  目标工具 Z         = {np.round(-normal, 8).tolist()}')

    if args.dry_run:
        print('\n--dry-run 模式，不写入配置。')
    else:
        confirm = input('\n将以上中心点与纸面姿态写入 franka_config.yaml？(y/N): ')
        if confirm.strip().lower() != 'y':
            print('已取消写入。')
            return 0
        save_config(center, normal, x_axis)
        print(f'\n已更新配置: {CFG_PATH}')

    print('\n下一步: python 03_check_config.py 复验配置，然后运行 07_first_move.py。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
