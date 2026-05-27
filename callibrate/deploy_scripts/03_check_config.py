#!/usr/bin/env python3
"""
阶段0 · Step 3 ── 配置文件检查 + 坐标预览（无需机器人）

检查 franka_config.yaml 中 workspace_center 是否合理，
并预览：给定配置后，NPZ 坐标将被映射到机器人的哪个物理位置。

⚠️ workspace_center 必须是你实际测量的纸面中心坐标（Franka 基坐标系），
   不能用默认值直接上机！
"""

import argparse
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(ROOT, 'franka_config.yaml')

try:
    import yaml
except ImportError:
    print('❌ pyyaml 未安装 → pip install pyyaml')
    sys.exit(1)


def load_config():
    if not os.path.exists(CFG_PATH):
        print(f'❌ 找不到配置文件: {CFG_PATH}')
        sys.exit(1)
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def config_vector(section, key):
    value = section.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f'paper_orientation.{key} 必须包含 x/y/z')
    return np.array([float(value['x']), float(value['y']), float(value['z'])])


def check_config(cfg):
    print('=' * 60)
    print('配置文件检查 · 阶段0 Step 3')
    print('=' * 60)
    print(f'\n配置路径: {CFG_PATH}\n')

    wc = cfg.get('workspace_center', {})
    x = wc.get('x')
    y = wc.get('y')
    z = wc.get('z')

    print(f'  workspace_center:')
    print(f'    x = {x} m  （机器人基坐标系，前方）')
    print(f'    y = {y} m  （机器人基坐标系，左/右）')
    print(f'    z = {z} m  （纸面高度）')

    issues = []
    if x is None or y is None or z is None:
        issues.append('❌ workspace_center 字段不完整，请填写 x/y/z')
    else:
        # Franka 典型可达范围检查
        if not (0.2 <= x <= 0.8):
            issues.append(f'⚠️ x={x} 超出 Franka 典型范围 [0.2, 0.8]m，请核实')
        if not (-0.4 <= y <= 0.4):
            issues.append(f'⚠️ y={y} 超出 Franka 典型范围 [-0.4, 0.4]m，请核实')
        if not (0.0 <= z <= 0.6):
            issues.append(f'⚠️ z={z} 超出 Franka 典型范围 [0.0, 0.6]m，请核实')

        # 安全边界预览
        safety = cfg.get('safety', {})
        max_xy = safety.get('max_xy_range', 0.15)
        max_z  = safety.get('max_z_range', 0.10)
        print(f'\n  安全边界 (相对于 workspace_center):')
        print(f'    X: [{x-max_xy:.3f}, {x+max_xy:.3f}] m  (±{max_xy}m)')
        print(f'    Y: [{y-max_xy:.3f}, {y+max_xy:.3f}] m  (±{max_xy}m)')
        print(f'    Z: [{z-max_z:.3f}, {z+max_z:.3f}] m   (±{max_z}m)')

    print('\n  paper_orientation (RoboControl2):')
    orientation = cfg.get('paper_orientation', {})
    try:
        normal = config_vector(orientation, 'normal')
        x_axis = config_vector(orientation, 'x_axis')
        normal_length = float(np.linalg.norm(normal))
        x_length = float(np.linalg.norm(x_axis))
        if normal_length < 1e-8 or x_length < 1e-8:
            raise ValueError('normal 和 x_axis 不能为零向量')
        normal = normal / normal_length
        x_axis = x_axis / x_length
        alignment = abs(float(np.dot(normal, x_axis)))
        if alignment > 0.98:
            raise ValueError('x_axis 与 normal 近似平行，无法确定纸面方向')
        toward_paper = bool(orientation.get('tool_z_points_toward_paper', True))
        target_tool_z = -normal if toward_paper else normal
        print(f'    normal = {np.round(normal, 6).tolist()}')
        print(f'    x_axis = {np.round(x_axis, 6).tolist()}')
        print(f'    target tool Z = {np.round(target_tool_z, 6).tolist()}')
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f'❌ paper_orientation 无效: {exc}；请运行 06b_teach_paper_orientation.py')

    other_keys = ['robot_ip', 'default_speed', 'default_acceleration']
    print(f'\n  其他配置:')
    for k in other_keys:
        print(f'    {k}: {cfg.get(k, "（未设置）")}')

    print()
    if issues:
        for iss in issues:
            print(f'  {iss}')
        return False
    else:
        print('  ✅ 配置文件格式正常')
        return True


def preview_mapping(cfg, npz_path):
    """预览 NPZ 坐标映射到机器人坐标后的结果"""
    if not os.path.exists(npz_path):
        return

    wc = cfg['workspace_center']
    wx, wy, wz = wc['x'], wc['y'], wc['z']

    d = np.load(npz_path)
    x, y, z = d['pos_3d_x'], d['pos_3d_y'], d['pos_3d_z']

    # 坐标映射（v2 NPZ 格式：纸面相对坐标，z 沿纸面法向）
    x_center = (x.min() + x.max()) / 2
    y_center = (y.min() + y.max()) / 2
    orientation = cfg['paper_orientation']
    normal = config_vector(orientation, 'normal')
    normal = normal / np.linalg.norm(normal)
    surface_x = config_vector(orientation, 'x_axis')
    surface_x = surface_x - np.dot(surface_x, normal) * normal
    surface_x = surface_x / np.linalg.norm(surface_x)
    surface_y = np.cross(normal, surface_x)
    positions = (
        np.array([wx, wy, wz])
        + (x - x_center)[:, None] * surface_x
        + (y - y_center)[:, None] * surface_y
        + z[:, None] * normal
    )
    x_robot, y_robot, z_robot = positions.T

    print(f'\n  坐标映射预览（文件: {os.path.basename(npz_path)}）:')
    print(f'    NPZ X [{x.min():.4f}, {x.max():.4f}] → 机器人 X [{x_robot.min():.4f}, {x_robot.max():.4f}] m')
    print(f'    NPZ Y [{y.min():.4f}, {y.max():.4f}] → 机器人 Y [{y_robot.min():.4f}, {y_robot.max():.4f}] m')
    print(f'    NPZ Z [{z.min():.4f}, {z.max():.4f}] → 机器人 Z [{z_robot.min():.4f}, {z_robot.max():.4f}] m')

    # 检查映射后是否在安全范围内
    safety = cfg.get('safety', {})
    max_xy = safety.get('max_xy_range', 0.15)
    max_z  = safety.get('max_z_range', 0.10)
    out_x = np.sum((x_robot < wx - max_xy) | (x_robot > wx + max_xy))
    out_y = np.sum((y_robot < wy - max_xy) | (y_robot > wy + max_xy))
    out_z = np.sum((z_robot < wz - max_z)  | (z_robot > wz + max_z))
    if out_x or out_y or out_z:
        print(f'\n    ⚠️ 部分点超出安全边界: X={out_x}点 Y={out_y}点 Z={out_z}点')
        print(f'       → 这些点会被 RoboControl2 自动截断到安全范围内')
    else:
        print(f'    ✅ 映射后所有点均在安全范围内')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='检查 Franka 工作区和纸面姿态配置')
    parser.add_argument('--npz', type=str, default=None, help='预览指定 NPZ 轨迹映射')
    args = parser.parse_args()
    cfg = load_config()
    ok = check_config(cfg)

    # 预览一个示例 NPZ 的映射结果
    sample = args.npz or os.path.join(ROOT, 'examples', 'ri.npz')
    if ok and os.path.exists(sample):
        preview_mapping(cfg, sample)

    print('\n' + '=' * 60)
    if not ok:
        print('❌ 配置存在问题，请修正后再继续。')
        print(f'   编辑: {CFG_PATH}')
    else:
        print('✅ 配置检查通过。')
        print('\n如果以上中心和姿态尚未实测，请先运行 06b_teach_paper_orientation.py。')
        print('若已完成三点示教，下一步运行 07_first_move.py。')
    print('=' * 60)
