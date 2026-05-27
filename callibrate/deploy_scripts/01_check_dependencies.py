#!/usr/bin/env python3
"""
阶段0 · Step 1 ── 依赖检查（无需机器人）

检查所有运行所需的 Python 包、配置文件、NPZ 文件是否就绪。
全部 ✅ 才可继续下一步。
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = []

def ok(msg):
    RESULTS.append(('✅', msg))
    print(f'  ✅  {msg}')

def warn(msg):
    RESULTS.append(('⚠️', msg))
    print(f'  ⚠️  {msg}')

def fail(msg):
    RESULTS.append(('❌', msg))
    print(f'  ❌  {msg}')


print('=' * 60)
print('依赖检查 · 阶段0 Step 1')
print('=' * 60)

# ── 1. Python 版本 ─────────────────────────────────────────────
print('\n[1] Python 版本')
v = sys.version_info
if v.major == 3 and v.minor >= 10:
    ok(f'Python {v.major}.{v.minor}.{v.micro}')
else:
    fail(f'Python {v.major}.{v.minor} — 需要 3.10+')

# ── 2. 必须包 ──────────────────────────────────────────────────
print('\n[2] 必须 Python 包')
required = ['numpy', 'yaml', 'matplotlib', 'scipy', 'cv2', 'skimage']
for pkg in required:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        ok(f'{pkg}  {ver}')
    except ImportError:
        fail(f'{pkg} 未安装 → pip install {pkg}')

# ── 3. Franka 控制库（franky 或 frankx）────────────────────────
print('\n[3] Franka 控制库')
has_franka = False
try:
    import franky
    ok(f'franky  {getattr(franky, "__version__", "?")}')
    has_franka = True
except ImportError:
    warn('franky 未安装')

if not has_franka:
    try:
        import frankx
        ok(f'frankx  {getattr(frankx, "__version__", "?")}')
        has_franka = True
    except ImportError:
        warn('frankx 未安装')

if not has_franka:
    fail('franky 和 frankx 均未安装 → pip install frankx')

# ── 4. MuJoCo（仿真用，可选但推荐）────────────────────────────
print('\n[4] MuJoCo（仿真验证用）')
try:
    import mujoco
    ok(f'mujoco  {mujoco.__version__}')
except ImportError:
    warn('mujoco 未安装（仿真验证将不可用）→ pip install mujoco')

# ── 5. 配置文件 ────────────────────────────────────────────────
print('\n[5] 配置文件')
cfg_path = os.path.join(ROOT, 'franka_config.yaml')
if os.path.exists(cfg_path):
    import yaml
    try:
        cfg = yaml.safe_load(open(cfg_path))
        wc = cfg.get('workspace_center', {})
        x, y, z = wc.get('x'), wc.get('y'), wc.get('z')
        ok(f'franka_config.yaml  workspace_center=({x}, {y}, {z})')
        # 合理性检查
        if x and not (0.2 <= x <= 0.8):
            warn(f'  workspace_center.x={x} 超出典型范围 [0.2, 0.8]，请核实')
        if z and not (0.0 <= z <= 0.5):
            warn(f'  workspace_center.z={z} 超出典型范围 [0.0, 0.5]，请核实')
        if x is None or y is None or z is None:
            fail('  franka_config.yaml 中 workspace_center 字段不完整！')
        orientation = cfg.get('paper_orientation', {})
        normal = orientation.get('normal', {})
        x_axis = orientation.get('x_axis', {})
        if all(key in normal and key in x_axis for key in ('x', 'y', 'z')):
            ok('franka_config.yaml  paper_orientation 已配置（RoboControl2）')
        else:
            fail('  paper_orientation 字段不完整 -> 运行 06b_teach_paper_orientation.py')
    except Exception as e:
        fail(f'franka_config.yaml 解析失败: {e}')
else:
    fail(f'franka_config.yaml 不存在: {cfg_path}')

# ── 6. NPZ 文件 ────────────────────────────────────────────────
print('\n[6] NPZ 轨迹文件')
npz_files = [
    'examples/example_永.npz',
    'examples/test_calibration.npz',
    'examples/simple_line.npz',
]
for rel in npz_files:
    path = os.path.join(ROOT, rel)
    if os.path.exists(path):
        import numpy as np
        d = np.load(path)
        n = len(d['pos_3d_x'])
        ok(f'{rel}  ({n} 个控制点)')
    else:
        warn(f'{rel} 不存在（可运行 generate_example_npz.py 生成）')

# ── 汇总 ───────────────────────────────────────────────────────
print('\n' + '=' * 60)
fails = [m for s, m in RESULTS if s == '❌']
warns = [m for s, m in RESULTS if s == '⚠️']
print(f'结果: {len(RESULTS)-len(fails)-len(warns)} ✅ | {len(warns)} ⚠️ | {len(fails)} ❌')
if fails:
    print('\n必须修复后再继续:')
    for m in fails:
        print(f'  ❌ {m}')
else:
    print('\n✅ 依赖检查通过，可进行下一步。')
print('=' * 60)
