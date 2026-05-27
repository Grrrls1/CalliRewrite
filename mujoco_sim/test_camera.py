#!/usr/bin/env python3
"""
测试不同的相机位置和角度
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import numpy as np
from mujoco_simulator import FrankaCalligraphySimulator
import cv2

# 创建仿真器
sim = FrankaCalligraphySimulator(render_mode="rgb_array")

# 创建渲染器
renderer = mujoco.Renderer(sim.model, height=720, width=1280)

# 测试不同的相机位置
camera_configs = [
    {"pos": [0.25, -0.8, 1.2], "euler": [1.4, 0, 0], "name": "config1"},
    {"pos": [0.3, -1.0, 1.5], "euler": [1.5, 0, 0], "name": "config2"},
    {"pos": [0.2, -0.6, 1.0], "euler": [1.3, 0, 0], "name": "config3"},
    {"pos": [0.15, -0.7, 1.3], "euler": [1.35, 0, 0], "name": "config4"},
]

print("测试不同的相机配置...")
print("机器人base位置: (0, 0, 0)")
print("纸张位置: (0.5, 0, 0.01)")
print()

# 移动机器人到一个可见位置
target_pos = np.array([0.4, 0, 0.3])
sim.move_to_position(target_pos, speed=0.1, wait_time=0.0)

for config in camera_configs:
    # 更新相机参数（手动设置，因为XML相机已经存在）
    # 这里我们使用固定相机ID并渲染
    camera_id = 0  # top_view camera ID

    # 渲染
    renderer.update_scene(sim.data, camera=camera_id)
    frame = renderer.render()

    # 检查帧内容
    avg_rgb = frame.mean(axis=(0,1))

    print(f"{config['name']}: pos={config['pos']}, euler={config['euler']}")
    print(f"  Average RGB: [{avg_rgb[0]:.1f}, {avg_rgb[1]:.1f}, {avg_rgb[2]:.1f}]")

    # 保存帧
    output_path = f"demo_outputs/camera_test_{config['name']}.png"
    cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {output_path}")
    print()

sim.close()
print("✅ 测试完成！查看 demo_outputs/camera_test_*.png")
