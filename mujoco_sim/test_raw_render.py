#!/usr/bin/env python3
"""
直接测试MuJoCo渲染 - 看看机器人在哪里
"""

import mujoco
import numpy as np
import cv2

# 加载模型
model = mujoco.MjModel.from_xml_path('models/franka_panda.xml')
data = mujoco.MjData(model)

# 前向动力学
mujoco.mj_forward(model, data)

# 创建渲染器
renderer = mujoco.Renderer(model, height=720, width=1280)

# 使用top_view相机
camera_id = 0

# 渲染
renderer.update_scene(data, camera=camera_id)
frame = renderer.render()

# 保存
cv2.imwrite('demo_outputs/raw_render_initial.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print(f"Saved: demo_outputs/raw_render_initial.png")
print(f"Frame average RGB: {frame.mean(axis=(0,1))}")

# 打印所有body的位置
print("\n所有body的位置:")
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  {body_name:20s}: {data.xpos[i]}")

# 打印所有geom
print("\n所有geom:")
for i in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    geom_type = model.geom_type[i]
    geom_size = model.geom_size[i]
    geom_rgba = model.geom_rgba[i]
    print(f"  {geom_name:20s}: type={geom_type}, size={geom_size}, rgba={geom_rgba}")
