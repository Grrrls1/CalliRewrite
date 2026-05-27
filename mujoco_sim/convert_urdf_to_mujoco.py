#!/usr/bin/env python3
"""
将Franka FR3 URDF转换为MuJoCo XML
"""

import mujoco
import os
import subprocess

# 首先处理xacro -> URDF
xacro_file = "temp_franka/robots/fr3/fr3.urdf.xacro"
urdf_file = "models/fr3_generated.urdf"

print("处理xacro文件...")
print(f"输入: {xacro_file}")
print(f"输出: {urdf_file}")

# 使用ros xacro命令处理
try:
    # 尝试使用ROS xacro
    result = subprocess.run([
        "ros2", "run", "xacro", "xacro",
        xacro_file,
        "hand:=false",  # 不要gripper
        "gazebo:=false",
        "ros2_control:=false"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        with open(urdf_file, 'w') as f:
            f.write(result.stdout)
        print(f"✅ URDF已生成: {urdf_file}")
    else:
        print(f"❌ xacro处理失败: {result.stderr}")
        print("尝试备用方案...")
        raise Exception("xacro failed")

except Exception as e:
    print(f"ROS xacro不可用: {e}")
    print("将使用简化的手动URDF...")

    # 备用方案：使用简化的手动创建的URDF
    # （如果没有ROS环境）
    print("❌ 需要ROS环境来处理xacro文件")
    print("建议：")
    print("1. 安装ROS 2: https://docs.ros.org/en/humble/Installation.html")
    print("2. 或者使用已经处理好的URDF文件")
