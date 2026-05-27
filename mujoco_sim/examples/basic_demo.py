#!/usr/bin/env python3
"""
基础演示 - MuJoCo 书法仿真

演示功能:
1. 加载 Franka Panda 模型
2. 执行简单轨迹
3. 可视化笔迹
4. 保存结果
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mujoco_simulator import FrankaCalligraphySimulator
import numpy as np


def demo_basic_trajectory():
    """演示基础轨迹执行"""
    print("\n" + "=" * 70)
    print("Demo 1: 基础轨迹执行")
    print("=" * 70 + "\n")

    # 创建仿真器
    sim = FrankaCalligraphySimulator(render_mode="human")

    # 使用示例 NPZ 文件
    npz_path = "../../callibrate/examples/simple_line.npz"

    if os.path.exists(npz_path):
        print(f"✅ Found example file: {npz_path}")
        sim.execute_trajectory(npz_path, speed=0.1, render=True)
    else:
        print(f"⚠️  Example file not found: {npz_path}")
        print("Creating simple test trajectory...")
        demo_create_trajectory(sim)

    sim.close()


def demo_create_trajectory(sim):
    """创建并执行简单测试轨迹"""
    print("\n创建测试轨迹: 直线")

    # 定义轨迹: 画一条直线
    x = np.linspace(0.5, 0.6, 50)  # 0.5m 到 0.6m (10cm)
    y = np.zeros(50)  # Y 方向不动
    z = np.concatenate(
        [
            [0.3],  # 起始: 抬笔
            [-0.09] * 48,  # 中间: 接触纸面
            [0.3],  # 结束: 抬笔
        ]
    )

    print(f"轨迹点数: {len(x)}")
    print(f"X 范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Z 范围: [{z.min():.3f}, {z.max():.3f}]")

    # 执行
    for i in range(len(x)):
        target = np.array([x[i], y[i], z[i]])
        sim.move_to_position(target, speed=0.1)

        if i % 10 == 0:
            print(f"Progress: {i}/{len(x)}")

    print("\n✅ 测试轨迹执行完成")
    sim._show_results()


def demo_multiple_strokes():
    """演示多笔画轨迹"""
    print("\n" + "=" * 70)
    print("Demo 2: 多笔画轨迹 (永字八法)")
    print("=" * 70 + "\n")

    sim = FrankaCalligraphySimulator()

    # 检查示例文件
    npz_path = "../../callibrate/examples/example_永.npz"

    if os.path.exists(npz_path):
        print(f"✅ 加载'永'字轨迹: {npz_path}")
        sim.execute_trajectory(npz_path, speed=0.05, render=True)
    else:
        print(f"⚠️  文件不存在: {npz_path}")
        print("请先运行: cd ../callibrate && python generate_example_npz.py")

    sim.close()


def demo_comparison():
    """演示不同速度对比"""
    print("\n" + "=" * 70)
    print("Demo 3: 速度对比")
    print("=" * 70 + "\n")

    npz_path = "../../callibrate/examples/simple_line.npz"

    if not os.path.exists(npz_path):
        print(f"⚠️  示例文件不存在: {npz_path}")
        return

    speeds = [0.02, 0.05, 0.1]  # 慢、中、快

    for i, speed in enumerate(speeds):
        print(f"\n--- 测试 {i+1}/3: 速度 = {speed} m/s ---")

        sim = FrankaCalligraphySimulator()
        sim.execute_trajectory(npz_path, speed=speed, render=False)

        # 保存结果
        import cv2

        output_path = f"outputs/result_speed_{speed:.2f}.png"
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(output_path, sim.paper_canvas)
        print(f"✅ 结果已保存: {output_path}")

        sim.close()

    print("\n✅ 所有速度测试完成")
    print("比较文件:")
    for speed in speeds:
        print(f"  - outputs/result_speed_{speed:.2f}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo 书法仿真演示")
    parser.add_argument(
        "--demo",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="演示编号 (1: 基础, 2: 多笔画, 3: 速度对比)",
    )

    args = parser.parse_args()

    if args.demo == 1:
        demo_basic_trajectory()
    elif args.demo == 2:
        demo_multiple_strokes()
    elif args.demo == 3:
        demo_comparison()
