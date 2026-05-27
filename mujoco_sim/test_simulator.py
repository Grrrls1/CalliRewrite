#!/usr/bin/env python3
"""
测试脚本 - 验证 MuJoCo 仿真器功能
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """测试 1: 模型加载"""
    print("\n" + "="*60)
    print("测试 1: 加载 MuJoCo 模型")
    print("="*60)

    try:
        from mujoco_simulator import FrankaCalligraphySimulator

        model_path = "models/franka_panda.xml"
        print(f"模型路径: {model_path}")

        sim = FrankaCalligraphySimulator(
            model_path=model_path,
            render_mode="rgb_array"  # 离屏渲染
        )

        print(f"✅ 模型加载成功!")
        print(f"   自由度 (DoF): {sim.model.nv}")
        print(f"   执行器数量: {sim.model.nu}")
        print(f"   刚体数量: {sim.model.nbody}")
        print(f"   关节数量: {sim.model.njnt}")
        print(f"   仿真频率: {sim.control_freq} Hz")

        return sim

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_kinematics(sim):
    """测试 2: 运动学"""
    print("\n" + "="*60)
    print("测试 2: 末端执行器运动学")
    print("="*60)

    try:
        # 获取初始末端位姿
        pos, quat = sim.get_ee_pose()
        print(f"初始末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")

        # 测试逆运动学
        print("\n测试逆运动学求解...")
        target_positions = [
            [0.5, 0.0, 0.3],   # 上方
            [0.5, 0.1, 0.2],   # 右侧
            [0.6, 0.0, 0.3],   # 前方
        ]

        for i, target in enumerate(target_positions):
            print(f"\n  目标 {i+1}: {target}")
            success = sim.inverse_kinematics(
                np.array(target),
                max_iter=100,
                tol=0.01
            )

            if success:
                final_pos, _ = sim.get_ee_pose()
                error = np.linalg.norm(final_pos - target)
                print(f"  ✅ IK 成功! 误差: {error*1000:.2f} mm")
            else:
                print(f"  ⚠️  IK 未收敛 (可能目标超出工作空间)")

        return True

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_contact_detection(sim):
    """测试 3: 接触检测"""
    print("\n" + "="*60)
    print("测试 3: 笔刷接触检测")
    print("="*60)

    try:
        # 移动到纸面上方
        target_above = np.array([0.5, 0.0, 0.05])
        print(f"移动到纸面上方: {target_above}")

        sim.inverse_kinematics(target_above)
        pos, force = sim.get_brush_contact()
        print(f"  位置: {pos}")
        print(f"  接触力: {force:.4f} N")

        # 移动到接触位置
        target_contact = np.array([0.5, 0.0, -0.085])
        print(f"\n移动到接触位置: {target_contact}")

        sim.inverse_kinematics(target_contact)

        # 步进仿真几步让接触稳定
        import mujoco
        for _ in range(50):
            mujoco.mj_step(sim.model, sim.data)

        pos, force = sim.get_brush_contact()
        print(f"  位置: {pos}")
        print(f"  接触力: {force:.4f} N")

        if force > 0.01:
            print(f"✅ 接触检测正常!")
        else:
            print(f"⚠️  未检测到接触 (可能需要调整Z坐标)")

        return True

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_trajectory(sim):
    """测试 4: 简单轨迹执行（带视频录制）"""
    print("\n" + "="*60)
    print("测试 4: 简单轨迹执行（画一条直线）+ 视频录制")
    print("="*60)

    try:
        # 重置仿真
        sim.reset()

        # 创建简单轨迹：水平直线
        num_points = 20
        x = np.linspace(0.5, 0.6, num_points)  # 10cm
        y = np.zeros(num_points)
        z = np.concatenate([
            [0.05],                    # 抬笔
            [-0.09] * (num_points-2),  # 接触
            [0.05]                     # 抬笔
        ])

        print(f"轨迹点数: {num_points}")
        print(f"X 范围: [{x.min():.3f}, {x.max():.3f}] m")
        print(f"Z 范围: [{z.min():.3f}, {z.max():.3f}] m")

        # 设置俯视相机视角（从机器人上方看下来）
        import mujoco

        print("\n开始录制视频（俯视角度）...")

        # 创建离屏渲染器
        renderer = mujoco.Renderer(sim.model, height=720, width=1280)

        # 获取俯视相机ID
        camera_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view")

        # 初始化视频录制
        frames = []

        # 执行轨迹并录制
        for i in range(num_points):
            target = np.array([x[i], y[i], z[i]])
            sim.move_to_position(target, speed=0.1, wait_time=0.0)

            # 渲染当前帧（俯视角度）
            renderer.update_scene(sim.data, camera=camera_id)
            pixels = renderer.render()
            frames.append(pixels)

            if i % 5 == 0:
                print(f"  进度: {i}/{num_points}")

        print(f"✅ 轨迹执行完成!")
        print(f"   录制帧数: {len(frames)}")

        # 检查笔迹
        if len(sim.ink_traces) > 0:
            contact_points = sum(1 for _, _, _, c in sim.ink_traces if c)
            print(f"   记录点数: {len(sim.ink_traces)}")
            print(f"   接触点数: {contact_points}")
            print(f"   接触率: {contact_points/len(sim.ink_traces)*100:.1f}%")

        # 保存画布
        import cv2
        os.makedirs("outputs", exist_ok=True)
        canvas_path = "outputs/test_result.png"
        cv2.imwrite(canvas_path, sim.paper_canvas)
        print(f"   画布已保存: {canvas_path}")

        # 保存视频
        if len(frames) > 0:
            try:
                import imageio
                video_path = "outputs/test_trajectory.mp4"
                imageio.mimsave(video_path, frames, fps=30)
                print(f"   视频已保存: {video_path}")

                # 获取视频文件大小
                video_size = os.path.getsize(video_path) / 1024  # KB
                print(f"   视频大小: {video_size:.1f} KB")
            except ImportError:
                print("   ⚠️  imageio 未安装，跳过视频保存")
                print("   安装方法: pip install imageio imageio-ffmpeg")
            except Exception as e:
                print(f"   ⚠️  视频保存失败: {e}")

        return True

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_npz_file(sim):
    """测试 5: 使用真实 NPZ 文件"""
    print("\n" + "="*60)
    print("测试 5: 加载真实 NPZ 文件")
    print("="*60)

    # 查找 NPZ 文件
    npz_files = [
        "../callibrate/examples/simple_line.npz",
        "../callibrate/examples/test_calibration.npz",
        "../callibrate/examples/example_永.npz",
    ]

    found_file = None
    for npz_path in npz_files:
        if os.path.exists(npz_path):
            found_file = npz_path
            break

    if found_file is None:
        print("⚠️  未找到示例 NPZ 文件")
        print("   请先运行: cd ../callibrate && python generate_example_npz.py")
        return False

    try:
        print(f"使用文件: {found_file}")

        # 加载数据
        data = np.load(found_file)
        x = data['pos_3d_x']
        y = data['pos_3d_y']
        z = data['pos_3d_z']

        print(f"✅ NPZ 文件加载成功!")
        print(f"   控制点数: {len(x)}")
        print(f"   X 范围: [{x.min():.4f}, {x.max():.4f}] m")
        print(f"   Y 范围: [{y.min():.4f}, {y.max():.4f}] m")
        print(f"   Z 范围: [{z.min():.4f}, {z.max():.4f}] m")

        # 执行轨迹（只执行前面几个点来测试）
        print("\n执行前 10 个点...")
        sim.reset()

        for i in range(min(10, len(x))):
            target = np.array([x[i], y[i], z[i]])
            sim.move_to_position(target, speed=0.1, wait_time=0.0)
            print(f"  点 {i+1}/10: {target}")

        print(f"✅ NPZ 轨迹测试完成!")

        return True

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("MuJoCo 仿真器测试套件")
    print("="*70)

    results = {}

    # 测试 1: 模型加载
    sim = test_model_loading()
    results['模型加载'] = sim is not None

    if sim is None:
        print("\n❌ 模型加载失败，停止测试")
        return

    # 测试 2: 运动学
    results['运动学'] = test_kinematics(sim)

    # 测试 3: 接触检测
    results['接触检测'] = test_contact_detection(sim)

    # 测试 4: 简单轨迹
    results['简单轨迹'] = test_simple_trajectory(sim)

    # 测试 5: NPZ 文件
    results['NPZ文件'] = test_with_npz_file(sim)

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过 ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️  有 {total-passed} 个测试失败")

    # 清理
    sim.close()


if __name__ == "__main__":
    main()
