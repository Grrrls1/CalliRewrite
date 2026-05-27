#!/usr/bin/env python3
"""
生成视频演示 - 从俯视角度录制书法轨迹
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _npz_output_stem(npz_path):
    """Return a usable output prefix from an NPZ path."""
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    return stem or "calligraphy_demo"


def resolve_default_outputs(
    npz_path,
    output_video=None,
    canvas_output=None,
    trajectory_output=None,
    result_output_dir=None,
):
    """Fill omitted output paths using the input NPZ filename as prefix."""
    stem = _npz_output_stem(npz_path)
    video_dir = "outputs"
    image_dir = result_output_dir or video_dir

    if output_video is None:
        output_video = os.path.join(video_dir, f"{stem}.mp4")
    if canvas_output is None:
        canvas_output = os.path.join(image_dir, f"{stem}_result.png")
    if trajectory_output is None:
        trajectory_output = os.path.join(image_dir, f"{stem}_3d.png")

    return output_video, canvas_output, trajectory_output

def generate_video(
    npz_path,
    output_video=None,
    speed=0.05,
    fps=30,
    camera_distance=2,
    camera_azimuth=45,
    camera_elevation=-30,
    canvas_output=None,
    trajectory_output=None,
    result_output_dir=None,
):
    """
    生成书法轨迹视频

    Args:
        npz_path: NPZ文件路径
        output_video: 输出视频路径，默认 outputs/<npz文件名>.mp4
        speed: 运动速度 (m/s)
        fps: 视频帧率
        camera_distance: 相机距离 (m)
        camera_azimuth: 相机水平角度 (度)
        camera_elevation: 相机俯仰角 (度)
        canvas_output: 画布输出路径，默认 outputs/<npz文件名>_result.png
        trajectory_output: 3D轨迹图输出路径，默认 outputs/<npz文件名>_3d.png
        result_output_dir: 结果图片默认输出目录
    """
    output_video, canvas_output, trajectory_output = resolve_default_outputs(
        npz_path,
        output_video=output_video,
        canvas_output=canvas_output,
        trajectory_output=trajectory_output,
        result_output_dir=result_output_dir,
    )

    print("=" * 70)
    print("MuJoCo 书法仿真 - 视频录制")
    print("=" * 70)
    print(f"\n输入文件: {npz_path}")
    print(f"输出视频: {output_video}")
    print(f"速度: {speed} m/s")
    print(f"帧率: {fps} FPS\n")

    import imageio
    import mujoco
    import numpy as np
    from mujoco_simulator_v2 import FrankaCalligraphySimulator

    # 加载 NPZ 数据
    data = np.load(npz_path)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    num_points = len(x)
    print(f"✅ 加载轨迹数据:")
    print(f"   控制点数: {num_points}")
    print(f"   X 范围: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"   Y 范围: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"   Z 范围: [{z.min():.4f}, {z.max():.4f}] m\n")

    # 创建仿真器（离屏模式）
    print("初始化 MuJoCo 仿真器...")
    sim = FrankaCalligraphySimulator(
        render_mode="rgb_array",
        camera_distance=camera_distance,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation
    )

    # 创建离屏渲染器
    print("创建离屏渲染器 (1280x720)...")
    renderer = mujoco.Renderer(sim.model, height=720, width=1280)

    # 创建自定义相机
    camera = mujoco.MjvCamera()
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation
    camera.lookat = np.array([sim.paper_offset[0] + 0.15, sim.paper_offset[1] + 0.21, 0.01])  # 看向纸张中心

    # 使用动态相机参数
    print(f"✅ 相机参数:")
    print(f"   距离: {camera_distance} m")
    print(f"   方位角: {camera_azimuth}°")
    print(f"   仰角: {camera_elevation}°")
    print(f"   看向: {camera.lookat}\n")

    # 初始化视频录制
    frames = []
    print("开始执行轨迹并录制视频...\n")

    # 执行轨迹并录制
    for i in range(num_points):
        sim.current_traj_z_m = float(z[i])
        sim.current_pen_depth_m = max(0.0, float(-z[i]))

        # NPZ 坐标是相对于纸张的，需要转换到世界坐标系。
        # 注意：NPZ 的 z 通常是“相对于纸面”的高度（z=0 为纸面，z<0 为按压）。
        # 需要转换到世界坐标并夹紧，避免 IK 试图到达纸面以下导致发散/抖动。
        z_world = z[i] + 0.011
        z_clamped = max(z_world, 0.007)

        # 纸张左下角在世界坐标 sim.paper_offset
        target_pos = np.array([
            x[i] + sim.paper_offset[0],  # 加上X偏移
            y[i] + sim.paper_offset[1],  # 加上Y偏移
            z_clamped,
        ])

        # 移动到目标位置
        sim.move_to_position(target_pos, speed=speed)

        # 渲染当前帧（使用动态相机参数）
        renderer.update_scene(sim.data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)

        # 显示进度
        if i % 10 == 0 or i == num_points - 1:
            progress = (i + 1) / num_points * 100
            print(f"  进度: {i+1}/{num_points} ({progress:.1f}%)")

    print(f"\n✅ 轨迹执行完成!")
    print(f"   总帧数: {len(frames)}")
    print(f"   视频时长: {len(frames)/fps:.2f} 秒")

    # 检查笔迹统计
    if len(sim.ink_traces) > 0:
        contact_points = sum(1 for _, _, _, c in sim.ink_traces if c)
        print(f"   记录点数: {len(sim.ink_traces)}")
        print(f"   接触点数: {contact_points}")
        print(f"   接触率: {contact_points/len(sim.ink_traces)*100:.1f}%")

    # 保存视频
    print(f"\n保存视频到: {output_video}")
    os.makedirs(os.path.dirname(output_video) if os.path.dirname(output_video) else ".", exist_ok=True)
    imageio.mimsave(output_video, frames, fps=fps)

    # 获取视频文件大小
    video_size_kb = os.path.getsize(output_video) / 1024
    video_size_mb = video_size_kb / 1024

    if video_size_mb >= 1:
        print(f"✅ 视频已保存! 大小: {video_size_mb:.2f} MB")
    else:
        print(f"✅ 视频已保存! 大小: {video_size_kb:.1f} KB")

    # 保存画布和 3D 轨迹图
    sim.save_results(
        canvas_output=canvas_output,
        trajectory_output=trajectory_output,
        output_dir=result_output_dir,
    )

    # 清理
    sim.close()

    print("\n" + "=" * 70)
    print("视频生成完成!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成书法轨迹视频（俯视角度）")
    parser.add_argument("npz_file", type=str, help="NPZ 轨迹文件路径")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出视频路径 (默认: outputs/<npz文件名>.mp4)")
    parser.add_argument("--speed", "-s", type=float, default=0.05,
                        help="运动速度 (m/s, 默认: 0.05)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="视频帧率 (默认: 30)")
    parser.add_argument("--camera-distance", type=float, default=1.5,
                        help="相机距离 (m, 默认: 1.5)")
    parser.add_argument("--camera-azimuth", type=float, default=90,
                        help="相机水平角度 (度, 默认: 45. 0=正前, 90=右侧, 180=背后, 270=左侧)")
    parser.add_argument("--camera-elevation", type=float, default=-30,
                        help="相机俯仰角 (度, 默认: -30. -90=俯视, 0=平视, 90=仰视)")
    parser.add_argument("--result-output-dir", type=str, default=None,
                        help="结果图片默认输出目录 (默认: outputs)")
    parser.add_argument("--canvas-output", type=str, default=None,
                        help="画布图片输出路径/文件名 (默认: outputs/<npz文件名>_result.png)")
    parser.add_argument("--trajectory-output", type=str, default=None,
                        help="3D轨迹图片输出路径/文件名 (默认: outputs/<npz文件名>_3d.png)")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.npz_file):
        print(f"❌ 错误: 文件不存在: {args.npz_file}")
        sys.exit(1)

    # 生成视频
    generate_video(
        args.npz_file,
        args.output,
        args.speed,
        args.fps,
        args.camera_distance,
        args.camera_azimuth,
        args.camera_elevation,
        args.canvas_output,
        args.trajectory_output,
        args.result_output_dir,
    )
    

'''
python generate_video_demo.py rl_npz/yong.npz \
  -o outputs/yong.mp4 \
  --canvas-output yong_result.png \
  --trajectory-output yong_3d.png \
  --result-output-dir outputs


'''    
