#!/usr/bin/env python3
"""
生成复合视频 - 机器人视图 + 画布叠加
"""

import sys
import os
import numpy as np
import mujoco
import imageio
import cv2

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mujoco_simulator import FrankaCalligraphySimulator

def generate_composite_video(npz_path, output_video, speed=0.05, fps=30,
                            canvas_scale=0.4, canvas_position="bottom_right",
                            frames_per_point=1, camera_name="top_view"):
    """
    生成复合视频（机器人 + 画布叠加）

    Args:
        npz_path: NPZ文件路径
        output_video: 输出视频路径
        speed: 运动速度 (m/s)
        fps: 视频帧率
        canvas_scale: 画布缩放比例（相对于视频宽度）
        canvas_position: 画布位置 ("bottom_right", "top_right", "bottom_left", "top_left")
        frames_per_point: 每个轨迹点录制的帧数（用于慢动作效果）
    """
    print("=" * 70)
    print("MuJoCo 书法仿真 - 复合视频录制")
    print("=" * 70)
    print(f"\n输入文件: {npz_path}")
    print(f"输出视频: {output_video}")
    print(f"速度: {speed} m/s")
    print(f"帧率: {fps} FPS")
    print(f"画布位置: {canvas_position}\n")

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
    sim = FrankaCalligraphySimulator(render_mode="rgb_array")

    # 修改_update_ink_trace方法以降低接触力阈值
    original_update = sim._update_ink_trace
    def patched_update():
        brush_pos, contact_force = sim.get_brush_contact()
        is_contact = contact_force > 0.001  # 从0.01降低到0.001
        sim.ink_traces.append((*brush_pos, is_contact))
        if is_contact:
            sim._draw_on_canvas(brush_pos)
        else:
            # 笔抬起时重置连线起点，避免跨笔画连线
            if hasattr(sim, '_last_canvas_pos'):
                delattr(sim, '_last_canvas_pos')
    sim._update_ink_trace = patched_update

    # 创建离屏渲染器
    video_width, video_height = 1280, 720
    print(f"创建离屏渲染器 ({video_width}x{video_height})...")
    renderer = mujoco.Renderer(sim.model, height=video_height, width=video_width)

    # 获取相机ID
    camera_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    print(f"✅ 使用相机: {camera_name} (ID={camera_id})\n")

    # 初始化视频录制
    frames = []
    print("开始执行轨迹并录制视频...\n")

    # 执行轨迹并录制
    for i in range(num_points):
        # NPZ坐标是相对于纸张的，需要转换到世界坐标系
        # Z坐标变换：NPZ z=0对应纸张表面(world z=0.011)，z>0表示抬起，z<0表示按压
        # 按压时夹紧到纸张表面-笔刷半径处(0.008)，让物理仿真产生接触力
        # 不夹紧的话IK会尝试到达地面以下(如z=-0.085)，求解失败导致机器人不动
        z_world = z[i] + 0.011  # 纸张表面偏移
        z_clamped = max(z_world, 0.008)  # 夹紧：按压时笔刷中心最低到0.008m
        target_pos = np.array([
            x[i] + sim.paper_offset[0],
            y[i] + sim.paper_offset[1],
            z_clamped
        ])

        # 移动到目标位置
        sim.move_to_position(target_pos, speed=speed, wait_time=0.0)

        # 每个点录制多帧（用于慢动作效果）
        for _ in range(frames_per_point):
            # 渲染机器人视图
            renderer.update_scene(sim.data, camera=camera_id)
            robot_frame = renderer.render()

            # 获取当前画布状态并叠加
            composite_frame = overlay_canvas(robot_frame, sim.paper_canvas,
                                            scale=canvas_scale,
                                            position=canvas_position)

            frames.append(composite_frame)

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

    # 保存画布图像
    canvas_path = output_video.replace('.mp4', '_canvas.png')
    cv2.imwrite(canvas_path, sim.paper_canvas)
    canvas_size = os.path.getsize(canvas_path) / 1024
    print(f"✅ 画布已保存: {canvas_path} ({canvas_size:.1f} KB)")

    # 清理
    sim.close()

    print("\n" + "=" * 70)
    print("复合视频生成完成!")
    print("=" * 70)


def overlay_canvas(robot_frame, canvas, scale=0.3, position="bottom_right"):
    """
    在机器人视图上叠加画布图像

    Args:
        robot_frame: 机器人视图帧 (H, W, 3) RGB
        canvas: 画布图像 (H, W) grayscale
        scale: 画布缩放比例（相对于视频宽度）
        position: 画布位置

    Returns:
        composite_frame: 复合帧 (H, W, 3) RGB
    """
    # 复制机器人帧
    composite = robot_frame.copy()

    # 将画布转换为彩色（灰度转RGB）
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # 计算画布尺寸
    video_height, video_width = robot_frame.shape[:2]
    canvas_width = int(video_width * scale)
    canvas_height = int(canvas_width * canvas.shape[0] / canvas.shape[1])

    # 缩放画布
    canvas_resized = cv2.resize(canvas_rgb, (canvas_width, canvas_height))

    # 添加边框
    border_thickness = 2
    cv2.rectangle(canvas_resized, (0, 0), (canvas_width-1, canvas_height-1),
                 (0, 0, 0), border_thickness)

    # 计算位置
    margin = 10
    if position == "bottom_right":
        y_offset = video_height - canvas_height - margin
        x_offset = video_width - canvas_width - margin
    elif position == "top_right":
        y_offset = margin
        x_offset = video_width - canvas_width - margin
    elif position == "bottom_left":
        y_offset = video_height - canvas_height - margin
        x_offset = margin
    elif position == "top_left":
        y_offset = margin
        x_offset = margin
    else:
        raise ValueError(f"Unknown position: {position}")

    # 叠加画布（使用alpha混合）
    alpha = 0.9  # 画布透明度
    composite[y_offset:y_offset+canvas_height, x_offset:x_offset+canvas_width] = \
        (alpha * canvas_resized + (1-alpha) * composite[y_offset:y_offset+canvas_height, x_offset:x_offset+canvas_width]).astype(np.uint8)

    return composite


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成复合视频（机器人 + 画布叠加）")
    parser.add_argument("npz_file", type=str, help="NPZ 轨迹文件路径")
    parser.add_argument("--output", "-o", type=str, default="outputs/composite_demo.mp4",
                        help="输出视频路径 (默认: outputs/composite_demo.mp4)")
    parser.add_argument("--speed", "-s", type=float, default=0.05,
                        help="运动速度 (m/s, 默认: 0.05)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="视频帧率 (默认: 30)")
    parser.add_argument("--canvas-scale", type=float, default=0.4,
                        help="画布缩放比例 (默认: 0.4)")
    parser.add_argument("--canvas-position", type=str, default="bottom_right",
                        choices=["bottom_right", "top_right", "bottom_left", "top_left"],
                        help="画布位置 (默认: bottom_right)")
    parser.add_argument("--slow", type=int, default=1,
                        help="慢动作倍数 - 每个点录制N帧 (默认: 1, 慢5倍则用5)")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.npz_file):
        print(f"❌ 错误: 文件不存在: {args.npz_file}")
        sys.exit(1)

    # 生成两个视角的复合视频
    cameras = [
        ("side_view", "_side"),
        ("front_view", "_front"),
    ]

    for camera_name, suffix in cameras:
        # 构造输出文件名
        base_name = args.output.replace('.mp4', '')
        output_path = f"{base_name}{suffix}.mp4"

        print(f"\n{'='*70}")
        print(f"生成 {camera_name} 视角视频")
        print(f"{'='*70}\n")

        generate_composite_video(args.npz_file, output_path, args.speed, args.fps,
                               args.canvas_scale, args.canvas_position, args.slow, camera_name)
