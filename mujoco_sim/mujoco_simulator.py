#!/usr/bin/env python3
"""
MuJoCo 仿真控制器 - CalliRewrite 系统

支持功能:
1. 加载 Franka Panda 机器人模型
2. 执行书法轨迹 (从 .npz 文件)
3. 可视化笔迹和机器人运动
4. 记录仿真视频
5. 碰撞检测和力反馈

作者: CalliRewrite Team
"""

import numpy as np
import mujoco
import mujoco.viewer
import cv2
import os
from pathlib import Path
from typing import Optional, Tuple, List
import time
import matplotlib.pyplot as plt


class FrankaCalligraphySimulator:
    """Franka Panda 书法仿真器"""

    def __init__(
        self,
        model_path: str = None,
        render_mode: str = "human",  # "human", "offscreen", "rgb_array"
        camera_distance: float = 1.5,
        camera_azimuth: float = 45,
        camera_elevation: float = -30,
    ):
        """
        初始化仿真器

        Args:
            model_path: MuJoCo XML 模型路径
            render_mode: 渲染模式
            camera_distance: 相机距离
            camera_azimuth: 相机方位角
            camera_elevation: 相机仰角
        """
        if model_path is None:
            # 默认使用真实的FR3v2模型
            current_dir = Path(__file__).parent
            model_path = current_dir / "models" / "franka_fr3v2_calligraphy.xml"

        self.model_path = str(model_path)
        self.render_mode = render_mode

        # 加载模型
        print(f"Loading MuJoCo model from: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # 相机设置
        self.camera_distance = camera_distance
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation

        # 控制参数
        self.dt = self.model.opt.timestep
        self.control_freq = 500  # Hz
        self.control_dt = 1.0 / self.control_freq

        # 笔迹记录
        self.ink_traces = []  # [(x, y, z, contact), ...]
        self.current_pen_depth_m = 0.0
        self.current_traj_z_m = 0.05
        self.stroke_radius_min_px = 7
        self.stroke_radius_max_px = 12
        self.stroke_depth_ref_m = 0.003  # 4mm depth maps to max radius
        self.contact_force_threshold = 0.5
        self.traj_contact_z_threshold = -1e-4
        self.paper_canvas = None
        self.paper_size = (0.3, 0.42)  # A3 纸 (米)
        self.canvas_resolution = (1200, 1680)  # 像素

        # 初始化画布
        self._init_canvas()

        # Viewer
        self.viewer = None

        print("✅ MuJoCo simulator initialized")
        self._print_model_info()

    def _print_model_info(self):
        """打印模型信息"""
        print("\n" + "=" * 60)
        print("MuJoCo Model Information")
        print("=" * 60)
        print(f"DoF: {self.model.nv}")
        print(f"Actuators: {self.model.nu}")
        print(f"Bodies: {self.model.nbody}")
        print(f"Joints: {self.model.njnt}")
        print(f"Timestep: {self.dt * 1000:.2f} ms")
        print(f"Control frequency: {self.control_freq} Hz")
        print("=" * 60 + "\n")

    def _init_canvas(self):
        """初始化画布"""
        self.paper_canvas = np.ones(self.canvas_resolution, dtype=np.uint8) * 255
        # 纸张中心在世界坐标 (0.5, 0, 0.01)，尺寸为 0.3 x 0.42
        # 所以纸张左下角在世界坐标 (0.5 - 0.15, 0 - 0.21) = (0.35, -0.21)
        self.paper_offset = np.array([0.35, -0.21])  # 纸张左下角的世界坐标

    def reset(self, qpos: Optional[np.ndarray] = None):
        """
        重置仿真

        Args:
            qpos: 初始关节位置 (可选)
        """
        mujoco.mj_resetData(self.model, self.data)

        if qpos is not None:
            self.data.qpos[:] = qpos

        # 重置画布
        self._init_canvas()
        self.ink_traces = []
        self.current_pen_depth_m = 0.0
        self.current_traj_z_m = 0.05

        mujoco.mj_forward(self.model, self.data)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器位姿

        Returns:
            position: (3,) 位置 [x, y, z]
            quaternion: (4,) 四元数 [w, x, y, z]
        """
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        position = self.data.site_xpos[ee_site_id].copy()
        quaternion = self.data.site_xmat[ee_site_id].reshape(3, 3)
        return position, quaternion

    def get_brush_contact(self) -> Tuple[np.ndarray, float]:
        """
        获取笔刷接触信息

        Returns:
            position: (3,) 笔刷位置
            force: 接触力大小
        """
        brush_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "brush_contact"
        )
        position = self.data.site_xpos[brush_site_id].copy()

        # 检测接触力
        touch_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "brush_touch"
        )
        touch_value = self.data.sensordata[touch_sensor_id]

        return position, touch_value

    # Home 位姿（无碰撞的标准姿态，用于零空间优化）
    HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    def inverse_kinematics(
        self, target_pos: np.ndarray, max_iter: int = 200, tol: float = 1e-3
    ) -> bool:
        """
        逆运动学求解 - 带零空间优化，利用冗余自由度向 home 位姿靠拢以规避自碰撞

        FR3v2 是 7DOF 机器人，求解 3D 位置只需 3 个约束，剩余 4 个自由度
        构成零空间，可用于在不影响末端位置的情况下优化关节构型。

        Args:
            target_pos: (3,) 目标位置
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            success: 是否成功
        """
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        # 零空间优化增益：将关节拉向 home 位姿
        # 注意：此值必须远小于主任务增益(0.5)，否则会妨碍末端位置收敛
        null_gain = 0.01

        for _ in range(max_iter):
            # 当前末端位置
            current_pos = self.data.site_xpos[ee_site_id]

            # 计算误差
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                return True

            # 计算雅可比矩阵（只取前7个关节自由度）
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, ee_site_id)
            J = jacp[:, :7]

            # 伪逆求解主任务
            J_pinv = np.linalg.pinv(J)
            dq_primary = J_pinv @ error * 0.5

            # 零空间项：把关节角拉向 home 位姿，降低自碰撞风险
            # null_proj = (I - J⁺J)，零空间投影矩阵
            null_proj = np.eye(7) - J_pinv @ J
            q_err_to_home = self.HOME_QPOS - self.data.qpos[:7]
            dq_null = null_proj @ (q_err_to_home * null_gain)

            # 合并更新
            dq = dq_primary + dq_null

            # 更新关节角度
            self.data.qpos[:7] += dq

            # 夹紧到关节限位
            for j in range(7):
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"fr3v2_joint{j+1}"
                )
                qmin = self.model.jnt_range[joint_id, 0]
                qmax = self.model.jnt_range[joint_id, 1]
                self.data.qpos[j] = np.clip(self.data.qpos[j], qmin, qmax)

            mujoco.mj_forward(self.model, self.data)

        return False

    def move_to_position(
        self, target_pos: np.ndarray, speed: float = 0.05, wait_time: float = 0.01
    ):
        """
        移动到目标位置

        Args:
            target_pos: (3,) 目标位置
            speed: 移动速度 (m/s)
            wait_time: 到达后等待时间 (秒)
        """
        # IK 求解
        success = self.inverse_kinematics(target_pos)
        if not success:
            print(f"⚠️  IK failed for target: {target_pos}")
            return

        # 执行运动 (简化版：直接设置目标，让 PD 控制器跟踪)
        target_qpos = self.data.qpos[:7].copy()

        # 平滑插值
        current_ee_pos, _ = self.get_ee_pose()
        distance = np.linalg.norm(target_pos - current_ee_pos)
        duration = distance / speed
        steps = int(duration / self.control_dt)

        if steps == 0:
            steps = 1

        for i in range(steps):
            # 设置控制目标 (位置控制)
            self.data.ctrl[:7] = target_qpos

            # 步进仿真
            mujoco.mj_step(self.model, self.data)

            # 记录笔迹
            self._update_ink_trace()

            # 渲染
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(self.dt)

        # 等待稳定
        if wait_time > 0:
            wait_steps = int(wait_time / self.dt)
            for _ in range(wait_steps):
                mujoco.mj_step(self.model, self.data)
                if self.viewer is not None:
                    self.viewer.sync()
                    time.sleep(self.dt)

    def _update_ink_trace(self):
        """更新笔迹记录"""
        brush_pos, contact_force = self.get_brush_contact()
        is_contact = contact_force > self.contact_force_threshold
        self.ink_traces.append((*brush_pos, is_contact))
        # print(
        #     f"Brush position: {brush_pos}, Contact force: {contact_force:.2f}, "
        #     f"traj_z: {self.current_traj_z_m:.4f}, Contact: {is_contact}"
        # )

        # 如果接触，绘制到画布；否则重置连线起点，避免跨笔画连线
        if is_contact:
            self._draw_on_canvas(brush_pos)
        else:
            if hasattr(self, '_last_canvas_pos'):
                delattr(self, '_last_canvas_pos')

    def _compute_stroke_radius_px(self) -> int:
        """根据当前落笔深度计算笔迹半径（像素）。"""
        depth = float(max(0.0, self.current_pen_depth_m))
        depth_ref = max(float(self.stroke_depth_ref_m), 1e-6)
        t = min(depth / depth_ref, 1.0)
        radius = self.stroke_radius_min_px + t * (
            self.stroke_radius_max_px - self.stroke_radius_min_px
        )
        return int(round(radius))

    def _draw_on_canvas(self, pos: np.ndarray):
        """在画布上绘制"""
        # 转换到画布坐标（相对于纸张左下角）
        paper_x = pos[0] - self.paper_offset[0]
        paper_y = pos[1] - self.paper_offset[1]

        # 归一化到 [0, 1] - 直接除以纸张尺寸，不需要中心对齐
        u = paper_x / self.paper_size[0]
        v = paper_y / self.paper_size[1]

        # 转换到像素坐标
        px = int(u * self.canvas_resolution[0])
        py = int(v * self.canvas_resolution[1])

        # 边界检查
        if 0 <= px < self.canvas_resolution[0] and 0 <= py < self.canvas_resolution[1]:
            radius_px = self._compute_stroke_radius_px()
            line_thickness = max(2, radius_px * 2)
            cv2.circle(self.paper_canvas, (px, py), radius_px, 0, -1)

            # 如果有上一个点，画线连接以确保连续性
            if hasattr(self, '_last_canvas_pos'):
                last_px, last_py = self._last_canvas_pos
                cv2.line(
                    self.paper_canvas,
                    (last_px, last_py),
                    (px, py),
                    0,
                    line_thickness,
                )

            self._last_canvas_pos = (px, py)
        else:
            # 如果超出边界，重置上一个位置
            if hasattr(self, '_last_canvas_pos'):
                delattr(self, '_last_canvas_pos')

    def execute_trajectory(
        self, npz_path: str, speed: float = 0.05, render: bool = True
    ):
        """
        执行书法轨迹

        Args:
            npz_path: NPZ 文件路径
            speed: 移动速度 (m/s)
            render: 是否可视化
        """
        print(f"\n{'=' * 60}")
        print(f"Executing trajectory from: {npz_path}")
        print(f"{'=' * 60}\n")

        # 加载轨迹
        data = np.load(npz_path)
        x = data["pos_3d_x"]
        y = data["pos_3d_y"]
        z = data["pos_3d_z"]

        num_points = len(x)
        print(f"Total control points: {num_points}")

        # 重置仿真
        self.reset()

        # 启动 Viewer (如果需要)
        if render and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # 逐点执行
        start_time = time.time()
        for i in range(num_points):
            self.current_traj_z_m = float(z[i])
            self.current_pen_depth_m = max(0.0, float(-z[i]))

            # Z坐标变换：NPZ z=0对应纸张表面，z<0表示按压，z>0表示抬起
            # 夹紧防止IK尝试到达地面以下导致求解失败
            z_world = z[i] + 0.011
            z_clamped = max(z_world, 0.007)  # 笔刷中心最低到0.007m
            target_pos = np.array([
                x[i] + self.paper_offset[0],
                y[i] + self.paper_offset[1],
                z_clamped
            ])

            if i % 10 == 0:
                print(
                    f"Progress: {i}/{num_points} ({i/num_points*100:.1f}%) - "
                    f"NPZ: [{x[i]:.4f}, {y[i]:.4f}, {z[i]:.4f}] → "
                    f"World: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]"
                )

            self.move_to_position(target_pos, speed=speed)

        elapsed = time.time() - start_time
        print(f"\n✅ Trajectory execution completed in {elapsed:.2f}s")
        print(f"Average speed: {num_points / elapsed:.1f} points/s")

        # 显示结果
        self._show_results()

    def _show_results(self):
        """显示执行结果"""
        print("\n" + "=" * 60)
        print("Execution Results")
        print("=" * 60)
        print(f"Total ink points: {len(self.ink_traces)}")
        contact_points = sum(1 for _, _, _, c in self.ink_traces if c)
        print(f"Contact points: {contact_points}")
        print(f"Contact ratio: {contact_points / len(self.ink_traces) * 100:.1f}%")
        print("=" * 60)

        # 保存画布
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        canvas_path = output_dir / "calligraphy_result.png"
        cv2.imwrite(str(canvas_path), self.paper_canvas)
        print(f"\n📄 Canvas saved to: {canvas_path}")

        # 可视化笔迹 3D
        self._plot_trajectory_3d(output_dir / "trajectory_3d.png")

    def _plot_trajectory_3d(self, save_path: str):
        """绘制 3D 轨迹图"""
        if not self.ink_traces:
            return

        traces = np.array(self.ink_traces)
        x, y, z, contact = traces[:, 0], traces[:, 1], traces[:, 2], traces[:, 3]

        fig = plt.figure(figsize=(15, 5))

        # 3D 轨迹
        ax1 = fig.add_subplot(131, projection="3d")
        contact_idx = contact > 0.5
        ax1.plot(x, y, z, "b-", alpha=0.3, linewidth=0.5, label="All points")
        ax1.scatter(
            x[contact_idx],
            y[contact_idx],
            z[contact_idx],
            c="red",
            s=1,
            label="Contact",
        )
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("3D Trajectory")
        ax1.legend()

        # 俯视图
        ax2 = fig.add_subplot(132)
        ax2.plot(x, y, "b-", alpha=0.3, linewidth=0.5)
        ax2.scatter(x[contact_idx], y[contact_idx], c="red", s=1)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("Top View (X-Y)")
        ax2.axis("equal")
        ax2.grid(True, alpha=0.3)

        # Z 高度变化
        ax3 = fig.add_subplot(133)
        ax3.plot(z, "b-", linewidth=1)
        ax3.fill_between(
            range(len(z)), z, -0.1, where=contact > 0.5, alpha=0.3, color="red"
        )
        ax3.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax3.set_xlabel("Point Index")
        ax3.set_ylabel("Z (m)")
        ax3.set_title("Z Height Profile")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 3D trajectory plot saved to: {save_path}")
        plt.close()

    def record_video(
        self, npz_path: str, output_path: str, speed: float = 0.05, fps: int = 30
    ):
        """
        录制仿真视频

        Args:
            npz_path: NPZ 轨迹文件
            output_path: 输出视频路径
            speed: 移动速度
            fps: 视频帧率
        """
        print(f"\n{'=' * 60}")
        print(f"Recording video to: {output_path}")
        print(f"{'=' * 60}\n")

        # 设置离屏渲染
        self.render_mode = "rgb_array"

        # 创建 offscreen 渲染器
        renderer = mujoco.Renderer(self.model, height=720, width=1280)

        # 加载轨迹
        data_npz = np.load(npz_path)
        x = data_npz["pos_3d_x"]
        y = data_npz["pos_3d_y"]
        z = data_npz["pos_3d_z"]

        # 重置
        self.reset()

        # 视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

        frame_interval = int(self.control_freq / fps)
        frame_count = 0

        # 执行并录制
        for i in range(len(x)):
            self.current_traj_z_m = float(z[i])
            self.current_pen_depth_m = max(0.0, float(-z[i]))
            target_pos = np.array([x[i], y[i], z[i]])

            # IK 求解
            self.inverse_kinematics(target_pos)
            self.data.ctrl[:7] = self.data.qpos[:7]

            # 步进仿真
            mujoco.mj_step(self.model, self.data)
            self._update_ink_trace()

            # 录制帧
            if frame_count % frame_interval == 0:
                renderer.update_scene(self.data)
                pixels = renderer.render()
                # MuJoCo 返回 RGB，OpenCV 需要 BGR
                frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            frame_count += 1

            if i % 50 == 0:
                print(f"Recording progress: {i}/{len(x)} ({i/len(x)*100:.1f}%)")

        video_writer.release()
        print(f"\n✅ Video saved to: {output_path}")

    def close(self):
        """关闭仿真器"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def main():
    """主函数 - 示例用法"""
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo Calligraphy Simulator")
    parser.add_argument(
        "npz_file", type=str, help="Path to NPZ trajectory file", nargs="?"
    )
    parser.add_argument("--speed", type=float, default=0.05, help="Movement speed (m/s)")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--record", type=str, help="Record video to path")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to MuJoCo XML model (default: models/franka_panda.xml)",
    )

    args = parser.parse_args()

    # 创建仿真器
    sim = FrankaCalligraphySimulator(model_path=args.model)

    if args.npz_file:
        if args.record:
            # 录制视频
            sim.record_video(args.npz_file, args.record, speed=args.speed)
        else:
            # 执行轨迹
            sim.execute_trajectory(
                args.npz_file, speed=args.speed, render=not args.no_render
            )

        # 保持窗口打开
        if not args.no_render and sim.viewer is not None:
            print("\nPress Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nExiting...")
    else:
        # 交互模式
        print("\n" + "=" * 60)
        print("MuJoCo Calligraphy Simulator - Interactive Mode")
        print("=" * 60)
        print("\nUsage:")
        print("  python mujoco_simulator.py <npz_file> [--speed 0.05] [--record video.mp4]")
        print("\nExample:")
        print(
            "  python mujoco_simulator.py ../callibrate/examples/simple_line.npz --speed 0.1"
        )
        print("\nNo NPZ file provided. Starting in test mode...")

        # 测试模式：简单运动
        sim.reset()
        sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

        print("\nMoving to test positions...")
        test_positions = [
            [0.5, 0.0, 0.3],  # 上方
            [0.5, 0.0, 0.01],  # 接触纸面
            [0.6, 0.0, 0.01],  # 画线
            [0.6, 0.0, 0.3],  # 抬起
        ]

        for i, pos in enumerate(test_positions):
            print(f"\nTarget {i+1}: {pos}")
            sim.move_to_position(np.array(pos), speed=0.1, wait_time=0.5)

        print("\nTest completed. Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")

    sim.close()


if __name__ == "__main__":
    main()
