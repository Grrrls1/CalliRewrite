"""
RoboControl2.py - Franka calligraphy control with paper-normal alignment.

This version keeps the trajectory and safety behavior of RoboControl.py, but
commands an absolute end-effector orientation on every Cartesian move. The
configured tool Z axis is aligned perpendicular to the paper surface.
"""

import platform
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from .RoboControl import (
        FrankaCalligraphyController as _BaseController,
        _load_franka_config,
        _workspace_center_from_config,
        visualize_trajectory,
    )
except ImportError:
    from RoboControl import (
        FrankaCalligraphyController as _BaseController,
        _load_franka_config,
        _workspace_center_from_config,
        visualize_trajectory,
    )


def _unit_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError(f"{name} cannot be a zero vector")
    return vector / length


def _vector_from_config(
    section: dict, key: str, default: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    value = section.get(key)
    if isinstance(value, dict):
        try:
            return float(value["x"]), float(value["y"]), float(value["z"])
        except (KeyError, TypeError, ValueError):
            return default
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return default
    return default


def _orientation_from_surface(
    paper_normal: Sequence[float],
    paper_x_axis: Sequence[float],
    tool_z_points_toward_paper: bool,
) -> np.ndarray:
    """Return the tool rotation matrix expressed in the robot base frame."""
    normal = _unit_vector(paper_normal, "paper_normal")
    tool_z = -normal if tool_z_points_toward_paper else normal

    x_hint = _unit_vector(paper_x_axis, "paper_x_axis")
    tool_x = x_hint - np.dot(x_hint, tool_z) * tool_z
    if float(np.linalg.norm(tool_x)) < 1e-8:
        raise ValueError("paper_x_axis must not be parallel to paper_normal")
    tool_x = _unit_vector(tool_x, "projected paper_x_axis")
    tool_y = _unit_vector(np.cross(tool_z, tool_x), "tool_y")
    tool_x = _unit_vector(np.cross(tool_y, tool_z), "tool_x")

    return np.column_stack((tool_x, tool_y, tool_z))


def _quaternion_xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quaternion = np.asarray((qx, qy, qz, qw), dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def _zyx_angles_from_matrix(rotation: np.ndarray) -> Tuple[float, float, float]:
    """Return frankx Affine angular arguments (Z, Y, X) in radians."""
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-8:
        z_angle = np.arctan2(rotation[1, 0], rotation[0, 0])
        y_angle = np.arctan2(-rotation[2, 0], sy)
        x_angle = np.arctan2(rotation[2, 1], rotation[2, 2])
    else:
        z_angle = np.arctan2(-rotation[0, 1], rotation[1, 1])
        y_angle = np.arctan2(-rotation[2, 0], sy)
        x_angle = 0.0
    return float(z_angle), float(y_angle), float(x_angle)


class FrankaCalligraphyController(_BaseController):
    """
    Franka controller whose tool axis remains normal to the writing surface.

    By default, the paper normal points upward (+Z in the robot base frame)
    and the tool's local +Z axis points from the holder toward the pen tip.
    The target tool +Z is therefore set to -paper_normal.
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        default_speed: float = 0.05,
        default_acceleration: float = 0.05,
        workspace_center: Optional[Tuple[float, float, float]] = None,
        use_gripper: bool = False,
        config_path: Optional[str] = None,
        paper_normal: Optional[Tuple[float, float, float]] = None,
        paper_x_axis: Optional[Tuple[float, float, float]] = None,
        tool_z_points_toward_paper: Optional[bool] = None,
    ):
        cfg = _load_franka_config(config_path)
        orientation_cfg = cfg.get("paper_orientation", {})
        if not isinstance(orientation_cfg, dict):
            orientation_cfg = {}

        paper_normal = paper_normal or _vector_from_config(
            orientation_cfg, "normal", (0.0, 0.0, 1.0)
        )
        paper_x_axis = paper_x_axis or _vector_from_config(
            orientation_cfg, "x_axis", (1.0, 0.0, 0.0)
        )
        if tool_z_points_toward_paper is None:
            tool_z_points_toward_paper = bool(
                orientation_cfg.get("tool_z_points_toward_paper", True)
            )

        super().__init__(
            robot_ip=robot_ip,
            default_speed=default_speed,
            default_acceleration=default_acceleration,
            workspace_center=workspace_center,
            use_gripper=use_gripper,
            config_path=config_path,
        )

        self.paper_normal = _unit_vector(paper_normal, "paper_normal")
        self.paper_x_axis = _unit_vector(paper_x_axis, "paper_x_axis")
        self.tool_z_points_toward_paper = tool_z_points_toward_paper
        self.target_rotation = _orientation_from_surface(
            self.paper_normal, self.paper_x_axis, tool_z_points_toward_paper
        )
        self.target_quaternion_xyzw = _quaternion_xyzw_from_matrix(self.target_rotation)
        self.target_angles_zyx = _zyx_angles_from_matrix(self.target_rotation)
        self.surface_x_axis = self.target_rotation[:, 0]
        self.surface_y_axis = _unit_vector(
            np.cross(self.paper_normal, self.surface_x_axis), "surface_y_axis"
        )

        tool_z = self.target_rotation[:, 2]
        print("   Orientation constraint: tool Z perpendicular to paper")
        print(f"   Paper normal: {np.round(self.paper_normal, 6).tolist()}")
        print(f"   Target tool Z: {np.round(tool_z, 6).tolist()}")

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        speed: Optional[float] = None,
    ) -> bool:
        """Move to an absolute position while enforcing paper-normal orientation."""
        self._check_connected()
        x, y, z = self._validate_position(x, y, z)
        speed = speed if speed is not None else self.default_speed

        try:
            if self._library == "franky":
                from franky import Affine, CartesianMotion

                target = Affine([float(x), float(y), float(z)], self.target_quaternion_xyzw)
                motion = CartesianMotion(target)
                old_speed = self._robot.relative_dynamics_factor
                self._robot.relative_dynamics_factor = speed
                try:
                    self._robot.move(motion)
                finally:
                    self._robot.relative_dynamics_factor = old_speed
                return True

            if self._library == "frankx":
                import frankx

                rz, ry, rx = self.target_angles_zyx
                target = frankx.Affine(float(x), float(y), float(z), rz, ry, rx)
                motion = frankx.LinearMotion(target)
                if hasattr(frankx, "MotionData"):
                    self._robot.move(motion, frankx.MotionData(speed))
                else:
                    self._robot.set_dynamic_rel(speed)
                    self._robot.move(motion)
                return True

            raise RuntimeError("Unsupported Franka control library")
        except Exception as exc:
            print(f"Motion failed: {exc}")
            return False

    def paper_to_robot(
        self, x_offset: float, y_offset: float, height: float
    ) -> Tuple[float, float, float]:
        """
        Map paper-frame coordinates to an absolute robot-base position.

        `height` is positive away from the paper and negative into the paper.
        """
        position = (
            self.workspace_center
            + float(x_offset) * self.surface_x_axis
            + float(y_offset) * self.surface_y_axis
            + float(height) * self.paper_normal
        )
        return float(position[0]), float(position[1]), float(position[2])

    def move_surface(
        self,
        x_offset: float,
        y_offset: float,
        height: float,
        speed: Optional[float] = None,
    ) -> bool:
        """Move using paper-frame coordinates while enforcing perpendicular pose."""
        return self.move_cartesian(*self.paper_to_robot(x_offset, y_offset, height), speed=speed)

    def execute_surface_trajectory(
        self,
        x_points: np.ndarray,
        y_points: np.ndarray,
        z_points: np.ndarray,
        speed: float = 0.05,
        wait_time: float = 0.01,
        center_xy: bool = True,
    ) -> bool:
        """Execute relative paper-frame points as absolute perpendicular poses."""
        x_points = np.asarray(x_points, dtype=float)
        y_points = np.asarray(y_points, dtype=float)
        z_points = np.asarray(z_points, dtype=float)
        if center_xy:
            x_points = x_points - (x_points.min() + x_points.max()) / 2.0
            y_points = y_points - (y_points.min() + y_points.max()) / 2.0

        points = np.asarray(
            [self.paper_to_robot(x, y, z) for x, y, z in zip(x_points, y_points, z_points)]
        )
        return self.execute_trajectory(
            points[:, 0], points[:, 1], points[:, 2], speed=speed, wait_time=wait_time
        )

    def move_to_home(self) -> bool:
        """Move 10 cm away from the paper along its measured normal."""
        print("Moving to home position above the paper surface...")
        return self.move_surface(0.0, 0.0, 0.10, speed=0.15)


def Control(npz_path: str, robot_ip: Optional[str] = None, speed: float = 0.05) -> bool:
    """Execute an NPZ trajectory with the end-effector normal to the paper."""
    print("\n" + "=" * 70)
    print("FRANKA CALLIGRAPHY CONTROL 2 - PAPER-NORMAL END EFFECTOR")
    print("=" * 70)
    print(f"Python environment: {platform.architecture()}")
    print(f"Platform: {platform.system()}")

    if not Path(npz_path).exists():
        print(f"Error: File not found: {npz_path}")
        return False

    data = np.load(npz_path)
    try:
        x = np.asarray(data["pos_3d_x"], dtype=float)
        y = np.asarray(data["pos_3d_y"], dtype=float)
        z = np.asarray(data["pos_3d_z"], dtype=float)
    except KeyError as exc:
        print(f"Error: missing trajectory field: {exc}")
        return False

    print(f"Loading trajectory: {npz_path}")
    print(f"   Points: {len(x)}")
    print(f"   X range: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"   Y range: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"   Z range: [{z.min():.4f}, {z.max():.4f}] m")

    cfg = _load_franka_config()
    workspace_center = _workspace_center_from_config(cfg) or (0.4, 0.0, 0.3)
    if robot_ip is None:
        robot_ip = str(cfg.get("robot_ip", "172.16.0.2"))

    controller = FrankaCalligraphyController(
        robot_ip=robot_ip,
        default_speed=speed,
        workspace_center=workspace_center,
        use_gripper=False,
    )
    if not controller.connect():
        print("Failed to connect to robot")
        return False

    try:
        print("\nMoving above the writing surface and applying target orientation...")
        if not controller.move_to_home():
            return False
        time.sleep(1)

        print("\nStarting calligraphy execution...")
        success = controller.execute_surface_trajectory(x, y, z, speed=speed)
        if success:
            print("\nReturning above the writing surface...")
            controller.move_to_home()
        return success
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False
    finally:
        controller.disconnect()
        print("\n" + "=" * 70)
        print("Session ended")
        print("=" * 70 + "\n")


def load_npz_trajectory(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one NPZ trajectory and validate required fields."""
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path)
    try:
        x = np.asarray(data["pos_3d_x"], dtype=float)
        y = np.asarray(data["pos_3d_y"], dtype=float)
        z = np.asarray(data["pos_3d_z"], dtype=float)
    except KeyError as exc:
        raise KeyError(f"missing trajectory field in {npz_path}: {exc}") from exc

    if len(x) == 0 or len(x) != len(y) or len(y) != len(z):
        raise ValueError(f"invalid trajectory lengths in {npz_path}")
    return x, y, z


def execute_npz_on_controller(
    controller: FrankaCalligraphyController,
    npz_path: str,
    speed: float,
    x_offset: float = 0.0,
    center_xy: bool = True,
) -> bool:
    """Execute one NPZ on an already-connected controller."""
    try:
        x, y, z = load_npz_trajectory(npz_path)
    except Exception as exc:
        print(f"Error loading {npz_path}: {exc}")
        return False

    print(f"\nLoading trajectory: {npz_path}")
    print(f"   Points: {len(x)}")
    print(f"   X range: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"   Y range: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"   Z range: [{z.min():.4f}, {z.max():.4f}] m")
    if x_offset:
        print(f"   Sequence X offset: {x_offset * 1000:.1f} mm")

    if center_xy:
        x = x - (x.min() + x.max()) / 2.0
        y = y - (y.min() + y.max()) / 2.0
    x = x + float(x_offset)

    points = np.asarray(
        [controller.paper_to_robot(px, py, pz) for px, py, pz in zip(x, y, z)]
    )
    return controller.execute_trajectory(
        points[:, 0], points[:, 1], points[:, 2], speed=speed, wait_time=0.01
    )


def ControlMany(
    npz_paths: Sequence[str],
    robot_ip: Optional[str] = None,
    speed: float = 0.05,
    spacing_m: float = 0.0,
    home_between: bool = False,
) -> bool:
    """Execute multiple NPZ trajectories in one robot session."""
    if not npz_paths:
        print("Error: no NPZ files provided")
        return False

    print("\n" + "=" * 70)
    print("FRANKA CALLIGRAPHY CONTROL 2 - MULTI NPZ SEQUENCE")
    print("=" * 70)
    print(f"Python environment: {platform.architecture()}")
    print(f"Platform: {platform.system()}")
    print(f"Files: {len(npz_paths)}")
    print(f"Speed: {speed:.3f}")
    if spacing_m:
        print(f"Spacing: {spacing_m * 1000:.1f} mm")

    cfg = _load_franka_config()
    workspace_center = _workspace_center_from_config(cfg) or (0.4, 0.0, 0.3)
    if robot_ip is None:
        robot_ip = str(cfg.get("robot_ip", "172.16.0.2"))

    controller = FrankaCalligraphyController(
        robot_ip=robot_ip,
        default_speed=speed,
        workspace_center=workspace_center,
        use_gripper=False,
    )
    if not controller.connect():
        print("Failed to connect to robot")
        return False

    try:
        print("\nMoving above the writing surface and applying target orientation...")
        if not controller.move_to_home():
            return False
        time.sleep(1)

        for index, npz_path in enumerate(npz_paths):
            print("\n" + "-" * 70)
            print(f"Sequence {index + 1}/{len(npz_paths)}")
            x_offset = index * float(spacing_m)
            if not execute_npz_on_controller(controller, npz_path, speed, x_offset=x_offset):
                return False
            if home_between and index < len(npz_paths) - 1:
                print("\nMoving above the writing surface before next NPZ...")
                if not controller.move_to_home():
                    return False

        print("\nReturning above the writing surface...")
        controller.move_to_home()
        return True
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False
    finally:
        controller.disconnect()
        print("\n" + "=" * 70)
        print("Session ended")
        print("=" * 70 + "\n")


def test_connection(robot_ip: Optional[str] = None) -> bool:
    """Test connection and display the configured perpendicular orientation."""
    cfg = _load_franka_config()
    robot_ip = robot_ip or str(cfg.get("robot_ip", "172.16.0.2"))
    controller = FrankaCalligraphyController(robot_ip=robot_ip)
    if not controller.connect():
        return False
    controller.disconnect()
    return True


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Execute one or more NPZ calligraphy trajectories with paper-normal orientation.",
        epilog=(
            "Legacy single-file form is still supported:\n"
            "  python RoboControl2.py file.npz 172.16.0.2 0.05\n\n"
            "Multi-file form:\n"
            "  python RoboControl2.py a.npz b.npz c.npz --robot-ip 172.16.0.2 --speed 0.05"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("args", nargs="*", help="NPZ file(s), or legacy: <npz_file> [robot_ip] [speed]")
    parser.add_argument("--robot-ip", dest="robot_ip", default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument(
        "--spacing-mm",
        type=float,
        default=0.0,
        help="Offset each later NPZ along paper X by this many millimeters.",
    )
    parser.add_argument(
        "--home-between",
        action="store_true",
        help="Move above the paper between NPZ files instead of writing continuously.",
    )
    parser.add_argument("--test", action="store_true", help="Only test robot connection.")
    parsed = parser.parse_args()

    if parsed.test:
        legacy_ip = parsed.args[0] if parsed.args else None
        ip = parsed.robot_ip or legacy_ip
        sys.exit(0 if test_connection(ip) else 1)

    if not parsed.args:
        parser.print_help()
        sys.exit(1)

    npz_paths = list(parsed.args)
    robot_ip = parsed.robot_ip
    motion_speed = parsed.speed if parsed.speed is not None else 0.05

    # Backward compatibility: python RoboControl2.py file.npz [robot_ip] [speed]
    if len(npz_paths) >= 2 and not npz_paths[1].lower().endswith(".npz"):
        robot_ip = robot_ip or npz_paths[1]
        if len(npz_paths) >= 3:
            motion_speed = parsed.speed if parsed.speed is not None else float(npz_paths[2])
        npz_paths = [npz_paths[0]]

    success = ControlMany(
        npz_paths,
        robot_ip=robot_ip,
        speed=motion_speed,
        spacing_m=parsed.spacing_mm / 1000.0,
        home_between=parsed.home_between,
    )
    sys.exit(0 if success else 1)
