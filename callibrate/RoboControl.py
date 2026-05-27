"""
RoboControl.py - Franka Emika Robot Control for CalliRewrite Project

This module provides trajectory control for Franka robots (Panda, FR3, etc.)
Compatible with both franky and frankx libraries.

Author: Generated for CalliRewrite × Franka Integration
License: MIT
"""

import numpy as np
import time
import platform
from pathlib import Path
from typing import Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def _load_franka_config(config_path: Optional[str] = None) -> dict:
    """Load franka_config.yaml. Return {} when unavailable."""
    if yaml is None:
        return {}

    path = Path(config_path).expanduser().resolve() if config_path else Path(__file__).resolve().parent / "franka_config.yaml"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _workspace_center_from_config(cfg: dict) -> Optional[Tuple[float, float, float]]:
    wc = cfg.get("workspace_center")
    if not isinstance(wc, dict):
        return None
    try:
        return float(wc["x"]), float(wc["y"]), float(wc["z"])
    except (TypeError, ValueError, KeyError):
        return None


class FrankaCalligraphyController:
    """
    Franka robot controller for calligraphy tasks.

    Supports:
    - Cartesian space trajectory execution
    - Pen pressure control via Z-axis positioning
    - Safe motion limits and collision avoidance
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        default_speed: float = 0.05,
        default_acceleration: float = 0.05,
        workspace_center: Optional[Tuple[float, float, float]] = None,
        use_gripper: bool = False,
        config_path: Optional[str] = None,
    ):
        """
        Initialize Franka calligraphy controller.

        Args:
            robot_ip: IP address of Franka robot
            default_speed: Motion speed factor (0.0-1.0)
            default_acceleration: Motion acceleration factor (0.0-1.0)
            workspace_center: (x, y, z) in meters - center of writing surface.
                If None, read from franka_config.yaml.
            use_gripper: Whether to use gripper (set False for pen holder)
            config_path: Optional path to franka_config.yaml
        """
        cfg = _load_franka_config(config_path)
        if workspace_center is None:
            workspace_center = _workspace_center_from_config(cfg) or (0.4, 0.0, 0.3)

        self.robot_ip = robot_ip
        self.default_speed = default_speed
        self.default_acceleration = default_acceleration
        self.workspace_center = np.array(workspace_center)
        self.use_gripper = use_gripper

        self._robot = None
        self._gripper = None
        self._library = None
        self._connected = False

        # Safety limits (relative to workspace_center)
        self.max_xy_range = 0.15  # ±15cm from center in X/Y
        self.max_z_range = 0.10   # ±10cm from center in Z

        print(f"🖌️  Franka Calligraphy Controller Initialized")
        print(f"   Robot IP: {robot_ip}")
        print(f"   Workspace Center: {workspace_center}")

    def connect(self) -> bool:
        """
        Connect to Franka robot.

        Returns:
            True if connection successful
        """
        if self._connected:
            print("⚠️  Already connected")
            return True

        print(f"📡 Connecting to Franka robot at {self.robot_ip}...")

        # Try franky first (recommended for newer systems)
        try:
            import franky

            self._robot = franky.Robot(self.robot_ip)
            self._library = "franky"
            print(f"✅ Connected using franky library")

            # Set motion parameters
            self._robot.relative_dynamics_factor = self.default_speed
            self._connected = True
            return True

        except ImportError:
            print("   franky not found, trying frankx...")
        except Exception as e:
            print(f"   franky connection failed: {e}")

        # Try frankx as fallback
        try:
            import frankx

            self._robot = frankx.Robot(self.robot_ip)
            self._robot.set_default_behavior()
            self._library = "frankx"
            print(f"✅ Connected using frankx library")

            # Initialize gripper if needed
            if self.use_gripper:
                try:
                    self._gripper = frankx.Gripper(self.robot_ip)
                    self._gripper.homing()
                    print("   Gripper initialized")
                except Exception as e:
                    print(f"   Warning: Gripper init failed: {e}")

            self._connected = True
            return True

        except ImportError:
            print("❌ Error: Neither franky nor frankx library found!")
            print("   Install with: pip install frankx")
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from robot."""
        if self._robot is not None:
            print("🔌 Disconnecting from robot...")
            self._robot = None
            self._gripper = None
            self._connected = False
            print("✅ Disconnected")

    def _check_connected(self):
        """Raise error if not connected."""
        if not self._connected:
            raise RuntimeError("Robot not connected! Call connect() first.")

    def _validate_position(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Validate and clamp position to safe workspace limits.

        Args:
            x, y, z: Absolute positions in meters

        Returns:
            Clamped (x, y, z) positions
        """
        # Calculate relative position from workspace center
        rel_x = x - self.workspace_center[0]
        rel_y = y - self.workspace_center[1]
        rel_z = z - self.workspace_center[2]

        # Clamp to safe range
        rel_x = np.clip(rel_x, -self.max_xy_range, self.max_xy_range)
        rel_y = np.clip(rel_y, -self.max_xy_range, self.max_xy_range)
        rel_z = np.clip(rel_z, -self.max_z_range, self.max_z_range)

        # Convert back to absolute
        x_safe = self.workspace_center[0] + rel_x
        y_safe = self.workspace_center[1] + rel_y
        z_safe = self.workspace_center[2] + rel_z

        return x_safe, y_safe, z_safe

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        speed: Optional[float] = None,
    ) -> bool:
        """
        Move end-effector to Cartesian position.

        Args:
            x, y, z: Target position in meters (workspace coordinates)
            speed: Optional speed factor (0.0-1.0), uses default if None

        Returns:
            True if motion completed successfully
        """
        self._check_connected()

        # Validate and clamp position
        x, y, z = self._validate_position(x, y, z)

        # Use default speed if not specified
        speed = speed if speed is not None else self.default_speed

        try:
            if self._library == "franky":
                from franky import CartesianMotion, Affine, ReferenceType

                # 读取当前末端姿态，保留朝向，仅替换位置
                # 直接用 Affine 的 relative 方式：计算当前位置到目标的 delta
                curr = self._robot.state.O_T_EE.translation
                dx, dy, dz = x - float(curr[0]), y - float(curr[1]), z - float(curr[2])
                target = Affine([dx, dy, dz])
                motion = CartesianMotion(target, ReferenceType.Relative)

                # Set speed
                old_speed = self._robot.relative_dynamics_factor
                self._robot.relative_dynamics_factor = speed

                # Execute motion
                self._robot.move(motion)

                # Restore speed
                self._robot.relative_dynamics_factor = old_speed
                return True

            elif self._library == "frankx":
                import frankx

                # Create Cartesian motion
                motion = frankx.CartesianMotion(
                    frankx.Affine(x, y, z)
                )
                motion.with_dynamic_rel(speed, self.default_acceleration)

                # Execute motion (blocking)
                self._robot.move(motion, asynchronous=False)
                return True

        except Exception as e:
            print(f"❌ Motion failed: {e}")
            return False

    def execute_trajectory(
        self,
        x_points: np.ndarray,
        y_points: np.ndarray,
        z_points: np.ndarray,
        speed: float = 0.05,
        wait_time: float = 0.01,
    ) -> bool:
        """
        Execute a calligraphy trajectory.

        Args:
            x_points: X positions in meters
            y_points: Y positions in meters
            z_points: Z positions in meters (height/pressure)
            speed: Motion speed factor (0.0-1.0)
            wait_time: Time to wait between points (seconds)

        Returns:
            True if trajectory completed successfully
        """
        self._check_connected()

        n_points = len(x_points)
        print(f"🖊️  Executing trajectory with {n_points} points...")
        print(f"   Speed: {speed:.2f}, Wait time: {wait_time:.3f}s")

        start_time = time.time()

        for i in range(n_points):
            # Move to point
            success = self.move_cartesian(
                x_points[i],
                y_points[i],
                z_points[i],
                speed=speed,
            )

            if not success:
                print(f"❌ Trajectory failed at point {i}/{n_points}")
                return False

            # Wait between points
            if wait_time > 0:
                time.sleep(wait_time)

            # Progress indicator every 10%
            if (i + 1) % max(1, n_points // 10) == 0:
                progress = (i + 1) / n_points * 100
                print(f"   Progress: {progress:.0f}% ({i+1}/{n_points})")

        elapsed = time.time() - start_time
        print(f"✅ Trajectory completed in {elapsed:.1f}s")
        return True

    def move_to_home(self) -> bool:
        """
        Move robot to safe home position above workspace.

        Returns:
            True if motion successful
        """
        print("🏠 Moving to home position...")
        home_x, home_y, home_z = self.workspace_center
        home_z += 0.1  # 10cm above workspace center

        return self.move_cartesian(home_x, home_y, home_z, speed=0.15)


def Control(npz_path: str, robot_ip: Optional[str] = None, speed: float = 0.05) -> bool:
    """
    Main control function for CalliRewrite compatibility.

    Execute trajectory from .npz file on Franka robot.

    Args:
        npz_path: Path to .npz file with pos_3d_x, pos_3d_y, pos_3d_z
        robot_ip: IP address of Franka robot
        speed: Motion speed factor (0.0-1.0)

    Returns:
        True if execution successful

    Example:
        >>> from RoboControl import Control
        >>> Control('./test/test.npz')
    """
    # Print system info
    print("\n" + "="*70)
    print("FRANKA CALLIGRAPHY CONTROL - CalliRewrite Project")
    print("="*70)
    print(f"Python environment: {platform.architecture()}")
    print(f"Platform: {platform.system()}")

    # Load trajectory data
    if not Path(npz_path).exists():
        print(f"❌ Error: File not found: {npz_path}")
        return False

    print(f"\n📂 Loading trajectory: {npz_path}")
    data = np.load(npz_path)

    # Extract control points
    x = np.array(data['pos_3d_x'])
    y = np.array(data['pos_3d_y'])
    z = np.array(data['pos_3d_z'])

    print(f"   Points: {len(x)}")
    print(f"   X range: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"   Y range: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"   Z range: [{z.min():.4f}, {z.max():.4f}] m")

    # Load workspace center from config
    cfg = _load_franka_config()
    workspace_center = _workspace_center_from_config(cfg)
    if workspace_center is None:
        print("⚠️  franka_config.yaml 未提供 workspace_center，回退到默认值 (0.4, 0.0, 0.3)")
        workspace_center = (0.4, 0.0, 0.3)

    if robot_ip is None:
        robot_ip = str(cfg.get("robot_ip", "172.16.0.2"))

    # Initialize controller
    controller = FrankaCalligraphyController(
        robot_ip=robot_ip,
        default_speed=speed,
        workspace_center=workspace_center,
        use_gripper=False,
    )

    # Connect to robot
    if not controller.connect():
        print("❌ Failed to connect to robot")
        return False

    try:
        # Move to home position
        print("\n🏠 Moving to start position...")
        controller.move_to_home()
        time.sleep(1)

        # Execute trajectory
        print("\n🖊️  Starting calligraphy execution...")
        success = controller.execute_trajectory(x, y, z, speed=speed)

        # Return to home
        if success:
            print("\n🏠 Returning to home position...")
            controller.move_to_home()

        return success

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always disconnect
        controller.disconnect()
        print("\n" + "="*70)
        print("Session ended")
        print("="*70 + "\n")


# Additional utility functions

def test_connection(robot_ip: Optional[str] = None) -> bool:
    """
    Test connection to Franka robot.

    Args:
        robot_ip: IP address of robot

    Returns:
        True if connection successful
    """
    print("🔧 Testing Franka connection...")
    if robot_ip is None:
        cfg = _load_franka_config()
        robot_ip = str(cfg.get("robot_ip", "172.16.0.2"))

    controller = FrankaCalligraphyController(robot_ip=robot_ip)

    if controller.connect():
        print("✅ Connection test passed!")
        controller.disconnect()
        return True
    else:
        print("❌ Connection test failed!")
        return False


def visualize_trajectory(npz_path: str):
    """
    Visualize trajectory from .npz file using matplotlib.

    Args:
        npz_path: Path to .npz trajectory file
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("❌ matplotlib not installed. Install with: pip install matplotlib")
        return

    # Load data
    data = np.load(npz_path)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    # Create 3D plot
    fig = plt.figure(figsize=(12, 5))

    # 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=1, alpha=0.6)
    ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # Top view (X-Y)
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y, 'b-', linewidth=1, alpha=0.6)
    ax2.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start')
    ax2.scatter(x[-1], y[-1], c='red', s=100, marker='x', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y)')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"📊 Trajectory statistics:")
    print(f"   Total points: {len(x)}")
    print(f"   X: [{x.min():.4f}, {x.max():.4f}] m, range: {x.max()-x.min():.4f} m")
    print(f"   Y: [{y.min():.4f}, {y.max():.4f}] m, range: {y.max()-y.min():.4f} m")
    print(f"   Z: [{z.min():.4f}, {z.max():.4f}] m, range: {z.max()-z.min():.4f} m")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python RoboControl.py <npz_file> [robot_ip] [speed]")
        print("\nExamples:")
        print("  python RoboControl.py ./test/test.npz")
        print("  python RoboControl.py ./data/永.npz 172.16.0.2 0.05")
        print("\nOr test connection:")
        print("  python RoboControl.py --test [robot_ip]")
        sys.exit(1)

    if sys.argv[1] == "--test":
        robot_ip = sys.argv[2] if len(sys.argv) > 2 else None
        test_connection(robot_ip)
    else:
        npz_path = sys.argv[1]
        robot_ip = sys.argv[2] if len(sys.argv) > 2 else None
        speed = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

        success = Control(npz_path, robot_ip, speed)
        sys.exit(0 if success else 1)
