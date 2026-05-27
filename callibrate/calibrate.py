"""
Calibration Script for Dobot Magician - CalliRewrite Project

This script calibrates the relationship between brush radius (r) and robot height (z)
for accurate calligraphy reproduction.

Usage:
    1. Modify tool parameters (max_z, min_z, its)
    2. Run generate_calibration_data() to create test trajectory
    3. Execute trajectory on robot to draw calibration lines
    4. Manually measure stroke widths and update 'widths' list
    5. Run fit_calibration_function() to get r-z mapping
    6. Use convert_rl_to_npz() to convert RL output to robot commands
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import optimize


# ============================================================================
# STEP 1: Tool Calibration - Generate Test Data
# ============================================================================

def generate_calibration_data(tool_type='brush', save_dir='./test/'):
    """
    Generate calibration test data for different tools.

    Args:
        tool_type: 'brush' or 'fude' or 'marker'
        save_dir: Directory to save calibration npz file

    Returns:
        zs: Array of z-heights for calibration
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Tool-specific parameters
    if tool_type == 'brush':
        print("=== Calibrating Calligraphy Brush ===")
        max_z = 0.01      # Pen touching paper (m)
        min_z = -0.006    # Maximum pressure (m)
        its = 17          # Number of samples

    elif tool_type == 'fude':
        print("=== Calibrating Fude Pen ===")
        max_z = 0.04602
        min_z = 0.044250
        its = 8

    elif tool_type == 'marker':
        print("=== Calibrating Flat Tip Marker ===")
        max_z = 0.015
        min_z = 0.005
        its = 10

    else:
        raise ValueError(f"Unknown tool type: {tool_type}")

    # Generate z-height samples
    zs = np.linspace(max_z, min_z, its).round(7)
    zs = np.sort(np.unique(zs))[::-1]
    print(f"Z-heights: {zs}")

    # Generate test trajectory
    generate_npz(zs, max_z, save_dir)

    return zs, max_z


def generate_npz(zs, max_z, save_dir):
    """
    Generate 'its' number of horizontal lines at different z heights.
    This creates the calibration pattern for measuring r-z relationship.

    Args:
        zs: Array of z-heights to test
        max_z: Maximum z (pen lifted)
        save_dir: Output directory
    """
    x = []
    y = []
    z = []

    for i in range(zs.shape[0]):
        # Begin point (lifted)
        x.append(0)
        y.append(i * 0.004)  # 4mm spacing between lines
        z.append(max_z + 0.03)

        # Left point with current z (pen down)
        x.append(0)
        y.append(i * 0.004)
        z.append(zs[i])

        # Right point with current z (draw line)
        x.append(0.05)  # 5cm horizontal line
        y.append(i * 0.004)
        z.append(zs[i])

        # Lift pen
        x.append(0.05)
        y.append(i * 0.004)
        z.append(max_z + 0.03)

    print(f"Generating calibration data with {zs.shape[0]} test lines...")
    output_path = os.path.join(save_dir, "test.npz")
    np.savez(output_path, pos_3d_x=x, pos_3d_y=y, pos_3d_z=z)
    print(f"Saved to: {output_path}")
    print("\n>>> Now execute this file on your Dobot robot:")
    print(f">>> from RoboControl import *")
    print(f">>> Control('{output_path}')")


# ============================================================================
# STEP 2: Execute on Robot (requires RoboControl.py)
# ============================================================================

def execute_calibration(npz_path='./test/test.npz'):
    """
    Execute calibration trajectory on Dobot robot.

    NOTE: This requires RoboControl.py to be implemented.
    """
    try:
        from RoboControl import Control
        print(f"Executing calibration trajectory: {npz_path}")
        Control(npz_path)
        print("✓ Calibration complete! Now measure the stroke widths.")
    except ImportError:
        print("ERROR: RoboControl.py not found!")
        print("You need to implement RoboControl.py with Control() function.")
        print("See DobotDllType.py for API reference.")


# ============================================================================
# STEP 3: Measure and Fit Calibration Function
# ============================================================================

def piecewise_linear3(x, x0, x1, y0, y1, k0, k1):
    """3-segment piecewise linear function for fitting."""
    return np.piecewise(
        x,
        [x <= x0, np.logical_and(x0 < x, x <= x1), x > x1],
        [
            lambda x: k0 * (x - x0) + y0,
            lambda x: (x - x0) * (y1 - y0) / (x1 - x0) + y0,
            lambda x: k1 * (x - x1) + y1
        ]
    )


def piecewise_linear4(x, x0, x1, x2, y0, y1, y2, k0, k1):
    """4-segment piecewise linear function for fitting."""
    return np.piecewise(
        x,
        [
            x <= x0,
            np.logical_and(x0 < x, x <= x1),
            np.logical_and(x1 < x, x <= x2),
            x > x2
        ],
        [
            lambda x: k0 * (x - x0) + y0,
            lambda x: (x - x0) * (y1 - y0) / (x1 - x0) + y0,
            lambda x: (x - x1) * (y2 - y1) / (x2 - x1) + y1,
            lambda x: k1 * (x - x2) + y2
        ]
    )


def fit_calibration_function(widths, zs, tool_type='brush', plot=True):
    """
    Fit piecewise linear function to measured data.

    Args:
        widths: List of measured stroke widths (in meters)
        zs: Corresponding z-heights (from generate_calibration_data)
        tool_type: 'brush' or 'fude'
        plot: Whether to plot the fitted curve

    Returns:
        p: Fitted parameters
        func: Calibrated function r -> z
    """
    rs = np.array(widths) / 2  # Convert width to radius

    print(f"\n=== Fitting {tool_type} calibration function ===")
    print(f"Measured radii (m): {rs}")
    print(f"Z-heights (m): {zs}")

    # Fit based on tool type
    if tool_type == 'brush':
        # 4-segment fit for brush (more complex dynamics)
        bounds = (
            [0.0007, 0.0015, 0.00215, 0.0038, 0.0025, -0.002, -20, -20],
            [0.001, 0.0019, 0.0025, 0.005, 0.003, -0.0008, -0.5, -0.5]
        )
        p, e = optimize.curve_fit(piecewise_linear4, rs, zs, bounds=bounds)

        print(f"Fitted parameters: {p}")

        # Create function
        def func_brush(radii):
            if radii >= 0 and radii <= p[0]:
                z = p[6] * (radii - p[0]) + p[3]
            elif radii > p[0] and radii <= p[1]:
                z = (radii - p[0]) * (p[4] - p[3]) / (p[1] - p[0]) + p[3]
            elif radii > p[1] and radii <= p[2]:
                z = (radii - p[1]) * (p[5] - p[4]) / (p[2] - p[1]) + p[4]
            elif radii > p[2] and radii <= 0.0045:
                z = p[7] * (radii - p[2]) + p[5]
            else:
                return -0.006  # Overflow protection
            return z

        func = func_brush

    elif tool_type == 'fude':
        # 3-segment fit for fude pen
        bounds = (
            [0.0001, 0.0005, 0, 0, -20, -20],
            [0.0003, 0.0006, 0.0456, 0.04675, -0.5, -0.5]
        )
        p, e = optimize.curve_fit(piecewise_linear3, rs, zs, bounds=bounds)

        print(f"Fitted parameters: {p}")

        # Create function
        def func_fude(radii):
            if radii >= 0 and radii <= p[0]:
                z = p[4] * (radii - p[0]) + p[2]
            elif radii > p[0] and radii <= p[1]:
                z = (radii - p[0]) * (p[3] - p[2]) / (p[1] - p[0]) + p[2]
            elif radii > p[1] and radii <= 0.00071875:
                z = p[5] * (radii - p[1]) + p[3]
            else:
                return 0.04425  # Overflow protection
            return z

        func = func_fude

    else:
        raise ValueError(f"Unknown tool type: {tool_type}")

    # Plot if requested
    if plot:
        xd = np.linspace(rs[0], rs[-1], 100)
        plt.figure(figsize=(8, 6))
        plt.scatter(rs, zs, s=100, label='Measured data', zorder=3)

        if tool_type == 'brush':
            plt.plot(xd, piecewise_linear4(xd, *p), 'r-', linewidth=2, label='Fitted curve')
        else:
            plt.plot(xd, piecewise_linear3(xd, *p), 'r-', linewidth=2, label='Fitted curve')

        plt.xlabel("Radius r (m)", fontsize=12)
        plt.ylabel("Z-height (m)", fontsize=12)
        plt.title(f"{tool_type.capitalize()} Calibration: r-z Relationship", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./test/{tool_type}_calibration.png', dpi=150)
        print(f"Plot saved to: ./test/{tool_type}_calibration.png")
        plt.show()

    return p, func


# ============================================================================
# Pre-calibrated Functions (from paper)
# ============================================================================

def func_brush_precalibrated(radii):
    """
    Pre-calibrated function for calligraphy brush.
    Re-calibrate for your own robot!
    """
    if radii >= 0 and radii <= 7.72667536e-04:
        z = -5.9701493 * (radii - 7.72667536e-04) + 4.34974603e-03
    elif radii > 7.72667536e-04 and radii <= 1.78125854e-03:
        z = -1.538473 * (radii - 7.72667536e-04) + 4.34974603e-03
    elif radii > 1.78125854e-03 and radii <= 2.45866277e-03:
        z = -6.028019 * (radii - 1.78125854e-03) + 2.79805600e-03
    elif radii > 2.45866277e-03 and radii < 0.0045:
        z = -2.37843574 * (radii - 2.45866277e-03) - 1.28534957e-03
    else:
        return -0.006
    return z


def func_fude_precalibrated(radii):
    """
    Pre-calibrated function for fude pen.
    Re-calibrate for your own robot!
    """
    if radii >= 0 and radii <= 0.00026555923:
        z = -3.06968833 * (radii - 0.00026555923) + 0.0453094035
    elif radii > 0.00026555923 and radii <= 0.000553636283:
        z = -1.444032 * (radii - 0.00026555923) + 0.0453094035
    elif radii > 0.000553636283 and radii <= 0.00071875:
        z = -3.66324702 * (radii - 0.000553636283) + 0.044893411
    else:
        return 0.04425
    return z


# ============================================================================
# STEP 4: Convert RL States to Robot Control Points
# ============================================================================

def convert_rl_to_npz(npy_path, output_path, calibration_func,
                     alpha=0.04, beta=0.5, style_type=0,
                     max_step_mm=None):
    """
    Convert RL fine-tuned states (.npy) to Dobot control points (.npz).

    Args:
        npy_path: Path to RL output .npy file
        output_path: Path to save .npz control points
        calibration_func: Calibrated r->z function
        alpha: Character size in meters (e.g., 0.04 = 4cm width)
        beta: Stroke width contrast adjustment (0.5 = normal)
        style_type: 0=Lishu (clerical), 1=Kaishu (regular)

    Returns:
        x, y, z: Lists of control points (meters)
    """
    # Load RL states
    data = np.load(npy_path)
    print(f"Loaded RL states: {data.shape}")
    print(f"Character size: {alpha*100} cm")
    print(f"Stroke width adjustment: {beta}")

    record_x = []
    record_y = []
    record_z = []

    for i in range(data.shape[0]):
        p_t, x, y, r = data[i]

        # Scale to real size
        x_ = x * alpha
        y_ = y * alpha
        r_ = r * alpha * beta

        # Convert r to z using calibration function
        h = calibration_func(r_)
        h = h - 0.09  # Adjust for table height

        if p_t == 0:
            # Normal stroke point
            record_x.append(x_)
            record_y.append(y_)
            record_z.append(h)

        elif p_t == 1:
            # New stroke beginning
            if i == data.shape[0] - 1:
                continue  # Skip if last point

            if style_type == 1:
                # Kaishu: enter from top-left
                record_x.append(x_ - 2 * r_)
                record_y.append(y_ - 2 * r_)
                record_z.append(0.05)  # Lift pen
            else:
                # Lishu: enter from above
                record_x.append(x_)
                record_y.append(y_)
                record_z.append(0.05)

            # Add current point (pen down)
            record_x.append(x_)
            record_y.append(y_)
            record_z.append(h)

            if style_type == 0:
                # Lishu: backtrack stroke for entry
                nxt_vec = data[i + 1][1:3] - data[i][1:3]
                nxt_vec = nxt_vec / np.linalg.norm(nxt_vec)
                record_x.append(x_ - 2 * r_ * nxt_vec[0])
                record_y.append(y_ - 2 * r_ * nxt_vec[1])
                record_z.append(h)
                record_x.append(x_)
                record_y.append(y_)
                record_z.append(h)

    # Gradual pen lift at end
    for _ in range(5):
        record_x.append(record_x[-1])
        record_y.append(record_y[-1])
        record_z.append(record_z[-1] + 0.015)

    # Optional: densify to increase control points (limit max step length)
    if max_step_mm is not None:
        max_step_m = float(max_step_mm) / 1000.0
        if max_step_m <= 0:
            raise ValueError(f"max_step_mm must be > 0, got {max_step_mm}")
        before = len(record_x)
        x2, y2, z2 = [float(record_x[0])], [float(record_y[0])], [float(record_z[0])]
        for i in range(1, len(record_x)):
            x0, y0, z0 = float(record_x[i - 1]), float(record_y[i - 1]), float(record_z[i - 1])
            x1, y1, z1 = float(record_x[i]), float(record_y[i]), float(record_z[i])
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            n_seg = int(np.ceil(dist / max_step_m)) if dist > 0 else 1
            for k in range(1, n_seg + 1):
                t = k / n_seg
                x2.append(x0 + dx * t)
                y2.append(y0 + dy * t)
                z2.append(z0 + dz * t)
        record_x, record_y, record_z = x2, y2, z2
        print(f"Densify: {before} -> {len(record_x)} (max_step={max_step_mm}mm)")

    # Save to npz
    np.savez(output_path,
             pos_3d_x=record_x,
             pos_3d_y=record_y,
             pos_3d_z=record_z)

    print(f"Converted {len(record_x)} control points")
    print(f"Z range: [{np.min(record_z):.4f}, {np.max(record_z):.4f}] m")
    print(f"Saved to: {output_path}")

    return record_x, record_y, record_z


# ============================================================================
# Main Workflow Examples
# ============================================================================

def example_calibration_workflow():
    """
    Example: Complete calibration workflow for calligraphy brush.
    """
    print("\n" + "="*70)
    print("CALIBRATION WORKFLOW EXAMPLE - Calligraphy Brush")
    print("="*70)

    # Step 1: Generate calibration test data
    print("\n[Step 1] Generating calibration test data...")
    zs, max_z = generate_calibration_data(tool_type='brush', save_dir='./test/')

    # Step 2: Execute on robot (manual step)
    print("\n[Step 2] Execute calibration on robot...")
    print(">>> Run the following in Python:")
    print(">>> from RoboControl import *")
    print(">>> Control('./test/test.npz')")
    print("\n>>> Then measure stroke widths with a ruler!")

    # Step 3: After measuring, fit calibration function
    print("\n[Step 3] After measuring, run fit_calibration_function()...")

    # Example measured widths (in meters)
    widths_example = [
        0.00025, 0.00085, 0.001, 0.0012, 0.002, 0.0033, 0.0038,
        0.0042, 0.0045, 0.0048, 0.0055, 0.0065, 0.0072, 0.0078, 0.009
    ]
    zs_example = np.array([
        0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002,
        0.001, 0., -0.001, -0.002, -0.003, -0.004, -0.005, -0.006
    ])

    print("Example measured widths:", widths_example)
    # p, func = fit_calibration_function(widths_example, zs_example, 'brush', plot=False)

    print("\n✓ Calibration complete!")
    print("Use the returned 'func' to convert RL states to robot commands.")


def example_rl_to_robot():
    """
    Example: Convert RL output to robot control points.
    """
    print("\n" + "="*70)
    print("RL TO ROBOT CONVERSION EXAMPLE")
    print("="*70)

    # Use pre-calibrated function
    calibration_func = func_brush_precalibrated

    # Example: Convert RL output for character "永"
    npy_path = './data/永.npy'
    output_path = './data/永.npz'

    if os.path.exists(npy_path):
        x, y, z = convert_rl_to_npz(
            npy_path=npy_path,
            output_path=output_path,
            calibration_func=calibration_func,
            alpha=0.04,  # 4cm character
            beta=0.5,    # Normal stroke width
            style_type=0  # Lishu style
        )

        print("\n>>> Now execute on robot:")
        print(">>> from RoboControl import *")
        print(f">>> Control('{output_path}')")
    else:
        print(f"ERROR: {npy_path} not found!")
        print("First run RL fine-tuning to generate .npy files.")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Dobot Calibration Tool')
    parser.add_argument('--mode', type=str, default='help',
                       choices=['generate', 'fit', 'convert', 'help'],
                       help='Operation mode')
    parser.add_argument('--tool', type=str, default='brush',
                       choices=['brush', 'fude', 'marker'],
                       help='Tool type')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--alpha', type=float, default=0.04,
                       help='Character size (meters)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Stroke width adjustment')
    parser.add_argument('--max-step-mm', type=float, default=None,
                       help='Optional densify: limit consecutive point max 3D distance (mm)')

    args = parser.parse_args()

    if args.mode == 'generate':
        # Generate calibration test data
        generate_calibration_data(tool_type=args.tool)

    elif args.mode == 'fit':
        # Fit calibration function (requires manual measurement)
        print("Please manually measure stroke widths and update the code!")
        print("See example_calibration_workflow() for reference.")

    elif args.mode == 'convert':
        # Convert RL states to robot commands
        if not args.input or not args.output:
            print("ERROR: --input and --output required for convert mode")
        else:
            calibration_func = func_brush_precalibrated if args.tool == 'brush' else func_fude_precalibrated
            convert_rl_to_npz(
                args.input,
                args.output,
                calibration_func,
                alpha=args.alpha,
                beta=args.beta,
                max_step_mm=args.max_step_mm
            )

    else:
        # Show help and examples
        print("\n" + "="*70)
        print("DOBOT CALIBRATION TOOL - CalliRewrite Project")
        print("="*70)
        print("\nUsage Examples:")
        print("\n1. Generate calibration test data:")
        print("   python calibrate.py --mode generate --tool brush")
        print("\n2. Convert RL output to robot commands:")
        print("   python calibrate.py --mode convert --tool brush \\")
        print("      --input data/永.npy --output data/永.npz --alpha 0.04")
        print("\n3. Run example workflows:")
        print("   python -c 'from calibrate import *; example_calibration_workflow()'")
        print("   python -c 'from calibrate import *; example_rl_to_robot()'")
        print("\n" + "="*70)
