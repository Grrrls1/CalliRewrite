#!/usr/bin/env python3
"""
3D trajectory visualization for .npz files.
Supports both static plots and animated playback.

Usage:
    # Static 3D plot
    python visualize_3d_trajectory.py input.npz
    
    # Animated playback with frame updates
    python visualize_3d_trajectory.py input.npz --animate --fps 30
    
    # Save to HTML (interactive)
    python visualize_3d_trajectory.py input.npz --html output.html
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def load_trajectory(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectory from .npz file."""
    data = np.load(npz_path)
    
    # Try different possible key names
    x_keys = ['pos_3d_x', 'x', 'X']
    y_keys = ['pos_3d_y', 'y', 'Y']
    z_keys = ['pos_3d_z', 'z', 'Z']
    
    x = None
    y = None
    z = None
    
    for key in x_keys:
        if key in data:
            x = data[key]
            break
    
    for key in y_keys:
        if key in data:
            y = data[key]
            break
    
    for key in z_keys:
        if key in data:
            z = data[key]
            break
    
    if x is None or y is None or z is None:
        raise KeyError(f"Could not find trajectory data in {npz_path}. Available keys: {list(data.keys())}")
    
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    
    # Flip x-axis to correct left-right orientation
    x = -x
    
    return x, y, z


def plot_static_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, output: str = None) -> None:
    """Create a static 3D plot of the trajectory."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory with color gradient
    colors = np.linspace(0, 1, len(x))
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=10, alpha=0.6)
    
    # Plot line
    ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax.scatter(*[x[0], y[0], z[0]], color='green', s=100, marker='o', label='Start')
    ax.scatter(*[x[-1], y[-1], z[-1]], color='red', s=100, marker='s', label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Time progression')
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"✅ Saved static plot to: {output}")
    else:
        plt.show()


def plot_interactive_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, output: str = None) -> None:
    """Create an interactive 3D plot using plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️  plotly not installed. Install with: pip install plotly")
        print("Falling back to static matplotlib plot...")
        plot_static_3d(x, y, z, output)
        return
    
    # Create color array based on progression
    colors = list(range(len(x)))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time")
        ),
        line=dict(color='blue', width=2),
        name='Trajectory'
    )])
    
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='End'
    ))
    
    fig.update_layout(
        title='3D Trajectory (Interactive)',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        width=1200,
        height=900
    )
    
    if output:
        fig.write_html(output)
        print(f"✅ Saved interactive plot to: {output}")
    else:
        fig.show()


def animate_trajectory(x: np.ndarray, y: np.ndarray, z: np.ndarray, fps: int = 30, output: str = None) -> None:
    """Create an animated 3D trajectory playback."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    ax.set_xlim(x.min() - 0.01, x.max() + 0.01)
    ax.set_ylim(y.min() - 0.01, y.max() + 0.01)
    ax.set_zlim(z.min() - 0.01, z.max() + 0.01)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Animation')
    
    # Plot elements
    line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
    point, = ax.plot([], [], [], 'ro', markersize=8)
    
    def update(frame):
        # Draw trajectory up to current frame
        line.set_data(x[:frame+1], y[:frame+1])
        line.set_3d_properties(z[:frame+1])
        
        # Draw current point
        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])
        
        # Update title with progress
        ax.set_title(f'3D Trajectory Animation - Frame {frame+1}/{len(x)}')
        
        return line, point
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(x), interval=1000/fps, blit=True, repeat=True
    )
    
    if output:
        # Save animation
        print(f"💾 Saving animation to {output}... (this may take a while)")
        anim.save(output, writer='ffmpeg', fps=fps)
        print(f"✅ Saved animation to: {output}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize 3D trajectory from .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Static 3D plot (opens in window)
  python visualize_3d_trajectory.py data.npz
  
  # Save static plot to image
  python visualize_3d_trajectory.py data.npz -o plot.png
  
  # Interactive HTML plot
  python visualize_3d_trajectory.py data.npz --html -o plot.html
  
  # Animated playback (requires ffmpeg)
  python visualize_3d_trajectory.py data.npz --animate -o trajectory.mp4 --fps 30
        """
    )
    
    parser.add_argument("input", type=str, help="Input .npz trajectory file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("--html", action="store_true", help="Create interactive HTML plot (requires plotly)")
    parser.add_argument("--animate", action="store_true", help="Create animation (requires ffmpeg)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for animation (default 30)")
    
    args = parser.parse_args()
    
    # Load trajectory
    print(f"📂 Loading trajectory from: {args.input}")
    try:
        x, y, z = load_trajectory(args.input)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return 1
    
    print(f"✓ Loaded {len(x)} points")
    print(f"  X range: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"  Y range: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"  Z range: [{z.min():.4f}, {z.max():.4f}] m")
    
    # Choose visualization type
    try:
        if args.animate:
            animate_trajectory(x, y, z, fps=args.fps, output=args.output)
        elif args.html:
            plot_interactive_3d(x, y, z, output=args.output)
        else:
            plot_static_3d(x, y, z, output=args.output)
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


'''

python visualize_3d_trajectory.py rl_npz/yong.npz --html -o rl_npz/yong.html

'''