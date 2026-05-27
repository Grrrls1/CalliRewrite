#!/usr/bin/env python3
"""
可视化NPZ轨迹 - 生成2D轨迹图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def visualize_npz_trajectory(npz_file, output_image):
    """可视化NPZ文件中的轨迹"""

    # 加载数据
    data = np.load(npz_file)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    print(f"加载轨迹: {npz_file}")
    print(f"  控制点数: {len(x)}")
    print(f"  X范围: [{x.min()*1000:.1f}, {x.max()*1000:.1f}] mm")
    print(f"  Y范围: [{y.min()*1000:.1f}, {y.max()*1000:.1f}] mm")
    print(f"  Z范围: [{z.min()*1000:.1f}, {z.max()*1000:.1f}] mm")

    # 创建图形
    fig = plt.figure(figsize=(12, 10))

    # 1. XY平面轨迹（俯视图）
    ax1 = plt.subplot(2, 2, 1)
    # 根据Z值判断是否接触（Z < 0表示接触纸面）
    contact_mask = z < 0

    # 画非接触部分（虚线）
    if np.any(~contact_mask):
        ax1.plot(x[~contact_mask]*1000, y[~contact_mask]*1000, 'gray',
                linestyle='--', linewidth=1, alpha=0.5, label='抬笔')

    # 画接触部分（实线）
    if np.any(contact_mask):
        ax1.plot(x[contact_mask]*1000, y[contact_mask]*1000, 'black',
                linewidth=2, label='书写')

    # 标记起点和终点
    ax1.plot(x[0]*1000, y[0]*1000, 'go', markersize=8, label='起点')
    ax1.plot(x[-1]*1000, y[-1]*1000, 'ro', markersize=8, label='终点')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('XY平面轨迹（俯视图）')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    ax1.invert_yaxis()  # Y轴反转，使其与纸面方向一致

    # 2. XZ平面轨迹（侧视图）
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x*1000, z*1000, 'b-', linewidth=1.5)
    ax2.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='纸面')
    ax2.plot(x[0]*1000, z[0]*1000, 'go', markersize=8, label='起点')
    ax2.plot(x[-1]*1000, z[-1]*1000, 'ro', markersize=8, label='终点')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('XZ平面轨迹（侧视图）')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. YZ平面轨迹（前视图）
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(y*1000, z*1000, 'r-', linewidth=1.5)
    ax3.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='纸面')
    ax3.plot(y[0]*1000, z[0]*1000, 'go', markersize=8, label='起点')
    ax3.plot(y[-1]*1000, z[-1]*1000, 'ro', markersize=8, label='终点')
    ax3.set_xlabel('Y (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('YZ平面轨迹（前视图）')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Z坐标随时间变化
    ax4 = plt.subplot(2, 2, 4)
    time_points = np.arange(len(z))
    ax4.plot(time_points, z*1000, 'purple', linewidth=1.5)
    ax4.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='纸面')
    ax4.fill_between(time_points, z*1000, 0, where=(z<0), alpha=0.3, color='blue', label='接触区域')
    ax4.set_xlabel('控制点序号')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('笔尖高度变化')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"✅ 轨迹可视化已保存: {output_image}")
    plt.close()

    # 统计信息
    contact_points = np.sum(contact_mask)
    print(f"\n统计信息:")
    print(f"  接触点数: {contact_points}/{len(z)} ({contact_points/len(z)*100:.1f}%)")
    print(f"  轨迹长度: {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))*1000:.1f} mm")

if __name__ == "__main__":
    import sys
    import os

    # 可视化"一"字
    print("=" * 70)
    print("可视化NPZ轨迹")
    print("=" * 70)
    print()

    # 如果提供了参数，使用参数指定的文件
    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
        # 如果指定了输出文件名，使用指定的；否则根据npz文件名自动生成
        if len(sys.argv) > 2:
            output_image = sys.argv[2]
        else:
            # 自动生成输出文件名：将.npz替换为_trajectory.png
            output_image = npz_file.replace('.npz', '_trajectory.png')
        visualize_npz_trajectory(npz_file, output_image)
    else:
        # 默认可视化demo_outputs目录下的所有npz文件
        demo_dir = 'demo_outputs'
        if os.path.exists(demo_dir):
            npz_files = [f for f in os.listdir(demo_dir) if f.endswith('.npz')]
            if npz_files:
                for npz_filename in sorted(npz_files):
                    npz_file = os.path.join(demo_dir, npz_filename)
                    # 自动生成对应的轨迹图文件名
                    output_image = npz_file.replace('.npz', '_trajectory.png')
                    visualize_npz_trajectory(npz_file, output_image)
                    print()
            else:
                print(f"❌ 在 {demo_dir} 目录中没有找到NPZ文件")
                print("请先运行 create_demo_npz.py 创建NPZ文件")
        else:
            print(f"❌ 目录 {demo_dir} 不存在")
            print("请先运行 create_demo_npz.py 创建NPZ文件")
