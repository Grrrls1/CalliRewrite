#!/usr/bin/env python3
"""
生成示例 NPZ 文件用于演示和测试

这个脚本会生成：
1. test_calibration.npz - 校准测试文件（17条水平线）
2. example_永.npz - 模拟"永"字书法轨迹
3. simple_line.npz - 简单的一条线（最基础示例）

也可以用于可视化现有的 NPZ 文件：
python generate_example_npz.py --visualize path/to/file.npz
"""

import numpy as np
import os
import argparse

def generate_calibration_npz(save_path='./examples/test_calibration.npz'):
    """
    生成校准测试 NPZ 文件
    包含 17 条不同 z 高度的水平线，用于测试笔压-宽度关系
    """
    print("\n" + "="*70)
    print("生成校准测试文件")
    print("="*70)

    # 校准参数（毛笔）
    max_z = 0.01      # 笔尖刚接触纸面
    min_z = -0.006    # 最大压力
    its = 17          # 采样点数

    # 生成 z 高度序列
    zs = np.linspace(max_z, min_z, its)

    x = []
    y = []
    z = []

    for i in range(len(zs)):
        # 每条线 4 个控制点

        # 1. 起点（抬笔）
        x.append(0)
        y.append(i * 0.004)  # 4mm 间隔
        z.append(max_z + 0.03)  # 抬高 3cm

        # 2. 左端点（下笔）
        x.append(0)
        y.append(i * 0.004)
        z.append(zs[i])

        # 3. 右端点（画线）
        x.append(0.05)  # 5cm 长的线
        y.append(i * 0.004)
        z.append(zs[i])

        # 4. 终点（抬笔）
        x.append(0.05)
        y.append(i * 0.004)
        z.append(max_z + 0.03)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, pos_3d_x=x, pos_3d_y=y, pos_3d_z=z)

    print(f"✅ 已生成: {save_path}")
    print(f"   线条数量: {its}")
    print(f"   控制点总数: {len(x)}")
    print(f"   Z 范围: [{min(z):.4f}, {max(z):.4f}] 米")

    return save_path


def generate_simple_line_npz(save_path='./examples/simple_line.npz'):
    """
    生成最简单的一条线
    用于快速测试机器人连接
    """
    print("\n" + "="*70)
    print("生成简单直线测试文件")
    print("="*70)

    x = []
    y = []
    z = []

    # 1. 移动到起点上方
    x.append(0.0)
    y.append(0.0)
    z.append(0.05)  # 抬笔 5cm

    # 2. 下笔
    x.append(0.0)
    y.append(0.0)
    z.append(-0.09)  # 接触纸面（假设纸面在 -0.09）

    # 3. 画一条 3cm 的水平线
    for i in range(1, 31):  # 30 个点
        x.append(i * 0.001)  # 每步 1mm
        y.append(0.0)
        z.append(-0.09)

    # 4. 抬笔
    x.append(0.03)
    y.append(0.0)
    z.append(0.05)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, pos_3d_x=x, pos_3d_y=y, pos_3d_z=z)

    print(f"✅ 已生成: {save_path}")
    print(f"   线长: 3cm")
    print(f"   控制点总数: {len(x)}")

    return save_path


def generate_character_yong_npz(save_path='./examples/example_永.npz'):
    """
    生成模拟"永"字的书法轨迹
    使用简化的笔画结构
    """
    print("\n" + "="*70)
    print("生成'永'字示例文件")
    print("="*70)

    x = []
    y = []
    z = []

    alpha = 0.04  # 字符宽度 4cm
    z_lift = 0.05  # 抬笔高度
    z_base = -0.09  # 纸面基准

    # "永"字的 8 个基本笔画（简化版）

    # === 笔画 1: 点（侧点）===
    # 起笔
    x.append(0.015 * alpha)
    y.append(0.02 * alpha)
    z.append(z_lift)

    # 下笔
    x.append(0.015 * alpha)
    y.append(0.02 * alpha)
    z.append(z_base - 0.002)  # 轻压

    # 加重
    x.append(0.018 * alpha)
    y.append(0.025 * alpha)
    z.append(z_base - 0.004)  # 重压

    # 抬笔
    x.append(0.018 * alpha)
    y.append(0.025 * alpha)
    z.append(z_lift)

    # === 笔画 2: 横（勒）===
    # 移动到起点
    x.append(0.01 * alpha)
    y.append(0.035 * alpha)
    z.append(z_lift)

    # 下笔
    x.append(0.01 * alpha)
    y.append(0.035 * alpha)
    z.append(z_base - 0.001)

    # 画横
    for i in range(10):
        t = i / 10
        x.append((0.01 + 0.03 * t) * alpha)
        y.append(0.035 * alpha)
        z.append(z_base - 0.001 - 0.002 * np.sin(t * np.pi))  # 中间稍重

    # 抬笔
    x.append(0.04 * alpha)
    y.append(0.035 * alpha)
    z.append(z_lift)

    # === 笔画 3: 竖（弯钩）===
    # 移动到起点
    x.append(0.025 * alpha)
    y.append(0.04 * alpha)
    z.append(z_lift)

    # 下笔
    x.append(0.025 * alpha)
    y.append(0.04 * alpha)
    z.append(z_base - 0.002)

    # 画竖（带弯钩）
    for i in range(15):
        t = i / 15
        x.append((0.025 + 0.01 * t) * alpha)  # 稍向右弯
        y.append((0.04 + 0.055 * t) * alpha)  # 向下
        z.append(z_base - 0.002 - 0.003 * t)  # 逐渐加压

    # 钩
    for i in range(5):
        t = i / 5
        x.append((0.035 - 0.008 * t) * alpha)  # 向左钩
        y.append((0.095 + 0.005 * t) * alpha)
        z.append(z_base - 0.005 + 0.003 * t)  # 逐渐减压

    # 抬笔
    x.append(0.027 * alpha)
    y.append(0.1 * alpha)
    z.append(z_lift)

    # === 笔画 4: 撇 ===
    x.append(0.03 * alpha)
    y.append(0.05 * alpha)
    z.append(z_lift)

    x.append(0.03 * alpha)
    y.append(0.05 * alpha)
    z.append(z_base - 0.003)

    for i in range(12):
        t = i / 12
        x.append((0.03 - 0.025 * t) * alpha)
        y.append((0.05 + 0.04 * t) * alpha)
        z.append(z_base - 0.003 + 0.002 * t)  # 逐渐提笔

    x.append(0.005 * alpha)
    y.append(0.09 * alpha)
    z.append(z_lift)

    # === 笔画 5: 捺 ===
    x.append(0.035 * alpha)
    y.append(0.055 * alpha)
    z.append(z_lift)

    x.append(0.035 * alpha)
    y.append(0.055 * alpha)
    z.append(z_base - 0.001)

    for i in range(15):
        t = i / 15
        x.append((0.035 + 0.055 * t) * alpha)
        y.append((0.055 + 0.035 * t) * alpha)
        z.append(z_base - 0.001 - 0.004 * np.sin(t * np.pi))  # 中间重两头轻

    x.append(0.09 * alpha)
    y.append(0.09 * alpha)
    z.append(z_lift)

    # === 笔画 6-8: 其他笔画（简化）===
    # 添加几个简单的短横和点

    # 短横 1
    x.extend([0.015 * alpha, 0.015 * alpha, 0.022 * alpha, 0.022 * alpha])
    y.extend([0.06 * alpha, 0.06 * alpha, 0.06 * alpha, 0.06 * alpha])
    z.extend([z_lift, z_base - 0.002, z_base - 0.002, z_lift])

    # 短横 2
    x.extend([0.055 * alpha, 0.055 * alpha, 0.065 * alpha, 0.065 * alpha])
    y.extend([0.065 * alpha, 0.065 * alpha, 0.065 * alpha, 0.065 * alpha])
    z.extend([z_lift, z_base - 0.002, z_base - 0.002, z_lift])

    # 点
    x.extend([0.045 * alpha, 0.045 * alpha, 0.046 * alpha, 0.046 * alpha])
    y.extend([0.075 * alpha, 0.075 * alpha, 0.077 * alpha, 0.077 * alpha])
    z.extend([z_lift, z_base - 0.003, z_base - 0.003, z_lift])

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, pos_3d_x=x, pos_3d_y=y, pos_3d_z=z)

    print(f"✅ 已生成: {save_path}")
    print(f"   字符大小: {alpha*100:.1f} cm")
    print(f"   控制点总数: {len(x)}")
    print(f"   笔画数: 8 (简化版)")

    return save_path


def visualize_npz(npz_path):
    """可视化 NPZ 文件内容"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过可视化")
        return

    print(f"\n可视化文件: {npz_path}")

    data = np.load(npz_path)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    fig = plt.figure(figsize=(15, 5))

    # 3D 视图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=1, alpha=0.6)
    ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='起点')
    ax1.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='x', label='终点')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D 轨迹')
    ax1.legend()

    # 俯视图 (X-Y)
    ax2 = fig.add_subplot(132)
    ax2.plot(x, y, 'b-', linewidth=1, alpha=0.6)
    ax2.scatter(x[0], y[0], c='green', s=100, marker='o', label='起点')
    ax2.scatter(x[-1], y[-1], c='red', s=100, marker='x', label='终点')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('俯视图 (X-Y)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Z 高度变化
    ax3 = fig.add_subplot(133)
    ax3.plot(range(len(z)), z, 'b-', linewidth=1.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, label='参考平面', alpha=0.5)
    ax3.axhline(y=-0.09, color='orange', linestyle='--', linewidth=1, label='纸面', alpha=0.5)
    ax3.fill_between(range(len(z)), z, -0.09, where=(np.array(z) < -0.08), alpha=0.3, color='blue', label='笔压区')
    ax3.set_xlabel('控制点序号')
    ax3.set_ylabel('Z 高度 (m)')
    ax3.set_title('Z 轴高度变化')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    # 保存图片
    save_name = npz_path.replace('.npz', '_visualization.png')
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"📊 可视化图片已保存: {save_name}")

    # plt.show()  # 取消注释以显示图形


def print_npz_info(npz_path):
    """打印 NPZ 文件详细信息"""
    print("\n" + "="*70)
    print(f"NPZ 文件信息: {npz_path}")
    print("="*70)

    data = np.load(npz_path)
    x = data['pos_3d_x']
    y = data['pos_3d_y']
    z = data['pos_3d_z']

    print(f"\n📊 统计信息:")
    print(f"   控制点数量: {len(x)}")
    print(f"   文件大小: {os.path.getsize(npz_path) / 1024:.2f} KB")

    print(f"\n📐 坐标范围:")
    print(f"   X: [{x.min():.4f}, {x.max():.4f}] m  (范围: {x.max()-x.min():.4f} m = {(x.max()-x.min())*100:.2f} cm)")
    print(f"   Y: [{y.min():.4f}, {y.max():.4f}] m  (范围: {y.max()-y.min():.4f} m = {(y.max()-y.min())*100:.2f} cm)")
    print(f"   Z: [{z.min():.4f}, {z.max():.4f}] m  (范围: {z.max()-z.min():.4f} m = {(z.max()-z.min())*100:.2f} cm)")

    print(f"\n🖊️  笔画分析:")
    # 分析抬笔/落笔次数
    z_array = np.array(z)
    lift_threshold = 0.0  # Z > 0 认为是抬笔
    is_lifted = z_array > lift_threshold
    transitions = np.diff(is_lifted.astype(int))
    num_strokes = np.sum(transitions == -1)  # 从抬笔到落笔的次数
    print(f"   估计笔画数: {num_strokes}")
    print(f"   抬笔点数: {np.sum(is_lifted)}")
    print(f"   接触点数: {np.sum(~is_lifted)}")

    print(f"\n📋 前 5 个控制点:")
    print("   索引    X (m)      Y (m)      Z (m)     状态")
    print("   " + "-"*55)
    for i in range(min(5, len(x))):
        status = "抬笔" if z[i] > 0 else "接触"
        print(f"   {i:3d}   {x[i]:8.4f}   {y[i]:8.4f}   {z[i]:8.4f}   {status}")

    if len(x) > 10:
        print("   ...")
        print(f"\n📋 最后 3 个控制点:")
        print("   索引    X (m)      Y (m)      Z (m)     状态")
        print("   " + "-"*55)
        for i in range(len(x)-3, len(x)):
            status = "抬笔" if z[i] > 0 else "接触"
            print(f"   {i:3d}   {x[i]:8.4f}   {y[i]:8.4f}   {z[i]:8.4f}   {status}")

    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成示例NPZ文件或可视化现有NPZ文件')
    parser.add_argument('--visualize', '-v', type=str, help='可视化指定的NPZ文件路径')
    parser.add_argument('--generate', '-g', action='store_true', help='生成示例NPZ文件（默认行为）')
    
    args = parser.parse_args()
    
    if args.visualize:
        # 可视化模式
        npz_path = args.visualize
        if not os.path.exists(npz_path):
            print(f"❌ 文件不存在: {npz_path}")
            exit(1)
        
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*20 + "NPZ 文件可视化工具" + " "*28 + "║")
        print("║" + " "*15 + "CalliRewrite × Franka Integration" + " "*20 + "║")
        print("╚" + "="*68 + "╝")
        
        print_npz_info(npz_path)
        visualize_npz(npz_path)
        
        print("\n✅ 可视化完成！")
        
    else:
        # 生成模式（默认）
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*20 + "NPZ 示例文件生成器" + " "*28 + "║")
        print("║" + " "*15 + "CalliRewrite × Franka Integration" + " "*20 + "║")
        print("╚" + "="*68 + "╝")

        # 生成三个示例文件
        files = []

        # 1. 校准测试文件
        files.append(generate_calibration_npz())

        # 2. 简单直线
        files.append(generate_simple_line_npz())

        # 3. "永"字示例
        files.append(generate_character_yong_npz())

        # 打印每个文件的详细信息
        for f in files:
            print_npz_info(f)
            visualize_npz(f)

        print("\n✅ 所有示例文件生成完成！")
        print("\n使用方法:")
        print("   # 查看文件内容")
        print("   python -c \"import numpy as np; data=np.load('./examples/simple_line.npz'); print(list(data.keys()))\"")
        print("\n   # 在机器人上执行")
        print("   python RoboControl.py ./examples/simple_line.npz")
        print("\n   # 可视化轨迹")
        print("   python generate_example_npz.py --visualize ./examples/example_永.npz")
        print()
