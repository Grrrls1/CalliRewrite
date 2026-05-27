#!/usr/bin/env python3
"""
创建演示用的NPZ文件 - 简单的"一"字横线
"""

import numpy as np
import os

def create_simple_horizontal_stroke():
    """创建一个简单的横线笔画"""

    # 定义轨迹参数
    num_points = 200

    # 横线：从左到右
    # X: 0 到 30mm (0.03m)
    # Y: 固定在中间 15mm (0.015m)
    # Z: 抬笔 -> 落笔 -> 书写 -> 抬笔

    x = np.linspace(0.0, 0.12, num_points)  # 0到30mm
    y = np.ones(num_points) * 0.015  # 固定在15mm

    # Z坐标：模拟笔的起落
    z = np.zeros(num_points)
    z[0] = 0.05  # 起始：抬笔 (50mm高)
    z[1:3] = np.linspace(0.05, -0.09, 2)  # 落笔
    z[3:-3] = -0.09  # 书写 (压下9mm)
    z[-3:-1] = np.linspace(-0.09, 0.05, 2)  # 抬笔
    z[-1] = 0.05  # 结束：抬笔

    return x, y, z

def create_character_yi(output_name="yi_character"):
    """创建"一"字 - 单个横笔画

    Args:
        output_name: 输出文件名（不含扩展名），默认为"yi_character"
    """

    print("创建'一'字轨迹...")
    x, y, z = create_simple_horizontal_stroke()

    output_file = f"demo_outputs/{output_name}.npz"
    os.makedirs("demo_outputs", exist_ok=True)

    np.savez(output_file,
             pos_3d_x=x,
             pos_3d_y=y,
             pos_3d_z=z)

    print(f"✅ NPZ文件已创建: {output_file}")
    print(f"   控制点数: {len(x)}")
    print(f"   X范围: [{x.min():.4f}, {x.max():.4f}] m ({(x.max()-x.min())*1000:.1f} mm)")
    print(f"   Y范围: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"   Z范围: [{z.min():.4f}, {z.max():.4f}] m")

    return output_file

def create_character_shi(output_name="shi_character"):
    """创建"十"字 - 横竖两笔

    Args:
        output_name: 输出文件名（不含扩展名），默认为"shi_character"
    """

    print("创建'十'字轨迹...")

    # 横笔
    x1, y1, z1 = create_simple_horizontal_stroke()

    # 竖笔：从上到下
    num_points = 50
    x2 = np.ones(num_points) * 0.015  # 固定在15mm (中间)
    y2 = np.linspace(0.0, 0.03, num_points)  # 0到30mm

    z2 = np.zeros(num_points)
    z2[0] = 0.05  # 抬笔
    z2[1:3] = np.linspace(0.05, -0.09, 2)  # 落笔
    z2[3:-3] = -0.09  # 书写
    z2[-3:-1] = np.linspace(-0.09, 0.05, 2)  # 抬笔
    z2[-1] = 0.05

    # 合并两笔
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    z = np.concatenate([z1, z2])

    output_file = f"demo_outputs/{output_name}.npz"
    os.makedirs("demo_outputs", exist_ok=True)

    np.savez(output_file,
             pos_3d_x=x,
             pos_3d_y=y,
             pos_3d_z=z)

    print(f"✅ NPZ文件已创建: {output_file}")
    print(f"   控制点数: {len(x)} (横笔:{len(x1)}, 竖笔:{len(x2)})")
    print(f"   X范围: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"   Y范围: [{y.min():.4f}, {y.max():.4f}] m")
    print(f"   Z范围: [{z.min():.4f}, {z.max():.4f}] m")

    return output_file

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("创建演示用NPZ文件")
    print("=" * 70)
    print()

    # 支持命令行参数指定输出文件名
    if len(sys.argv) > 1:
        # 使用命令行参数作为输出文件名
        # 用法: python create_demo_npz.py yi custom_yi
        # 用法: python create_demo_npz.py shi custom_shi
        character_type = sys.argv[1]
        output_name = sys.argv[2] if len(sys.argv) > 2 else f"{character_type}_character"

        if character_type == "yi":
            output_file = create_character_yi(output_name)
        elif character_type == "shi":
            output_file = create_character_shi(output_name)
        else:
            print(f"❌ 未知的字符类型: {character_type}")
            print("支持的类型: yi, shi")
            sys.exit(1)

        print()
        print("=" * 70)
        print("✅ NPZ文件创建完成！")
        print("=" * 70)
        print()
        print("生成的文件:")
        print(f"  {output_file}")
        print()
        print("使用方法:")
        base_name = output_name
        print(f"  python generate_video_demo.py {output_file} --output demo_outputs/{base_name}_video.mp4")
        print(f"  python visualize_trajectory.py {output_file}")
    else:
        # 默认行为：创建两个默认文件
        yi_file = create_character_yi()
        print()

        shi_file = create_character_shi()
        print()

        print("=" * 70)
        print("✅ 所有NPZ文件创建完成！")
        print("=" * 70)
        print()
        print("生成的文件:")
        print(f"  1. {yi_file}")
        print(f"  2. {shi_file}")
        print()
        print("使用方法:")
        print(f"  python generate_video_demo.py {yi_file} --output demo_outputs/yi_video.mp4")
        print(f"  python generate_video_demo.py {shi_file} --output demo_outputs/shi_video.mp4")
        print()
        print("自定义输出文件名:")
        print(f"  python create_demo_npz.py yi my_custom_name")
