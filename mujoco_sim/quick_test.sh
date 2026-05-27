#!/bin/bash
# MuJoCo 仿真快速测试脚本

echo "================================"
echo "MuJoCo Simulation Quick Test"
echo "================================"
echo ""

# 检查依赖
echo "1. Checking dependencies..."
python -c "import mujoco; print(f'✅ MuJoCo {mujoco.__version__}')" || {
    echo "❌ MuJoCo not installed"
    echo "Run: pip install mujoco"
    exit 1
}

python -c "import cv2; print('✅ OpenCV')" || {
    echo "❌ OpenCV not installed"
    echo "Run: pip install opencv-python"
    exit 1
}

python -c "import matplotlib; print('✅ Matplotlib')" || {
    echo "❌ Matplotlib not installed"
    echo "Run: pip install matplotlib"
    exit 1
}

echo ""
echo "2. Generating example NPZ files..."
cd ../callibrate
python generate_example_npz.py || {
    echo "❌ Failed to generate examples"
    exit 1
}

echo ""
echo "3. Running basic simulation..."
cd ../mujoco_sim
python mujoco_simulator.py ../callibrate/examples/simple_line.npz --speed 0.1 --no-render

echo ""
echo "4. Checking outputs..."
if [ -f "outputs/calligraphy_result.png" ]; then
    echo "✅ Canvas: outputs/calligraphy_result.png"
else
    echo "❌ Canvas not found"
fi

if [ -f "outputs/trajectory_3d.png" ]; then
    echo "✅ 3D Plot: outputs/trajectory_3d.png"
else
    echo "❌ 3D plot not found"
fi

echo ""
echo "================================"
echo "✅ Quick test completed!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. View results: open outputs/*.png"
echo "  2. Try interactive: python mujoco_simulator.py ../callibrate/examples/example_永.npz"
echo "  3. Record video: python mujoco_simulator.py <npz> --record video.mp4"
echo ""
