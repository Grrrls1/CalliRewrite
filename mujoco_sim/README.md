# MuJoCo 仿真模块 - CalliRewrite

## 📚 模块概述

**mujoco_sim** 是 CalliRewrite 系统的仿真模块，提供在 MuJoCo 物理引擎中测试和可视化书法轨迹的能力，无需真实机器人硬件。

**主要功能**:
- 🤖 Franka Panda 机器人完整仿真
- 🖌️ 真实笔刷物理建模（变形、墨水扩散）
- 💪 力控制和阻抗控制
- 📹 轨迹可视化和视频录制
- 📊 接触力和笔迹分析
- ⚡ 高性能渲染

---

## 🏗️ 架构

```
mujoco_sim/
├── models/
│   └── franka_panda.xml          # Franka Panda MJCF 模型
├── mujoco_simulator.py           # 基础仿真器
├── advanced_simulator.py         # 高级仿真器（力控制）
├── examples/                     # 示例脚本
│   ├── basic_demo.py
│   ├── force_control_demo.py
│   └── video_recording.py
└── outputs/                      # 仿真输出
    ├── calligraphy_result.png
    ├── trajectory_3d.png
    └── videos/
```

---

## 🚀 快速开始

### 安装依赖

```bash
# MuJoCo (3.0+)
pip install mujoco

# 其他依赖
pip install numpy opencv-python matplotlib

# 可选: 用于高级功能
pip install imageio imageio-ffmpeg
```

### 基础使用

```bash
# 运行基础仿真
cd mujoco_sim
python mujoco_simulator.py ../callibrate/examples/simple_line.npz --speed 0.1

# 录制视频
python mujoco_simulator.py ../callibrate/examples/example_永.npz \
    --record outputs/yong_video.mp4 --speed 0.05

# 高级力控制仿真
python advanced_simulator.py ../callibrate/examples/real_character_0.npz
```

---

## 📖 详细说明

### 1. 基础仿真器 (`mujoco_simulator.py`)

#### 特性

- ✅ 逆运动学 (IK) 求解
- ✅ 笛卡尔空间轨迹执行
- ✅ 接触检测和笔迹记录
- ✅ 实时 3D 可视化
- ✅ 视频录制

#### 使用示例

```python
from mujoco_simulator import FrankaCalligraphySimulator

# 创建仿真器
sim = FrankaCalligraphySimulator(
    model_path="models/franka_panda.xml",
    render_mode="human"
)

# 执行轨迹
sim.execute_trajectory(
    npz_path="path/to/trajectory.npz",
    speed=0.05,  # 0.05 m/s
    render=True
)

# 查看结果
# - outputs/calligraphy_result.png (画布)
# - outputs/trajectory_3d.png (3D轨迹图)
```

#### 命令行选项

```bash
python mujoco_simulator.py <npz_file> [options]

Options:
  --speed SPEED         Movement speed in m/s (default: 0.05)
  --no-render          Disable real-time rendering
  --record PATH        Record video to specified path
  --model PATH         Custom MuJoCo XML model path
```

---

### 2. 高级仿真器 (`advanced_simulator.py`)

#### 特性

- ✅ 阻抗控制 (力位混合控制)
- ✅ 真实笔刷变形模型
- ✅ 墨水扩散仿真
- ✅ 高分辨率画布 (2400×3360)
- ✅ 力反馈分析

#### 笔刷物理模型

```python
@dataclass
class BrushModel:
    stiffness: float = 5000.0        # N/m (刚度)
    damping: float = 50.0            # N·s/m (阻尼)
    max_deformation: float = 0.005   # m (最大变形 5mm)
    radius_base: float = 0.003       # m (基础半径 3mm)
    ink_flow_rate: float = 0.01      # 墨水流速
```

**物理方程**:

```
变形: δ = F / k                    (胡克定律)
阻尼力: F_d = c * v                (线性阻尼)
笔画半径: r = r_base + δ           (与变形相关)
墨水浓度: I = 255 * (1 - F/F_max)  (与力相关)
```

#### 阻抗控制

```python
# 空中: 纯位置控制
if contact_force < 0.1:
    control = K_p * (target_pos - current_pos)

# 接触: 力位混合控制
else:
    # 法向: 力控制
    normal_control = K_f * (target_force - contact_force)

    # 切向: 位置控制
    tangent_control = K_p * tangent_error

    control = normal_control + tangent_control
```

#### 使用示例

```python
from advanced_simulator import AdvancedCalligraphySimulator, BrushModel

# 自定义笔刷
brush = BrushModel(
    stiffness=8000.0,      # 更硬的笔刷
    damping=100.0,
    max_deformation=0.003
)

# 创建仿真器
sim = AdvancedCalligraphySimulator(
    model_path="models/franka_panda.xml",
    brush_model=brush,
    canvas_size=(3000, 4200)  # 超高分辨率
)

# 执行轨迹
sim.execute_trajectory_with_force_control(
    npz_path="path/to/trajectory.npz",
    render=True
)
```

---

## 🎨 MuJoCo 模型详解

### Franka Panda 模型 (`franka_panda.xml`)

#### 模型结构

```xml
<mujoco model="franka_panda_calligraphy">
  <worldbody>
    <!-- 纸面 -->
    <body name="paper" pos="0.5 0 0.01">
      <geom name="paper_surface" type="box"
            size="0.15 0.21 0.001" material="paper"/>
    </body>

    <!-- 机器人 -->
    <body name="panda_link0" pos="0 0 0">
      <!-- 7个关节 -->
      <joint name="joint1" ... range="-2.8973 2.8973"/>
      ...
      <joint name="joint7" ... range="-2.8973 2.8973"/>

      <!-- 末端执行器 + 笔刷 -->
      <body name="end_effector">
        <body name="brush_holder">
          <body name="brush_tip">
            <geom name="brush" type="sphere" size="0.003"/>
            <site name="brush_contact"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 执行器 -->
  <actuator>
    <motor name="motor1" joint="joint1" gear="100"/>
    ...
  </actuator>

  <!-- 传感器 -->
  <sensor>
    <framepos name="ee_pos" objtype="site" objname="ee_site"/>
    <touch name="brush_touch" site="brush_contact"/>
    <force name="brush_force" site="brush_contact"/>
  </sensor>
</mujoco>
```

#### 关键参数

| 参数 | 值 | 说明 |
|------|----|----|
| 时间步长 | 0.002s | 500Hz 仿真频率 |
| 控制频率 | 500Hz | 与时间步长匹配 |
| 笔刷半径 | 0.003m | 3mm 球形笔头 |
| 纸面尺寸 | 0.3×0.42m | A3 纸 |
| 摩擦系数 | 0.9 | 笔-纸摩擦 |

---

## 📊 仿真输出

### 1. 画布图像

```
outputs/calligraphy_result.png
```

- 分辨率: 1200×1680 (基础) / 2400×3360 (高级)
- 格式: PNG，RGB+Alpha
- 包含所有接触点的笔迹

### 2. 3D 轨迹图

```
outputs/trajectory_3d.png
```

包含三个子图:
- **3D 轨迹**: 完整空间路径
- **俯视图 (X-Y)**: 平面笔迹
- **Z 高度曲线**: 笔压变化

### 3. 视频

```
outputs/videos/<timestamp>.mp4
```

- 帧率: 30 FPS
- 分辨率: 1280×720
- 编码: H.264
- 包含机器人运动和笔迹绘制过程

---

## 🔬 仿真分析

### 接触力分析

```python
# 在仿真过程中记录
contact_forces = []

for step in trajectory:
    _, force = sim.get_brush_contact()
    contact_forces.append(force)

# 统计
print(f"平均接触力: {np.mean(contact_forces):.2f} N")
print(f"最大接触力: {np.max(contact_forces):.2f} N")
print(f"接触点数: {np.sum(np.array(contact_forces) > 0.1)}")
```

### 轨迹平滑度

```python
# 加速度分析
positions = np.array([pos for pos, _, _, _ in sim.ink_traces])
velocities = np.diff(positions, axis=0) / sim.dt
accelerations = np.diff(velocities, axis=0) / sim.dt

smoothness = np.mean(np.linalg.norm(accelerations, axis=1))
print(f"轨迹平滑度: {smoothness:.4f} m/s²")
```

---

## 🎯 使用场景

### 场景 1: 轨迹验证

在真实机器人上执行前，先在 MuJoCo 中验证:

```bash
# 1. 生成 NPZ 文件
cd callibrate
python calibrate.py --mode convert --tool brush \
    --input ../rl_finetune/results/永.npy \
    --output test_trajectory.npz

# 控制点太少/想更密：用 --max-step-mm 增密（例如 0.3mm）
# python calibrate.py --mode convert --tool brush \
#     --input ../rl_finetune/results/永.npy \
#     --output test_trajectory_dense.npz \
#     --max-step-mm 0.3

# 2. MuJoCo 仿真验证
cd ../mujoco_sim
python mujoco_simulator.py ../callibrate/test_trajectory.npz

# 3. 检查结果
# - 无碰撞
# - 笔迹质量
# - 轨迹平滑

# 4. 部署到真实机器人
cd ../callibrate
python RoboControl.py test_trajectory.npz 172.16.0.2 0.05
```

### 场景 2: 参数优化

调整校准参数并观察效果:

```python
# 测试不同的 alpha (字符大小)
for alpha in [0.03, 0.04, 0.05]:
    npz_path = f"trajectory_alpha_{alpha}.npz"
    # convert_rl_to_npz(..., alpha=alpha, ...)

    sim.execute_trajectory(npz_path)
    # 比较画布输出
```

### 场景 3: 力控制研究

```python
from advanced_simulator import AdvancedCalligraphySimulator

sim = AdvancedCalligraphySimulator("models/franka_panda.xml")

# 测试不同目标力
for target_force in [1.0, 2.0, 3.0, 5.0]:
    sim.target_force = target_force
    sim.execute_trajectory_with_force_control(npz_path)

    # 分析笔画粗细和质量
```

---

## 🔧 自定义和扩展

### 添加新的机器人

创建新的 MJCF 模型:

```xml
<!-- models/my_robot.xml -->
<mujoco model="my_robot_calligraphy">
  <worldbody>
    <!-- 你的机器人定义 -->
  </worldbody>
</mujoco>
```

使用:

```python
sim = FrankaCalligraphySimulator(
    model_path="models/my_robot.xml"
)
```

### 自定义笔刷

```python
class CustomBrush(BrushModel):
    def __init__(self):
        super().__init__(
            stiffness=10000.0,      # 硬笔
            damping=20.0,
            radius_base=0.001,      # 细笔尖
            ink_flow_rate=0.005     # 慢墨水流动
        )
```

### 添加新的传感器

在 XML 中:

```xml
<sensor>
  <framepos name="custom_sensor" objtype="site" objname="my_site"/>
  <torque name="joint_torque" joint="joint1"/>
</sensor>
```

在代码中读取:

```python
sensor_id = mujoco.mj_name2id(
    self.model, mujoco.mjtObj.mjOBJ_SENSOR, "custom_sensor"
)
sensor_data = self.data.sensordata[sensor_id]
```

---

## 🐛 常见问题

### Q: IK 求解失败?

**A**: 检查目标位置是否在工作空间内:

```python
# Franka Panda 工作空间
x_range = [0.3, 0.85]
y_range = [-0.5, 0.5]
z_range = [0.0, 0.8]

# 验证
if not (x_range[0] <= x <= x_range[1] and ...):
    print("Target out of workspace!")
```

### Q: 仿真运行慢?

**A**: 优化选项:

```python
# 1. 禁用可视化
sim.execute_trajectory(npz_path, render=False)

# 2. 降低控制频率
sim.control_freq = 200  # 从 500 降到 200 Hz

# 3. 使用更大的时间步长
# 在 XML 中: <option timestep="0.005" ...>
```

### Q: 笔迹不连续?

**A**: 增加轨迹密度:

```python
# 在转换 NPZ 时插值
from scipy.interpolate import interp1d

# 对轨迹进行上采样
x_interp = interp1d(range(len(x)), x, kind='cubic')
x_dense = x_interp(np.linspace(0, len(x)-1, len(x)*5))
```

---

## 📚 API 参考

### FrankaCalligraphySimulator

```python
class FrankaCalligraphySimulator:
    def __init__(model_path, render_mode, camera_distance, ...)
    def reset(qpos=None)
    def get_ee_pose() -> (position, quaternion)
    def get_brush_contact() -> (position, force)
    def inverse_kinematics(target_pos, max_iter, tol) -> success
    def move_to_position(target_pos, speed, wait_time)
    def execute_trajectory(npz_path, speed, render)
    def record_video(npz_path, output_path, speed, fps)
    def close()
```

### AdvancedCalligraphySimulator

```python
class AdvancedCalligraphySimulator:
    def __init__(model_path, brush_model, canvas_size)
    def impedance_control(target_pos, target_force) -> control_input
    def update_brush_deformation(contact_force)
    def draw_ink(pos, contact_force)
    def execute_trajectory_with_force_control(npz_path, render)
```

---

## 📈 性能基准

| 指标 | 基础仿真器 | 高级仿真器 |
|------|------------|------------|
| **IK 求解** | ~0.5 ms | ~0.5 ms |
| **步进速度** | 500 Hz | 500 Hz |
| **渲染 FPS** | 30+ | 25+ |
| **内存占用** | ~200 MB | ~300 MB |
| **GPU 加速** | 支持 | 支持 |

---

## 🔗 相关资源

- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)
- [Franka Panda 规格](https://www.franka.de/robot-system)
- [CalliRewrite 论文](https://arxiv.org/abs/2024.xxxxx)

---

**总结**: MuJoCo 仿真模块提供了一个安全、高效的环境来测试和优化书法轨迹，是真实机器人部署前的重要验证工具！
