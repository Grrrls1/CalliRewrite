# NPZ 文件格式详解 - CalliRewrite 系统

## 📊 文件格式总览

在 CalliRewrite 系统中，有两种关键的 NumPy 文件格式：

| 格式 | 来源 | 用途 | 包含数据 |
|------|------|------|----------|
| **`.npy`** | RL 强化学习输出 | 虚拟环境中的笔画状态 | `[p_t, x, y, r, ...]` |
| **`.npz`** | 校准转换输出 | 机器人控制点 | `{pos_3d_x, pos_3d_y, pos_3d_z}` |

---

## 🔄 完整数据流

```
书法图像
   ↓
[Seq Extract] LSTM 粗分割
   ↓
笔画序列.npy (虚拟坐标)
   ↓
[RL Finetune] 强化学习优化
   ↓
优化后.npy (虚拟坐标 + 笔画半径)
   ↓
[calibrate.py] convert_rl_to_npz() ← 使用校准函数
   ↓
机器人控制.npz (真实3D坐标)
   ↓
[RoboControl.py] 机器人执行
   ↓
实物书法作品 ✍️
```

---

## 📁 `.npy` 文件格式（RL 输出）

### 数据结构

```python
data.shape = (N, 7)  # N个控制点，每个7维

# 每一行的格式：
[p_t, x, y, r, col1, col2, col3]
```

### 列的含义

| 列索引 | 名称 | 范围 | 含义 |
|--------|------|------|------|
| 0 | `p_t` | 0 或 1 | 笔状态：0=继续画，1=新笔画起点 |
| 1 | `x` | 0-255 | 虚拟X坐标（像素） |
| 2 | `y` | 0-255 | 虚拟Y坐标（像素） |
| 3 | `r` | 0-255 | 笔画半径（像素） |
| 4-6 | - | - | 其他状态（颜色、角度等） |

### 实际示例

```python
import numpy as np

# 加载 RL 输出
data = np.load('./rl_finetune/data/test_data/0.npy')
print(data.shape)  # (27, 7)

# 查看前几行
print(data[:5])
# [[  1.00  53.50  54.50  54.78  67.53  66.45  78.12]  ← 新笔画起点
#  [  0.00  53.50  54.50  54.78  67.53  66.45  78.12]  ← 继续画
#  [  0.00  66.07  78.41  68.59  76.82 127.93  76.32]  ← 继续画
#  [  0.00 127.83  76.12 144.73  75.89 185.29  75.59]  ← 继续画
#  [  1.00 110.08 158.07 118.02 154.86 219.59 152.45]] ← 新笔画起点
```

### 关键特性

1. **`p_t` 标志位**：
   - `p_t = 1`：新笔画的第一个点（需要抬笔、移动、下笔）
   - `p_t = 0`：当前笔画的连续点（笔不离纸）

2. **虚拟坐标系**：
   - X, Y 坐标在图像像素空间（通常 0-255）
   - 需要缩放到真实物理尺寸

3. **笔画半径 `r`**：
   - 虚拟环境中的笔画粗细
   - 需要通过校准函数转换为机器人 Z 轴高度

---

## 📦 `.npz` 文件格式（机器人控制）

### 数据结构

`.npz` 是 NumPy 的压缩归档格式，包含多个命名数组：

```python
npz_file = {
    'pos_3d_x': np.array([...]),  # X 坐标数组（米）
    'pos_3d_y': np.array([...]),  # Y 坐标数组（米）
    'pos_3d_z': np.array([...]),  # Z 坐标数组（米）
}

# 所有数组长度相同，表示 N 个控制点
```

### 坐标系统

| 轴 | 单位 | 范围（示例） | 含义 |
|----|------|-------------|------|
| `pos_3d_x` | 米 | 0.0 - 0.08 | 左右方向（字符宽度） |
| `pos_3d_y` | 米 | 0.0 - 0.08 | 前后方向（字符高度） |
| `pos_3d_z` | 米 | -0.096 - 0.05 | 上下方向（笔压 + 抬笔） |

### 实际示例

```python
import numpy as np

# 加载机器人控制文件
data = np.load('./data/永.npz')

print(list(data.keys()))
# ['pos_3d_x', 'pos_3d_y', 'pos_3d_z']

x = data['pos_3d_x']
y = data['pos_3d_y']
z = data['pos_3d_z']

print(f"控制点数量: {len(x)}")
# 控制点数量: 150

print(f"X 范围: [{x.min():.4f}, {x.max():.4f}] 米")
# X 范围: [0.0000, 0.0400] 米

print(f"Z 范围: [{z.min():.4f}, {z.max():.4f}] 米")
# Z 范围: [-0.0960, 0.0500] 米
```

### Z 轴含义

```
Z 轴高度 (米)
    ^
    |
0.05 |  ████████  抬笔状态（离开纸面）
    |
0.01 |  ▓▓▓▓▓▓▓▓  笔尖轻触
    |
0.00 |  ━━━━━━━━  纸面高度
    |
-0.002|  ▒▒▒▒▒▒▒▒  轻压
    |
-0.006|  ░░░░░░░░  重压（最大笔画宽度）
    |
-0.09 |            工作台偏移补偿
```

---

## 🔧 转换过程详解

### 在 `calibrate.py` 中的转换函数

```python
def convert_rl_to_npz(npy_path, output_path, calibration_func,
                     alpha=0.04, beta=0.5, style_type=0):
    """
    将 RL 虚拟状态转换为机器人真实控制点

    参数:
        npy_path: RL 输出 .npy 文件
        output_path: 保存的 .npz 文件
        calibration_func: r -> z 的校准函数
        alpha: 字符大小（米），如 0.04 = 4cm 宽的字
        beta: 笔画宽度调节系数，如 0.5 = 正常，1.0 = 加粗
        style_type: 0=隶书（回锋），1=楷书（斜入）
    """
```

### 转换步骤

#### 步骤 1: 加载 RL 数据

```python
data = np.load(npy_path)  # Shape: (N, 7)

for i in range(data.shape[0]):
    p_t, x, y, r = data[i][:4]
    # p_t: 笔状态
    # x, y: 虚拟坐标（0-255）
    # r: 虚拟半径（0-255）
```

#### 步骤 2: 缩放到真实尺寸

```python
# 将虚拟坐标 (0-255) 缩放到真实米制
x_real = (x / 255) * alpha        # alpha = 0.04 → 4cm 字符
y_real = (y / 255) * alpha
r_real = (r / 255) * alpha * beta  # beta = 0.5 → 笔画调节

# 示例：
# x = 127.5 (虚拟中点)
# alpha = 0.04 (4cm)
# x_real = (127.5/255) * 0.04 = 0.02 米 = 2cm (字符中心)
```

#### 步骤 3: 半径转高度（核心校准）

```python
# 使用校准函数将笔画半径转换为 Z 轴高度
z = calibration_func(r_real)

# 校准函数示例（分段线性）:
def func_brush(r):
    if r <= 0.00077:
        z = -5.97 * (r - 0.00077) + 0.00435
    elif r <= 0.00178:
        z = -1.54 * (r - 0.00077) + 0.00435
    # ... 更多段
    return z

# 补偿工作台高度
z = z - 0.09  # 工作台偏移
```

#### 步骤 4: 处理笔画起点（书法风格）

```python
if p_t == 1:  # 新笔画起点
    if style_type == 0:  # 隶书：从正上方进入 + 回锋
        # 1. 抬笔到起点上方
        record_x.append(x_real)
        record_y.append(y_real)
        record_z.append(0.05)  # 5cm 高抬笔

        # 2. 下笔
        record_x.append(x_real)
        record_y.append(y_real)
        record_z.append(z)

        # 3. 回锋（向下一点方向反向移动）
        next_point = data[i+1]
        direction = (next_point[1:3] - data[i][1:3]) / norm(...)
        record_x.append(x_real - 2*r_real*direction[0])
        record_y.append(y_real - 2*r_real*direction[1])
        record_z.append(z)

        # 4. 回到起点
        record_x.append(x_real)
        record_y.append(y_real)
        record_z.append(z)

    elif style_type == 1:  # 楷书：从左上角斜入
        # 1. 从左上角接近
        record_x.append(x_real - 2*r_real)
        record_y.append(y_real - 2*r_real)
        record_z.append(0.05)

        # 2. 斜向下笔到起点
        record_x.append(x_real)
        record_y.append(y_real)
        record_z.append(z)

else:  # p_t == 0，正常笔画点
    record_x.append(x_real)
    record_y.append(y_real)
    record_z.append(z)
```

#### 步骤 5: 渐进抬笔

```python
# 在轨迹末尾添加渐进抬笔，避免甩墨
for _ in range(5):
    record_x.append(record_x[-1])  # X 不变
    record_y.append(record_y[-1])  # Y 不变
    record_z.append(record_z[-1] + 0.015)  # Z 逐渐上升
```

#### 步骤 6: 保存为 NPZ

```python
np.savez(output_path,
         pos_3d_x=record_x,
         pos_3d_y=record_y,
         pos_3d_z=record_z)
```

---

## 📐 坐标转换示例

### 完整示例：从虚拟到真实

```python
# ===== RL 输出 (.npy) =====
# 虚拟坐标系：256x256 像素
p_t = 1      # 新笔画起点
x = 127.5    # 中心点 X
y = 191.25   # 3/4 高度 Y
r = 25.5     # 笔画半径（像素）

# ===== 参数设置 =====
alpha = 0.04   # 写 4cm 宽的字
beta = 0.5     # 笔画正常粗细

# ===== 转换计算 =====
# 1. 归一化 (0-1)
x_norm = 127.5 / 255 = 0.5
y_norm = 191.25 / 255 = 0.75
r_norm = 25.5 / 255 = 0.1

# 2. 缩放到真实尺寸（米）
x_real = 0.5 * 0.04 = 0.02 米 = 2cm
y_real = 0.75 * 0.04 = 0.03 米 = 3cm
r_real = 0.1 * 0.04 * 0.5 = 0.002 米 = 2mm

# 3. 半径转高度（使用校准函数）
z = func_brush(0.002)
z ≈ -0.0015 米  # 查表或插值得到

# 4. 工作台补偿
z_final = -0.0015 - 0.09 = -0.0915 米

# ===== 最终机器人控制点 (.npz) =====
pos_3d_x = 0.02 米   # 2cm (字符中心偏右)
pos_3d_y = 0.03 米   # 3cm (字符 3/4 高度)
pos_3d_z = -0.0915 米 # 工作台下方 9.15cm（笔压约 1.5mm）
```

---

## 🎨 书法风格的实现

### 隶书（Lishu, style_type=0）

```
      ③↑抬笔
       |
①抬笔→ ②↓下笔 ④←回锋 ⑤→前进
           ↓
         ━━━━━━━━━━→ 笔画主体
```

**特点**：
- 从正上方垂直下笔
- 回锋起笔（向后再向前）
- 笔画沉稳有力

### 楷书（Kaishu, style_type=1）

```
①抬笔在左上
   ╲
    ╲ ②斜向下笔
     ╲
      ━━━━━━━━━→ 笔画主体
```

**特点**：
- 从左上角斜向进入
- 自然流畅
- 适合快速书写

---

## 🔍 调试技巧

### 1. 检查 .npy 文件

```python
import numpy as np

data = np.load('your_file.npy')
print(f"Shape: {data.shape}")
print(f"笔画起点数量: {np.sum(data[:, 0] == 1)}")
print(f"X 范围: [{data[:, 1].min()}, {data[:, 1].max()}]")
print(f"Y 范围: [{data[:, 2].min()}, {data[:, 2].max()}]")
print(f"R 范围: [{data[:, 3].min()}, {data[:, 3].max()}]")
```

### 2. 检查 .npz 文件

```python
import numpy as np

data = np.load('your_file.npz')
x, y, z = data['pos_3d_x'], data['pos_3d_y'], data['pos_3d_z']

print(f"控制点数量: {len(x)}")
print(f"X 范围: [{x.min():.4f}, {x.max():.4f}] 米")
print(f"Y 范围: [{y.min():.4f}, {y.max():.4f}] 米")
print(f"Z 范围: [{z.min():.4f}, {z.max():.4f}] 米")

# 检查是否有异常值
if z.max() > 0.1:
    print("⚠️  警告：Z 轴过高，可能抬笔过度")
if z.min() < -0.1:
    print("⚠️  警告：Z 轴过低，可能压坏笔或纸")
```

### 3. 可视化轨迹

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load('your_file.npz')
x, y, z = data['pos_3d_x'], data['pos_3d_y'], data['pos_3d_z']

fig = plt.figure(figsize=(15, 5))

# 3D 视图
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(x, y, z, 'b-', linewidth=0.5)
ax1.scatter(x[0], y[0], z[0], c='green', s=100, label='起点')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D 轨迹')

# 俯视图 (X-Y)
ax2 = fig.add_subplot(132)
ax2.plot(x, y, 'b-', linewidth=0.5)
ax2.scatter(x[0], y[0], c='green', s=100)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('俯视图')
ax2.axis('equal')

# Z 高度变化
ax3 = fig.add_subplot(133)
ax3.plot(z, 'b-', linewidth=1)
ax3.axhline(y=0, color='r', linestyle='--', label='纸面')
ax3.set_xlabel('点序号')
ax3.set_ylabel('Z (m)')
ax3.set_title('Z 轴高度变化')
ax3.legend()

plt.tight_layout()
plt.show()
```

---

## 💡 常见问题

### Q: 为什么需要两种文件格式？

**A**: 分工明确：
- `.npy` 是 RL 在虚拟环境中的输出，与物理机器人无关
- `.npz` 是经过校准转换后的真实物理坐标，直接用于机器人控制
- 这样可以在不同机器人上复用同一个 `.npy` 文件，只需重新校准

### Q: `alpha` 和 `beta` 参数如何选择？

**A**:
- `alpha`：根据纸张大小和字符复杂度
  - 简单字（如"永"）：0.04-0.06 米 (4-6cm)
  - 复杂字：0.08-0.10 米 (8-10cm)
- `beta`：根据艺术效果
  - 瘦金体：0.3-0.5（细笔画）
  - 正常：0.5-0.7
  - 厚重：0.8-1.0

### Q: Z 轴的 -0.09 偏移是什么？

**A**: 这是工作台高度补偿
- 校准函数输出的 z 是相对于笔尖接触点的高度
- 实际机器人需要知道相对于基座的绝对高度
- `-0.09` 表示纸面在机器人基座下方 9cm 处
- **这个值需要根据你的实际设置调整！**

### Q: 如何验证转换是否正确？

**A**: 三步验证：
1. **数值检查**：Z 范围应该在 -0.1 到 0.1 米之间
2. **可视化**：使用上面的绘图代码检查轨迹形状
3. **仿真测试**：先在虚拟环境或空中执行（Z+0.1），确认运动正确

---

## 📚 相关函数快速参考

| 函数 | 输入 | 输出 | 功能 |
|------|------|------|------|
| `generate_calibration_data()` | 工具类型 | `zs`, NPZ | 生成校准测试轨迹 |
| `fit_calibration_function()` | 宽度, z高度 | 校准函数 | 拟合 r→z 映射 |
| `convert_rl_to_npz()` | NPY + 校准函数 | NPZ | RL 状态 → 机器人控制 |
| `Control()` | NPZ 文件 | 执行结果 | 在机器人上执行轨迹 |

---

**总结**：`.npz` 文件是连接虚拟 AI 和真实机器人的关键桥梁，通过校准函数将虚拟笔画参数精确转换为物理机器人的 3D 坐标！
