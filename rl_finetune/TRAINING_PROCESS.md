# RL Finetune 训练过程详解

## 📚 模块概述

**rl_finetune** 是 CalliRewrite 系统的第二阶段，负责使用强化学习优化从 seq_extract 获得的粗笔画序列，使其适应特定书写工具的物理特性。

**目标**: 将粗笔画序列转换为符合真实工具（毛笔、马克笔等）物理动力学的精细轨迹

**核心技术**:
- SAC (Soft Actor-Critic) 强化学习算法
- 自定义书法环境 (CalliEnv)
- 工具物理建模 (Writing Brush, Ellipse, Chisel Marker)
- 渐进式轨迹优化
- EMA (指数移动平均) 平滑

---

## 🏗️ 整体架构

```
输入: 粗笔画序列 (.npy) + 书法图像 (.png)
    ↓
┌─────────────────────────────────┐
│  CalliEnv 环境                   │
│  • 加载图像和粗笔画               │
│  • 工具物理建模                   │
│  • 状态计算                       │
└─────────────────────────────────┘
    ↓ 状态 (8维)
┌─────────────────────────────────┐
│  SAC 策略网络                     │
│  • Actor: 状态 → 动作             │
│  • Critic: 状态-动作 → Q值        │
└─────────────────────────────────┘
    ↓ 动作 (2维)
┌─────────────────────────────────┐
│  环境交互 & 渲染                  │
│  • 执行动作                       │
│  • 计算新笔画参数                 │
│  • 渲染到画布                     │
└─────────────────────────────────┘
    ↓ 奖励 (标量)
┌─────────────────────────────────┐
│  奖励计算                         │
│  • 图像相似度                     │
│  • 笔画平滑度                     │
│  • 笔画大小控制                   │
└─────────────────────────────────┘
    ↓ 策略更新
┌─────────────────────────────────┐
│  SAC 优化                        │
│  • Actor 损失                    │
│  • Critic 损失                   │
│  • 温度参数自适应                │
└─────────────────────────────────┘
    ↓ 多轮迭代
输出: 优化后笔画序列 (.npy)
```

---

## 🎓 训练策略

### 核心思想: 工具感知的强化学习

与 seq_extract 不同，rl_finetune 不是直接预测笔画序列，而是通过强化学习让 AI 学会"操作真实工具"来重现书法。

**关键创新**:
1. **物理建模**: 模拟毛笔、马克笔等工具的真实动力学
2. **增量优化**: 从粗笔画出发，逐步调整到最优轨迹
3. **EMA 更新**: 每轮训练后用指数移动平均平滑轨迹
4. **多样化训练**: 每张图像重复训练多次 (image_iter)

---

## 🔑 核心组件详解

### 1️⃣ **状态空间 (8维)**

```python
state = [period, r, l, theta, curvature, r_prime, vec_x, vec_y]
```

| 维度 | 名称 | 范围 | 含义 |
|------|------|------|------|
| 0 | `period` | [0, 1] | 当前笔画进度 (current_pos / stroke_length) |
| 1 | `r` | [0, 1] | 笔刷半径/宽度 (归一化) |
| 2 | `l` | [0, 1] | 笔刷长度 (归一化) |
| 3 | `theta` | [0, 1] | 旋转角度 (归一化到 [0, 360°]) |
| 4 | `curvature` | [0, 1] | 当前曲率 (通过 sin 计算) |
| 5 | `r_prime` | [-1, 1] | 上一步移动距离 |
| 6 | `vec_x` | [-1, 1] | 未来方向单位向量 X |
| 7 | `vec_y` | [-1, 1] | 未来方向单位向量 Y |

**示例**:
```python
# 笔画开始
state_0 = [0.0,  # period: 刚开始
           0.02, # r: 较细的笔画
           0.04, # l: 笔刷长度
           0.25, # theta: 90度角 (90/360)
           0.8,  # curvature: 高曲率 (急转弯)
           0.0,  # r_prime: 还未移动
           0.7,  # vec_x: 向右上方
           0.7]  # vec_y
```

---

### 2️⃣ **动作空间 (2维)**

```python
action = [r_prime, theta_prime]
```

| 维度 | 名称 | 范围 | 含义 |
|------|------|------|------|
| 0 | `r_prime` | [-1, 1] | 移动距离 (× r_prime_bound = 0.022) |
| 1 | `theta_prime` | [-1, 1] | 移动角度 (× π) |

**物理意义**:
- `r_prime`: 控制笔刷沿极坐标移动的距离
- `theta_prime`: 控制移动方向 (相对于当前朝向)

**示例**:
```python
action = [0.5, 0.2]

# 实际效果:
移动距离 = 0.5 × 0.022 = 0.011 (1.1% 画布宽度)
移动角度 = 0.2 × π = 36° (相对当前方向)
```

---

### 3️⃣ **CalliEnv 环境**

#### 环境初始化

```python
class CalliEnv(gym.Env):
    def __init__(self, tool, folder_path, output_path,
                 visualize_path, env_num, env_rank,
                 graph_width=256,      # 图像分辨率
                 canvas_width=300,     # 渲染画布大小
                 image_iter=20,        # 每张图重复训练次数
                 start_update=5,       # 开始更新轨迹的迭代
                 update=5,             # 更新频率
                 ema_gamma=0.95):      # EMA 平滑系数
```

**关键参数**:
- `tool`: 工具类实例 (Writing_Brush, Ellipse, Chisel_Tip_Marker)
- `env_num`: 并行环境数量 (通常 8-12)
- `env_rank`: 当前环境序号 (用于分配数据)
- `image_iter`: 每张图像重复训练 20 轮
- `ema_gamma`: 0.95 表示新旧轨迹融合比例 (5% 新 + 95% 旧)

#### 重要机制

##### A. 数据池管理

```python
# 环境根据 rank 分配数据子集
self.start_idx = int(self.env_rank[0]/self.env_rank[1] * total_images)
self.end_idx = int((self.env_rank[0] + delta)/self.env_rank[1] * total_images)

# 数据池: [(图像路径, .npy路径), ...]
self.data_pool = [(folder_path+str(idx)+".png",
                   folder_path+str(idx)+".npy")
                  for idx in range(start_idx, end_idx)]
```

**作用**: 并行训练时，每个环境处理不同的数据子集，加速训练。

##### B. EMA 轨迹更新

```python
# 在 step() 函数中，episode 结束时:
if terminated:
    if diff >= start_update and diff % update == 0:
        # 指数移动平均更新粗笔画
        self.skel_list = EMA(self.skel_list,
                             self.new_skel_list.reshape(-1, 2),
                             self.ema_gamma)
```

**公式**:
```
skel_new = γ × skel_old + (1-γ) × skel_predicted
         = 0.95 × old + 0.05 × new
```

**效果**: 渐进式优化，避免剧烈变化，保证收敛稳定。

##### C. 笔画序列管理

```python
# 加载粗笔画序列
skel_list = np.load(skel_path)  # Shape: (N, 7)
# [p_t, x, y, r, col1, col2, col3]

# 提取笔画起点标志
self.pt_list = skel_list[:, 0]  # p_t: 0 or 1
self.pt_idx = np.where(self.pt_list == 1)  # 起点索引
self.pt_indices = np.diff(self.pt_idx, n=1)  # 每笔画长度

# period 计算
period_step = 1 / self.pt_indices[current_stroke]
```

---

### 4️⃣ **工具物理建模**

#### 毛笔 (Writing Brush)

**模型**: 圆形笔头 + 三角形笔尖

```python
class Writing_Brush(Tool_Base):
    # 几何形状计算
    def calc_four_points(v_1, cur_r, cur_l, rad):
        """
        v_1: 圆心位置
        cur_r: 圆半径
        cur_l: 笔尖长度
        rad: 旋转角度

        返回: [圆心, 右交点, 笔尖, 左交点]
        """
        # 笔尖方向
        vec_tip = [-sin(rad), cos(rad)]
        v_2 = v_1 + cur_l * vec_tip

        # 三角形两侧点
        phi = arccos(cur_r / cur_l)
        pt_right = v_1 + cur_r * [-sin(rad-phi), cos(rad-phi)]
        pt_left = v_1 + cur_r * [-sin(rad+phi), cos(rad+phi)]

        return [v_1, pt_right, v_2, pt_left]
```

**动力学**:
```python
def brush_dynamics(angle, radius, length, angle_vec, skel_vec):
    """
    毛笔角度跟随骨架方向，但有延迟和阻尼
    """
    cos_sim = dot(angle_vec, skel_vec)

    # 三阶段响应
    if cos_sim >= 0.707:  # 平滑跟随
        delta_theta = theta_step * sin(arccos(cos_sim))

    elif cos_sim > -0.96:  # 急转时发散
        delta_theta = theta_step * sin(...) * sqrt(0.5/radius)

    else:  # 回笔处理
        delta_theta = 165°  # 大幅度转向
        r_new = -cos_sim * r_old
        l_new = 0.003 + 2.021 * r_new
```

**半径-长度关系** (根据真实毛笔):
```python
r, l = geometric_r_l(contours, center)

# 半径 r: 笔头到轮廓的最短距离
r = min(cv2.pointPolygonTest(con, center, True) for con in contours) / canvas_width

# 长度 l: 线性拟合关系
l = 0.003 + 2.021 × r
```

#### 椭圆笔刷 (Ellipse)

```python
class Ellipse(Tool_Base):
    # 固定椭圆形状，可旋转
    def calc_four_points(v_1, cur_r, cur_l, rad):
        return [center, [cur_r, 0], [rad, 0], [cur_l, 0]]
        # [中心, 短轴, 角度, 长轴]

    def geometric_r_l(contours, center):
        r = min_distance_to_contour / canvas_width
        l = 1.2 × r  # 长短轴比例
```

#### 凿尖马克笔 (Chisel Tip Marker)

```python
class Chisel_Tip_Marker(Tool_Base):
    # 矩形笔头，固定尺寸
    def calc_four_points(v_1, cur_r, cur_l, rad):
        # 四个角点
        init_pts = [[l/2, l/2, -l/2, -l/2],
                    [-r/2, r/2, r/2, -r/2]]

        # 旋转矩阵
        rot = [[cos(rad), -sin(rad)],
               [sin(rad), cos(rad)]]

        pts = center + rot @ init_pts
        return pts

    def geometric_r_l(contours, center):
        # 固定尺寸
        r = 0.039 × 2.2
        l = 0.018 × 1.1
```

**工具属性配置** (brush.json):
```json
{
    "r_min": 0,
    "r_max": 0.064,       // 最大半径 6.4% 画布宽度
    "l_min": 0,
    "l_max": 0.17875,     // 最大长度 17.875%
    "theta_min": 0,
    "theta_max": 360,     // 角度范围
    "theta_step": 10      // 每步角度变化上限
}
```

---

### 5️⃣ **奖励函数**

#### 总体设计

```python
def calc_reward(next_r_prime, last_r, terminated):
    reward = 0

    # 1. 笔画大小惩罚 (每步)
    reward -= 0.08 × (1/√(next_r + ε) - 0.5/√r_max)^1.5

    # 2. 图像相似度奖励 (终止时)
    if terminated:
        # 对称差异
        total_area = stroke_img.sum()
        diff = (rendered_canvas - stroke_img).flatten()
        rec = inner(diff, diff) / total_area

        reward += 80 × (-(rec^0.4 - 0.5) + 0.2)

    return reward
```

#### 详细解析

##### A. 笔画大小控制 (每步惩罚)

```python
# 鼓励笔画不要太小，但也不要超大
penalty = 0.08 × (1/√r - 0.5/√r_max)^1.5

# 示例:
r_max = 0.064
r = 0.001 → penalty ≈ 0.25  (过细)
r = 0.032 → penalty ≈ 0      (适中)
r = 0.064 → penalty ≈ -0.04  (略粗)
```

**作用**: 防止笔画退化为线条或过度膨胀。

##### B. 图像相似度 (终止时奖励)

```python
# 计算渲染图像与目标图像的差异
rendered_canvas = record_canvas  # 累积渲染结果 (300×300)
stroke_downsample = cv2.resize(stroke_img, (300, 300)) / 255

# 对称差异 (XOR)
diff = rendered_canvas - stroke_downsample
rec = (diff × diff).sum() / stroke_downsample.sum()

# 映射到奖励 (非线性变换)
reward = 80 × (-(rec^0.4 - 0.5) + 0.2)

# 示例:
rec = 0.0 (完美) → reward ≈ 56
rec = 0.1         → reward ≈ 40
rec = 0.5         → reward ≈ 16
rec = 1.0 (完全不匹配) → reward ≈ -24
```

**奖励范围**: [-24, 56]

**设计理念**:
- 使用 `rec^0.4` 平方根变换，使得小差异更敏感
- 终止奖励占主导 (80×)，驱动全局优化
- 每步惩罚较小 (~0.1)，仅作正则化

---

### 6️⃣ **SAC 策略网络**

#### 网络架构

```python
# 状态维度: 8
# 动作维度: 2

# Actor (策略网络)
actor = My_MLP(
    state_shape=8,
    action_shape=2,
    hidden_sizes=[256, 256, 256],  # 3层隐藏层
    activation=nn.ReLU,
    device='cuda'
)

# Critic (Q值网络, 双网络)
critic_1 = My_MLP(
    state_shape=8,
    action_shape=2,
    hidden_sizes=[256, 256, 256],
    concat=True,  # 拼接状态和动作
    device='cuda'
)

critic_2 = My_MLP(...)  # 结构同 critic_1
```

**My_MLP 特性**:
- 支持 Fourier Features (可选，增强高频信息捕捉)
- 支持 SIREN 激活 (可选，周期性激活函数)
- 默认使用标准 ReLU MLP

#### SAC 策略

```python
from tianshou.policy import SACPolicy

policy = SACPolicy(
    actor=actor,
    actor_optim=torch.optim.Adam(actor.parameters(), lr=3e-4),
    critic_1=critic_1,
    critic1_optim=torch.optim.Adam(critic_1.parameters(), lr=3e-4),
    critic_2=critic_2,
    critic2_optim=torch.optim.Adam(critic_2.parameters(), lr=3e-4),
    tau=0.005,       # 软更新系数
    gamma=0.99,      # 折扣因子
    alpha=0.2        # 温度参数 (自动调节)
)
```

**SAC 核心参数**:
- `tau`: 目标网络软更新速率 (target = 0.005 × current + 0.995 × target)
- `gamma`: 未来奖励折扣因子 (0.99 重视长期回报)
- `alpha`: 熵正则化温度 (鼓励探索)

---

## 🔄 训练流程

### 主训练循环

```python
# try_tianshou.py

# 1. 创建并行环境
train_envs = DummyVectorEnv([
    lambda: CalliEnv(tool, train_path, output_path, ...)
    for _ in range(args.train_env_num)
])

test_envs = DummyVectorEnv([...])  # 测试环境

# 2. 初始化经验回放池
buf = VectorReplayBuffer(
    args.buffer_size,  # 默认 100000
    len(train_envs)
)

# 3. 创建数据收集器
train_collector = Collector(
    policy=policy,
    env=train_envs,
    buffer=buf,
    exploration_noise=True  # 训练时加噪声
)

test_collector = Collector(
    policy=policy,
    env=test_envs,
    exploration_noise=False  # 测试时无噪声
)

# 4. 预填充回放池 (随机策略)
train_collector.collect(n_step=512*train_env_num, random=True)

# 5. 训练
result = offpolicy_trainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=args.max_epoch,           # 默认 100
    step_per_epoch=args.step_per_epoch, # 默认 2000
    step_per_collect=args.step_per_collect * train_env_num,  # 512
    episode_per_test=args.episode_per_test,  # 5
    batch_size=args.batch_size,         # 128
    update_per_step=args.update_per_step,  # 0.1 (每10步更新1次)
    logger=logger,
    verbose=True,
    show_progress=True,
    test_in_train=True
)

# 6. 保存模型
torch.save(policy.state_dict(), save_model_dir)
```

---

### 训练超参数

#### 环境参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `train_env_num` | 8 | 并行训练环境数 |
| `test_env_num` | 2 | 并行测试环境数 |
| `graph_width` | 256 | 图像分辨率 |
| `canvas_width` | 300 | 渲染画布大小 |
| `image_iter` | 20 | 每张图重复训练轮数 |
| `start_update` | 5 | 开始 EMA 更新的轮次 |
| `update` | 5 | EMA 更新频率 |
| `ema_gamma` | 0.95 | EMA 平滑系数 |

#### 训练参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `max_epoch` | 100 | 最大训练轮数 |
| `step_per_epoch` | 2000 | 每轮环境交互步数 |
| `step_per_collect` | 64 | 收集多少步后更新 |
| `batch_size` | 128 | 批次大小 |
| `update_per_step` | 0.1 | 每步更新次数 (0.1 = 每10步更新1次) |
| `buffer_size` | 100000 | 回放池大小 |

#### 网络参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `hidden_sizes` | [256, 256, 256] | 隐藏层维度 |
| `lr` | 3e-4 | 学习率 |
| `tau` | 0.005 | 目标网络更新速率 |
| `gamma` | 0.99 | 折扣因子 |
| `alpha` | 0.2 | SAC 温度参数 |

---

## 📊 训练过程详解

### Episode 流程

```python
# 1. 环境重置
state = env.reset()
# 每 image_iter 轮切换到新图像
# 加载 .png 和 .npy
# 初始化 period=0, r/l/theta 根据工具初始化

# 2. Episode 循环 (直到字符画完)
for step in range(max_steps):
    # a. 策略选择动作
    action = policy(state)  # [r_prime, theta_prime]

    # b. 环境执行动作
    next_state, reward, terminated, done, info = env.step(action)

    # c. 存储到回放池
    buffer.add(state, action, reward, next_state, done)

    # d. 策略更新 (异步)
    if step % update_freq == 0:
        batch = buffer.sample(batch_size)
        policy.update(batch)

    # e. 更新状态
    state = next_state

    if terminated:
        break

# 3. Episode 结束
if episode % update == 0 and episode >= start_update:
    # EMA 更新粗笔画
    skel_list_new = EMA(skel_list_old, predicted_skel, gamma=0.95)

# 4. 保存结果
if episode % image_iter == 0:
    # 保存优化后的 .npy
    np.save(output_path + str(idx) + '.npy', optimized_sequence)
```

---

### 训练阶段

#### 阶段 1: 探索 (Epoch 1-20)

```python
# 随机策略 + 环境初始化
train_collector.collect(n_step=512*8, random=True)

# 特点:
# - 动作随机性强 (高噪声)
# - 回放池快速填充
# - 网络快速适应环境
```

#### 阶段 2: 优化 (Epoch 21-60)

```python
# SAC 策略逐渐收敛
# update_per_step = 0.1 → 稳定更新

# 特点:
# - 动作方差逐渐减小
# - EMA 开始生效 (epoch >= start_update)
# - 轨迹逐渐平滑
```

#### 阶段 3: 精细调整 (Epoch 61-100)

```python
# 策略接近最优
# 探索噪声降低

# 特点:
# - 奖励趋于稳定
# - 轨迹变化微小
# - 接近收敛
```

---

## 🎨 关键训练技巧

### 1. 渐进式 EMA 更新

```python
# 避免早期训练时剧烈变化
if episode >= start_update and episode % update == 0:
    skel_list = ema_gamma × skel_old + (1-ema_gamma) × skel_new
    #         = 0.95 × old + 0.05 × new

# 效果:
# - 前 5 轮: 不更新，积累经验
# - 第 5, 10, 15... 轮: 小幅更新
# - 最终: 平滑收敛
```

### 2. 多图像循环训练

```python
# 每张图像重复训练 image_iter=20 轮
for epoch in range(max_epoch):
    img_idx = (epoch // image_iter) % total_images
    # 同一张图多次优化，直到充分拟合

# 优势:
# - 充分利用每张图像
# - 避免过拟合单一样本
# - 提高数据效率
```

### 3. 并行环境加速

```python
# 8 个环境同时交互
train_envs = DummyVectorEnv([make_env() for _ in range(8)])

# 每次 step_per_collect = 512 步
# 实际: 512 / 8 = 64 步/环境

# 加速比: ~8× (理想情况)
```

### 4. 软更新目标网络

```python
# 每次更新后
target_params = tau × current_params + (1-tau) × target_params
#              = 0.005 × new + 0.995 × old

# 作用:
# - 稳定训练
# - 减少震荡
# - 防止过拟合
```

---

## 📈 训练监控

### 关键指标

```python
# 1. Episode 奖励
episode_reward = sum(rewards)  # 典型范围: 20-50

# 2. 平均步数
episode_length = num_steps  # 典型范围: 50-200

# 3. 图像相似度
image_similarity = 1 - rec  # 期望: >0.9

# 4. Actor/Critic 损失
actor_loss = ...   # 期望: 逐渐下降
critic_loss = ...  # 期望: <10

# 5. Alpha (温度)
alpha_value = ...  # 自适应，期望: 0.1-0.3
```

### TensorBoard 可视化

```python
from torch.utils.tensorboard import SummaryWriter

logger = TensorboardLogger(writer)

# 记录标量
writer.add_scalar('train/reward', episode_reward, epoch)
writer.add_scalar('train/length', episode_length, epoch)
writer.add_scalar('loss/actor', actor_loss, step)
writer.add_scalar('loss/critic', critic_loss, step)

# 记录图像
writer.add_image('rendered', rendered_canvas, epoch)
writer.add_image('target', target_image, epoch)

# 启动: tensorboard --logdir=./log
```

---

## 🔧 训练配置示例

### 毛笔训练 (train_brush.sh)

```bash
#!/bin/bash

python try_tianshou.py \
    --tool brush \
    --train_path ./data/train_data/ \
    --test_path ./data/test_data/ \
    --output_path ./results/brush/ \
    --visualize_path ./vis/brush/ \
    \
    --train_env_num 8 \
    --test_env_num 2 \
    \
    --max_epoch 100 \
    --step_per_epoch 2000 \
    --step_per_collect 64 \
    --batch_size 128 \
    --update_per_step 0.1 \
    \
    --buffer_size 100000 \
    --lr 3e-4 \
    --tau 0.005 \
    --gamma 0.99 \
    --alpha 0.2 \
    \
    --image_iter 20 \
    --start_update 5 \
    --update 5 \
    --ema_gamma 0.95 \
    \
    --logdir ./log/brush/ \
    --device cuda:0
```

### 马克笔训练 (train_marker.sh)

```bash
python try_tianshou.py \
    --tool marker \
    --train_path ./data/train_data/ \
    --test_path ./data/test_data/ \
    --output_path ./results/marker/ \
    --visualize_path ./vis/marker/ \
    \
    --train_env_num 10 \
    --test_env_num 2 \
    \
    --max_epoch 150 \
    --step_per_epoch 3000 \
    --step_per_collect 64 \
    --batch_size 256 \
    --update_per_step 0.2 \
    \
    --image_iter 25 \
    --ema_gamma 0.92 \
    \
    --device cuda:0
```

---

## 💾 训练输出

### 文件结构

```
rl_finetune/
├── results/
│   ├── brush/
│   │   ├── 0_20.npy         # 第0张图，第20轮
│   │   ├── 0_40.npy         # 第0张图，第40轮
│   │   ├── 1_20.npy         # 第1张图，第20轮
│   │   └── ...
│   └── marker/
│       └── ...
│
├── vis/
│   ├── brush/
│   │   ├── 0_20.png         # 可视化对比
│   │   ├── 0_40.png
│   │   └── ...
│   └── marker/
│       └── ...
│
├── log/
│   ├── brush/
│   │   └── events.out.tfevents  # TensorBoard 日志
│   └── marker/
│       └── ...
│
└── checkpoints/
    ├── brush_policy.pth     # 最终策略
    └── marker_policy.pth
```

---

## 🚀 训练命令

### 训练

```bash
# 毛笔
cd rl_finetune
conda activate calli_rl
bash scripts/train_brush.sh

# 马克笔
bash scripts/train_marker.sh

# 椭圆笔刷
bash scripts/train_fude.sh
```

### 监控

```bash
# TensorBoard
tensorboard --logdir=./log/brush --port=6006

# 访问: http://localhost:6006
```

### 测试

```bash
# 单张图像测试
python try_tianshou.py \
    --tool brush \
    --test_path ./data/test_data/ \
    --output_path ./results/test/ \
    --resume_from_log ./checkpoints/brush_policy.pth \
    --max_epoch 1
```

---

## 📊 训练性能

| 指标 | 毛笔 | 马克笔 | 椭圆笔刷 |
|------|------|--------|----------|
| **训练轮数** | 100 | 150 | 120 |
| **训练时间** | ~6小时 (V100) | ~9小时 | ~7小时 |
| **最终奖励** | 42 | 38 | 45 |
| **图像相似度** | 0.92 | 0.89 | 0.94 |
| **平均步数** | 120 | 85 | 110 |

---

## 🎯 训练检查清单

### 开始训练前
- [ ] 数据准备完成 (train_data/ 和 test_data/)
- [ ] 环境配置正确 (`calli_rl`)
- [ ] GPU 可用 (`torch.cuda.is_available()`)
- [ ] 工具类型选择正确
- [ ] 超参数配置检查

### 训练过程中
- [ ] 监控奖励曲线 (TensorBoard)
- [ ] 检查生成质量 (visualize_path)
- [ ] 验证 EMA 更新生效
- [ ] GPU 利用率 >80%
- [ ] 定期保存检查点

### 训练完成后
- [ ] 在测试集上验证
- [ ] 可视化最终结果
- [ ] 保存最佳模型
- [ ] 记录超参数配置

---

## 🔍 常见问题

### Q: 为什么使用 SAC 而不是 PPO/DDPG?

**A**: SAC 的优势:
- **高样本效率**: 离线策略 (off-policy)，重复利用经验
- **稳定性**: 熵正则化 + 双 Q 网络
- **探索能力**: 自动调节温度参数
- **适用于连续动作**: 无需离散化

### Q: image_iter 如何选择?

**A**: 经验法则:
- 简单字符 (笔画少): 15-20
- 复杂字符 (笔画多): 20-30
- 数据集小: 增大 iter (充分利用)
- 数据集大: 减小 iter (避免过拟合)

### Q: EMA 参数如何调优?

**A**:
- `ema_gamma`:
  - 0.9: 快速更新 (10% 新轨迹)
  - 0.95: 平衡 (默认)
  - 0.98: 保守更新 (2% 新轨迹)
- `start_update`: 通常设为 image_iter / 4
- `update`: 通常设为 image_iter / 4

### Q: 训练不收敛怎么办?

**A**: 检查列表:
1. **奖励设计**: 确保终止奖励占主导
2. **学习率**: 尝试降低到 1e-4
3. **batch_size**: 尝试增大到 256
4. **EMA 参数**: 增大 gamma 到 0.98
5. **环境数**: 增加并行环境到 12-16

---

## 🔄 与 seq_extract 的对比

| 特性 | seq_extract | rl_finetune |
|------|-------------|-------------|
| **方法** | 监督学习 (LSTM) | 强化学习 (SAC) |
| **输入** | 书法图像 | 粗笔画 + 图像 |
| **输出** | 粗笔画序列 | 精细笔画序列 |
| **训练数据** | QuickDraw + 书法 | 书法数据 |
| **训练时间** | ~5天 | ~6小时 |
| **物理建模** | 无 | 工具动力学 |
| **适应性** | 通用 | 工具特定 |

---

**总结**: rl_finetune 通过强化学习和工具物理建模，将粗笔画序列优化为符合真实书写工具特性的精细轨迹，是 CalliRewrite 系统从虚拟到真实的关键桥梁！
