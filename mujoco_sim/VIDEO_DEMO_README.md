# MuJoCo 视频演示说明

## 🎥 生成的视频

所有视频都是从**机器人上方俯视角度**录制的，清晰展示了书写过程。

### 可用的演示视频

| 文件 | 时长 | 大小 | 描述 | 接触率 |
|------|------|------|------|--------|
| **永_fixed.mp4** | 2.83秒 | 16KB | "永"字完整书写（85个控制点）| 75.9% |
| **line_demo.mp4** | 1.10秒 | 11KB | 简单直线演示（33个控制点）| 55.9% |
| **calibration_demo.mp4** | 2.27秒 | 14KB | 校准测试轨迹（68个控制点）| 23.9% |
| **test_trajectory.mp4** | 0.67秒 | 9KB | 单元测试直线（20个点）| 80.3% |

### 对应的画布图像

每个视频都有对应的画布PNG图像，显示最终的书写结果：

- `永_fixed_canvas.png` (12KB) - "永"字笔迹
- `line_demo_canvas.png` (9.3KB) - 直线笔迹
- `calibration_demo_canvas.png` (4.5KB) - 校准轨迹
- `test_result.png` (5.8KB) - 测试轨迹

## 🛠️ 如何生成视频

### 使用专用脚本（推荐）

```bash
# 基本用法
python generate_video_demo.py <npz文件>

# 示例：生成"永"字视频
python generate_video_demo.py ../callibrate/examples/example_永.npz \
    --output outputs/永_demo.mp4 --speed 0.03

# 调整速度和帧率
python generate_video_demo.py ../callibrate/examples/simple_line.npz \
    --output outputs/line.mp4 --speed 0.02 --fps 30
```

### 参数说明

- `npz_file`: NPZ轨迹文件路径（必需）
- `--output, -o`: 输出视频路径（默认: outputs/<npz文件名>.mp4）
- `--canvas-output`: 画布图片输出路径（默认: outputs/<npz文件名>_result.png）
- `--trajectory-output`: 3D轨迹图片输出路径（默认: outputs/<npz文件名>_3d.png）
- `--speed, -s`: 机器人运动速度，单位 m/s（默认: 0.05）
- `--fps, -f`: 视频帧率（默认: 30）

### 使用内置录制功能

```bash
# 使用 mujoco_simulator.py 的 --record 参数
python mujoco_simulator.py ../callibrate/examples/example_永.npz \
    --record outputs/video.mp4 --speed 0.05
```

## 📐 坐标系统说明

### NPZ文件坐标
- NPZ文件中的坐标是**相对于纸张左下角**的坐标
- 范围：X: 0-0.05m, Y: 0-0.064m, Z: -0.1-0.1m
- 例如：`[0.003, 0.002, -0.09]` 表示距离纸张左下角3mm, 2mm，笔尖压下

### MuJoCo世界坐标
- 纸张中心位于世界坐标 `(0.5, 0, 0.01)`
- 转换公式：
  ```python
  world_x = npz_x + 0.5
  world_y = npz_y + 0.0
  world_z = npz_z  # Z坐标保持不变
  ```

### 自动坐标转换

从2026-01-18开始，`mujoco_simulator.py` 和 `generate_video_demo.py` 已经**自动处理坐标转换**：

- ✅ 读取NPZ文件中的相对坐标
- ✅ 自动添加纸张偏移量（0.5, 0, 0）
- ✅ 转换为MuJoCo世界坐标
- ✅ 正确渲染笔迹到画布

## 🎬 视频特性

### 相机设置
- **名称**: `top_view`
- **位置**: (0.5, 0, 1.2) - 纸张正上方1.2米
- **朝向**: 俯视（euler: -π, 0, 0）
- **视野**: 45度

### 渲染设置
- **分辨率**: 1280x720 (720p HD)
- **帧率**: 30 FPS
- **编码**: MP4 (H.264)
- **颜色空间**: RGB

### 画布设置
- **分辨率**: 1200x1680 像素
- **对应纸张**: A3尺寸 (0.3m × 0.42m)
- **背景**: 白色 (255)
- **笔迹**: 黑色 (0)
- **笔刷大小**: 3像素半径

## 🐛 已解决的问题

### 问题1: 画布全白，视频全蓝
**原因**: NPZ坐标没有转换到世界坐标系
**解决**: 在 `execute_trajectory` 和 `generate_video_demo.py` 中添加坐标偏移

### 问题2: 离屏渲染缓冲区错误
**原因**: XML模型中未配置足够大的离屏缓冲区
**解决**: 在 `franka_panda.xml` 中添加：
```xml
<global offwidth="1280" offheight="720"/>
```

### 问题3: 重复的global元素错误
**原因**: XML中不能有两个 `<global>` 元素
**解决**: 合并属性到一个元素中

## 📊 性能统计

| 轨迹 | 控制点 | 记录点 | 插值率 | 接触率 | 视频大小 | 画布大小 |
|------|--------|--------|--------|--------|----------|----------|
| 永字 | 85 | 967 | 11.4× | 75.9% | 16KB | 12KB |
| 直线 | 33 | 608 | 18.4× | 55.9% | 11KB | 9.3KB |
| 校准 | 68 | 1200 | 17.6× | 23.9% | 14KB | 4.5KB |
| 测试 | 20 | 71 | 3.6× | 80.3% | 9KB | 5.8KB |

### 说明
- **控制点**: NPZ文件中的原始点数
- **记录点**: 仿真中实际执行的点数（包含插值）
- **插值率**: 记录点数 / 控制点数
- **接触率**: 检测到笔-纸接触的点的百分比
- **视频大小**: 生成的MP4文件大小
- **画布大小**: PNG图像文件大小

## 🎯 使用建议

### 推荐速度设置
- **慢速演示**: 0.02 m/s - 适合展示细节
- **正常速度**: 0.05 m/s - 平衡速度和清晰度
- **快速预览**: 0.10 m/s - 快速验证轨迹

### 最佳实践
1. **首次测试**: 使用 `simple_line.npz` 快速验证
2. **完整字符**: 使用较慢速度（0.03-0.05 m/s）
3. **长轨迹**: 考虑增加帧率到60 FPS
4. **调试**: 使用 `--no-render` 加快速度

## 📝 更新日志

### 2026-01-18
- ✅ 添加俯视相机到MuJoCo模型
- ✅ 创建 `generate_video_demo.py` 脚本
- ✅ 修复NPZ坐标到世界坐标的转换
- ✅ 配置离屏渲染缓冲区
- ✅ 生成多个演示视频
- ✅ 更新测试套件支持视频录制

## 🔗 相关文档

- [README.md](README.md) - MuJoCo模块完整文档
- [TEST_REPORT.md](TEST_REPORT.md) - 测试报告
- [models/franka_panda.xml](models/franka_panda.xml) - 机器人模型
- [mujoco_simulator.py](mujoco_simulator.py) - 核心仿真器代码

---

**提示**: 所有视频都保存在 `outputs/` 目录中，可以直接在视频播放器中查看！🎥
