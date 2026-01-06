# 三维海气立方体动态可视化实现计划

## 一、项目概述

基于 `velocity_3D_vector_optimized.py` 实现三维海气立方体的动态可视化，展示10小时（每帧1小时）的标量场和矢量场变化。

## 二、核心功能需求

### 2.1 时间数据获取
- 参考 `detect14month2.py` 的时间数据读取方法
- 实现多时间步数据的批量加载
- 支持10个时间步（10小时，每帧1小时）

### 2.2 标量场动态效果
- **温度（颜色）**：随时间平滑变化
- **盐度（透明度）**：随时间平滑变化
- 使用插值确保帧间平滑过渡

### 2.3 矢量场动态效果

#### 2.3.1 静止帧（暂停时）
- 箭头亮度沿流速方向从尾到头传递
- 每个周期：
  - 初始：尾部最亮，头部最暗
  - 结束：尾部最暗，头部最亮
- 亮度变化遵循线性函数，相位按方向传递

#### 2.3.2 播放帧（时间推进时）
- 每个时间帧内：
  1. 箭头从尾部到头部逐渐出现（渐显）
  2. 箭头完全出现后保持
  3. 箭头从尾部到头部逐渐消失（渐隐）
- 下一个时间帧承接动画效果

## 三、技术实现方案

### 3.1 数据加载模块
```python
# 参考 detect14month2.py 的时间数据读取
def load_time_series_data(db, time_start, time_end, time_step=1):
    """
    加载时间序列数据
    Args:
        db: 数据集对象
        time_start: 起始时间步
        time_end: 结束时间步
        time_step: 时间步间隔（默认1小时）
    Returns:
        time_series_data: 时间序列数据字典
    """
```

### 3.2 标量场动画模块
```python
def update_scalar_field(frame_idx, time_series_data):
    """
    更新标量场（温度、盐度）
    - 使用线性插值实现帧间平滑过渡
    - 更新体积渲染的颜色和透明度
    """
```

### 3.3 矢量场动画模块

#### 3.3.1 静止帧亮度传递
```python
def apply_flow_animation(arrows, time_cycle, flow_direction):
    """
    应用流动动画效果（静止帧）
    - 计算箭头上每点沿流速方向的参数化位置
    - 根据周期时间和位置计算亮度
    - 使用线性函数：brightness = f(t, s)
    """
```

#### 3.3.2 播放帧渐显渐隐
```python
def apply_temporal_animation(arrows, frame_idx, total_frames):
    """
    应用时间动画效果（播放帧）
    - 计算当前帧在时间周期内的位置
    - 根据位置决定箭头的渐显/渐隐状态
    - 从尾部到头部逐渐出现/消失
    """
```

### 3.4 动画控制模块
```python
class AnimationController:
    """
    动画控制器
    - 播放/暂停控制
    - 帧率控制
    - 时间步切换
    """
    def __init__(self, total_frames=10, fps=1.0):
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame = 0
        self.is_playing = False
        self.cycle_time = 0.0  # 用于静止帧的周期动画
    
    def play(self):
        """开始播放"""
    
    def pause(self):
        """暂停播放"""
    
    def next_frame(self):
        """下一帧"""
    
    def update(self, dt):
        """更新动画状态"""
```

## 四、实现步骤

### 阶段1：基础框架搭建
1. ✅ 创建主程序文件 `ocean_3d_animation.py`
2. ✅ 实现时间序列数据加载
3. ✅ 实现基础动画控制器
4. ✅ 集成 PyVista 交互式窗口

### 阶段2：标量场动画
1. ✅ 实现多时间步数据插值
2. ✅ 实现体积渲染的动态更新
3. ✅ 测试标量场平滑过渡

### 阶段3：矢量场动画（静止帧）
1. ✅ 实现箭头参数化（沿流速方向）
2. ✅ 实现亮度传递函数（线性）
3. ✅ 实现周期动画控制

### 阶段4：矢量场动画（播放帧）
1. ✅ 实现箭头渐显渐隐效果
2. ✅ 实现从尾到头的动画顺序
3. ✅ 实现时间帧切换逻辑

### 阶段5：优化与集成
1. ✅ 性能优化（减少不必要的重绘）
2. ✅ UI 控制界面（播放/暂停/速度控制）
3. ✅ 导出动画功能（可选）

## 五、关键技术点

### 5.1 时间数据插值
- 使用 `scipy.interpolate.interp1d` 进行时间维度插值
- 确保帧间数据平滑过渡

### 5.2 箭头参数化
- 计算箭头上每点沿流速方向的归一化位置 s ∈ [0, 1]
- 尾部 s=0，头部 s=1

### 5.3 亮度函数（静止帧）
```
brightness(s, t) = 1 - (s + t_cycle) % 1
```
其中：
- s: 沿箭头方向的归一化位置
- t_cycle: 周期时间（0-1）
- 线性函数：尾部最亮→头部最暗→尾部最暗→头部最亮

### 5.4 渐显渐隐函数（播放帧）
```
alpha(s, t_frame) = {
    if t_frame < 0.5:
        alpha = 1 if s <= 2*t_frame else 0  # 渐显
    else:
        alpha = 1 if s > 2*(t_frame-0.5) else 0  # 渐隐
}
```

## 六、文件结构

```
AddedExperiments/
├── ocean_3d_animation.py          # 主程序文件
├── ocean_3d_animation_plan.md    # 实现计划（本文件）
└── ocean_3d_animation_output/     # 输出目录（可选）
```

## 七、依赖项

- `pyvista`: 3D可视化
- `numpy`: 数值计算
- `scipy`: 插值和滤波
- `OpenVisus`: 数据加载
- `matplotlib`: 颜色映射（可选）

## 八、注意事项

1. **性能考虑**：动态更新体积渲染可能较慢，考虑使用LOD优化
2. **内存管理**：10个时间步的数据量较大，注意内存使用
3. **动画流畅度**：确保帧率稳定，避免卡顿
4. **交互性**：保持原有的旋转、缩放等交互功能

