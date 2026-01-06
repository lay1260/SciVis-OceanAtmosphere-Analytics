# 三维海气立方体动态可视化
# - 基于 velocity_3D_vector_optimized.py 实现
# - 展示10小时（每帧1小时）的标量场和矢量场变化
# - 标量场：温度、盐度随时间平滑变化
# - 矢量场：静止帧时亮度传递，播放帧时渐显渐隐
import OpenVisus as ov
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("警告：VTK不可用，将使用PyVista高层API")

# 矢量场优化：模式1 - 弯曲箭头所需依赖
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告：SciPy不可用，将使用直线箭头")

# ----------------------------
# 1️⃣ 模拟数据生成（10x10x10立方体，10个时间帧）
# ----------------------------
def generate_simulated_data(nx=10, ny=10, nz=10, n_frames=10):
    """
    生成模拟的10x10x10立方体数据，10个时间帧
    
    Args:
        nx, ny, nz: 立方体尺寸（默认10x10x10）
        n_frames: 时间帧数（默认10）
    
    Returns:
        time_series_data: 字典，包含每个时间帧的数据
    """
    print(f"\n正在生成模拟数据（{nx}x{ny}x{nz}立方体，{n_frames}个时间帧）...")
    
    # 设置随机种子，确保可重复性
    np.random.seed(42)
    
    # 创建空间坐标网格（归一化到0-1范围，然后缩放）
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    time_series_data = {
        'time_steps': list(range(n_frames)),
        'U': [],
        'V': [],
        'W': [],
        'Salt': [],
        'Theta': []
    }
    
    for t in range(n_frames):
        # 时间相关的相位（0到2π）
        time_phase = 2 * np.pi * t / n_frames
        
        # 生成速度场（U, V, W）- 随时间变化，变化更明显
        # U: 水平方向速度，包含时间相关的波动
        U = 0.5 * np.sin(2 * np.pi * X * 2 + time_phase) * np.cos(2 * np.pi * Y * 2)
        U += 0.2 * np.sin(2 * np.pi * Z * 2 + time_phase * 0.7)
        U += 0.1 * np.sin(2 * np.pi * (X + Y + Z) + time_phase * 1.3)
        U += 0.05 * np.random.randn(nx, ny, nz)  # 添加随机噪声
        
        # V: 水平方向速度
        V = 0.5 * np.cos(2 * np.pi * X * 2 + time_phase) * np.sin(2 * np.pi * Y * 2)
        V += 0.2 * np.cos(2 * np.pi * Z * 2 + time_phase * 0.7)
        V += 0.1 * np.cos(2 * np.pi * (X + Y + Z) + time_phase * 1.3)
        V += 0.05 * np.random.randn(nx, ny, nz)
        
        # W: 垂直方向速度（较小）
        W = 0.1 * np.sin(2 * np.pi * X * 2 + time_phase) * np.sin(2 * np.pi * Y * 2)
        W += 0.05 * np.sin(2 * np.pi * Z * 2 + time_phase)
        W += 0.02 * np.random.randn(nx, ny, nz)
        
        # 生成盐度场（Salt）- 随时间变化，变化更明显
        # 基础盐度分布：中心高，边缘低
        center_x, center_y, center_z = 0.5, 0.5, 0.5
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
        max_dist = np.sqrt(3) / 2
        
        # 盐度随时间波动（变化更明显）
        salt_base = 30.0 + 5.0 * (1 - dist_from_center / max_dist)
        salt_variation = 3.0 * np.sin(time_phase + dist_from_center * 5)
        salt_wave = 1.5 * np.sin(2 * np.pi * X * 3 + time_phase) * np.sin(2 * np.pi * Y * 3)
        Salt = salt_base + salt_variation + salt_wave
        Salt = np.clip(Salt, 0, 35)  # 限制在合理范围
        
        # 生成温度场（Theta）- 随时间变化，变化更明显
        # 温度分布：上层高，下层低，随时间波动
        depth_factor = 1 - Z  # 深度因子（0=底部，1=顶部）
        temp_base = 2.0 + 3.0 * depth_factor
        temp_variation = 2.0 * np.sin(time_phase + X * 5 + Y * 5)
        temp_wave = 1.0 * np.sin(2 * np.pi * X * 4 + time_phase) * np.cos(2 * np.pi * Y * 4)
        Theta = temp_base + temp_variation + temp_wave
        Theta = np.clip(Theta, -2, 5)  # 限制在合理范围
        
        time_series_data['U'].append(U.astype(np.float32))
        time_series_data['V'].append(V.astype(np.float32))
        time_series_data['W'].append(W.astype(np.float32))
        time_series_data['Salt'].append(Salt.astype(np.float32))
        time_series_data['Theta'].append(Theta.astype(np.float32))
        
        print(f"  生成时间帧 {t+1}/{n_frames}: U范围[{U.min():.3f}, {U.max():.3f}], "
              f"Salt范围[{Salt.min():.2f}, {Salt.max():.2f}], "
              f"Theta范围[{Theta.min():.2f}, {Theta.max():.2f}]")
    
    print(f"✅ 模拟数据生成完成，共 {n_frames} 个时间帧")
    return time_series_data

# ----------------------------
# 2️⃣ 局部区域参数（用于网格坐标）
# ----------------------------
nx, ny, nz = 10, 10, 10  # 立方体尺寸
scale_xy = 25  # 坐标缩放因子

# ----------------------------
# 3️⃣ 生成模拟时间序列数据
# ----------------------------
# 生成10x10x10立方体，10个时间帧的模拟数据
time_series_data = generate_simulated_data(nx=10, ny=10, nz=10, n_frames=10)

# 获取第一个时间步的数据用于初始化网格
U_local = time_series_data['U'][0]
V_local = time_series_data['V'][0]
W_local = time_series_data['W'][0]
Salt_local = time_series_data['Salt'][0]
Theta_local = time_series_data['Theta'][0]

print(f"\n网格尺寸: nx={nx}, ny={ny}, nz={nz}")

# ----------------------------
# 4️⃣ 构建网格坐标（10x10x10立方体）
# ----------------------------
# 创建空间坐标网格（与模拟数据一致）
x = np.linspace(0, 100, nx) * scale_xy
y = np.linspace(0, 100, ny) * scale_xy
z = np.linspace(0, 100, nz) * scale_xy
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
# Z轴向下（深度方向）
Z = -Z

# ----------------------------
# 6️⃣ 动画控制器
# ----------------------------
class AnimationController:
    """动画控制器"""
    def __init__(self, total_frames=10, fps=2.0):
        self.total_frames = total_frames
        self.fps = fps  # 提高fps，使动画播放更快（每帧0.5秒）
        self.current_frame = 0
        self.is_playing = False
        self.cycle_time = 0.0  # 用于静止帧的周期动画（0-1）
        self.cycle_speed = 0.5  # 周期动画速度
        self.frame_time = 0.0  # 当前帧内的时间（0-1），用于渐显渐隐
        self.last_update_time = time.time()
        
        # 帧过渡动画状态
        self.is_transitioning = False  # 是否正在过渡
        self.transition_start_frame = 0  # 过渡起始帧
        self.transition_target_frame = 0  # 过渡目标帧
        self.transition_start_time = 0.0  # 过渡开始时间
        self.transition_duration = 5.0  # 过渡持续时间（秒）
        self.transition_progress = 0.0  # 过渡进度（0-1）
    
    def play(self):
        """开始播放"""
        self.is_playing = True
        self.last_update_time = time.time()
        # 停止任何正在进行的过渡
        self.is_transitioning = False
    
    def pause(self):
        """暂停播放"""
        self.is_playing = False
    
    def next_frame(self):
        """下一帧（启动过渡动画）"""
        target_frame = (self.current_frame + 1) % self.total_frames
        self.start_transition(target_frame)
    
    def prev_frame(self):
        """上一帧（启动过渡动画）"""
        target_frame = (self.current_frame - 1) % self.total_frames
        self.start_transition(target_frame)
    
    def start_transition(self, target_frame):
        """启动帧过渡动画"""
        # 如果目标帧和当前帧相同，不启动过渡
        if target_frame == self.current_frame and not self.is_transitioning:
            print(f"⚠️ 目标帧 {target_frame} 与当前帧相同，跳过过渡")
            return
        
        if self.is_transitioning:
            # 如果已经在过渡，从当前实际帧（可能是插值位置）继续过渡到新目标
            # 使用当前插值位置作为新的起始帧，确保连续性
            current_interp_frame = self.transition_start_frame + (
                self.transition_target_frame - self.transition_start_frame
            ) * self.transition_progress
            # 使用当前插值帧作为新的起始帧（保留小数部分用于平滑过渡）
            self.transition_start_frame = current_interp_frame
        else:
            # 不在过渡中，使用当前帧作为起始帧
            self.transition_start_frame = float(self.current_frame)
        
        self.transition_target_frame = target_frame
        self.transition_start_time = time.time()
        self.transition_progress = 0.0
        self.is_transitioning = True
        print(f"🔄 开始过渡动画: 帧 {self.transition_start_frame:.2f} -> {self.transition_target_frame} (5秒)")
    
    def set_frame(self, frame_idx):
        """设置当前帧（立即切换，不过渡）"""
        self.current_frame = np.clip(frame_idx, 0, self.total_frames - 1)
        self.frame_time = 0.0
        self.is_transitioning = False
    
    def update(self, dt=None):
        """更新动画状态"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
        
        # 更新帧过渡动画
        if self.is_transitioning:
            elapsed = current_time - self.transition_start_time
            self.transition_progress = min(elapsed / self.transition_duration, 1.0)
            
            if self.transition_progress >= 1.0:
                # 过渡完成
                old_frame = self.current_frame
                self.current_frame = int(self.transition_target_frame)
                self.is_transitioning = False
                self.transition_progress = 0.0
                self.transition_start_frame = float(self.current_frame)  # 重置起始帧为当前帧
                print(f"✅ 过渡完成: 到达帧 {self.current_frame} (从 {old_frame})")
        
        if self.is_playing:
            # 播放模式：更新帧内时间，自动连续播放所有帧
            # 注意：播放时不应该有过渡动画
            if self.is_transitioning:
                self.is_transitioning = False  # 停止过渡，开始播放
            
            self.frame_time += dt * self.fps
            # 当帧内时间超过1.0时，自动切换到下一帧
            while self.frame_time >= 1.0:
                self.frame_time -= 1.0  # 保留超出部分
                self.current_frame = (self.current_frame + 1) % self.total_frames
                # 如果回到第0帧，说明完成一轮循环
                if self.current_frame == 0:
                    self.frame_time = 0.0  # 重置帧内时间
        else:
            # 静止模式：更新周期动画时间
            self.cycle_time = (self.cycle_time + dt * self.cycle_speed) % 1.0
    
    def get_interpolated_frame_index(self):
        """获取插值后的帧索引（用于标量场插值）"""
        if self.is_transitioning:
            # 过渡模式：使用插值计算中间帧
            # transition_start_frame 可能是浮点数（如果是从另一个过渡继续）
            interp_frame = self.transition_start_frame + (
                self.transition_target_frame - self.transition_start_frame
            ) * self.transition_progress
            return interp_frame
        elif self.is_playing:
            # 播放模式：使用帧内时间进行插值
            return self.current_frame + self.frame_time
        else:
            # 静止模式：使用当前帧
            return float(self.current_frame)

# ----------------------------
# 7️⃣ 标量场透明度映射策略（策略19）
# ----------------------------
def opacity_strategy_19(salt_data, salt_gradient_norm, salt_min_global=None, salt_max_global=None):
    """策略19：低（30%）阈值 + 不常见-平方根（1.0系数）+ 0~0.25透明度（降低透明度）
    低过滤 + 高透明，温和保留中低盐细节
    
    Args:
        salt_data: 当前帧的盐度数据
        salt_gradient_norm: 当前帧的盐度梯度归一化值
        salt_min_global: 全局盐度最小值（可选，用于统一映射）
        salt_max_global: 全局盐度最大值（可选，用于统一映射）
    """
    # 使用全局范围（如果提供）或当前帧范围
    if salt_min_global is not None and salt_max_global is not None:
        salt_range = salt_max_global - salt_min_global
        salt_threshold = salt_min_global + 0.3 * salt_range
        salt_norm = np.clip((salt_data - salt_threshold) / (salt_max_global - salt_threshold), 0.0, 1.0)
    else:
        salt_threshold = np.percentile(salt_data, 30)
        salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    
    # 降低透明度范围：从0~0.35降低到0~0.25
    base_opacity = 0 + 0.25 * np.sqrt(salt_norm)
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.25)
    return final_opacity

# ----------------------------
# 8️⃣ 标量场插值函数
# ----------------------------
def interpolate_scalar_field(time_series_data, frame_idx, field_name):
    """
    在时间维度上插值标量场或矢量场
    
    Args:
        time_series_data: 时间序列数据字典
        field_name: 字段名称（'Salt', 'Theta', 'U', 'V', 'W'）
        frame_idx: 帧索引（可以是浮点数，用于插值）
    
    Returns:
        interpolated_data: 插值后的数据
    """
    field_data = time_series_data[field_name]
    time_steps = np.arange(len(field_data))
    
    # 如果frame_idx是整数，直接返回
    if isinstance(frame_idx, (int, np.integer)) and 0 <= frame_idx < len(field_data):
        return field_data[frame_idx]
    
    # 否则进行线性插值
    frame_idx = np.clip(frame_idx, 0, len(field_data) - 1)
    
    if frame_idx == int(frame_idx):
        return field_data[int(frame_idx)]
    
    idx_low = int(np.floor(frame_idx))
    idx_high = int(np.ceil(frame_idx))
    t = frame_idx - idx_low
    
    if idx_high >= len(field_data):
        return field_data[-1]
    
    # 线性插值
    interpolated = (1 - t) * field_data[idx_low] + t * field_data[idx_high]
    return interpolated

# ----------------------------
# 9️⃣ 矢量场模式1：弯曲箭头生成函数（参考velocity_3D_vector_optimized.py）
# ----------------------------
def get_neighbors(sample_points, target_idx, k=5):
    """获取目标采样点的k个空间最近邻（含自身）"""
    target_point = sample_points[target_idx]
    distances = np.linalg.norm(sample_points - target_point, axis=1)
    neighbor_indices = np.argsort(distances)[:k]
    return neighbor_indices

def smooth_velocity_field(sample_points, velocities, sigma=1.0):
    """高斯卷积平滑速度场（x/y/z三个分量分别平滑）"""
    smoothed_vel = np.zeros_like(velocities)
    for i in range(3):
        smoothed_vel[:, i] = gaussian_filter1d(velocities[:, i], sigma=sigma)
    return smoothed_vel

def create_bent_arrows(sample_points, velocities, speeds, arrow_scale=60.0, 
                      k_neighbors=4, spline_degree=3, max_bend_factor=0.3):
    """生成三维弯曲箭头（模式1）- 参考velocity_3D_vector_optimized.py的实现"""
    if not SCIPY_AVAILABLE:
        print("⚠️  SciPy不可用，无法生成弯曲箭头")
        return None
    
    # 计算速度大小用于缩放箭头
    speed_range = [np.min(speeds), np.max(speeds)]
    print(f"  速度范围: [{speed_range[0]:.4f}, {speed_range[1]:.4f}]")
    
    # 自适应箭头长度缩放
    if speed_range[1] > 0:
        scale_factor = arrow_scale / speed_range[1]
    else:
        scale_factor = arrow_scale
    
    arrows = []
    success_count = 0
    fail_count = 0
    
    # 为每个采样点生成弯曲箭头
    for i in range(len(sample_points)):
        try:
            # 获取当前点和邻域点
            current_point = sample_points[i]
            current_vel = velocities[i]
            speed = speeds[i]
            
            # 跳过速度过小的点，避免箭头过短
            if speed < 0.01 * speed_range[1]:  # 忽略速度小于1%最大值的点
                fail_count += 1
                continue
            
            # 获取邻域点并平滑速度场
            neighbors = get_neighbors(sample_points, i, k=k_neighbors)
            neighbor_points = sample_points[neighbors]
            neighbor_vels = velocities[neighbors]
            
            # 平滑速度场 (降低sigma值减少过度弯曲)
            smoothed_vels = smooth_velocity_field(neighbor_points, neighbor_vels, sigma=0.8)
            
            # 生成曲线点 (减少采样点数避免过度弯曲)
            num_points = 5  # 减少点数使曲线更平缓
            curve_points = [current_point.copy()]
            current_pos = current_point.copy()
            
            # 计算总长度 (基于速度大小)
            total_length = speed * scale_factor
            
            # 沿平滑后的速度方向生成曲线点
            for j in range(1, num_points):
                # 插值获取当前段的速度方向
                t = j / (num_points - 1)
                vel_idx = min(int(t * len(smoothed_vels)), len(smoothed_vels) - 1)
                dir_vec = smoothed_vels[vel_idx]
                
                # 标准化方向向量并应用弯曲因子限制
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 0:
                    dir_vec = dir_vec / dir_norm
                    
                    # 与初始方向计算角度，限制最大弯曲角度
                    initial_dir = current_vel / np.linalg.norm(current_vel) if np.linalg.norm(current_vel) > 0 else dir_vec
                    angle = np.arccos(np.clip(np.dot(dir_vec, initial_dir), -1.0, 1.0))
                    
                    # 应用弯曲限制
                    max_angle = max_bend_factor * np.pi/2  # 最大弯曲角度为90度的max_bend_factor比例
                    if angle > max_angle:
                        # 限制方向向量，使其不超过最大弯曲角度
                        cross = np.cross(initial_dir, dir_vec)
                        cross_norm = np.linalg.norm(cross)
                        if cross_norm > 1e-6:
                            cross = cross / cross_norm
                            dir_vec = np.sin(max_angle) * np.cross(cross, initial_dir) + np.cos(max_angle) * initial_dir
                        else:
                            # 如果叉积为零，说明方向相同或相反，直接使用初始方向
                            dir_vec = initial_dir
                
                # 计算步长并更新位置
                step = dir_vec * (total_length / (num_points - 1))
                current_pos += step
                curve_points.append(current_pos.copy())
            
            # 创建弯曲箭杆 (使用PolyData正确处理多点曲线)
            if len(curve_points) >= 2:
                # 正确创建多点曲线
                poly = pv.PolyData()
                poly.points = np.array(curve_points)
                
                # 创建线段连接
                lines = np.empty((len(curve_points)-1, 3), dtype=int)
                lines[:, 0] = 2  # 每个线段有2个点
                for j in range(len(curve_points)-1):
                    lines[j, 1] = j
                    lines[j, 2] = j + 1
                
                poly.lines = lines
                
                # 创建管状箭杆 (增大半径使箭头更明显)
                tube_radius = 0.05 * scale_factor * (speed / speed_range[1]) if speed_range[1] > 0 else 0.05
                arrow_shaft = poly.tube(radius=tube_radius, n_sides=12)
                
                # 创建箭头头部 (圆锥)
                # 计算箭头头部方向（曲线终点的切线方向）
                if len(curve_points) >= 2:
                    tip_direction = (curve_points[-1] - curve_points[-2])
                    tip_norm = np.linalg.norm(tip_direction)
                    if tip_norm > 1e-6:
                        tip_direction = tip_direction / tip_norm
                    else:
                        tip_direction = (curve_points[-1] - curve_points[0]) / np.linalg.norm(curve_points[-1] - curve_points[0])
                else:
                    tip_direction = np.array([1, 0, 0])
                
                cone_length = 0.3 * total_length  # 头部长度为总长度的30%
                cone_radius = 3 * tube_radius     # 头部半径为箭杆的3倍
                # 调整圆锥位置：将圆锥底部（大端）放在曲线终点，向前延伸，完整包裹箭杆前端
                forward_offset = cone_length * 0.2  # 向前偏移20%的圆锥长度，确保完整包裹箭杆前端
                cone_center = curve_points[-1] + tip_direction * (cone_length * 0.5 + forward_offset)
                cone = pv.Cone(
                    center=cone_center,
                    direction=tip_direction,
                    height=cone_length,
                    radius=cone_radius,
                    resolution=8
                )
                
                # 合并箭杆和头部
                arrow = arrow_shaft.merge(cone)
                
                # 设置颜色 (基于速度大小)
                arrow['speed'] = np.full(arrow.n_points, speed)
                arrow['velocity'] = np.tile(current_vel, (arrow.n_points, 1))
                arrows.append(arrow)
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            if fail_count < 3:
                print(f"   警告：箭头创建失败（点{i}）: {str(e)}")
            fail_count += 1
            continue
    
    print(f"  箭头创建统计：成功={success_count}，失败={fail_count}")
    
    # 合并所有箭头（改进合并逻辑，确保兼容性）
    if arrows and len(arrows) > 0:
        try:
            # 尝试使用merge方法合并所有箭头
            combined = arrows[0]
            for arrow in arrows[1:]:
                combined = combined.merge(arrow)
            return combined
        except Exception as e1:
            try:
                # 备选合并方法：使用MultiBlock
                from pyvista import MultiBlock
                block = MultiBlock(arrows)
                return block.combine()
            except Exception as e2:
                print(f"   警告：合并箭头失败: {str(e1)}, {str(e2)}")
                return None
    else:
        return None

# ----------------------------
# 🔟 矢量场动画函数
# ----------------------------

def parameterize_arrow_points(arrow_points, arrow_direction, start_point=None):
    """
    参数化箭头上的点（沿流速方向）
    
    Args:
        arrow_points: 箭头上的点坐标 (N, 3)
        arrow_direction: 箭头方向向量 (3,)
        start_point: 箭头起点（尾部），如果为None则使用第一个点
    
    Returns:
        s: 归一化位置参数 (N,)，尾部s=0，头部s=1
    """
    if len(arrow_points) == 0:
        return np.array([])
    
    # 计算每个点沿箭头方向的投影
    arrow_dir_norm = arrow_direction / (np.linalg.norm(arrow_direction) + 1e-6)
    
    # 找到起点（尾部）
    if start_point is None:
        # 使用投影最小的点作为起点
        relative_pos = arrow_points - arrow_points[0]
        projections = np.dot(relative_pos, arrow_dir_norm)
        start_idx = np.argmin(projections)
        start_point = arrow_points[start_idx]
    
    # 计算每个点相对于起点的投影距离
    relative_pos = arrow_points - start_point
    projections = np.dot(relative_pos, arrow_dir_norm)
    
    # 归一化到 [0, 1]
    proj_min = projections.min()
    proj_max = projections.max()
    if proj_max > proj_min:
        s = (projections - proj_min) / (proj_max - proj_min)
    else:
        s = np.zeros(len(arrow_points))
    
    return s

def extract_arrow_segments(arrows_mesh, sample_points_coords, sample_velocities, arrow_scale_factor=50.0):
    """
    从箭头mesh中提取每个采样点对应的箭头段
    
    Args:
        arrows_mesh: PyVista PolyData，包含所有箭头
        sample_points_coords: 采样点坐标 (N, 3)
        sample_velocities: 采样点速度向量 (N, 3)
        arrow_scale_factor: 箭头缩放因子
    
    Returns:
        arrow_segments: 列表，每个元素是一个字典，包含：
            - 'points': 箭头点坐标
            - 'direction': 箭头方向
            - 'start_point': 起点（采样点）
            - 'indices': 在arrows_mesh中的点索引
    """
    arrow_segments = []
    arrows_points = arrows_mesh.points
    
    # 为每个采样点找到最近的箭头点作为起点
    for i, (sample_point, vel) in enumerate(zip(sample_points_coords, sample_velocities)):
        vel_norm = np.linalg.norm(vel)
        if vel_norm < 1e-6:
            continue
        
        vel_dir = vel / vel_norm
        
        # 找到距离采样点最近的箭头点
        distances = np.linalg.norm(arrows_points - sample_point, axis=1)
        start_idx = np.argmin(distances)
        start_point = arrows_points[start_idx]
        
        # 计算所有点沿箭头方向的投影
        relative_pos = arrows_points - start_point
        projections = np.dot(relative_pos, vel_dir)
        
        # 找到属于这个箭头的点（投影距离在合理范围内）
        # 箭头长度大约是速度大小 * arrow_scale_factor
        # 使用速度范围来估算箭头长度
        speed_max = np.max([np.linalg.norm(v) for v in sample_velocities]) if len(sample_velocities) > 0 else 1.0
        arrow_length = (vel_norm / speed_max) * arrow_scale_factor if speed_max > 0 else arrow_scale_factor
        valid_mask = (projections >= -arrow_length * 0.1) & (projections <= arrow_length * 1.1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # 按投影距离排序
            valid_projections = projections[valid_indices]
            sort_idx = np.argsort(valid_projections)
            sorted_indices = valid_indices[sort_idx]
            
            arrow_segments.append({
                'points': arrows_points[sorted_indices],
                'direction': vel_dir,
                'start_point': start_point,
                'indices': sorted_indices,
                'sample_idx': i
            })
    
    return arrow_segments

def compute_flow_brightness(s, cycle_time):
    """
    计算静止帧时的流动亮度（从尾部到头部闪动效果）
    
    Args:
        s: 沿箭头方向的归一化位置 (N,)，0=尾部，1=头部
        cycle_time: 周期时间 (0-1)
    
    Returns:
        brightness: 亮度值 (N,)，范围 [0, 1]
    """
    # 闪动效果：亮度波从尾部向头部传播
    # 使用正弦波，相位沿箭头方向传播
    # phase = (s + cycle_time) % 1.0 表示波从尾部(s=0)向头部(s=1)传播
    phase = (s + cycle_time) % 1.0
    # 使用正弦波实现平滑的闪动效果
    brightness = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi))
    return brightness

def compute_temporal_alpha(s, frame_time):
    """
    计算播放帧时的渐显渐隐透明度
    箭头从尾部到头部逐渐出现，然后从尾部到头部逐渐消失
    
    Args:
        s: 沿箭头方向的归一化位置 (N,)，0=尾部，1=头部
        frame_time: 帧内时间 (0-1)
    
    Returns:
        alpha: 透明度值 (N,)，范围 [0, 1]
    """
    if frame_time < 0.5:
        # 渐显阶段（0-0.5）：从尾部到头部逐渐出现
        # frame_time从0到0.5，threshold从0到1
        threshold = 2.0 * frame_time  # 0到1
        # s <= threshold的点显示，s > threshold的点隐藏
        alpha = np.where(s <= threshold, 1.0, 0.0)
    else:
        # 渐隐阶段（0.5-1.0）：从尾部到头部逐渐消失
        # frame_time从0.5到1.0，threshold从0到1
        threshold = 2.0 * (frame_time - 0.5)  # 0到1
        # s > threshold的点显示，s <= threshold的点隐藏（从尾部开始消失）
        alpha = np.where(s > threshold, 1.0, 0.0)
    
    return alpha

def compute_transition_alpha(s, transition_progress):
    """
    计算过渡动画时的渐显透明度
    箭头从尾部到头部逐渐出现，在过渡完成时（progress=1.0）完全显现
    
    Args:
        s: 沿箭头方向的归一化位置 (N,)，0=尾部，1=头部
        transition_progress: 过渡进度 (0-1)，0=开始，1=完成
    
    Returns:
        alpha: 透明度值 (N,)，范围 [0, 1]
    """
    # transition_progress从0到1，threshold从0到1
    # 当progress=0时，threshold=0，只有s=0的点（尾部）显示
    # 当progress=1时，threshold=1，所有点（s<=1）都显示
    threshold = transition_progress  # 0到1
    
    # s <= threshold的点显示，s > threshold的点隐藏
    # 使用平滑过渡，避免硬边界
    # 在threshold附近添加一个小的过渡区域，使渐显更平滑
    transition_width = 0.1  # 过渡区域宽度（10%）
    alpha = np.clip((threshold - s) / transition_width + 0.5, 0.0, 1.0)
    
    return alpha

# ----------------------------
# 9️⃣ 主程序：创建可视化
# ----------------------------
print("\n" + "="*60)
print("三维海气立方体动态可视化")
print("="*60)

# 创建动画控制器
anim_controller = AnimationController(total_frames=len(time_series_data['time_steps']), fps=1.0)

# 创建Plotter
plotter = pv.Plotter(window_size=(1400, 900))

# 设置背景色
plotter.background_color = (0.08, 0.12, 0.18)

# 启用深度剥离
try:
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
    print("✅ 已启用深度剥离")
except Exception:
    pass

# 初始化网格和体积渲染
grid = pv.StructuredGrid(X, Y, Z)
combined_volume = pv.StructuredGrid(X, Y, Z)

# 初始化标量场数据
theta_data = Theta_local.flatten(order="F")
salt_data = Salt_local.flatten(order="F")
combined_volume["Temperature"] = theta_data
combined_volume["Salinity"] = salt_data

# 计算所有时间帧的全局数据范围（确保整个动画过程中映射规则统一）
print("正在计算所有时间帧的全局数据范围...")
all_salt_data = np.concatenate([data.flatten() for data in time_series_data['Salt']])
all_theta_data = np.concatenate([data.flatten() for data in time_series_data['Theta']])

salt_min_val = np.min(all_salt_data)
salt_max_val = np.max(all_salt_data)
temp_min_val = np.min(all_theta_data)
temp_max_val = np.max(all_theta_data)

print(f"全局盐度范围: [{salt_min_val:.4f}, {salt_max_val:.4f}]")
print(f"全局温度范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")

# 计算盐度梯度（用于策略19）
print("正在计算盐度梯度...")
salt_3d = salt_data.reshape(nx, ny, nz, order='F')
grad_x, grad_y, grad_z = np.gradient(salt_3d)
salt_gradient = np.stack([
    grad_x.flatten(order='F'),
    grad_y.flatten(order='F'),
    grad_z.flatten(order='F')
], axis=1)
salt_gradient_mag = np.linalg.norm(salt_gradient, axis=1)
if salt_gradient_mag.max() > salt_gradient_mag.min():
    salt_gradient_norm = (salt_gradient_mag - salt_gradient_mag.min()) / (salt_gradient_mag.max() - salt_gradient_mag.min())
else:
    salt_gradient_norm = np.zeros_like(salt_gradient_mag)

# 使用策略19计算初始透明度（使用全局盐度范围）
final_opacity = opacity_strategy_19(salt_data, salt_gradient_norm, 
                                     salt_min_global=salt_min_val, 
                                     salt_max_global=salt_max_val)
print(f"策略19透明度范围: [{final_opacity.min():.4f}, {final_opacity.max():.4f}]")

# 计算全局盐度梯度范围（用于策略19，可选，用于后续分析）
all_salt_gradient_norms = []
for salt_frame in time_series_data['Salt']:
    salt_3d_frame = salt_frame.reshape(nx, ny, nz, order='F')
    grad_x, grad_y, grad_z = np.gradient(salt_3d_frame)
    salt_gradient_frame = np.stack([
        grad_x.flatten(order='F'),
        grad_y.flatten(order='F'),
        grad_z.flatten(order='F')
    ], axis=1)
    salt_gradient_mag_frame = np.linalg.norm(salt_gradient_frame, axis=1)
    if salt_gradient_mag_frame.max() > salt_gradient_mag_frame.min():
        salt_gradient_norm_frame = (salt_gradient_mag_frame - salt_gradient_mag_frame.min()) / (salt_gradient_mag_frame.max() - salt_gradient_mag_frame.min())
    else:
        salt_gradient_norm_frame = np.zeros_like(salt_gradient_mag_frame)
    all_salt_gradient_norms.append(salt_gradient_norm_frame)

# 计算全局梯度归一化范围（用于后续归一化）
global_gradient_min = min([np.min(g) for g in all_salt_gradient_norms])
global_gradient_max = max([np.max(g) for g in all_salt_gradient_norms])
print(f"全局盐度梯度归一化范围: [{global_gradient_min:.4f}, {global_gradient_max:.4f}]")

# 添加体积渲染
volume_actor = plotter.add_volume(
    combined_volume,
    scalars="Temperature",
    cmap="hot",
    opacity=0.1,
    opacity_unit_distance=5,
    show_scalar_bar=True,
    scalar_bar_args={'title': '温度 (Temperature) - 颜色'},
    shade=True,
    ambient=0.1,
    blending='composite'
)

# 使用VTK底层API设置策略19的透明度映射
if VTK_AVAILABLE:
    try:
        mapper = volume_actor.GetMapper()
        vtk_volume = mapper.GetInput()
        volume_property = volume_actor.GetProperty()
        
        # 确保盐度数据在PointData中
        salt_vtk_array = vtk_volume.GetPointData().GetArray("Salinity")
        if salt_vtk_array is None:
            salt_vtk_array = numpy_to_vtk(salt_data.astype(np.float32), array_type=vtk.VTK_FLOAT)
            salt_vtk_array.SetName("Salinity")
            vtk_volume.GetPointData().AddArray(salt_vtk_array)
        
        # 创建透明度传递函数（基于温度值，但使用策略19计算的透明度）
        n_bins = 512
        opacity_func = vtk.vtkPiecewiseFunction()
        temp_vals = np.linspace(temp_min_val, temp_max_val, n_bins)
        temp_tolerance = (temp_max_val - temp_min_val) / n_bins * 2
        
        for t in temp_vals:
            temp_mask = np.abs(theta_data - t) <= temp_tolerance
            if np.any(temp_mask):
                corresponding_opacities = final_opacity[temp_mask]
                avg_opacity = np.mean(corresponding_opacities)
                avg_opacity = np.clip(avg_opacity, final_opacity.min(), final_opacity.max())
                opacity_func.AddPoint(t, avg_opacity)
            else:
                temp_norm = (t - temp_min_val) / (temp_max_val - temp_min_val) if (temp_max_val - temp_min_val) > 0 else 0
                opacity = final_opacity.min() + (final_opacity.max() - final_opacity.min()) * temp_norm
                opacity = np.clip(opacity, final_opacity.min(), final_opacity.max())
                opacity_func.AddPoint(t, opacity)
        
        # 设置边界值
        min_temp_mask = np.abs(theta_data - temp_min_val) < temp_tolerance
        if np.any(min_temp_mask):
            min_opacity = np.mean(final_opacity[min_temp_mask])
            opacity_func.AddPoint(temp_min_val, np.clip(min_opacity, final_opacity.min(), final_opacity.max()))
        else:
            opacity_func.AddPoint(temp_min_val, final_opacity.min())
        
        max_temp_mask = np.abs(theta_data - temp_max_val) < temp_tolerance
        if np.any(max_temp_mask):
            max_opacity = np.mean(final_opacity[max_temp_mask])
            opacity_func.AddPoint(temp_max_val, np.clip(max_opacity, final_opacity.min(), final_opacity.max()))
        else:
            opacity_func.AddPoint(temp_max_val, final_opacity.max())
        
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.SetScalarOpacityUnitDistance(5.0)
        
        # 自适应颜色映射（5%-95%分位数）- 使用全局范围
        # 定义为全局变量，以便在update_animation中使用
        global temp_percentile_5, temp_percentile_95
        temp_percentile_5 = np.percentile(all_theta_data, 5)
        temp_percentile_95 = np.percentile(all_theta_data, 95)
        
        try:
            import matplotlib.colormaps as cmaps
            hot_r_cmap = cmaps['hot_r']
        except (ImportError, KeyError):
            hot_r_cmap = plt.cm.get_cmap('hot_r')
        
        color_func = vtk.vtkColorTransferFunction()
        if (temp_max_val - temp_min_val) > 0:
            n_control_points = 10
            temp_vals = np.linspace(temp_percentile_5, temp_percentile_95, n_control_points)
            mid_start_idx = 0
            mid_end_idx = int(n_control_points * 0.7)
            mid_temp_vals = temp_vals[mid_start_idx:mid_end_idx]
            mid_cmap_vals = np.linspace(0.1, 0.7, len(mid_temp_vals))
            extreme_temp_vals = temp_vals[mid_end_idx:]
            extreme_cmap_vals = np.linspace(0.7, 0.9, len(extreme_temp_vals))
            
            for temp_val, cmap_val in zip(mid_temp_vals, mid_cmap_vals):
                rgba = hot_r_cmap(cmap_val)
                color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
            
            for temp_val, cmap_val in zip(extreme_temp_vals, extreme_cmap_vals):
                rgba = hot_r_cmap(cmap_val)
                color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
            
            rgba_min = hot_r_cmap(0.1)
            rgba_max = hot_r_cmap(0.9)
            color_func.AddRGBPoint(temp_percentile_5, rgba_min[0], rgba_min[1], rgba_min[2])
            color_func.AddRGBPoint(temp_percentile_95, rgba_max[0], rgba_max[1], rgba_max[2])
        else:
            color_func.AddRGBPoint(temp_min_val, 0.5, 0.5, 0.5)
        
        volume_property.SetColor(color_func)
        
        # 启用三线性插值
        try:
            volume_property.SetInterpolationTypeToLinear()
        except:
            try:
                volume_property.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
            except:
                pass
        
        print("✅ 策略19透明度映射已应用")
    except Exception as e:
        print(f"警告：VTK底层API设置失败: {e}")

# 初始化矢量场（使用第一个时间步的数据）
vectors = np.stack([
    U_local.flatten(order="F"),
    V_local.flatten(order="F"),
    W_local.flatten(order="F")
], axis=1)
grid["velocity"] = vectors

# 创建采样点（参考 velocity_3D_vector_optimized.py）
# 优化：对于10x10x10立方体，减少采样点数量以提高性能
sampling_points_per_edge = 5  # 从10减少到5，采样点从1000减少到125
n_samples_x = min(sampling_points_per_edge, nx)
n_samples_y = min(sampling_points_per_edge, ny)
n_samples_z = min(sampling_points_per_edge, nz)

x_indices = np.linspace(0, nx-1, n_samples_x, dtype=int) if nx > 1 else np.array([0])
y_indices = np.linspace(0, ny-1, n_samples_y, dtype=int) if ny > 1 else np.array([0])
z_indices = np.linspace(0, nz-1, n_samples_z, dtype=int) if nz > 1 else np.array([0])

X_idx, Y_idx, Z_idx = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
X_idx = X_idx.flatten()
Y_idx = Y_idx.flatten()
Z_idx = Z_idx.flatten()

sample_points_coords = []
sample_velocities = []
sample_speeds = []

for i in range(len(X_idx)):
    x_idx, y_idx, z_idx = X_idx[i], Y_idx[i], Z_idx[i]
    x_idx = np.clip(x_idx, 0, nx-1)
    y_idx = np.clip(y_idx, 0, ny-1)
    z_idx = np.clip(z_idx, 0, nz-1)
    
    u_val = U_local[x_idx, y_idx, z_idx]
    v_val = V_local[x_idx, y_idx, z_idx]
    w_val = W_local[x_idx, y_idx, z_idx]
    
    vel = np.array([u_val, v_val, w_val])
    speed = np.linalg.norm(vel)
    
    point_idx = x_idx + y_idx * nx + z_idx * nx * ny
    coords = grid.points[point_idx]
    
    sample_points_coords.append(coords)
    sample_velocities.append(vel)
    sample_speeds.append(speed)

sample_points_coords = np.array(sample_points_coords)
sample_velocities = np.array(sample_velocities)
sample_speeds = np.array(sample_speeds)

# 创建采样点PolyData
sample_points = pv.PolyData(sample_points_coords)
sample_points["velocity"] = sample_velocities
sample_points["speed"] = sample_speeds

# 创建箭头（使用模式1：弯曲箭头）
speed_max = np.max(sample_speeds) if len(sample_speeds) > 0 else 1.0
arrow_scale_factor = 60.0 / speed_max if speed_max > 0 else 1.0

print("正在生成弯曲箭头（模式1）...")
arrows = create_bent_arrows(
    sample_points_coords,
    sample_velocities,
    sample_speeds,
    arrow_scale=60.0,
    k_neighbors=4,
    spline_degree=3,
    max_bend_factor=0.3
)

# 如果弯曲箭头生成失败，使用直线箭头
if arrows is None or arrows.n_points == 0:
    print("⚠️ 弯曲箭头生成失败，使用直线箭头")
    arrows = sample_points.glyph(
        orient='velocity',
        scale='speed',
        factor=arrow_scale_factor
    )

# 添加箭头到场景
arrow_actor = plotter.add_mesh(
    arrows,
    scalars='speed',
    cmap='cool',
    opacity=1.0,
    show_scalar_bar=True,
    scalar_bar_args={'title': '流速 (Speed)'},
    pickable=True,
    render_lines_as_tubes=True
)

# 存储箭头数据用于动画
arrow_points_data = arrows.points.copy()
arrow_velocities_data = sample_velocities.copy()

# 存储箭头actor引用（用于更新）
arrow_actor_ref = {'actor': arrow_actor, 'last_frame': -1}

print(f"✅ 初始场景创建完成")
print(f"   时间步数: {anim_controller.total_frames}")
print(f"   采样点数: {len(sample_points_coords)}")
print(f"   箭头数: {arrows.n_points}")

# ----------------------------
# 🔟 动画更新回调函数
# ----------------------------
# 缓存上次的更新状态，避免不必要的更新
last_update_state = {
    'frame_idx': -1,
    'current_frame': -1,
    'cycle_time': -1,
    'frame_time': -1,
    'is_transitioning': False,
    'transition_progress': 0.0
}

def update_animation():
    """更新动画（在交互循环中调用，实时更新插值后的标量场和矢量场）"""
    global last_update_state, arrows, arrow_points_data, arrow_velocities_data, arrow_actor_ref
    global sample_points, sample_velocities, sample_speeds, plotter, volume_actor
    global temp_min_val, temp_max_val, salt_min_val, salt_max_val
    global temp_percentile_5, temp_percentile_95, all_theta_data, update_count
    
    # 更新动画控制器（确保状态正确更新）
    # 注意：如果已经在timer_callback中调用过，这里会再次调用，但这是安全的（基于时间差）
    anim_controller.update()
    
    # 更新标量场
    current_frame = anim_controller.current_frame
    
    # 获取插值后的帧索引（支持过渡动画）
    frame_idx = anim_controller.get_interpolated_frame_index()
    
    # 检查是否需要更新标量场（帧变化时强制更新，或插值变化时更新）
    frame_changed = (int(current_frame) != int(last_update_state.get('current_frame', -1)))
    transition_changed = anim_controller.is_transitioning != last_update_state.get('is_transitioning', False)
    
    # 更新条件：帧变化、过渡状态变化、或插值变化超过阈值（实现平滑过渡）
    # 过渡时总是更新，确保平滑过渡
    # 降低阈值，确保过渡时每帧都更新（实现真正的平滑过渡）
    frame_idx_diff = abs(frame_idx - last_update_state.get('frame_idx', -1))
    need_update_scalar = (
        frame_changed or  # 帧切换时强制更新
        transition_changed or  # 过渡状态变化时强制更新
        anim_controller.is_transitioning or  # 过渡时总是更新，确保平滑过渡
        anim_controller.is_playing or  # 播放时总是更新，确保平滑过渡
        frame_idx_diff > 0.001  # 插值变化超过0.1%（进一步降低阈值，更频繁更新，确保平滑）
    )
    
    # 调试信息（仅在帧变化时打印）
    if frame_changed:
        print(f"🔄 帧变化: {last_update_state.get('current_frame', -1)} -> {current_frame}, 需要更新标量场: {need_update_scalar}")
    
    # 过渡时总是更新标量场，确保平滑过渡
    # 降低更新阈值，确保过渡时每帧都更新
    if need_update_scalar:
        # 插值标量场数据（支持过渡动画）
        # 仅在关键状态变化时打印调试信息，避免过度输出
        if frame_changed or transition_changed or (anim_controller.is_transitioning and update_count % 20 == 0):
            if anim_controller.is_transitioning:
                print(f"📊 正在更新标量场数据（过渡中: {anim_controller.transition_start_frame:.2f} -> {anim_controller.transition_target_frame}, "
                      f"进度: {anim_controller.transition_progress*100:.1f}%, frame_idx={frame_idx:.3f}）...")
            else:
                print(f"📊 正在更新标量场数据（帧 {current_frame}，frame_idx={frame_idx:.2f}）...")
        
        # 使用插值后的帧索引获取标量场数据（实时插值，确保平滑过渡）
        theta_interp = interpolate_scalar_field(time_series_data, frame_idx, 'Theta')
        salt_interp = interpolate_scalar_field(time_series_data, frame_idx, 'Salt')
        
        if frame_changed or transition_changed:
            print(f"   温度范围: [{np.min(theta_interp):.4f}, {np.max(theta_interp):.4f}]")
            print(f"   盐度范围: [{np.min(salt_interp):.4f}, {np.max(salt_interp):.4f}]")
        
        # 更新体积渲染数据
        theta_data_new = theta_interp.flatten(order="F")
        salt_data_new = salt_interp.flatten(order="F")
        
        combined_volume["Temperature"] = theta_data_new
        combined_volume["Salinity"] = salt_data_new
        
        # 重新计算盐度梯度（用于策略19）
        salt_3d_new = salt_data_new.reshape(nx, ny, nz, order='F')
        grad_x, grad_y, grad_z = np.gradient(salt_3d_new)
        salt_gradient_new = np.stack([
            grad_x.flatten(order='F'),
            grad_y.flatten(order='F'),
            grad_z.flatten(order='F')
        ], axis=1)
        salt_gradient_mag_new = np.linalg.norm(salt_gradient_new, axis=1)
        if salt_gradient_mag_new.max() > salt_gradient_mag_new.min():
            salt_gradient_norm_new = (salt_gradient_mag_new - salt_gradient_mag_new.min()) / (salt_gradient_mag_new.max() - salt_gradient_mag_new.min())
        else:
            salt_gradient_norm_new = np.zeros_like(salt_gradient_mag_new)
        
        # 使用策略19重新计算透明度（使用全局盐度范围）
        final_opacity_new = opacity_strategy_19(salt_data_new, salt_gradient_norm_new, 
                                                 salt_min_global=salt_min_val, 
                                                 salt_max_global=salt_max_val)
        
        # 更新体积渲染actor（通过VTK底层API）
        if VTK_AVAILABLE:
            try:
                mapper = volume_actor.GetMapper()
                vtk_volume = mapper.GetInput()
                volume_property = volume_actor.GetProperty()
                if vtk_volume is not None:
                    # 更新温度数据
                    temp_array = numpy_to_vtk(theta_data_new.astype(np.float32), array_type=vtk.VTK_FLOAT)
                    temp_array.SetName("Temperature")
                    vtk_volume.GetPointData().SetScalars(temp_array)
                    
                    # 更新盐度数据
                    salt_array = numpy_to_vtk(salt_data_new.astype(np.float32), array_type=vtk.VTK_FLOAT)
                    salt_array.SetName("Salinity")
                    vtk_volume.GetPointData().AddArray(salt_array)
                    
                    # 更新透明度传递函数（策略19）- 使用全局温度范围
                    n_bins = 512
                    opacity_func = vtk.vtkPiecewiseFunction()
                    temp_vals = np.linspace(temp_min_val, temp_max_val, n_bins)  # 使用全局范围
                    temp_tolerance = (temp_max_val - temp_min_val) / n_bins * 2
                    
                    # 定义全局透明度范围（策略19：0~0.25，已降低）
                    opacity_min_global = 0.0
                    opacity_max_global = 0.25  # 策略19的最大透明度（已降低）
                    
                    # 使用全局温度范围构建透明度映射
                    for t in temp_vals:
                        temp_mask = np.abs(theta_data_new - t) <= temp_tolerance
                        if np.any(temp_mask):
                            corresponding_opacities = final_opacity_new[temp_mask]
                            avg_opacity = np.mean(corresponding_opacities)
                            # 使用全局透明度范围进行归一化
                            avg_opacity = np.clip(avg_opacity, opacity_min_global, opacity_max_global)
                            opacity_func.AddPoint(t, avg_opacity)
                        else:
                            # 如果没有匹配的温度值，使用线性插值
                            temp_norm = (t - temp_min_val) / (temp_max_val - temp_min_val) if (temp_max_val - temp_min_val) > 0 else 0
                            opacity = opacity_min_global + (opacity_max_global - opacity_min_global) * temp_norm
                            opacity = np.clip(opacity, opacity_min_global, opacity_max_global)
                            opacity_func.AddPoint(t, opacity)
                    
                    # 设置边界值（使用全局范围）
                    min_temp_mask = np.abs(theta_data_new - temp_min_val) < temp_tolerance
                    if np.any(min_temp_mask):
                        min_opacity = np.mean(final_opacity_new[min_temp_mask])
                        opacity_func.AddPoint(temp_min_val, np.clip(min_opacity, opacity_min_global, opacity_max_global))
                    else:
                        opacity_func.AddPoint(temp_min_val, opacity_min_global)
                    
                    max_temp_mask = np.abs(theta_data_new - temp_max_val) < temp_tolerance
                    if np.any(max_temp_mask):
                        max_opacity = np.mean(final_opacity_new[max_temp_mask])
                        opacity_func.AddPoint(temp_max_val, np.clip(max_opacity, opacity_min_global, opacity_max_global))
                    else:
                        opacity_func.AddPoint(temp_max_val, opacity_max_global)
                    
                    volume_property.SetScalarOpacity(opacity_func)
                    
                    # 更新颜色映射函数（使用全局温度范围，基于插值后的温度数据更新颜色）
                    try:
                        import matplotlib.colormaps as cmaps
                        hot_r_cmap = cmaps['hot_r']
                    except (ImportError, KeyError):
                        hot_r_cmap = plt.cm.get_cmap('hot_r')
                    
                    color_func = vtk.vtkColorTransferFunction()
                    # 使用全局温度范围，但基于当前插值后的温度数据更新颜色
                    if (temp_max_val - temp_min_val) > 0:
                        n_control_points = 10
                        temp_vals_color = np.linspace(temp_percentile_5, temp_percentile_95, n_control_points)
                        mid_start_idx = 0
                        mid_end_idx = int(n_control_points * 0.7)
                        mid_temp_vals = temp_vals_color[mid_start_idx:mid_end_idx]
                        mid_cmap_vals = np.linspace(0.1, 0.7, len(mid_temp_vals))
                        extreme_temp_vals = temp_vals_color[mid_end_idx:]
                        extreme_cmap_vals = np.linspace(0.7, 0.9, len(extreme_temp_vals))
                        
                        for temp_val, cmap_val in zip(mid_temp_vals, mid_cmap_vals):
                            rgba = hot_r_cmap(cmap_val)
                            color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
                        
                        for temp_val, cmap_val in zip(extreme_temp_vals, extreme_cmap_vals):
                            rgba = hot_r_cmap(cmap_val)
                            color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
                        
                        rgba_min = hot_r_cmap(0.1)
                        rgba_max = hot_r_cmap(0.9)
                        color_func.AddRGBPoint(temp_percentile_5, rgba_min[0], rgba_min[1], rgba_min[2])
                        color_func.AddRGBPoint(temp_percentile_95, rgba_max[0], rgba_max[1], rgba_max[2])
                    else:
                        color_func.AddRGBPoint(temp_min_val, 0.5, 0.5, 0.5)
                    
                    volume_property.SetColor(color_func)
                    
                    # 强制更新VTK渲染管道
                    vtk_volume.Modified()
                    mapper.Modified()
                    volume_actor.Modified()
                    
                    if frame_changed:
                        print(f"✅ 标量场已更新（帧 {current_frame}）")
            except Exception as e:
                print(f"警告：更新体积渲染失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 更新缓存状态
        last_update_state['frame_idx'] = frame_idx
        last_update_state['current_frame'] = current_frame
        last_update_state['frame_time'] = anim_controller.frame_time if anim_controller.is_playing else -1
        last_update_state['is_transitioning'] = anim_controller.is_transitioning
        last_update_state['transition_progress'] = anim_controller.transition_progress
    
    # 更新矢量场数据（帧变化时强制重建箭头，过渡时也需要更新）
    # 过渡时也需要更新矢量场，确保箭头也平滑过渡
    need_rebuild_arrows = (
        (arrow_actor_ref['last_frame'] != current_frame) or 
        frame_changed or 
        anim_controller.is_transitioning  # 过渡时也需要更新矢量场
    )
    
    # 插值矢量场数据（支持播放模式和过渡模式的时间插值）
    # 过渡时也使用插值，确保矢量场平滑过渡
    if anim_controller.is_playing or anim_controller.is_transitioning:
        # 播放模式或过渡模式：使用插值后的帧索引进行插值
        U_frame = interpolate_scalar_field(time_series_data, frame_idx, 'U')
        V_frame = interpolate_scalar_field(time_series_data, frame_idx, 'V')
        W_frame = interpolate_scalar_field(time_series_data, frame_idx, 'W')
    else:
        # 静止模式：使用当前帧
        if current_frame < len(time_series_data['U']):
            U_frame = time_series_data['U'][current_frame]
            V_frame = time_series_data['V'][current_frame]
            W_frame = time_series_data['W'][current_frame]
        else:
            # 如果超出范围，使用最后一帧
            U_frame = time_series_data['U'][-1]
            V_frame = time_series_data['V'][-1]
            W_frame = time_series_data['W'][-1]
    
    # 更新矢量场（如果数据有效）
    if 'U_frame' in locals() and 'V_frame' in locals() and 'W_frame' in locals() and U_frame is not None:
        # 更新采样点速度
        for i in range(len(X_idx)):
            x_idx, y_idx, z_idx = X_idx[i], Y_idx[i], Z_idx[i]
            x_idx = np.clip(x_idx, 0, nx-1)
            y_idx = np.clip(y_idx, 0, ny-1)
            z_idx = np.clip(z_idx, 0, nz-1)
            
            u_val = U_frame[x_idx, y_idx, z_idx]
            v_val = V_frame[x_idx, y_idx, z_idx]
            w_val = W_frame[x_idx, y_idx, z_idx]
            
            vel = np.array([u_val, v_val, w_val])
            speed = np.linalg.norm(vel)
            
            sample_velocities[i] = vel
            sample_speeds[i] = speed
        
        # 更新网格速度向量
        vectors_new = np.stack([
            U_frame.flatten(order="F"),
            V_frame.flatten(order="F"),
            W_frame.flatten(order="F")
        ], axis=1)
        grid["velocity"] = vectors_new
        
        # 只在需要时重新生成箭头（使用模式1：弯曲箭头）
        if need_rebuild_arrows:
            sample_points["velocity"] = sample_velocities
            sample_points["speed"] = sample_speeds
            
            # 使用模式1生成弯曲箭头
            arrows_new = create_bent_arrows(
                sample_points_coords,
                sample_velocities,
                sample_speeds,
                arrow_scale=60.0,
                k_neighbors=4,
                spline_degree=3,
                max_bend_factor=0.3
            )
            
            # 如果弯曲箭头生成失败，使用直线箭头
            if arrows_new is None or arrows_new.n_points == 0:
                arrows_new = sample_points.glyph(
                    orient='velocity',
                    scale='speed',
                    factor=arrow_scale_factor
                )
            
            # 更新存储的箭头数据
            arrows = arrows_new
            arrow_points_data = arrows.points.copy()
            arrow_velocities_data = sample_velocities.copy()
            arrow_actor_ref['last_frame'] = current_frame
            
            # 更新箭头actor（直接更新mesh数据，而不是移除和重新添加）
            current_arrow_actor = arrow_actor_ref['actor']
            try:
                # 直接更新箭头mesh的数据（更高效，避免闪烁）
                mapper = current_arrow_actor.GetMapper()
                if mapper is not None:
                    mapper_input = mapper.GetInput()
                    if mapper_input is not None:
                        # 更新点坐标
                        mapper_input.SetPoints(pv.convert_array(arrows_new.points))
                        # 更新速度标量
                        if 'speed' in arrows_new.array_names:
                            speed_array = pv.convert_array(arrows_new['speed'])
                            speed_array.SetName('speed')
                            mapper_input.GetPointData().SetScalars(speed_array)
                        mapper_input.Modified()
                        mapper.Modified()
                        current_arrow_actor.Modified()
            except Exception as e:
                # 如果直接更新失败，尝试移除并重新添加
                try:
                    plotter.remove_actor(current_arrow_actor)
                    # 重新添加箭头
                    new_arrow_actor = plotter.add_mesh(
                        arrows_new,
                        scalars='speed',
                        cmap='cool',
                        opacity=1.0,
                        show_scalar_bar=False,  # 不重复显示标量条
                        pickable=True,
                        render_lines_as_tubes=True
                    )
                    # 更新引用
                    arrow_actor_ref['actor'] = new_arrow_actor
                    
                    # 调整箭头渲染属性
                    try:
                        arrow_property = new_arrow_actor.GetProperty()
                        arrow_property.SetOpacity(1.0)
                        if hasattr(arrow_property, 'SetRenderLinesAsTubes'):
                            arrow_property.SetRenderLinesAsTubes(True)
                        if hasattr(arrow_property, 'SetLineWidth'):
                            arrow_property.SetLineWidth(4.5)
                        if hasattr(arrow_property, 'SetDepthWrite'):
                            arrow_property.SetDepthWrite(False)
                    except:
                        pass
                except Exception as e2:
                    print(f"警告：无法更新箭头: {e}, {e2}")
            
            if frame_changed:
                print(f"✅ 矢量场已更新（帧 {current_frame}，箭头数: {arrows_new.n_points}）")
        else:
            # 使用现有箭头
            arrows_new = arrows
        
        # 提取箭头段（用于精确动画应用）
        arrow_segments = extract_arrow_segments(arrows_new, sample_points_coords, sample_velocities, arrow_scale_factor)
        
        # 初始化动画值数组（用于存储每个箭头点的动画值）
        arrow_animation_values = np.ones(arrows_new.n_points)  # 默认值（完全不透明）
        
        # 更新箭头动画效果
        if anim_controller.is_transitioning:
            # 过渡模式：箭头从尾部到头部逐渐出现（5秒过渡动画）
            # 确保所有箭头点都被处理
            processed_indices = set()
            for segment in arrow_segments:
                if len(segment['points']) == 0 or len(segment['indices']) == 0:
                    continue
                
                # 计算箭头点沿流速方向的参数化位置
                s = parameterize_arrow_points(
                    segment['points'],
                    segment['direction'],
                    segment['start_point']
                )
                
                # 计算过渡渐显透明度（基于transition_progress）
                alphas = compute_transition_alpha(s, anim_controller.transition_progress)
                
                # 应用到对应的箭头点（确保索引匹配）
                for idx, arrow_idx in enumerate(segment['indices']):
                    if arrow_idx < len(arrow_animation_values) and idx < len(alphas):
                        arrow_animation_values[arrow_idx] = alphas[idx]
                        processed_indices.add(arrow_idx)
                    elif arrow_idx < len(arrow_animation_values):
                        # 如果alphas长度不足，使用最后一个值或默认值
                        arrow_animation_values[arrow_idx] = alphas[-1] if len(alphas) > 0 else 1.0
                        processed_indices.add(arrow_idx)
            
            # 对于未处理的点，使用默认值（完全不透明）
            # 这不应该发生，但为了安全起见
            if len(processed_indices) < arrows_new.n_points:
                unprocessed = set(range(arrows_new.n_points)) - processed_indices
                for idx in unprocessed:
                    arrow_animation_values[idx] = 1.0
        elif anim_controller.is_playing:
            # 播放模式：应用渐显渐隐效果
            # 确保所有箭头点都被处理
            processed_indices = set()
            for segment in arrow_segments:
                if len(segment['points']) == 0 or len(segment['indices']) == 0:
                    continue
                
                # 计算箭头点沿流速方向的参数化位置
                s = parameterize_arrow_points(
                    segment['points'],
                    segment['direction'],
                    segment['start_point']
                )
                
                # 计算渐显渐隐透明度
                alphas = compute_temporal_alpha(s, anim_controller.frame_time)
                
                # 应用到对应的箭头点（确保索引匹配）
                for idx, arrow_idx in enumerate(segment['indices']):
                    if arrow_idx < len(arrow_animation_values) and idx < len(alphas):
                        arrow_animation_values[arrow_idx] = alphas[idx]
                        processed_indices.add(arrow_idx)
                    elif arrow_idx < len(arrow_animation_values):
                        # 如果alphas长度不足，使用最后一个值或默认值
                        arrow_animation_values[arrow_idx] = alphas[-1] if len(alphas) > 0 else 1.0
                        processed_indices.add(arrow_idx)
            
            # 对于未处理的点，使用默认值（完全不透明）
            # 这不应该发生，但为了安全起见
            if len(processed_indices) < arrows_new.n_points:
                unprocessed = set(range(arrows_new.n_points)) - processed_indices
                for idx in unprocessed:
                    arrow_animation_values[idx] = 1.0
        else:
            # 静止模式：应用流动亮度效果
            for segment in arrow_segments:
                if len(segment['points']) == 0:
                    continue
                
                # 计算箭头点沿流速方向的参数化位置
                s = parameterize_arrow_points(
                    segment['points'],
                    segment['direction'],
                    segment['start_point']
                )
                
                # 计算流动亮度
                brightnesses = compute_flow_brightness(s, anim_controller.cycle_time)
                
                # 应用到对应的箭头点（通过调整颜色值来模拟亮度）
                for idx, arrow_idx in enumerate(segment['indices']):
                    if arrow_idx < len(arrow_animation_values):
                        arrow_animation_values[arrow_idx] = brightnesses[idx] if idx < len(brightnesses) else 1.0
        
        # 检查是否需要更新箭头动画（更频繁更新以实现平滑动画）
        # 过渡时和播放时总是更新箭头动画，确保渐显效果
        need_update_arrow_anim = (
            anim_controller.is_transitioning or  # 过渡时总是更新
            anim_controller.is_playing or  # 播放时总是更新
            abs(anim_controller.cycle_time - last_update_state.get('cycle_time', -1)) > 0.01  # 静止模式：只在周期时间变化时更新
        )
        
        # 应用动画效果到箭头mesh
        if arrows_new.n_points > 0 and len(arrow_animation_values) == arrows_new.n_points:
            # 获取当前箭头actor
            current_arrow_actor = arrow_actor_ref['actor']
            
            if anim_controller.is_transitioning or anim_controller.is_playing:
                # 过渡模式或播放模式：使用动画值作为透明度因子（渐显效果）
                # 过渡时：箭头从尾部到头部逐渐出现（5秒过渡）
                # 播放时：箭头渐显渐隐效果
                # 总是更新，确保动画效果持续
                # 通过调整颜色值来模拟渐显效果
                # 获取原始速度值（从sample_speeds获取，而不是从arrows_new）
                # 重建速度值数组，确保与箭头点一一对应
                speed_values = []
                for segment in arrow_segments:
                    sample_idx = segment.get('sample_idx', 0)
                    if sample_idx < len(sample_speeds):
                        segment_speed = sample_speeds[sample_idx]
                        # 为这个段的所有点分配相同的速度
                        speed_values.extend([segment_speed] * len(segment['indices']))
                
                # 如果长度不匹配，使用默认值填充
                if len(speed_values) != arrows_new.n_points:
                    # 使用arrows_new中的speed字段（如果存在）
                    if 'speed' in arrows_new.array_names:
                        speed_values = arrows_new['speed'].copy()
                    else:
                        speed_values = np.ones(arrows_new.n_points) * np.mean(sample_speeds)
                else:
                    speed_values = np.array(speed_values)
                
                # 将动画值应用到速度值（模拟透明度效果）
                # arrow_animation_values范围是[0,1]，直接应用到速度值
                # 确保数组长度匹配
                if len(arrow_animation_values) == len(speed_values):
                    modified_speeds = speed_values * arrow_animation_values
                else:
                    # 如果长度不匹配，使用默认值
                    modified_speeds = speed_values * 1.0
                
                # 更新箭头mesh（总是更新，确保动画效果持续）
                # 过渡时和播放时都需要实时更新箭头动画
                if need_update_arrow_anim or anim_controller.is_transitioning:
                    try:
                        mapper_input = current_arrow_actor.GetMapper().GetInput()
                        if mapper_input is not None:
                            # 创建速度标量数组
                            speed_array = numpy_to_vtk(modified_speeds.astype(np.float32), array_type=vtk.VTK_FLOAT)
                            speed_array.SetName('speed')
                            mapper_input.GetPointData().SetScalars(speed_array)
                            mapper_input.Modified()
                            current_arrow_actor.GetMapper().Modified()
                            current_arrow_actor.Modified()
                    except Exception as e:
                        print(f"警告：更新箭头动画失败: {e}")
                
                # 更新状态缓存
                if anim_controller.is_transitioning:
                    last_update_state['transition_progress'] = anim_controller.transition_progress
                else:
                    last_update_state['frame_time'] = anim_controller.frame_time
            else:
                # 静止模式：使用动画值作为亮度因子
                if need_update_arrow_anim:
                    # 通过调整颜色值来模拟亮度流动效果
                    # 获取当前速度值
                    if 'speed' in arrows_new.array_names:
                        speed_values = arrows_new['speed']
                    else:
                        speed_values = np.ones(arrows_new.n_points) * np.mean(sample_speeds)
                    
                    # 将亮度值应用到速度值（模拟亮度效果）
                    # 亮度值范围[0,1]，映射到速度值的[0.3, 1.0]范围，保持可见性
                    brightness_factor = 0.3 + 0.7 * arrow_animation_values
                    modified_speeds = speed_values * brightness_factor
                    arrows_new['speed'] = modified_speeds
                    
                    # 更新箭头mesh
                    current_arrow_actor.GetMapper().GetInput().GetPointData().SetScalars(
                        numpy_to_vtk(modified_speeds.astype(np.float32), array_type=vtk.VTK_FLOAT)
                    )
                    current_arrow_actor.GetMapper().GetInput().Modified()
                    last_update_state['cycle_time'] = anim_controller.cycle_time

# 强制更新函数（用于键盘事件）
def force_update():
    """强制更新动画和渲染"""
    global last_update_state
    try:
        # 强制重置更新状态，确保触发更新
        old_frame = last_update_state.get('current_frame', -1)
        last_update_state['current_frame'] = -1
        last_update_state['frame_idx'] = -1
        
        # 调用更新函数
        update_animation()
        
        # 强制渲染
        plotter.render()
        if hasattr(plotter, 'renderer') and plotter.renderer is not None:
            plotter.renderer.GetRenderWindow().Render()
        
        print(f"✅ 强制更新完成（帧: {anim_controller.current_frame}）")
    except Exception as e:
        print(f"警告：强制更新失败: {e}")
        import traceback
        traceback.print_exc()

# 添加键盘回调（为每个按键创建独立的无参数回调函数）
def key_press_space():
    """空格键回调：播放/暂停"""
    if anim_controller.is_playing:
        anim_controller.pause()
        print("⏸ 暂停")
    else:
        anim_controller.play()
        print("▶ 播放")
        # 重置last_update_time，确保动画从当前时间开始
        anim_controller.last_update_time = time.time()
        # 重置frame_time，确保从当前帧开始播放
        anim_controller.frame_time = 0.0
    # 触发一次更新（确保状态变化后立即更新）
    # 定时器会持续更新，这里只触发一次初始更新
    update_animation()
    plotter.render()

def key_press_right():
    """右箭头键回调：下一帧（启动5秒过渡动画）"""
    # 确保过渡已完成（如果正在过渡，先完成它）
    if anim_controller.is_transitioning:
        # 如果正在过渡，直接跳到目标帧，然后开始新的过渡
        anim_controller.current_frame = anim_controller.transition_target_frame
        anim_controller.is_transitioning = False
        anim_controller.transition_progress = 0.0
        print(f"⏩ 中断当前过渡，跳到帧 {anim_controller.current_frame}")
    
    old_frame = anim_controller.current_frame
    target_frame = (old_frame + 1) % anim_controller.total_frames
    anim_controller.next_frame()  # 这会启动过渡动画
    print(f"⏭ 下一帧: {old_frame} -> {target_frame} (5秒过渡)")
    # 触发一次更新，开始过渡动画
    update_animation()
    plotter.render()

def key_press_left():
    """左箭头键回调：上一帧（启动5秒过渡动画）"""
    # 确保过渡已完成（如果正在过渡，先完成它）
    if anim_controller.is_transitioning:
        # 如果正在过渡，直接跳到目标帧，然后开始新的过渡
        anim_controller.current_frame = anim_controller.transition_target_frame
        anim_controller.is_transitioning = False
        anim_controller.transition_progress = 0.0
        print(f"⏩ 中断当前过渡，跳到帧 {anim_controller.current_frame}")
    
    old_frame = anim_controller.current_frame
    target_frame = (old_frame - 1) % anim_controller.total_frames
    anim_controller.prev_frame()  # 这会启动过渡动画
    print(f"⏮ 上一帧: {old_frame} -> {target_frame} (5秒过渡)")
    # 触发一次更新，开始过渡动画
    update_animation()
    plotter.render()

# 使用PyVista的键盘事件系统（无参数回调）
# 注意：PyVista的add_key_event需要在show()之前注册，但实际事件处理在show()之后
try:
    plotter.add_key_event('space', key_press_space)
    plotter.add_key_event('Right', key_press_right)
    plotter.add_key_event('Left', key_press_left)
    print("✅ 键盘事件已注册（PyVista方法）")
except Exception as e:
    print(f"警告：PyVista键盘事件注册失败: {e}")
    print("   将使用VTK add_observer方法作为备用")
    
    # 备用方法：使用VTK的AddObserver（在show()之后添加）
    def setup_keyboard_observer():
        """设置键盘事件观察者（备用方法）"""
        try:
            if hasattr(plotter, 'iren') and plotter.iren is not None:
                def key_press_observer(obj, event):
                    """键盘事件观察者（备用方法）"""
                    try:
                        key = plotter.iren.GetKeySym()
                        if key == 'space' or key == ' ':
                            key_press_space()
                        elif key == 'Right':
                            key_press_right()
                        elif key == 'Left':
                            key_press_left()
                    except Exception as e2:
                        pass
                
                plotter.iren.AddObserver("KeyPressEvent", key_press_observer)
                print("✅ 键盘事件已注册（VTK observer方法）")
        except Exception as e:
            print(f"警告：VTK键盘事件注册也失败: {e}")
    
    # 在show()之后添加观察者（与定时器设置合并）
    # 注意：这个会在show_with_timer中被调用
    pass  # setup_keyboard_observer将在show_with_timer中调用

print("\n" + "="*60)
print("控制说明：")
print("  空格键：播放/暂停")
print("  右箭头：下一帧")
print("  左箭头：上一帧")
print("="*60)

# 添加定时器回调以实现自动更新
update_count = 0
last_fps_time = time.time()

def timer_callback():
    """定时器回调函数（每帧调用，实时更新插值后的标量场和矢量场）"""
    global update_count, last_fps_time
    try:
        # 更新动画控制器状态（更新过渡进度、播放状态等）
        anim_controller.update()
        
        # 更新动画（这会实时更新插值后的标量场和矢量场）
        # update_animation()内部会：
        # 1. 获取插值后的帧索引（支持过渡动画）
        # 2. 实时插值标量场（温度、盐度）
        # 3. 实时更新体积渲染数据
        # 4. 实时插值矢量场（U、V、W）
        # 5. 实时更新箭头数据
        update_animation()
        
        # 总是渲染，确保动画效果可见
        plotter.render()
        if hasattr(plotter, 'renderer') and plotter.renderer is not None:
            plotter.renderer.GetRenderWindow().Render()
        
        # 每100次更新输出一次FPS信息
        update_count += 1
        if update_count % 100 == 0:
            current_time = time.time()
            fps = 100.0 / (current_time - last_fps_time)
            last_fps_time = current_time
            frame_info = f"帧 {anim_controller.current_frame}/{anim_controller.total_frames-1}"
            if anim_controller.is_transitioning:
                frame_info += f" (过渡中: {anim_controller.transition_progress*100:.1f}%)"
            elif anim_controller.is_playing:
                frame_info += f" (播放中, 帧内时间: {anim_controller.frame_time:.3f})"
            else:
                frame_info += f" (暂停, 周期时间: {anim_controller.cycle_time:.3f})"
            print(f"动画状态: {frame_info}, FPS: {fps:.1f}")
    except Exception as e:
        print(f"警告：动画更新异常: {e}")
        import traceback
        traceback.print_exc()

# 添加定时器（每50ms更新一次，约20fps）
# 优先使用PyVista的add_callback方法，更符合PyVista的设计
def setup_timer():
    """设置定时器（在show()之后调用）"""
    try:
        # 优先使用PyVista的add_callback方法
        if hasattr(plotter, 'add_callback'):
            try:
                # PyVista的add_callback会在每次渲染时调用
                plotter.add_callback(timer_callback, interval=50)  # 50ms间隔
                print("✅ 定时器已添加（PyVista add_callback方法，50ms间隔，约20fps）")
                return True
            except Exception as e:
                print(f"警告：PyVista add_callback失败: {e}，尝试VTK方法")
        
        # 备用方法：使用VTK底层API添加定时器
        if hasattr(plotter, 'iren') and plotter.iren is not None:
            # 使用VTK的AddObserver添加定时器
            def timer_observer(obj, event):
                try:
                    timer_callback()
                except Exception as e:
                    print(f"警告：定时器回调异常: {e}")
            plotter.iren.AddObserver("TimerEvent", timer_observer)
            timer_id = plotter.iren.CreateRepeatingTimer(50)  # 50ms间隔
            print("✅ 定时器已添加（VTK方法，50ms间隔，约20fps）")
            return timer_id
        else:
            print("警告：plotter.iren不可用，无法添加定时器")
            return None
    except Exception as e:
        print(f"警告：无法添加定时器: {e}")
        import traceback
        traceback.print_exc()
        return None

# 包装show()函数，在显示窗口后添加定时器和键盘事件
original_show = plotter.show
def show_with_timer():
    """显示窗口并设置定时器和键盘事件"""
    result = original_show()
    # 延迟设置定时器和键盘事件，确保窗口已完全初始化
    import threading
    def delayed_setup():
        import time
        time.sleep(0.2)  # 等待0.2秒确保窗口初始化完成
        setup_timer()
        # 如果PyVista键盘事件注册失败，使用VTK observer方法
        if not hasattr(plotter, '_keyboard_registered') or not plotter._keyboard_registered:
            try:
                if hasattr(plotter, 'iren') and plotter.iren is not None:
                    def key_press_observer(obj, event):
                        """键盘事件观察者（备用方法）"""
                        try:
                            key = plotter.iren.GetKeySym()
                            if key == 'space' or key == ' ':
                                key_press_space()
                            elif key == 'Right':
                                key_press_right()
                            elif key == 'Left':
                                key_press_left()
                        except Exception as e2:
                            pass
                    
                    plotter.iren.AddObserver("KeyPressEvent", key_press_observer)
                    print("✅ 键盘事件已注册（VTK observer方法，备用）")
            except Exception as e:
                print(f"警告：VTK键盘事件注册也失败: {e}")
    threading.Thread(target=delayed_setup, daemon=True).start()
    return result
plotter.show = show_with_timer

# 添加坐标轴
plotter.add_axes()

# 显示窗口
print("\n✅ 启动交互式窗口...")
print("   提示：动画将自动更新，使用空格键控制播放/暂停")
plotter.show()

