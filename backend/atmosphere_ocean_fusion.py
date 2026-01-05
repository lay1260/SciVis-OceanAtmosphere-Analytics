"""
大气-海洋立方体融合可视化
在海洋可视化基础上，添加对应经纬位置的大气立方体可视化
"""

import numpy as np
import OpenVisus as ov
import pyvista as pv
import matplotlib.pyplot as plt

# 检查依赖库
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告：SciPy不可用，将使用直线箭头（模式1优化需要SciPy）")

try:
    from scipy.integrate import solve_ivp
    SCIPY_INTEGRATE_AVAILABLE = True
except ImportError:
    SCIPY_INTEGRATE_AVAILABLE = False
    print("警告：scipy.integrate不可用，模式2（三维流线）将不可用")

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("警告：VTK不可用，将使用PyVista高层API")

# 导入数据提取函数
from data_extractor import (
    extract_data, 
    ATMOSPHERE_VARIABLES, 
    OCEAN_VARIABLES,
    get_all_face_coordinates,
    find_points_in_range
)

# ============================================================
# 1. 数据加载配置
# ============================================================

# 海洋数据URL
OCEAN_BASE_URL = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

def load_ocean_dataset(variable):
    """加载海洋数据集"""
    if variable in ["theta", "w"]:
        base_dir = f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
    elif variable == "u":
        base_dir = "mit_output/llc2160_arco/visus.idx"
    else:
        base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
    dataset_url = OCEAN_BASE_URL + base_dir
    return ov.LoadDataset(dataset_url)

# ============================================================
# 2. 大气数据提取（使用data_extractor的方法）
# ============================================================

def extract_atmosphere_data_for_region(
    lon_min, lon_max, lat_min, lat_max, 
    time_step, layer_min=0, layer_max=50,
    quality=-6
):
    """
    提取指定经纬范围的大气数据
    
    Returns:
        dict: 包含大气变量数据的字典
    """
    # 需要提取的大气变量
    atm_vars = ['T', 'QI', 'QL', 'RI', 'RL', 'U', 'V', 'W']
    
    print(f"\n{'='*60}")
    print("提取大气数据")
    print(f"  经纬范围: [{lon_min:.2f}, {lon_max:.2f}] × [{lat_min:.2f}, {lat_max:.2f}]")
    print(f"  时间步: {time_step}")
    print(f"  层数范围: [{layer_min}, {layer_max}]")
    print(f"{'='*60}\n")
    
    # 使用extract_data函数提取大气数据
    data = extract_data(
        variables=atm_vars,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_step=time_step,
        layer_min=layer_min,
        layer_max=layer_max,
        quality=quality
    )
    
    return data

# ============================================================
# 3. 辅助函数：弯曲箭头生成（简化版，用于模式1）
# ============================================================

def get_neighbors(sample_points, target_idx, k=5):
    """获取目标采样点的k个空间最近邻（含自身）"""
    target_point = sample_points[target_idx]
    distances = np.linalg.norm(sample_points - target_point, axis=1)
    neighbor_indices = np.argsort(distances)[:k]
    return neighbor_indices

def smooth_velocity_field(sample_points, velocities, sigma=1.0):
    """高斯卷积平滑速度场"""
    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed_vel = np.zeros_like(velocities)
        for i in range(3):
            smoothed_vel[:, i] = gaussian_filter1d(velocities[:, i], sigma=sigma)
        return smoothed_vel
    except ImportError:
        return velocities

def create_bent_arrows(sample_points, velocities, speeds, arrow_scale=60.0, 
                      k_neighbors=4, spline_degree=3, max_bend_factor=0.2):
    """
    生成三维弯曲箭头（完整实现，参考velocity_3D_vector_optimized.py）
    """
    if not SCIPY_AVAILABLE:
        print("⚠️  SciPy不可用，无法生成弯曲箭头")
        return None
    
    speed_range = [np.min(speeds), np.max(speeds)]
    if speed_range[1] > 0:
        scale_factor = arrow_scale / speed_range[1]
    else:
        scale_factor = arrow_scale
    
    arrows = []
    success_count = 0
    fail_count = 0
    
    for i in range(len(sample_points)):
        try:
            current_point = sample_points[i]
            current_vel = velocities[i]
            speed = speeds[i]
            
            if speed < 0.01 * speed_range[1]:
                fail_count += 1
                continue
            
            neighbors = get_neighbors(sample_points, i, k=k_neighbors)
            neighbor_points = sample_points[neighbors]
            neighbor_vels = velocities[neighbors]
            
            smoothed_vels = smooth_velocity_field(neighbor_points, neighbor_vels, sigma=0.8)
            
            num_points = 5
            curve_points = [current_point.copy()]
            current_pos = current_point.copy()
            total_length = speed * scale_factor
            
            for j in range(1, num_points):
                t = j / (num_points - 1)
                vel_idx = min(int(t * len(smoothed_vels)), len(smoothed_vels) - 1)
                dir_vec = smoothed_vels[vel_idx]
                
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 0:
                    dir_vec = dir_vec / dir_norm
                    
                    initial_dir = current_vel / np.linalg.norm(current_vel) if np.linalg.norm(current_vel) > 0 else dir_vec
                    angle = np.arccos(np.clip(np.dot(dir_vec, initial_dir), -1.0, 1.0))
                    
                    max_angle = max_bend_factor * np.pi/2
                    if angle > max_angle:
                        cross = np.cross(initial_dir, dir_vec)
                        cross_norm = np.linalg.norm(cross)
                        if cross_norm > 1e-6:
                            cross = cross / cross_norm
                            dir_vec = np.sin(max_angle) * np.cross(cross, initial_dir) + np.cos(max_angle) * initial_dir
                        else:
                            dir_vec = initial_dir
                
                step = dir_vec * (total_length / (num_points - 1))
                current_pos += step
                curve_points.append(current_pos.copy())
            
            if len(curve_points) >= 2:
                poly = pv.PolyData()
                poly.points = np.array(curve_points)
                
                lines = np.empty((len(curve_points)-1, 3), dtype=int)
                lines[:, 0] = 2
                for j in range(len(curve_points)-1):
                    lines[j, 1] = j
                    lines[j, 2] = j + 1
                
                poly.lines = lines
                
                tube_radius = 0.05 * scale_factor * (speed / speed_range[1]) if speed_range[1] > 0 else 0.05
                arrow_shaft = poly.tube(radius=tube_radius, n_sides=12)
                
                if len(curve_points) >= 2:
                    tip_direction = (curve_points[-1] - curve_points[-2])
                    tip_norm = np.linalg.norm(tip_direction)
                    if tip_norm > 1e-6:
                        tip_direction = tip_direction / tip_norm
                    else:
                        tip_direction = (curve_points[-1] - curve_points[0]) / np.linalg.norm(curve_points[-1] - curve_points[0])
                else:
                    tip_direction = np.array([1, 0, 0])
                
                cone_length = 0.3 * total_length
                cone_radius = 3 * tube_radius
                forward_offset = cone_length * 0.2
                cone_center = curve_points[-1] + tip_direction * (cone_length * 0.5 + forward_offset)
                cone = pv.Cone(
                    center=cone_center,
                    direction=tip_direction,
                    height=cone_length,
                    radius=cone_radius,
                    resolution=8
                )
                
                arrow = arrow_shaft.merge(cone)
                arrow['speed'] = np.full(arrow.n_points, speed)
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
    
    if arrows and len(arrows) > 0:
        try:
            combined = arrows[0]
            for arrow in arrows[1:]:
                combined = combined.merge(arrow)
            return combined
        except Exception as e1:
            try:
                from pyvista import MultiBlock
                block = MultiBlock(arrows)
                return block.combine()
            except Exception as e2:
                print(f"   警告：合并箭头失败: {str(e1)}, {str(e2)}")
                return None
    else:
        return None

# ============================================================
# 4. 大气数据融合：计算可见水分子含量W
# ============================================================

def compute_water_content(QI, QL, RI, RL):
    """
    计算可见水分子含量W
    
    W = 0.7 * Q_L * min(R_L_prime / 10, 1) + 0.3 * Q_I * min(R_I_prime / 20, 1)
    
    Args:
        QI: 云冰水质量分数 (kg/kg)
        QL: 云液态水质量分数 (kg/kg)
        RI: 冰相粒子有效半径 (m)
        RL: 液相粒子有效半径 (m)
    
    Returns:
        W: 可见水分子含量
    """
    # 转换单位：R_L 和 R_I 从 m 转换为 μm
    R_L_prime = RL * 1e6  # m -> μm
    R_I_prime = RI * 1e6  # m -> μm
    
    # 计算W
    W = (0.7 * QL * np.minimum(R_L_prime / 10, 1.0) + 
         0.3 * QI * np.minimum(R_I_prime / 20, 1.0))
    
    return W

# ============================================================
# 4. 海洋数据读取（参考velocity_3D_vector_optimized.py）
# ============================================================

def read_ocean_region(
    db, lat_start, lat_end, lon_start, lon_end, 
    nz, quality=-6, skip=None, time_step=0
):
    """
    读取海洋局部区域数据（优化版，参考velocity_3D_vector_optimized.py）
    
    Args:
        db: OpenVisus数据集
        lat_start, lat_end: 纬度范围
        lon_start, lon_end: 经度范围
        nz: 深度层数
        quality: 数据质量等级
        skip: 采样间隔（None表示不采样）
        time_step: 时间步索引
    
    Returns:
        局部区域数据数组
    """
    print(f"    读取数据: 纬度[{lat_start}, {lat_end}], 经度[{lon_start}, {lon_end}], 层数{nz}, 采样间隔{skip}")
    
    try:
        # 读取完整数据（只读取一次）
        data_full = db.read(time=time_step, quality=quality)
        
        if data_full is None or data_full.size == 0:
            print(f"    警告: 数据为空")
            return np.array([])
        
        lat_dim, lon_dim, depth_dim = data_full.shape
        print(f"    数据形状: ({lat_dim}, {lon_dim}, {depth_dim})")
        
        # 计算索引范围
        lat_idx_start = max(0, int(lat_dim * (lat_start + 90) / 180))
        lat_idx_end = min(lat_dim, int(lat_dim * (lat_end + 90) / 180) + 1)
        lon_idx_start = max(0, int(lon_dim * (lon_start + 180) / 360))
        lon_idx_end = min(lon_dim, int(lon_dim * (lon_end + 180) / 360) + 1)
        
        print(f"    索引范围: lat=[{lat_idx_start}:{lat_idx_end}], lon=[{lon_idx_start}:{lon_idx_end}]")
        
        # 检查索引范围
        if lat_idx_end <= lat_idx_start or lon_idx_end <= lon_idx_start:
            print(f"    警告: 索引范围无效，使用默认范围")
            lat_idx_start = 0
            lat_idx_end = min(lat_dim, 100)  # 限制范围避免过大
            lon_idx_start = 0
            lon_idx_end = min(lon_dim, 100)
        
        # 应用采样
        if skip is None or skip <= 1:
            result = data_full[lat_idx_start:lat_idx_end, 
                              lon_idx_start:lon_idx_end, 
                              :nz]
        else:
            result = data_full[lat_idx_start:lat_idx_end:skip,
                              lon_idx_start:lon_idx_end:skip,
                              :nz]
        
        # 检查结果
        if result.size == 0:
            print(f"    警告: 采样后数据为空，尝试使用skip=2")
            result = data_full[lat_idx_start:lat_idx_end:2,
                              lon_idx_start:lon_idx_end:2,
                              :nz]
        
        print(f"    读取完成: 形状{result.shape}")
        return result
        
    except Exception as e:
        print(f"    错误: 读取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])

# ============================================================
# 5. 构建3D网格
# ============================================================

def build_3d_grid(lon_min, lon_max, lat_min, lat_max, 
                  z_min, z_max, nx, ny, nz):
    """
    构建3D结构化网格
    
    Args:
        lon_min, lon_max: 经度范围
        lat_min, lat_max: 纬度范围
        z_min, z_max: 深度范围（z_min为底部，z_max为顶部）
        nx, ny, nz: 网格点数
    
    Returns:
        X, Y, Z: 网格坐标数组
    """
    x = np.linspace(lon_min, lon_max, nx)
    y = np.linspace(lat_min, lat_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z

# ============================================================
# 6. 主可视化函数
# ============================================================

def visualize_atmosphere_ocean_fusion(
    lon_min, lon_max, lat_min, lat_max,
    time_step=0,
    resolution='medium',
    quality=-6,
    scale_xy=25,
    vector_mode=1
):
    """
    大气-海洋立方体融合可视化
    
    Args:
        lon_min, lon_max, lat_min, lat_max: 经纬范围
        time_step: 时间步索引
        resolution: 分辨率选择 ('low', 'medium', 'high')
            - 'low': 采样间隔skip=4, 海洋层数=5, 大气层数=5, 箭头数=200
            - 'medium': 采样间隔skip=2, 海洋层数=10, 大气层数=10, 箭头数=500
            - 'high': 采样间隔skip=1, 海洋层数=20, 大气层数=20, 箭头数=1000
        quality: 数据质量等级
        scale_xy: XY方向缩放系数
        vector_mode: 矢量场可视化模式（1=弯曲箭头，2=流线）
    """
    # 根据分辨率设置参数
    resolution_configs = {
        'low': {
            'skip': 4,
            'ocean_nz': 5,
            'atmosphere_nz': 5,
            'n_arrows': 200,
            'n_streamlines': 200
        },
        'medium': {
            'skip': 2,
            'ocean_nz': 10,
            'atmosphere_nz': 10,
            'n_arrows': 500,
            'n_streamlines': 400
        },
        'high': {
            'skip': 1,
            'ocean_nz': 20,
            'atmosphere_nz': 20,
            'n_arrows': 1000,
            'n_streamlines': 800
        }
    }
    
    if resolution not in resolution_configs:
        print(f"警告: 无效的分辨率 '{resolution}'，使用'medium'")
        resolution = 'medium'
    
    config = resolution_configs[resolution]
    skip = config['skip']
    ocean_nz = config['ocean_nz']
    atmosphere_nz = config['atmosphere_nz']
    n_arrows = config['n_arrows']
    n_streamlines = config['n_streamlines']
    
    print(f"\n分辨率设置: {resolution}")
    print(f"  采样间隔: {skip}")
    print(f"  海洋层数: {ocean_nz}")
    print(f"  大气层数: {atmosphere_nz}")
    print(f"  箭头数量: {n_arrows}")
    print(f"  流线数量: {n_streamlines}")
    """
    大气-海洋立方体融合可视化
    
    Args:
        lon_min, lon_max, lat_min, lat_max: 经纬范围
        time_step: 时间步索引
        ocean_nz: 海洋层数
        atmosphere_nz: 大气层数
        quality: 数据质量等级
        scale_xy: XY方向缩放系数
        vector_mode: 矢量场可视化模式（1=弯曲箭头，2=流线）
    """
    print("\n" + "="*60)
    print("大气-海洋立方体融合可视化")
    print("="*60)
    
    # ============================================================
    # 步骤1: 提取大气数据
    # ============================================================
    print("\n【步骤1】提取大气数据...")
    atm_data = extract_atmosphere_data_for_region(
        lon_min, lon_max, lat_min, lat_max,
        time_step=time_step,
        layer_min=0,
        layer_max=min(50, atmosphere_nz - 1),
        quality=quality
    )
    
    # 检查大气数据
    if len(atm_data.get('lon', [])) == 0:
        print("错误：无法提取大气数据")
        return
    
    # 提取大气变量
    atm_T = atm_data.get('T', np.array([]))
    atm_QI = atm_data.get('QI', np.array([]))
    atm_QL = atm_data.get('QL', np.array([]))
    atm_RI = atm_data.get('RI', np.array([]))
    atm_RL = atm_data.get('RL', np.array([]))
    atm_U = atm_data.get('U', np.array([]))
    atm_V = atm_data.get('V', np.array([]))
    atm_W = atm_data.get('W', np.array([]))
    atm_lon = atm_data.get('lon', np.array([]))
    atm_lat = atm_data.get('lat', np.array([]))
    
    # 计算大气可见水分子含量W
    if (len(atm_QI) > 0 and len(atm_QL) > 0 and 
        len(atm_RI) > 0 and len(atm_RL) > 0):
        atm_water = compute_water_content(atm_QI, atm_QL, atm_RI, atm_RL)
    else:
        print("警告：无法计算大气可见水分子含量，使用QI+QL作为替代")
        atm_water = atm_QI + atm_QL if len(atm_QI) > 0 else np.zeros_like(atm_T)
    
    # 计算大气速度
    if len(atm_U) > 0 and len(atm_V) > 0 and len(atm_W) > 0:
        atm_velocity = np.stack([
            atm_U.flatten() if atm_U.ndim > 1 else atm_U,
            atm_V.flatten() if atm_V.ndim > 1 else atm_V,
            atm_W.flatten() if atm_W.ndim > 1 else atm_W
        ], axis=1)
        atm_speed = np.linalg.norm(atm_velocity, axis=1)
    else:
        atm_velocity = np.zeros((len(atm_T), 3))
        atm_speed = np.zeros(len(atm_T))
    
    print(f"  大气数据点数: {len(atm_lon)}")
    print(f"  大气温度范围: [{np.min(atm_T):.2f}, {np.max(atm_T):.2f}]")
    print(f"  大气水含量范围: [{np.min(atm_water):.6f}, {np.max(atm_water):.6f}]")
    
    # ============================================================
    # 步骤2: 读取海洋数据
    # ============================================================
    print("\n【步骤2】读取海洋数据...")
    
    # 加载海洋数据集
    U_db = load_ocean_dataset("u")
    V_db = load_ocean_dataset("v")
    W_db = load_ocean_dataset("w")
    Salt_db = load_ocean_dataset("salt")
    Theta_db = load_ocean_dataset("theta")
    
    # 读取局部区域数据（使用采样间隔优化性能）
    print(f"  使用采样间隔: {skip}")
    U_local = read_ocean_region(U_db, lat_min, lat_max, lon_min, lon_max, ocean_nz, quality, skip=skip, time_step=time_step)
    V_local = read_ocean_region(V_db, lat_min, lat_max, lon_min, lon_max, ocean_nz, quality, skip=skip, time_step=time_step)
    W_local = read_ocean_region(W_db, lat_min, lat_max, lon_min, lon_max, ocean_nz, quality, skip=skip, time_step=time_step)
    Salt_local = read_ocean_region(Salt_db, lat_min, lat_max, lon_min, lon_max, ocean_nz, quality, skip=skip, time_step=time_step)
    Theta_local = read_ocean_region(Theta_db, lat_min, lat_max, lon_min, lon_max, ocean_nz, quality, skip=skip, time_step=time_step)
    
    # 检查数据是否成功加载
    if (U_local.size == 0 or Salt_local.size == 0 or Theta_local.size == 0):
        print("错误：海洋数据加载失败，数据为空！")
        print("尝试减小skip值或检查数据源")
        return
    
    print(f"  海洋数据形状: {U_local.shape}")
    
    # 获取海洋网格尺寸
    ocean_nx, ocean_ny, ocean_nz = U_local.shape
    print(f"  海洋网格尺寸: nx={ocean_nx}, ny={ocean_ny}, nz={ocean_nz}")
    
    # 检查维度是否有效
    if ocean_nx < 1 or ocean_ny < 1 or ocean_nz < 1:
        print(f"  错误: 海洋数据维度无效 ({ocean_nx}, {ocean_ny}, {ocean_nz})")
        return
    
    # PyVista的StructuredGrid需要至少2x2x2的网格才能进行体积渲染
    # 如果任何维度只有1个元素，需要扩展或跳过体积渲染
    ocean_nx_safe = max(2, ocean_nx)
    ocean_ny_safe = max(2, ocean_ny)
    ocean_nz_safe = max(2, ocean_nz)
    
    if ocean_nx < 2 or ocean_ny < 2 or ocean_nz < 2:
        print(f"  警告: 海洋数据维度太小 ({ocean_nx}, {ocean_ny}, {ocean_nz})，扩展为 ({ocean_nx_safe}, {ocean_ny_safe}, {ocean_nz_safe}) 以支持体积渲染")
        # 扩展数据到至少2x2x2
        U_local = np.pad(U_local, ((0, ocean_nx_safe - ocean_nx), (0, ocean_ny_safe - ocean_ny), (0, ocean_nz_safe - ocean_nz)), mode='edge')
        V_local = np.pad(V_local, ((0, ocean_nx_safe - ocean_nx), (0, ocean_ny_safe - ocean_ny), (0, ocean_nz_safe - ocean_nz)), mode='edge')
        W_local = np.pad(W_local, ((0, ocean_nx_safe - ocean_nx), (0, ocean_ny_safe - ocean_ny), (0, ocean_nz_safe - ocean_nz)), mode='edge')
        Theta_local = np.pad(Theta_local, ((0, ocean_nx_safe - ocean_nx), (0, ocean_ny_safe - ocean_ny), (0, ocean_nz_safe - ocean_nz)), mode='edge')
        Salt_local = np.pad(Salt_local, ((0, ocean_nx_safe - ocean_nx), (0, ocean_ny_safe - ocean_ny), (0, ocean_nz_safe - ocean_nz)), mode='edge')
        ocean_nx, ocean_ny, ocean_nz = ocean_nx_safe, ocean_ny_safe, ocean_nz_safe
    
    # ============================================================
    # 步骤3: 构建海洋立方体网格
    # ============================================================
    print("\n【步骤3】构建海洋立方体网格...")
    
    # 海洋深度范围（负数，向下）
    ocean_z_min = -1000  # 底部
    ocean_z_max = 0      # 海面
    
    # 构建海洋网格
    ocean_x = np.linspace(lon_min, lon_max, ocean_nx) * scale_xy
    ocean_y = np.linspace(lat_min, lat_max, ocean_ny) * scale_xy
    ocean_z = np.linspace(ocean_z_min, ocean_z_max, ocean_nz)
    
    ocean_X, ocean_Y, ocean_Z = np.meshgrid(ocean_x, ocean_y, ocean_z, indexing='ij')
    ocean_X = ocean_X.transpose(1, 0, 2)
    ocean_Y = ocean_Y.transpose(1, 0, 2)
    ocean_Z = -ocean_Z.transpose(1, 0, 2)  # 向下为负
    
    ocean_grid = pv.StructuredGrid(ocean_X, ocean_Y, ocean_Z)
    ocean_vectors = np.stack([
        U_local.flatten(order="F"),
        V_local.flatten(order="F"),
        W_local.flatten(order="F")
    ], axis=1)
    ocean_grid["velocity"] = ocean_vectors
    ocean_grid["Temperature"] = Theta_local.flatten(order="F")
    ocean_grid["Salinity"] = Salt_local.flatten(order="F")
    ocean_speed = np.linalg.norm(ocean_vectors, axis=1)
    ocean_grid["speed"] = ocean_speed
    
    print(f"  海洋网格点数: {ocean_grid.n_points}")
    
    # ============================================================
    # 步骤4: 构建大气立方体网格
    # ============================================================
    print("\n【步骤4】构建大气立方体网格...")
    
    # 大气高度范围（正数，向上）
    # 大气立方体放在海洋立方体上方
    atmosphere_z_min = 0      # 海面/地面
    atmosphere_z_max = 20000  # 顶部（20km）
    
    # 需要将大气数据插值到规则网格
    # 大气数据是 (n_points, n_layers) 格式，需要转换为3D网格
    if len(atm_lon) > 0 and len(atm_T) > 0:
        # 获取大气数据的实际点数
        atm_n_points = len(atm_lon)
        
        # 确定大气网格尺寸（与海洋网格匹配）
        # 使用与海洋相同的水平分辨率
        atm_nx = ocean_nx
        atm_ny = ocean_ny
        atm_nz = atmosphere_nz
        
        # PyVista的StructuredGrid需要至少2x2x2的网格才能进行体积渲染
        # 如果任何维度只有1个元素，需要扩展
        atm_nx_original = atm_nx
        atm_ny_original = atm_ny
        atm_nz_original = atm_nz
        
        atm_nx_safe = max(2, atm_nx)
        atm_ny_safe = max(2, atm_ny)
        atm_nz_safe = max(2, atm_nz)
        
        need_expand = (atm_nx < 2 or atm_ny < 2 or atm_nz < 2)
        if need_expand:
            print(f"  警告: 大气数据维度太小 ({atm_nx}, {atm_ny}, {atm_nz})，扩展为 ({atm_nx_safe}, {atm_ny_safe}, {atm_nz_safe}) 以支持体积渲染")
            atm_nx, atm_ny, atm_nz = atm_nx_safe, atm_ny_safe, atm_nz_safe
        
        # 构建大气网格坐标
        atm_x = np.linspace(lon_min, lon_max, atm_nx) * scale_xy
        atm_y = np.linspace(lat_min, lat_max, atm_ny) * scale_xy
        atm_z = np.linspace(atmosphere_z_min, atmosphere_z_max, atm_nz)
        
        atm_X, atm_Y, atm_Z = np.meshgrid(atm_x, atm_y, atm_z, indexing='ij')
        atm_X = atm_X.transpose(1, 0, 2)
        atm_Y = atm_Y.transpose(1, 0, 2)
        atm_Z = atm_Z.transpose(1, 0, 2)
        
        # 创建大气网格
        atmosphere_grid = pv.StructuredGrid(atm_X, atm_Y, atm_Z)
        
        # 将大气数据插值到规则网格
        # 大气数据是 (n_points, n_layers) 格式，需要插值到 (nx, ny, nz)
        print(f"  插值大气数据: {atm_n_points}个点 -> ({atm_nx}, {atm_ny}, {atm_nz})网格")
        
        # 将大气数据插值到规则网格
        # 大气数据是 (n_points, n_layers) 格式，需要转换为3D网格 (nx, ny, nz)
        print(f"  处理大气数据: {atm_n_points}个点，{atm_T.shape[1] if atm_T.ndim == 2 else 1}层")
        
        # 简化方法：如果数据点数匹配，直接reshape；否则需要插值
        if atm_T.ndim == 2:
            n_layers = atm_T.shape[1]
            
            # 如果层数匹配，直接reshape
            if n_layers == atm_nz and atm_n_points == atm_nx * atm_ny:
                # 直接reshape（假设数据已经是规则排列）
                atm_T_3d = atm_T.reshape(atm_nx, atm_ny, atm_nz, order='F')
                atm_water_3d = atm_water.reshape(atm_nx, atm_ny, atm_nz, order='F')
                
                # 处理速度数据
                if atm_velocity.ndim == 2 and atm_velocity.shape[0] == atm_n_points:
                    # 速度是 (n_points, 3) 格式，需要扩展到所有层
                    atm_U_3d = np.zeros((atm_nx_original, atm_ny_original, atm_nz_original))
                    atm_V_3d = np.zeros((atm_nx_original, atm_ny_original, atm_nz_original))
                    atm_W_3d = np.zeros((atm_nx_original, atm_ny_original, atm_nz_original))
                    for k in range(atm_nz_original):
                        atm_U_3d[:, :, k] = atm_velocity[:, 0].reshape(atm_nx_original, atm_ny_original, order='F')
                        atm_V_3d[:, :, k] = atm_velocity[:, 1].reshape(atm_nx_original, atm_ny_original, order='F')
                        atm_W_3d[:, :, k] = atm_velocity[:, 2].reshape(atm_nx_original, atm_ny_original, order='F')
                    
                    # 如果需要扩展，使用边缘填充
                    if need_expand:
                        atm_U_3d = np.pad(atm_U_3d, ((0, atm_nx - atm_nx_original), (0, atm_ny - atm_ny_original), (0, atm_nz - atm_nz_original)), mode='edge')
                        atm_V_3d = np.pad(atm_V_3d, ((0, atm_nx - atm_nx_original), (0, atm_ny - atm_ny_original), (0, atm_nz - atm_nz_original)), mode='edge')
                        atm_W_3d = np.pad(atm_W_3d, ((0, atm_nx - atm_nx_original), (0, atm_ny - atm_ny_original), (0, atm_nz - atm_nz_original)), mode='edge')
                else:
                    atm_U_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_V_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_W_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                
                atm_velocity_3d = np.stack([atm_U_3d, atm_V_3d, atm_W_3d], axis=3)
                print("✅ 大气数据直接reshape完成")
            else:
                # 层数不匹配或点数不匹配，需要插值
                print(f"  警告：数据格式不匹配，进行插值（点数: {atm_n_points} vs {atm_nx*atm_ny}, 层数: {n_layers} vs {atm_nz}）")
                
                # 使用最近邻插值
                try:
                    from scipy.interpolate import griddata
                    
                    # 创建目标网格点（水平面）
                    target_xy = np.column_stack([
                        atm_X[:, :, 0].flatten(),
                        atm_Y[:, :, 0].flatten()
                    ])
                    
                    # 创建源数据点（使用大气数据的经纬度坐标）
                    source_xy = np.column_stack([
                        atm_lon * scale_xy,
                        atm_lat * scale_xy
                    ])
                    
                    # 对每一层进行插值
                    atm_T_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_water_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_U_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_V_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_W_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    
                    for k in range(atm_nz):
                        # 找到对应的源层
                        src_k = int(k * n_layers / atm_nz) if n_layers > 0 else 0
                        src_k = min(src_k, n_layers - 1)
                        
                        # 插值温度和水含量
                        if len(source_xy) > 0:
                            atm_T_3d[:, :, k] = griddata(
                                source_xy, atm_T[:, src_k], target_xy, 
                                method='nearest', fill_value=0
                            ).reshape(atm_nx, atm_ny)
                            atm_water_3d[:, :, k] = griddata(
                                source_xy, atm_water[:, src_k], target_xy, 
                                method='nearest', fill_value=0
                            ).reshape(atm_nx, atm_ny)
                            
                            # 插值速度
                            if atm_velocity.ndim == 2 and atm_velocity.shape[0] == atm_n_points:
                                atm_U_3d[:, :, k] = griddata(
                                    source_xy, atm_velocity[:, 0], target_xy, 
                                    method='nearest', fill_value=0
                                ).reshape(atm_nx, atm_ny)
                                atm_V_3d[:, :, k] = griddata(
                                    source_xy, atm_velocity[:, 1], target_xy, 
                                    method='nearest', fill_value=0
                                ).reshape(atm_nx, atm_ny)
                                atm_W_3d[:, :, k] = griddata(
                                    source_xy, atm_velocity[:, 2], target_xy, 
                                    method='nearest', fill_value=0
                                ).reshape(atm_nx, atm_ny)
                    
                    atm_velocity_3d = np.stack([atm_U_3d, atm_V_3d, atm_W_3d], axis=3)
                    print("✅ 大气数据插值完成")
                except Exception as e:
                    print(f"  警告：插值失败: {e}，使用零填充")
                    atm_T_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_water_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                    atm_velocity_3d = np.zeros((atm_nx, atm_ny, atm_nz, 3))
        else:
            print("  警告：大气数据格式不匹配，使用零填充")
            atm_T_3d = np.zeros((atm_nx, atm_ny, atm_nz))
            atm_water_3d = np.zeros((atm_nx, atm_ny, atm_nz))
            atm_velocity_3d = np.zeros((atm_nx, atm_ny, atm_nz, 3))
            # 简化方法：如果数据点数匹配，直接reshape
            if atm_T.ndim == 2 and atm_T.shape[0] == atm_n_points:
                n_layers = atm_T.shape[1]
                if n_layers == atm_nz:
                    # 直接reshape（假设数据已经是规则排列）
                    atm_T_3d = atm_T.reshape(atm_ny, atm_nx, atm_nz, order='F')
                    atm_water_3d = atm_water.reshape(atm_ny, atm_nx, atm_nz, order='F')
                    if atm_velocity.ndim == 2 and atm_velocity.shape[0] == atm_n_points:
                        atm_velocity_3d = atm_velocity.reshape(atm_ny, atm_nx, atm_nz, 3, order='F')
                    else:
                        atm_velocity_3d = np.zeros((atm_ny, atm_nx, atm_nz, 3))
                else:
                    # 层数不匹配，使用最近层
                    atm_T_3d = np.zeros((atm_ny, atm_nx, atm_nz))
                    atm_water_3d = np.zeros((atm_ny, atm_nx, atm_nz))
                    atm_velocity_3d = np.zeros((atm_ny, atm_nx, atm_nz, 3))
                    for k in range(atm_nz):
                        src_k = int(k * n_layers / atm_nz)
                        atm_T_3d[:, :, k] = atm_T[:, src_k].reshape(atm_ny, atm_nx, order='F')
                        atm_water_3d[:, :, k] = atm_water[:, src_k].reshape(atm_ny, atm_nx, order='F')
                        if atm_velocity.ndim == 2:
                            atm_velocity_3d[:, :, k, :] = atm_velocity[:, :].reshape(atm_ny, atm_nx, 3, order='F')
            else:
                print("  错误：无法处理大气数据格式")
                return
        
        # 设置大气网格数据
        atmosphere_grid["Temperature"] = atm_T_3d.flatten(order="F")
        atmosphere_grid["WaterContent"] = atm_water_3d.flatten(order="F")
        atmosphere_grid["velocity"] = atm_velocity_3d.reshape(-1, 3, order='F')
        atm_speed_3d = np.linalg.norm(atm_velocity_3d, axis=3)
        atmosphere_grid["speed"] = atm_speed_3d.flatten(order="F")
        
        print(f"  大气网格点数: {atmosphere_grid.n_points}")
        print(f"  大气温度范围: [{np.min(atm_T_3d):.2f}, {np.max(atm_T_3d):.2f}]")
        print(f"  大气水含量范围: [{np.min(atm_water_3d):.6f}, {np.max(atm_water_3d):.6f}]")
    else:
        print("  错误：无法构建大气网格（无数据点）")
        return
    
    # ============================================================
    # 步骤5: 创建分界面（不透明深蓝色）
    # ============================================================
    print("\n【步骤5】创建分界面...")
    
    # 在海面（z=0）创建分界面
    interface_x = np.linspace(lon_min, lon_max, max(ocean_nx, atm_nx)) * scale_xy
    interface_y = np.linspace(lat_min, lat_max, max(ocean_ny, atm_ny)) * scale_xy
    interface_X, interface_Y = np.meshgrid(interface_x, interface_y, indexing='ij')
    interface_X = interface_X.transpose(1, 0)
    interface_Y = interface_Y.transpose(1, 0)
    interface_Z = np.zeros_like(interface_X)
    
    # 创建分界面网格
    interface_grid = pv.StructuredGrid(interface_X, interface_Y, interface_Z)
    interface_grid["interface"] = np.ones(interface_grid.n_points)
    
    print(f"  分界面点数: {interface_grid.n_points}")
    
    # ============================================================
    # 步骤6: 创建Plotter并添加体积渲染
    # ============================================================
    print("\n【步骤6】创建可视化...")
    
    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.background_color = (0.08, 0.12, 0.18)
    
    try:
        plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        print("✅ 已启用深度剥离")
    except Exception as e:
        print(f"警告：无法启用深度剥离: {e}")
    
    # 计算全局数据范围（用于统一颜色映射）
    print(f"\n  [DEBUG] 开始计算全局数据范围...")
    print(f"  [DEBUG] Theta_local形状={Theta_local.shape}, atm_T_3d形状={atm_T_3d.shape}")
    print(f"  [DEBUG] Salt_local形状={Salt_local.shape}, atm_water_3d形状={atm_water_3d.shape}")
    
    # 检查 atm_water_3d 的形状，确保它是有效的数组
    if not isinstance(atm_water_3d, np.ndarray):
        print(f"  [DEBUG] 警告: atm_water_3d 不是 numpy 数组，类型={type(atm_water_3d)}")
        try:
            atm_water_3d = np.asarray(atm_water_3d)
        except Exception as e:
            print(f"  [DEBUG] 无法转换为 numpy 数组: {e}")
            atm_water_3d = np.zeros((atm_nx, atm_ny, atm_nz))
    
    print(f"  [DEBUG] atm_water_3d 最终形状={atm_water_3d.shape}, 大小={atm_water_3d.size}")
    
    try:
        temp_min = min(np.min(Theta_local), np.min(atm_T_3d))
        temp_max = max(np.max(Theta_local), np.max(atm_T_3d))
        salt_min = np.min(Salt_local)
        salt_max = np.max(Salt_local)
        
        # 安全地计算 water_min 和 water_max
        if atm_water_3d.size > 0:
            water_min = np.min(atm_water_3d)
            water_max = np.max(atm_water_3d)
        else:
            print(f"  [DEBUG] 警告: atm_water_3d 为空，使用默认值")
            water_min = 0.0
            water_max = 0.0
        
        print(f"  全局温度范围: [{temp_min:.2f}, {temp_max:.2f}]")
        print(f"  海洋盐度范围: [{salt_min:.2f}, {salt_max:.2f}]")
        print(f"  大气水含量范围: [{water_min:.6f}, {water_max:.6f}]")
    except Exception as e:
        print(f"  [DEBUG] 计算全局数据范围失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 添加海洋体积渲染（完整实现，参考velocity_3D_vector_optimized.py）
    print(f"\n  [DEBUG] 准备添加海洋体积渲染，ocean_grid.n_points={ocean_grid.n_points}")
    if ocean_grid.n_points > 0:
        # 计算盐度梯度（用于视觉权重优化）
        print(f"\n  [DEBUG] 开始计算盐度梯度")
        print(f"  [DEBUG] Salt_local形状={Salt_local.shape}, 大小={Salt_local.size}")
        print(f"  [DEBUG] 目标形状=({ocean_nx}, {ocean_ny}, {ocean_nz})")
        print(f"  [DEBUG] 维度检查: nx>=2? {ocean_nx >= 2}, ny>=2? {ocean_ny >= 2}, nz>=2? {ocean_nz >= 2}")
        
        # 检查维度是否足够计算梯度（至少需要2个元素）
        # 如果任何维度只有1个元素，使用零梯度
        if ocean_nx < 2 or ocean_ny < 2 or ocean_nz < 2:
            print(f"  [DEBUG] 警告: 数据维度太小 ({ocean_nx}, {ocean_ny}, {ocean_nz})，无法计算梯度，使用零梯度")
            salt_gradient_norm = np.zeros(Salt_local.size)
        else:
            try:
                print(f"  [DEBUG] 尝试reshape...")
                salt_3d = Salt_local.reshape(ocean_nx, ocean_ny, ocean_nz, order='F')
                print(f"  [DEBUG] salt_3d形状: {salt_3d.shape}")
                
                # 检查每个维度是否>=2，如果任何维度只有1个元素，使用零梯度
                if any(s < 2 for s in salt_3d.shape):
                    print(f"  [DEBUG] 警告: reshape后维度太小 {salt_3d.shape}，无法计算梯度，使用零梯度")
                    salt_gradient_norm = np.zeros(Salt_local.size)
                else:
                    print(f"  [DEBUG] 开始计算np.gradient...")
                    try:
                        grad_x, grad_y, grad_z = np.gradient(salt_3d)
                        print(f"  [DEBUG] 梯度计算成功")
                        salt_gradient = np.stack([
                            grad_x.flatten(order='F'),
                            grad_y.flatten(order='F'),
                            grad_z.flatten(order='F')
                        ], axis=1)
                        salt_gradient_mag = np.linalg.norm(salt_gradient, axis=1)
                        if salt_gradient_mag.max() > salt_gradient_mag.min():
                            salt_gradient_norm = (salt_gradient_mag - salt_gradient_mag.min()) / (salt_gradient_mag.max() - salt_gradient_mag.min() + 1e-6)
                        else:
                            salt_gradient_norm = np.zeros_like(salt_gradient_mag)
                        print(f"  [DEBUG] 盐度梯度计算完成")
                    except ValueError as e:
                        # 捕获梯度计算错误（维度太小），使用零梯度
                        if "too small" in str(e) or "gradient" in str(e).lower():
                            print(f"  [DEBUG] 梯度计算失败（维度太小）: {e}，使用零梯度")
                            salt_gradient_norm = np.zeros(Salt_local.size)
                        else:
                            raise  # 重新抛出其他错误
            except Exception as e:
                print(f"  [DEBUG] 异常捕获: 计算盐度梯度失败: {type(e).__name__}: {e}，使用零梯度")
                salt_gradient_norm = np.zeros(Salt_local.size)
        
        # 策略17：反幂函数映射（参考velocity_3D_vector_optimized.py）
        def opacity_strategy_17(salt_data, salt_gradient_norm):
            salt_threshold = np.percentile(salt_data, 80)
            salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold + 1e-6), 0.0, 1.0)
            base_opacity = 0 + 0.3 * (salt_norm ** 0.8)
            gradient_boost = 0.1 + 0.2 * salt_gradient_norm
            final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.3)
            return final_opacity
        
        final_opacity = opacity_strategy_17(Salt_local.flatten(order="F"), salt_gradient_norm)
        
        ocean_volume = plotter.add_volume(
            ocean_grid,
            scalars="Temperature",
            cmap="hot",
            opacity=0.1,
            opacity_unit_distance=5,
            show_scalar_bar=True,
            scalar_bar_args={'title': '温度 (Temperature) - 颜色'},
            shade=True,
            ambient=0.1,
            pickable=False,
            blending='composite'
        )
        print("✅ 已添加海洋体积渲染")
        
        # 使用VTK API设置透明度（基于盐度，策略17）
        if VTK_AVAILABLE:
            try:
                mapper = ocean_volume.GetMapper()
                vtk_volume = mapper.GetInput()
                volume_property = ocean_volume.GetProperty()
                
                # 添加盐度数据
                salt_vtk_array = vtk_volume.GetPointData().GetArray("Salinity")
                if salt_vtk_array is None:
                    salt_vtk_array = numpy_to_vtk(
                        Salt_local.flatten(order="F").astype(np.float32),
                        array_type=vtk.VTK_FLOAT
                    )
                    salt_vtk_array.SetName("Salinity")
                    vtk_volume.GetPointData().AddArray(salt_vtk_array)
                
                # 创建透明度映射（基于策略17）
                opacity_func = vtk.vtkPiecewiseFunction()
                n_bins = 512
                temp_vals = np.linspace(temp_min, temp_max, n_bins)
                temp_tolerance = (temp_max - temp_min) / n_bins * 2
                theta_flat = Theta_local.flatten(order="F")
                
                for t in temp_vals:
                    temp_mask = np.abs(theta_flat - t) <= temp_tolerance
                    if np.any(temp_mask):
                        avg_opacity = np.mean(final_opacity[temp_mask])
                    else:
                        avg_opacity = np.interp(t, [temp_min, temp_max], [final_opacity.min(), final_opacity.max()])
                    opacity_func.AddPoint(t, np.clip(avg_opacity, 0, 0.25))
                
                volume_property.SetScalarOpacity(opacity_func)
                volume_property.SetScalarOpacityUnitDistance(5)
                volume_property.SetInterpolationTypeToLinear()  # 三线性插值
                
                # 自适应颜色映射（分位数拉伸）
                temp_percentile_5 = np.percentile(Theta_local, 5)
                temp_percentile_95 = np.percentile(Theta_local, 95)
                
                try:
                    import matplotlib.colormaps as cmaps
                    hot_r_cmap = cmaps['hot_r']
                except (ImportError, KeyError):
                    try:
                        import matplotlib
                        if hasattr(matplotlib, 'colormaps'):
                            hot_r_cmap = matplotlib.colormaps['hot_r']
                        else:
                            hot_r_cmap = plt.cm.get_cmap('hot_r')
                    except (AttributeError, KeyError):
                        hot_r_cmap = plt.cm.hot_r
                
                color_func = vtk.vtkColorTransferFunction()
                if temp_max > temp_min:
                    n_control_points = 10
                    temp_vals = np.linspace(temp_percentile_5, temp_percentile_95, n_control_points)
                    mid_end_idx = int(n_control_points * 0.7)
                    mid_temp_vals = temp_vals[:mid_end_idx]
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
                
                volume_property.SetColor(color_func)
                
                # 光照设置
                volume_property.ShadeOn()
                volume_property.SetSpecular(0.05)
                avg_gradient_norm = np.mean(salt_gradient_norm)
                base_ambient = 0.15 + 0.35 * avg_gradient_norm
                base_diffuse = 0.4 + 0.6 * avg_gradient_norm
                high_gradient_ratio = np.mean(salt_gradient_norm > 0.5)
                salt_80_percentile = np.percentile(Salt_local, 80)
                high_salt_ratio = np.mean(Salt_local > salt_80_percentile)
                enhanced_ambient = min(base_ambient + 0.4 * high_gradient_ratio + 0.3 * high_salt_ratio + 0.15, 0.95)
                enhanced_diffuse = min(base_diffuse + 0.2, 1.0)
                volume_property.SetAmbient(enhanced_ambient)
                volume_property.SetDiffuse(enhanced_diffuse)
                
                print("✅ 已设置海洋透明度映射（策略17：反幂函数映射）")
            except Exception as e:
                print(f"警告：无法设置海洋透明度映射: {e}")
                import traceback
                traceback.print_exc()
    
    # 添加大气体积渲染（完整实现，类似海洋）
    if atmosphere_grid.n_points > 0:
        # 计算水含量梯度
        print(f"  计算水含量梯度: atm_water_3d形状={atm_water_3d.shape}, 目标形状=({atm_nx}, {atm_ny}, {atm_nz})")
        
        # 检查维度是否足够计算梯度（至少需要2个元素）
        # 如果任何维度只有1个元素，使用零梯度
        if atm_nx < 2 or atm_ny < 2 or atm_nz < 2:
            print(f"  警告: 数据维度太小 ({atm_nx}, {atm_ny}, {atm_nz})，无法计算梯度，使用零梯度")
            water_gradient_norm = np.zeros(atm_water_3d.size)
        else:
            try:
                water_3d = atm_water_3d.reshape(atm_nx, atm_ny, atm_nz, order='F')
                print(f"  water_3d形状: {water_3d.shape}")
                
                # 检查每个维度是否>=2，如果任何维度只有1个元素，使用零梯度
                if any(s < 2 for s in water_3d.shape):
                    print(f"  警告: reshape后维度太小 {water_3d.shape}，无法计算梯度，使用零梯度")
                    water_gradient_norm = np.zeros(atm_water_3d.size)
                else:
                    print(f"  [DEBUG] 开始计算np.gradient (water)...")
                    try:
                        grad_x, grad_y, grad_z = np.gradient(water_3d)
                        print(f"  [DEBUG] 梯度计算成功 (water)")
                        water_gradient = np.stack([
                            grad_x.flatten(order='F'),
                            grad_y.flatten(order='F'),
                            grad_z.flatten(order='F')
                        ], axis=1)
                        water_gradient_mag = np.linalg.norm(water_gradient, axis=1)
                        if water_gradient_mag.max() > water_gradient_mag.min():
                            water_gradient_norm = (water_gradient_mag - water_gradient_mag.min()) / (water_gradient_mag.max() - water_gradient_mag.min() + 1e-6)
                        else:
                            water_gradient_norm = np.zeros_like(water_gradient_mag)
                        print(f"  [DEBUG] 水含量梯度计算完成")
                    except ValueError as e:
                        # 捕获梯度计算错误（维度太小），使用零梯度
                        if "too small" in str(e) or "gradient" in str(e).lower():
                            print(f"  [DEBUG] 梯度计算失败（维度太小）: {e}，使用零梯度")
                            water_gradient_norm = np.zeros(atm_water_3d.size)
                        else:
                            raise  # 重新抛出其他错误
            except Exception as e:
                print(f"  警告: 计算水含量梯度失败: {e}，使用零梯度")
                water_gradient_norm = np.zeros(atm_water_3d.size)
        
        # 策略17：反幂函数映射（类似盐度）
        def opacity_strategy_17_water(water_data, water_gradient_norm):
            water_threshold = np.percentile(water_data, 80)
            water_norm = np.clip((water_data - water_threshold) / (water_data.max() - water_threshold + 1e-6), 0.0, 1.0)
            base_opacity = 0 + 0.3 * (water_norm ** 0.8)
            gradient_boost = 0.1 + 0.2 * water_gradient_norm
            final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.3)
            return final_opacity
        
        final_opacity_water = opacity_strategy_17_water(atm_water_3d.flatten(order="F"), water_gradient_norm)
        
        atm_volume = plotter.add_volume(
            atmosphere_grid,
            scalars="Temperature",
            cmap="hot",
            opacity=0.1,
            opacity_unit_distance=5,
            show_scalar_bar=False,
            shade=True,
            ambient=0.1,
            pickable=False,
            blending='composite'
        )
        print("✅ 已添加大气体积渲染")
        
        # 使用VTK API设置透明度（基于水含量，策略17）
        if VTK_AVAILABLE:
            try:
                mapper = atm_volume.GetMapper()
                vtk_volume = mapper.GetInput()
                volume_property = atm_volume.GetProperty()
                
                # 添加水含量数据
                water_vtk_array = vtk_volume.GetPointData().GetArray("WaterContent")
                if water_vtk_array is None:
                    water_vtk_array = numpy_to_vtk(
                        atm_water_3d.flatten(order="F").astype(np.float32),
                        array_type=vtk.VTK_FLOAT
                    )
                    water_vtk_array.SetName("WaterContent")
                    vtk_volume.GetPointData().AddArray(water_vtk_array)
                
                # 创建透明度映射（基于策略17）
                opacity_func = vtk.vtkPiecewiseFunction()
                n_bins = 512
                temp_vals = np.linspace(temp_min, temp_max, n_bins)
                temp_tolerance = (temp_max - temp_min) / n_bins * 2
                atm_T_flat = atm_T_3d.flatten(order="F")
                
                for t in temp_vals:
                    temp_mask = np.abs(atm_T_flat - t) <= temp_tolerance
                    if np.any(temp_mask):
                        avg_opacity = np.mean(final_opacity_water[temp_mask])
                    else:
                        avg_opacity = np.interp(t, [temp_min, temp_max], [final_opacity_water.min(), final_opacity_water.max()])
                    opacity_func.AddPoint(t, np.clip(avg_opacity, 0, 0.25))
                
                volume_property.SetScalarOpacity(opacity_func)
                volume_property.SetScalarOpacityUnitDistance(5)
                volume_property.SetInterpolationTypeToLinear()
                
                # 自适应颜色映射（类似海洋）
                temp_percentile_5 = np.percentile(atm_T_3d, 5)
                temp_percentile_95 = np.percentile(atm_T_3d, 95)
                
                try:
                    import matplotlib.colormaps as cmaps
                    hot_r_cmap = cmaps['hot_r']
                except (ImportError, KeyError):
                    try:
                        import matplotlib
                        if hasattr(matplotlib, 'colormaps'):
                            hot_r_cmap = matplotlib.colormaps['hot_r']
                        else:
                            hot_r_cmap = plt.cm.get_cmap('hot_r')
                    except (AttributeError, KeyError):
                        hot_r_cmap = plt.cm.hot_r
                
                color_func = vtk.vtkColorTransferFunction()
                if temp_max > temp_min:
                    n_control_points = 10
                    temp_vals = np.linspace(temp_percentile_5, temp_percentile_95, n_control_points)
                    mid_end_idx = int(n_control_points * 0.7)
                    mid_temp_vals = temp_vals[:mid_end_idx]
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
                
                volume_property.SetColor(color_func)
                
                # 光照设置
                volume_property.ShadeOn()
                volume_property.SetSpecular(0.05)
                avg_gradient_norm = np.mean(water_gradient_norm)
                base_ambient = 0.15 + 0.35 * avg_gradient_norm
                base_diffuse = 0.4 + 0.6 * avg_gradient_norm
                high_gradient_ratio = np.mean(water_gradient_norm > 0.5)
                water_80_percentile = np.percentile(atm_water_3d, 80)
                high_water_ratio = np.mean(atm_water_3d > water_80_percentile)
                enhanced_ambient = min(base_ambient + 0.4 * high_gradient_ratio + 0.3 * high_water_ratio + 0.15, 0.95)
                enhanced_diffuse = min(base_diffuse + 0.2, 1.0)
                volume_property.SetAmbient(enhanced_ambient)
                volume_property.SetDiffuse(enhanced_diffuse)
                
                print("✅ 已设置大气透明度映射（策略17：反幂函数映射）")
            except Exception as e:
                print(f"警告：无法设置大气透明度映射: {e}")
                import traceback
                traceback.print_exc()
    
    # 添加分界面
    plotter.add_mesh(
        interface_grid,
        color=(0.0, 0.2, 0.4),  # 深蓝色
        opacity=1.0,
        show_edges=False,
        pickable=False
    )
    print("✅ 已添加分界面（深蓝色，不透明）")
    
    # ============================================================
    # 步骤7: 添加矢量场可视化
    # ============================================================
    print("\n【步骤7】添加矢量场可视化...")
    
    # 使用传入的vector_mode参数（不再交互式输入）
    # 验证vector_mode有效性
    if vector_mode not in [1, 2]:
        print(f"⚠️  无效的矢量场模式 {vector_mode}，使用默认模式1")
        vector_mode = 1
    
    print(f"✅ 使用矢量场模式: {vector_mode} ({'弯曲箭头' if vector_mode == 1 else '三维流线'})")
    
    # 模式1: 弯曲箭头（完整实现，参考velocity_3D_vector_optimized.py）
    if vector_mode == 1:
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.interpolate import make_interp_spline
            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False
            print("警告：SciPy不可用，将使用直线箭头")
        
        # 海洋箭头
        if 'velocity' in ocean_grid.array_names and ocean_grid.n_points > 0:
            print(f"  生成海洋箭头（目标数量: {n_arrows}）...")
            
            # 创建采样点（均匀采样）
            sample_points_coords = []
            sample_velocities = []
            sample_speeds = []
            
            # 在立方体上均匀采样
            sampling_points_per_edge = int(np.ceil(n_arrows ** (1/3)))
            nx, ny, nz = ocean_nx, ocean_ny, ocean_nz
            
            # 计算采样索引
            x_indices = np.linspace(0, nx-1, min(sampling_points_per_edge, nx), dtype=int)
            y_indices = np.linspace(0, ny-1, min(sampling_points_per_edge, ny), dtype=int)
            z_indices = np.linspace(0, nz-1, min(sampling_points_per_edge, nz), dtype=int)
            
            X_idx, Y_idx, Z_idx = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
            X_idx = X_idx.flatten()
            Y_idx = Y_idx.flatten()
            Z_idx = Z_idx.flatten()
            
            # 获取采样点的速度和坐标
            for i in range(len(X_idx)):
                x_idx, y_idx, z_idx = X_idx[i], Y_idx[i], Z_idx[i]
                x_idx = np.clip(x_idx, 0, nx-1)
                y_idx = np.clip(y_idx, 0, ny-1)
                z_idx = np.clip(z_idx, 0, nz-1)
                
                point_idx = x_idx + y_idx * nx + z_idx * nx * ny
                if point_idx < ocean_grid.n_points:
                    coords = ocean_grid.points[point_idx]
                    vel = ocean_grid["velocity"][point_idx]
                    speed = ocean_grid["speed"][point_idx]
                    
                    sample_points_coords.append(coords)
                    sample_velocities.append(vel)
                    sample_speeds.append(speed)
            
            if len(sample_points_coords) > 0:
                sample_points_coords = np.array(sample_points_coords)
                sample_velocities = np.array(sample_velocities)
                sample_speeds = np.array(sample_speeds)
                
                # 创建采样点PolyData
                ocean_sample_points = pv.PolyData(sample_points_coords)
                ocean_sample_points["velocity"] = sample_velocities
                ocean_sample_points["speed"] = sample_speeds
                
                # 生成箭头
                if SCIPY_AVAILABLE:
                    # 使用弯曲箭头（完整实现）
                    ocean_arrows = create_bent_arrows(
                        sample_points_coords, 
                        sample_velocities, 
                        sample_speeds,
                        arrow_scale=60.0,
                        k_neighbors=4,
                        spline_degree=3,
                        max_bend_factor=0.3
                    )
                else:
                    # 使用直线箭头
                    ocean_arrows = ocean_sample_points.glyph(
                        orient='velocity',
                        scale='speed',
                        factor=50.0
                    )
                
                if ocean_arrows is not None and ocean_arrows.n_points > 0:
                    plotter.add_mesh(
                        ocean_arrows,
                        scalars='speed' if 'speed' in ocean_arrows.array_names else None,
                        cmap='cool',
                        opacity=1.0,
                        show_scalar_bar=True,
                        scalar_bar_args={'title': '海洋流速'},
                        render_lines_as_tubes=True
                    )
                    print(f"✅ 已添加海洋箭头（{len(sample_points_coords)}个采样点）")
        
        # 大气箭头（类似处理）
        if 'velocity' in atmosphere_grid.array_names and atmosphere_grid.n_points > 0:
            print(f"  生成大气箭头（目标数量: {n_arrows}）...")
            
            sample_points_coords = []
            sample_velocities = []
            sample_speeds = []
            
            nx, ny, nz = atm_nx, atm_ny, atm_nz
            sampling_points_per_edge = int(np.ceil(n_arrows ** (1/3)))
            
            x_indices = np.linspace(0, nx-1, min(sampling_points_per_edge, nx), dtype=int)
            y_indices = np.linspace(0, ny-1, min(sampling_points_per_edge, ny), dtype=int)
            z_indices = np.linspace(0, nz-1, min(sampling_points_per_edge, nz), dtype=int)
            
            X_idx, Y_idx, Z_idx = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
            X_idx = X_idx.flatten()
            Y_idx = Y_idx.flatten()
            Z_idx = Z_idx.flatten()
            
            for i in range(len(X_idx)):
                x_idx, y_idx, z_idx = X_idx[i], Y_idx[i], Z_idx[i]
                x_idx = np.clip(x_idx, 0, nx-1)
                y_idx = np.clip(y_idx, 0, ny-1)
                z_idx = np.clip(z_idx, 0, nz-1)
                
                point_idx = x_idx + y_idx * nx + z_idx * nx * ny
                if point_idx < atmosphere_grid.n_points:
                    coords = atmosphere_grid.points[point_idx]
                    vel = atmosphere_grid["velocity"][point_idx]
                    speed = atmosphere_grid["speed"][point_idx]
                    
                    sample_points_coords.append(coords)
                    sample_velocities.append(vel)
                    sample_speeds.append(speed)
            
            if len(sample_points_coords) > 0:
                sample_points_coords = np.array(sample_points_coords)
                sample_velocities = np.array(sample_velocities)
                sample_speeds = np.array(sample_speeds)
                
                atm_sample_points = pv.PolyData(sample_points_coords)
                atm_sample_points["velocity"] = sample_velocities
                atm_sample_points["speed"] = sample_speeds
                
                if SCIPY_AVAILABLE:
                    atm_arrows = create_bent_arrows(
                        sample_points_coords,
                        sample_velocities,
                        sample_speeds,
                        arrow_scale=60.0,
                        k_neighbors=4,
                        spline_degree=3,
                        max_bend_factor=0.3
                    )
                else:
                    atm_arrows = atm_sample_points.glyph(
                        orient='velocity',
                        scale='speed',
                        factor=50.0
                    )
                
                if atm_arrows is not None and atm_arrows.n_points > 0:
                    plotter.add_mesh(
                        atm_arrows,
                        scalars='speed' if 'speed' in atm_arrows.array_names else None,
                        cmap='cool',
                        opacity=1.0,
                        show_scalar_bar=False,
                        render_lines_as_tubes=True
                    )
                    print(f"✅ 已添加大气箭头（{len(sample_points_coords)}个采样点）")
    
    # 模式2: 三维流线（完整实现，参考velocity_3D_vector_optimized.py）
    elif vector_mode == 2:
        # 海洋流线
        if 'velocity' in ocean_grid.array_names and ocean_grid.n_points > 0:
            print(f"  生成海洋流线（目标数量: {n_streamlines}）...")
            try:
                # 优化种子点生成（高速度区域优先）
                min_effective_speed = 0.01
                speeds = ocean_grid['speed']
                high_speed_mask = speeds > min_effective_speed
                
                if np.any(high_speed_mask):
                    high_coords = ocean_grid.points[high_speed_mask]
                    n_high = min(n_streamlines // 2, len(high_coords))
                    if n_high > 0:
                        high_idx = np.random.choice(len(high_coords), size=n_high, replace=False)
                        high_seeds = high_coords[high_idx]
                        
                        low_coords = ocean_grid.points[~high_speed_mask]
                        n_low = min(n_streamlines - n_high, len(low_coords))
                        if n_low > 0:
                            low_idx = np.random.choice(len(low_coords), size=n_low, replace=(len(low_coords) < n_low))
                            low_seeds = low_coords[low_idx]
                            seed_points_coords = np.vstack([high_seeds, low_seeds])
                        else:
                            seed_points_coords = high_seeds
                    else:
                        seed_points_coords = ocean_grid.points[np.random.choice(len(ocean_grid.points), size=min(n_streamlines, len(ocean_grid.points)), replace=False)]
                else:
                    seed_points_coords = ocean_grid.points[np.random.choice(len(ocean_grid.points), size=min(n_streamlines, len(ocean_grid.points)), replace=False)]
                
                seed_points = pv.PolyData(seed_points_coords)
                print(f"    种子点数量: {seed_points.n_points}")
                
                ocean_streamlines = ocean_grid.streamlines_from_source(
                    source=seed_points,
                    vectors='velocity',
                    integration_direction='both',
                    initial_step_length=0.1,
                    terminal_speed=1e-3,
                    max_steps=2000
                )
                
                if ocean_streamlines.n_points > 0:
                    if 'velocity' in ocean_streamlines.array_names:
                        ocean_speed_stream = np.linalg.norm(ocean_streamlines['velocity'], axis=1)
                    else:
                        ocean_speed_stream = np.ones(ocean_streamlines.n_points) * np.mean(speeds)
                    ocean_streamlines['speed'] = ocean_speed_stream
                    
                    plotter.add_mesh(
                        ocean_streamlines,
                        scalars='speed',
                        cmap='cool',
                        line_width=2.0,
                        opacity=0.8,
                        show_scalar_bar=True,
                        scalar_bar_args={'title': '海洋流速'},
                        pickable=True
                    )
                    print(f"✅ 已添加海洋流线（{ocean_streamlines.n_points}个点）")
            except Exception as e:
                print(f"  警告：海洋流线生成失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 大气流线（类似处理）
        if 'velocity' in atmosphere_grid.array_names and atmosphere_grid.n_points > 0:
            print(f"  生成大气流线（目标数量: {n_streamlines}）...")
            try:
                speeds = atmosphere_grid['speed']
                high_speed_mask = speeds > 0.01
                
                if np.any(high_speed_mask):
                    high_coords = atmosphere_grid.points[high_speed_mask]
                    n_high = min(n_streamlines // 2, len(high_coords))
                    if n_high > 0:
                        high_idx = np.random.choice(len(high_coords), size=n_high, replace=False)
                        high_seeds = high_coords[high_idx]
                        
                        low_coords = atmosphere_grid.points[~high_speed_mask]
                        n_low = min(n_streamlines - n_high, len(low_coords))
                        if n_low > 0:
                            low_idx = np.random.choice(len(low_coords), size=n_low, replace=(len(low_coords) < n_low))
                            low_seeds = low_coords[low_idx]
                            seed_points_coords = np.vstack([high_seeds, low_seeds])
                        else:
                            seed_points_coords = high_seeds
                    else:
                        seed_points_coords = atmosphere_grid.points[np.random.choice(len(atmosphere_grid.points), size=min(n_streamlines, len(atmosphere_grid.points)), replace=False)]
                else:
                    seed_points_coords = atmosphere_grid.points[np.random.choice(len(atmosphere_grid.points), size=min(n_streamlines, len(atmosphere_grid.points)), replace=False)]
                
                seed_points = pv.PolyData(seed_points_coords)
                print(f"    种子点数量: {seed_points.n_points}")
                
                atm_streamlines = atmosphere_grid.streamlines_from_source(
                    source=seed_points,
                    vectors='velocity',
                    integration_direction='both',
                    initial_step_length=0.1,
                    terminal_speed=1e-3,
                    max_steps=2000
                )
                
                if atm_streamlines.n_points > 0:
                    if 'velocity' in atm_streamlines.array_names:
                        atm_speed_stream = np.linalg.norm(atm_streamlines['velocity'], axis=1)
                    else:
                        atm_speed_stream = np.ones(atm_streamlines.n_points) * np.mean(speeds)
                    atm_streamlines['speed'] = atm_speed_stream
                    
                    plotter.add_mesh(
                        atm_streamlines,
                        scalars='speed',
                        cmap='cool',
                        line_width=2.0,
                        opacity=0.8,
                        show_scalar_bar=False,
                        pickable=True
                    )
                    print(f"✅ 已添加大气流线（{atm_streamlines.n_points}个点）")
            except Exception as e:
                print(f"  警告：大气流线生成失败: {e}")
                import traceback
                traceback.print_exc()
    
    # ============================================================
    # 步骤8: 分别显示大气和海洋可视化，然后显示融合可视化
    # ============================================================
    print("\n" + "="*60)
    print("步骤8: 显示可视化")
    print("="*60)
    
    # 8.1: 先显示海洋可视化
    print("\n【8.1】显示海洋可视化...")
    ocean_plotter = pv.Plotter(window_size=(1200, 800))
    ocean_plotter.background_color = (0.08, 0.12, 0.18)
    ocean_plotter.add_text("海洋可视化", font_size=20, position='upper_left')
    
    try:
        ocean_plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
    except Exception as e:
        print(f"  警告：无法启用深度剥离: {e}")
    
    if ocean_grid.n_points > 0:
        # 添加海洋体积渲染（使用与融合可视化相同的设置）
        ocean_volume_solo = ocean_plotter.add_volume(
            ocean_grid,
            scalars="Temperature",
            cmap="hot",
            opacity=0.15,
            opacity_unit_distance=5,
            show_scalar_bar=True,
            scalar_bar_args={'title': '温度 (Temperature)'},
            shade=True,
            ambient=0.1,
            pickable=False,
            blending='composite'
        )
        
        # 如果之前计算了透明度映射，尝试应用
        if 'final_opacity' in locals() and VTK_AVAILABLE:
            try:
                mapper = ocean_volume_solo.GetMapper()
                vtk_volume = mapper.GetInput()
                volume_property = ocean_volume_solo.GetProperty()
                
                # 应用透明度映射（简化版）
                opacity_func = vtk.vtkPiecewiseFunction()
                opacity_func.AddPoint(temp_min, 0.0)
                opacity_func.AddPoint(temp_max, 0.25)
                volume_property.SetScalarOpacity(opacity_func)
                volume_property.SetScalarOpacityUnitDistance(5)
            except Exception as e:
                print(f"  警告：无法设置海洋透明度映射: {e}")
        
        # 添加海洋矢量场（简化版）
        if 'velocity' in ocean_grid.array_names:
            sample_indices = np.linspace(0, ocean_grid.n_points - 1, min(200, ocean_grid.n_points), dtype=int)
            sample_points = pv.PolyData(ocean_grid.points[sample_indices])
            sample_points["velocity"] = ocean_grid["velocity"][sample_indices]
            sample_points["speed"] = ocean_grid["speed"][sample_indices]
            ocean_arrows = sample_points.glyph(orient='velocity', scale='speed', factor=40.0)
            if ocean_arrows.n_points > 0:
                ocean_plotter.add_mesh(
                    ocean_arrows, 
                    scalars='speed', 
                    cmap='cool', 
                    opacity=0.9,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': '流速'}
                )
    
    ocean_plotter.add_axes()
    print("✅ 海洋可视化窗口已打开（关闭窗口后继续）")
    ocean_plotter.show(auto_close=False)
    
    # 8.2: 再显示大气可视化
    print("\n【8.2】显示大气可视化...")
    atmosphere_plotter = pv.Plotter(window_size=(1200, 800))
    atmosphere_plotter.background_color = (0.08, 0.12, 0.18)
    atmosphere_plotter.add_text("大气可视化", font_size=20, position='upper_left')
    
    try:
        atmosphere_plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
    except Exception as e:
        print(f"  警告：无法启用深度剥离: {e}")
    
    if atmosphere_grid.n_points > 0:
        # 添加大气体积渲染（使用与融合可视化相同的设置）
        atm_volume_solo = atmosphere_plotter.add_volume(
            atmosphere_grid,
            scalars="Temperature",
            cmap="hot",
            opacity=0.15,
            opacity_unit_distance=5,
            show_scalar_bar=True,
            scalar_bar_args={'title': '温度 (Temperature)'},
            shade=True,
            ambient=0.1,
            pickable=False,
            blending='composite'
        )
        
        # 如果之前计算了透明度映射，尝试应用
        if 'final_opacity_water' in locals() and VTK_AVAILABLE:
            try:
                mapper = atm_volume_solo.GetMapper()
                vtk_volume = mapper.GetInput()
                volume_property = atm_volume_solo.GetProperty()
                
                # 应用透明度映射（简化版）
                opacity_func = vtk.vtkPiecewiseFunction()
                opacity_func.AddPoint(temp_min, 0.0)
                opacity_func.AddPoint(temp_max, 0.25)
                volume_property.SetScalarOpacity(opacity_func)
                volume_property.SetScalarOpacityUnitDistance(5)
            except Exception as e:
                print(f"  警告：无法设置大气透明度映射: {e}")
        
        # 添加大气矢量场（简化版）
        if 'velocity' in atmosphere_grid.array_names:
            sample_indices = np.linspace(0, atmosphere_grid.n_points - 1, min(200, atmosphere_grid.n_points), dtype=int)
            sample_points = pv.PolyData(atmosphere_grid.points[sample_indices])
            sample_points["velocity"] = atmosphere_grid["velocity"][sample_indices]
            sample_points["speed"] = atmosphere_grid["speed"][sample_indices]
            atm_arrows = sample_points.glyph(orient='velocity', scale='speed', factor=40.0)
            if atm_arrows.n_points > 0:
                atmosphere_plotter.add_mesh(
                    atm_arrows, 
                    scalars='speed', 
                    cmap='cool', 
                    opacity=0.9,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': '流速'}
                )
    
    atmosphere_plotter.add_axes()
    print("✅ 大气可视化窗口已打开（关闭窗口后继续）")
    atmosphere_plotter.show(auto_close=False)
    
    # 8.3: 最后显示融合可视化
    print("\n【8.3】显示融合可视化...")
    plotter.add_axes()
    plotter.add_text("大气-海洋融合可视化", font_size=20, position='upper_left')
    print("✅ 融合可视化窗口已打开")
    plotter.show()

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("大气-海洋立方体融合可视化")
    print("="*60)
    
    # 用户输入参数
    print("\n【1】请输入经纬范围（左下角和右上角）:")
    lon_min = float(input("  左下角经度 (lon_min): "))
    lat_min = float(input("  左下角纬度 (lat_min): "))
    lon_max = float(input("  右上角经度 (lon_max): "))
    lat_max = float(input("  右上角纬度 (lat_max): "))
    
    print("\n【2】请输入时间点:")
    time_step = int(input("  时间步索引 (time_step，默认0): ") or "0")
    
    print("\n【3】请选择分辨率:")
    print("  low: 低分辨率（采样间隔4，海洋5层，大气5层，箭头200个）")
    print("  medium: 中分辨率（采样间隔2，海洋10层，大气10层，箭头500个）")
    print("  high: 高分辨率（采样间隔1，海洋20层，大气20层，箭头1000个）")
    resolution = input("  分辨率 (low/medium/high，默认medium): ").strip().lower() or "medium"
    
    if resolution not in ['low', 'medium', 'high']:
        print("  警告：无效的分辨率，使用medium")
        resolution = 'medium'
    
    # 执行可视化
    visualize_atmosphere_ocean_fusion(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_step=time_step,
        resolution=resolution,
        quality=-6,
        scale_xy=25
    )

