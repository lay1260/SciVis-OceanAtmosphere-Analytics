# 大气数据3D可视化（参考velocity_3D_vector_optimized_ocean.py）
# 矢量场优化版本：
# - 温度：只映射颜色值（hot色图）
# - 水含量：只映射透明度（使用策略17：反幂函数，0.8次，0~0.3透明度）
# - 箭头：完全不透明
# - 自动启用LOD优化（步骤5）
# - 自动启用环境光与背景优化（步骤6）
# - 使用VTK底层API实现真正的双标量独立控制
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import base64
import io
import imageio.v2 as iio

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
    print("警告：SciPy不可用，将使用直线箭头（模式1优化需要SciPy）")

# 矢量场优化：模式3 - 聚类区域大箭头所需依赖
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告：scikit-learn不可用，模式3（聚类区域大箭头）将退回到传统直线箭头")

# 矢量场优化：模式2 - 三维流线所需依赖
try:
    from scipy.integrate import solve_ivp
    SCIPY_INTEGRATE_AVAILABLE = True
except ImportError:
    SCIPY_INTEGRATE_AVAILABLE = False
    print("警告：scipy.integrate不可用，模式2（三维流线）将不可用")

# 导入数据提取函数
from data_extractor import (
    extract_data, 
    ATMOSPHERE_VARIABLES, 
    get_all_face_coordinates,
    find_points_in_range
)

# ----------------------------
# 主函数：大气数据3D可视化
# ----------------------------
def visualize_atmosphere_3d(
    lon_min, lon_max, lat_min, lat_max,
    time_step=0,
    layer_min=0,
    layer_max=50,
    data_quality=-6,
    scale_xy=25,
    atmosphere_nz=20,
    vector_mode=1,
    return_image=False
):
    """
    大气数据3D可视化
    
    Args:
        lon_min, lon_max, lat_min, lat_max: 经纬范围
        time_step: 时间步索引
        layer_min, layer_max: 大气层数范围
        data_quality: 数据质量等级
        scale_xy: XY方向缩放系数
        atmosphere_nz: 大气层数（用于可视化）
        vector_mode: 矢量场可视化模式（1=弯曲箭头，2=流线，3=直线箭头）
    """
    # ----------------------------
    # 1️⃣ 提取大气数据
    # ----------------------------
    print("\n" + "="*60)
    print("提取大气数据")
    print("="*60)
    print(f"经纬范围: [{lon_min}, {lon_max}] × [{lat_min}, {lat_max}]")
    print(f"时间步: {time_step}")
    print(f"层数范围: [{layer_min}, {layer_max}]")

    # 需要提取的大气变量
    atm_vars = ['T', 'QI', 'QL', 'RI', 'RL', 'U', 'V', 'W']

    # 使用extract_data函数提取大气数据
    atm_data = extract_data(
        variables=atm_vars,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_step=time_step,
        layer_min=layer_min,
        layer_max=layer_max,
        quality=data_quality
    )

    # 检查数据是否成功提取
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

    print(f"大气数据点数: {len(atm_lon)}")
    print(f"大气温度范围: [{np.min(atm_T):.2f}, {np.max(atm_T):.2f}]")

    # ----------------------------
    # 3️⃣ 计算可见水分子含量W
    # ----------------------------
    def compute_water_content(QI, QL, RI, RL):
        """
        计算可见水分子含量W
        
        W = 0.7 * Q_L * min(R_L_prime / 10, 1) + 0.3 * Q_I * min(R_I_prime / 20, 1)
        """
        R_L_prime = RL * 1e6  # m -> μm
        R_I_prime = RI * 1e6  # m -> μm
        
        W = (0.7 * QL * np.minimum(R_L_prime / 10, 1.0) + 
             0.3 * QI * np.minimum(R_I_prime / 20, 1.0))
        
        return W

    # 计算大气可见水分子含量
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

    print(f"大气水含量范围: [{np.min(atm_water):.6f}, {np.max(atm_water):.6f}]")
    print(f"大气速度范围: [{np.min(atm_speed):.4f}, {np.max(atm_speed):.4f}]")

    # ----------------------------
    # 4️⃣ 构建3D网格
    # ----------------------------
    print("\n" + "="*60)
    print("构建3D网格")
    print("="*60)

    # 获取大气数据的实际点数
    atm_n_points = len(atm_lon)

    # 确定大气网格尺寸
    # 需要将点数据转换为规则网格
    # 使用经纬度的唯一值来确定网格尺寸
    unique_lons = np.unique(atm_lon)
    unique_lats = np.unique(atm_lat)

    # 如果数据是规则网格，可以直接使用
    if len(unique_lons) * len(unique_lats) == atm_n_points:
        atm_nx = len(unique_lons)
        atm_ny = len(unique_lats)
        atm_nz = atmosphere_nz
        print(f"检测到规则网格: nx={atm_nx}, ny={atm_ny}, nz={atm_nz}")
    else:
        # 非规则网格，需要插值或采样
        # 使用合理的网格尺寸
        atm_nx = int(np.sqrt(atm_n_points))
        atm_ny = int(np.sqrt(atm_n_points))
        atm_nz = atmosphere_nz
        print(f"非规则网格，使用插值: nx={atm_nx}, ny={atm_ny}, nz={atm_nz}")

    # 确保网格维度至少为2x2x2
    atm_nx = max(2, atm_nx)
    atm_ny = max(2, atm_ny)
    atm_nz = max(2, atm_nz)

    # 大气高度范围（正数，向上）
    # 为了保持立方体比例，Z方向也需要缩放
    # 计算X和Y方向的缩放后范围
    lon_range = (lon_max - lon_min) * scale_xy
    lat_range = (lat_max - lat_min) * scale_xy
    # 使用X和Y范围的平均值作为Z方向的参考范围
    avg_xy_range = (lon_range + lat_range) / 2
    
    # 大气高度范围（按比例缩放，使其与X、Y方向的比例合理）
    atmosphere_z_min = 0
    atmosphere_z_max = avg_xy_range  # 使用与X、Y方向相同的缩放范围，保持立方体比例
    
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

    # ----------------------------
    # 5️⃣ 将大气数据插值到规则网格
    # ----------------------------
    print("\n" + "="*60)
    print("将大气数据插值到规则网格")
    print("="*60)

    # 将大气数据插值到规则网格
    # 大气数据是 (n_points, n_layers) 格式，需要转换为3D网格 (nx, ny, nz)
    print(f"处理大气数据: {atm_n_points}个点，{atm_T.shape[1] if atm_T.ndim == 2 else 1}层")

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
                atm_U_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                atm_V_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                atm_W_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                for k in range(atm_nz):
                    atm_U_3d[:, :, k] = atm_velocity[:, 0].reshape(atm_nx, atm_ny, order='F')
                    atm_V_3d[:, :, k] = atm_velocity[:, 1].reshape(atm_nx, atm_ny, order='F')
                    atm_W_3d[:, :, k] = atm_velocity[:, 2].reshape(atm_nx, atm_ny, order='F')
            else:
                atm_U_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                atm_V_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                atm_W_3d = np.zeros((atm_nx, atm_ny, atm_nz))
            
            atm_velocity_3d = np.stack([atm_U_3d, atm_V_3d, atm_W_3d], axis=3)
            print("✅ 大气数据直接reshape完成")
        else:
            # 层数不匹配或点数不匹配，需要插值
            print(f"警告：数据格式不匹配，进行插值（点数: {atm_n_points} vs {atm_nx*atm_ny}, 层数: {n_layers} vs {atm_nz}）")
            
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
                print(f"警告：插值失败: {e}，使用零填充")
                atm_T_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                atm_water_3d = np.zeros((atm_nx, atm_ny, atm_nz))
                atm_velocity_3d = np.zeros((atm_nx, atm_ny, atm_nz, 3))
    else:
        print("警告：大气数据格式不匹配，使用零填充")
        atm_T_3d = np.zeros((atm_nx, atm_ny, atm_nz))
        atm_water_3d = np.zeros((atm_nx, atm_ny, atm_nz))
        atm_velocity_3d = np.zeros((atm_nx, atm_ny, atm_nz, 3))

    # 设置大气网格数据
    atmosphere_grid["Temperature"] = atm_T_3d.flatten(order="F")
    atmosphere_grid["WaterContent"] = atm_water_3d.flatten(order="F")
    atmosphere_grid["velocity"] = atm_velocity_3d.reshape(-1, 3, order='F')
    atm_speed_3d = np.linalg.norm(atm_velocity_3d, axis=3)
    atmosphere_grid["speed"] = atm_speed_3d.flatten(order="F")

    print(f"大气网格点数: {atmosphere_grid.n_points}")
    print(f"大气温度范围: [{np.min(atm_T_3d):.2f}, {np.max(atm_T_3d):.2f}]")
    print(f"大气水含量范围: [{np.min(atm_water_3d):.6f}, {np.max(atm_water_3d):.6f}]")

    # ----------------------------
    # 6️⃣ 创建采样点（用于矢量场可视化）
    # ----------------------------
    print("\n" + "="*60)
    print("创建采样点")
    print("="*60)

    # 在立方体上均匀采样，每条边上10个点
    sampling_points_per_edge = 10

    # 计算每个维度的实际采样点数
    n_samples_x = min(sampling_points_per_edge, atm_nx)
    n_samples_y = min(sampling_points_per_edge, atm_ny)
    n_samples_z = min(sampling_points_per_edge, atm_nz)

    # 生成均匀分布的索引
    x_indices = np.linspace(0, atm_nx-1, n_samples_x, dtype=int)
    y_indices = np.linspace(0, atm_ny-1, n_samples_y, dtype=int)
    z_indices = np.linspace(0, atm_nz-1, n_samples_z, dtype=int)

    # 创建采样点的网格
    X_idx, Y_idx, Z_idx = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    X_idx = X_idx.flatten()
    Y_idx = Y_idx.flatten()
    Z_idx = Z_idx.flatten()

    print(f"采样索引: x={len(x_indices)}个点, y={len(y_indices)}个点, z={len(z_indices)}个点")
    print(f"实际采样点数: {len(X_idx)}")

    # 获取采样点的坐标和速度
    sample_points_coords = []
    sample_velocities = []
    sample_speeds = []

    for i in range(len(X_idx)):
        x_idx, y_idx, z_idx = X_idx[i], Y_idx[i], Z_idx[i]
        x_idx = np.clip(x_idx, 0, atm_nx-1)
        y_idx = np.clip(y_idx, 0, atm_ny-1)
        z_idx = np.clip(z_idx, 0, atm_nz-1)
        
        point_idx = x_idx + y_idx * atm_nx + z_idx * atm_nx * atm_ny
        if point_idx < atmosphere_grid.n_points:
            coords = atmosphere_grid.points[point_idx]
            vel = atmosphere_grid["velocity"][point_idx]
            speed = atmosphere_grid["speed"][point_idx]
            
            sample_points_coords.append(coords)
            sample_velocities.append(vel)
            sample_speeds.append(speed)

    sample_points_coords = np.array(sample_points_coords)
    sample_velocities = np.array(sample_velocities)
    sample_speeds = np.array(sample_speeds)

    # 创建采样点的PolyData
    sample_points = pv.PolyData(sample_points_coords)
    sample_points["velocity"] = sample_velocities
    sample_points["speed"] = sample_speeds

    print(f"✅ 采样点创建完成: {len(sample_points_coords)} 个点")
    print(f"速度范围: [{np.min(sample_speeds):.4f}, {np.max(sample_speeds):.4f}]")

    # ----------------------------
    # 7️⃣ 温度与水含量体积渲染（优化：单体积双标量绑定）
    # ----------------------------
    # 核心优化：单体积双标量绑定（替代双体积叠加）
    # 可视化逻辑：
    # - 温度：映射颜色值（hot色图）
    # - 水含量：映射透明度（使用策略17：反幂函数，0.8次，0~0.3透明度）
    # - 箭头：完全不透明

    # 创建单网格，同时绑定温度、水含量双标量
    combined_volume = atmosphere_grid
    theta_data = atm_T_3d.flatten(order="F")
    water_data = atm_water_3d.flatten(order="F")

    # 绑定温度（用于颜色映射）
    combined_volume["Temperature"] = theta_data
    # 绑定水含量（用于透明度映射）
    combined_volume["WaterContent"] = water_data

    # ----------------------------
    # 第五步优化：分级细节（LOD）渲染（自动启用）
    # ----------------------------
    print("\n" + "="*60)
    print("第五步优化：分级细节（LOD）渲染（自动启用）")
    print("="*60)

    use_lod = True  # 自动启用LOD优化
    print("✅ 已自动启用LOD优化")

    # 保存原始数据用于梯度计算
    theta_data_original = theta_data.copy()
    water_data_original = water_data.copy()
    combined_volume_original = combined_volume

    if use_lod:
        print("\n正在应用第五步优化：分级细节（LOD）渲染...")
        
        # 1. 划分核心区：y轴中间区 + 水含量高值区（>70%分位数）
        y_coords = combined_volume.points[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_mid_low = y_min + (y_max - y_min) * 0.3
        y_mid_high = y_min + (y_max - y_min) * 0.7
        
        # 水含量高值区（70%分位数以上）
        water_70_percentile = np.percentile(water_data, 70)
        high_water_mask = water_data >= water_70_percentile
        
        # 合并核心区：y轴中间区（30%~70%）或水含量高值区
        core_mask = ((y_coords >= y_mid_low) & (y_coords <= y_mid_high)) | high_water_mask
        edge_mask = ~core_mask
        
        # 统计信息
        n_core = np.sum(core_mask)
        n_edge = np.sum(edge_mask)
        n_total_original = len(core_mask)
        print(f"核心区点数: {n_core} ({n_core/n_total_original*100:.1f}%)")
        print(f"边缘区点数: {n_edge} ({n_edge/n_total_original*100:.1f}%)")
        print(f"Y轴核心区范围: [{y_mid_low:.2f}, {y_mid_high:.2f}]")
        print(f"水含量高值阈值: {water_70_percentile:.6f} (70%分位数)")
        
        # 2. 边缘区降采样（保留原逻辑，低水含量边缘区降采样50%）
        edge_indices = np.where(edge_mask)[0]
        edge_downsampled_indices = edge_indices[::2]  # 每2个点取1个，降采样50%
        core_indices = np.where(core_mask)[0]
        selected_indices = np.concatenate([core_indices, edge_downsampled_indices])
        selected_indices = np.sort(selected_indices)
        
        print(f"LOD后总点数: {len(selected_indices)} (原始: {n_total_original}, 降采样率: {len(selected_indices)/n_total_original*100:.1f}%)")
        
        # 3. 改进的LOD实现：在Y方向进行降采样，保持结构化网格
        print("   正在应用改进的LOD优化（保持结构化网格）...")
        
        # 将1D掩码映射回3D索引
        core_mask_3d = np.zeros((atm_nx, atm_ny, atm_nz), dtype=bool, order='F')
        
        for i in range(len(core_mask)):
            z_idx = i // (atm_nx * atm_ny)
            remainder = i % (atm_nx * atm_ny)
            y_idx = remainder // atm_nx
            x_idx = remainder % atm_nx
            if core_mask[i]:
                core_mask_3d[x_idx, y_idx, z_idx] = True
        
        # 识别核心Y层
        y_core_layers = np.zeros(atm_ny, dtype=bool)
        for y_idx in range(atm_ny):
            if np.any(core_mask_3d[:, y_idx, :]):
                y_core_layers[y_idx] = True
        
        # 边缘Y层降采样（每2个取1个）
        y_edge_layers = np.where(~y_core_layers)[0]
        y_edge_downsampled = y_edge_layers[::2]
        y_core_layers_indices = np.where(y_core_layers)[0]
        y_selected = np.sort(np.concatenate([y_core_layers_indices, y_edge_downsampled]))
        
        # 重新构建降采样后的网格（保持结构化）
        X_lod = atm_X[:, y_selected, :]
        Y_lod = atm_Y[:, y_selected, :]
        Z_lod = atm_Z[:, y_selected, :]
        
        # 重新构建StructuredGrid
        lod_volume = pv.StructuredGrid(X_lod, Y_lod, Z_lod)
        
        # 提取对应的数据
        theta_data_lod = atm_T_3d[:, y_selected, :].flatten(order='F')
        water_data_lod = atm_water_3d[:, y_selected, :].flatten(order='F')
        
        # 更新网格和数据
        lod_volume["Temperature"] = theta_data_lod
        lod_volume["WaterContent"] = water_data_lod
        
        # 更新维度
        lod_nx, lod_ny, lod_nz = X_lod.shape[0], X_lod.shape[1], X_lod.shape[2]
        
        print(f"   ✅ LOD网格已生成（保持结构化）")
        print(f"   原始尺寸: ({atm_nx}, {atm_ny}, {atm_nz}) = {atm_nx*atm_ny*atm_nz} 点")
        print(f"   LOD尺寸: ({lod_nx}, {lod_ny}, {lod_nz}) = {lod_nx*lod_ny*lod_nz} 点")
        print(f"   性能提升：点数减少 {((atm_nx*atm_ny*atm_nz - lod_nx*lod_ny*lod_nz) / (atm_nx*atm_ny*atm_nz) * 100):.1f}%")
        
        # 更新combined_volume和数据
        combined_volume = lod_volume
        theta_data = theta_data_lod
        water_data = water_data_lod
        atm_nx, atm_ny, atm_nz = lod_nx, lod_ny, lod_nz  # 更新维度
        
        print(f"✅ 第五步优化完成：LOD网格点数={combined_volume.n_points}")

    # 如果LOD被禁用，使用原始数据
    if not use_lod:
        combined_volume = combined_volume_original
        theta_data = theta_data_original
        water_data = water_data_original
        n_core = None
        n_edge = None
        print("✅ 使用原始完整网格（LOD已禁用）")

    # ----------------------------
    # 第三步优化：计算水含量梯度（用于视觉权重优化）
    # ----------------------------
    print("正在计算水含量梯度...")
    # 使用 NumPy 直接计算水含量梯度
    water_3d = water_data.reshape(atm_nx, atm_ny, atm_nz, order='F')

    # 检查维度是否足够计算梯度
    if atm_nx >= 2 and atm_ny >= 2 and atm_nz >= 2:
        try:
            grad_x, grad_y, grad_z = np.gradient(water_3d)
            
            # 展平并组合为梯度向量（保持F顺序）
            water_gradient = np.stack([
                grad_x.flatten(order='F'),
                grad_y.flatten(order='F'),
                grad_z.flatten(order='F')
            ], axis=1)
            
            water_gradient_mag = np.linalg.norm(water_gradient, axis=1)
            if water_gradient_mag.max() > water_gradient_mag.min():
                water_gradient_norm = (water_gradient_mag - water_gradient_mag.min()) / (water_gradient_mag.max() - water_gradient_mag.min())
            else:
                water_gradient_norm = np.zeros_like(water_gradient_mag)
            print(f"水含量梯度范围: [{water_gradient_mag.min():.4f}, {water_gradient_mag.max():.4f}]")
            print(f"水含量梯度归一化范围: [{water_gradient_norm.min():.4f}, {water_gradient_norm.max():.4f}]")
        except Exception as e:
            print(f"警告：计算水含量梯度失败: {e}，使用零梯度")
            water_gradient_norm = np.zeros(water_data.size)
    else:
        print(f"警告：数据维度太小 ({atm_nx}, {atm_ny}, {atm_nz})，无法计算梯度，使用零梯度")
        water_gradient_norm = np.zeros(water_data.size)

    # 检查数据范围和数据有效性
    print(f"水含量数据范围: [{np.min(water_data):.6f}, {np.max(water_data):.6f}]")
    print(f"温度数据范围: [{np.min(theta_data):.4f}, {np.max(theta_data):.4f}]")
    print(f"数据形状: water={water_data.shape}, theta={theta_data.shape}")

    # 计算数据范围
    water_min_val = np.min(water_data)
    water_max_val = np.max(water_data)
    water_range = water_max_val - water_min_val
    temp_min_val = np.min(theta_data)
    temp_max_val = np.max(theta_data)
    temp_range = temp_max_val - temp_min_val

    print(f"水含量数据范围: [{water_min_val:.6f}, {water_max_val:.6f}]")
    print(f"温度数据范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")

    # ----------------------------
    # 第六步优化：环境光与背景协同优化（自动启用）
    # ----------------------------
    print("\n" + "="*60)
    print("第六步优化：环境光与背景协同优化（自动启用）")
    print("="*60)

    use_env_lighting = True  # 自动启用环境光与背景优化
    print("✅ 已自动启用环境光与背景优化")

    # 创建Plotter（支持离屏渲染以返回截图）
    plotter = pv.Plotter(window_size=(1400, 900), off_screen=return_image)

    # 应用第六步优化：背景色设置
    if use_env_lighting:
        plotter.background_color = (0.08, 0.12, 0.18)  # 深暗蓝
        print("✅ 已设置背景色为深暗蓝色 (0.08, 0.12, 0.18)")
    else:
        plotter.background_color = 'white'

    try:
        plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        print("✅ 已启用深度剥离（depth peeling）以正确处理透明度混合")
    except Exception as e:
        print(f"警告：无法启用深度剥离: {e}")
        print("   将使用标准渲染模式")

    # ========================================
    # 使用VTK底层API实现真正的双标量独立控制（改进的温度-水含量映射）
    # ========================================
    if combined_volume.n_points > 0 and not np.isnan(theta_data).all() and not np.isnan(water_data).all():
        # 1. 获取PyVista网格的VTK数据对象
        vtk_grid = combined_volume.GetBlock(0) if hasattr(combined_volume, 'GetBlock') else combined_volume
        
        # 2. 使用PyVista创建volume，然后通过VTK API修改属性
        volume_actor = plotter.add_volume(
            combined_volume,
            scalars="Temperature",  # 先设置温度作为颜色标量
            cmap="hot",
            opacity=0.1,  # 临时透明度，稍后通过VTK API修改
            opacity_unit_distance=5,
            show_scalar_bar=True,
            scalar_bar_args={'title': '温度 (Temperature) - 颜色'},
            shade=True,
            ambient=0.1,
            pickable=False,
            blending='composite'
        )
        print("✅ 已设置体积渲染混合模式为复合模式（通过PyVista参数）")
        
        # 3. 通过VTK底层API实现真正的双标量独立控制
        if not VTK_AVAILABLE:
            print("警告：VTK不可用，使用PyVista高层API（近似方案）")
        else:
            try:
                # 获取VTK Volume和VolumeProperty
                mapper = volume_actor.GetMapper()
                vtk_volume = mapper.GetInput()
                volume_property = volume_actor.GetProperty()
                
                # 3.1 确保水含量数据在PointData中（作为第二个标量数组）
                water_vtk_array = vtk_volume.GetPointData().GetArray("WaterContent")
                if water_vtk_array is None:
                    water_vtk_array = numpy_to_vtk(
                        water_data.astype(np.float32),
                        array_type=vtk.VTK_FLOAT
                    )
                    water_vtk_array.SetName("WaterContent")
                    vtk_volume.GetPointData().AddArray(water_vtk_array)
                
                # 3.2 第三步优化：水含量主导的视觉权重优化（策略17：反幂函数，0.8次，0~0.3透明度）
                def opacity_mapping_strategy_17(water_data, water_gradient_norm):
                    """策略17：高（80%）阈值 + 不常见-反幂函数（0.8次）+ 0~0.3透明度"""
                    water_threshold = np.percentile(water_data, 80)
                    water_norm = np.clip((water_data - water_threshold) / (water_data.max() - water_threshold + 1e-6), 0.0, 1.0)
                    base_opacity = 0 + 0.3 * (water_norm ** 0.8)
                    gradient_boost = 0.1 + 0.2 * water_gradient_norm
                    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.3)
                    return final_opacity
                
                # 计算策略17的最终透明度
                final_opacity = opacity_mapping_strategy_17(water_data, water_gradient_norm)
                print("✅ 已选择策略17：反幂函数映射（80%阈值，0.8次，0~0.3透明度）")
                print(f"透明度范围: [{final_opacity.min():.4f}, {final_opacity.max():.4f}]")
                
                # 将计算好的透明度映射到温度值
                n_bins = 512
                opacity_func = vtk.vtkPiecewiseFunction()
                
                temp_vals = np.linspace(temp_min_val, temp_max_val, n_bins)
                temp_tolerance = (temp_max_val - temp_min_val) / n_bins * 2
                
                opacity_min = final_opacity.min()
                opacity_max = final_opacity.max()
                
                print("正在构建温度-透明度映射表（基于水含量主导的视觉权重，策略17）...")
                for i, t in enumerate(temp_vals):
                    temp_mask = np.abs(theta_data - t) <= temp_tolerance
                    
                    if np.any(temp_mask):
                        corresponding_opacities = final_opacity[temp_mask]
                        avg_opacity = np.mean(corresponding_opacities)
                        avg_opacity = np.clip(avg_opacity, opacity_min, opacity_max)
                        opacity_func.AddPoint(t, avg_opacity)
                    else:
                        temp_norm = (t - temp_min_val) / (temp_max_val - temp_min_val) if temp_range > 0 else 0
                        opacity = opacity_min + (opacity_max - opacity_min) * temp_norm
                        opacity = np.clip(opacity, opacity_min, opacity_max)
                        opacity_func.AddPoint(t, opacity)
                
                # 强制设置边界值
                min_temp_mask = np.abs(theta_data - temp_min_val) < temp_tolerance
                if np.any(min_temp_mask):
                    min_opacity = np.mean(final_opacity[min_temp_mask])
                    min_opacity = np.clip(min_opacity, opacity_min, opacity_max)
                    opacity_func.AddPoint(temp_min_val, min_opacity)
                else:
                    opacity_func.AddPoint(temp_min_val, opacity_min)
                
                max_temp_mask = np.abs(theta_data - temp_max_val) < temp_tolerance
                if np.any(max_temp_mask):
                    max_opacity = np.mean(final_opacity[max_temp_mask])
                    max_opacity = np.clip(max_opacity, opacity_min, opacity_max)
                    opacity_func.AddPoint(temp_max_val, max_opacity)
                else:
                    opacity_func.AddPoint(temp_max_val, opacity_max)
                
                # 3.3 设置透明度函数
                volume_property.SetScalarOpacity(opacity_func)
                if use_lod:
                    lod_opacity_unit_distance = 8.0
                else:
                    lod_opacity_unit_distance = 5.0
                volume_property.SetScalarOpacityUnitDistance(lod_opacity_unit_distance)
                
                # 启用三线性插值
                try:
                    volume_property.SetInterpolationTypeToLinear()
                    print("✅ 已启用三线性插值（Trilinear Interpolation）")
                except Exception as e:
                    print(f"⚠️  无法设置三线性插值: {e}")
                
                # 3.4 自适应颜色映射（基于数据分布的分位数拉伸）
                temp_percentile_5 = np.percentile(theta_data, 5)
                temp_percentile_95 = np.percentile(theta_data, 95)
                
                print(f"温度原始范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")
                print(f"温度分位数范围（5%-95%）: [{temp_percentile_5:.4f}, {temp_percentile_95:.4f}]")
                
                # 获取hot_r色图
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
                
                # 创建VTK颜色传递函数
                color_func = vtk.vtkColorTransferFunction()
                if temp_range > 0:
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
                else:
                    color_func.AddRGBPoint(temp_min_val, 0.5, 0.5, 0.5)
                
                volume_property.SetColor(color_func)
                
                # 3.5 光照设置
                try:
                    volume_property.ShadeOn()
                    volume_property.SetSpecular(0.05)
                    
                    avg_gradient_norm = np.mean(water_gradient_norm)
                    base_ambient = 0.15 + 0.35 * avg_gradient_norm
                    base_diffuse = 0.4 + 0.6 * avg_gradient_norm
                    
                    if use_env_lighting:
                        high_gradient_ratio = np.mean(water_gradient_norm > 0.5)
                        water_80_percentile = np.percentile(water_data, 80)
                        high_water_ratio = np.mean(water_data > water_80_percentile)
                        enhanced_ambient = min(base_ambient + 0.4 * high_gradient_ratio + 0.3 * high_water_ratio + 0.15, 0.95)
                        enhanced_diffuse = min(base_diffuse + 0.2, 1.0)
                        volume_property.SetAmbient(enhanced_ambient)
                        volume_property.SetDiffuse(enhanced_diffuse)
                        print("✅ 第四步优化：渲染混合模式优化已实现（亮度增强）")
                    else:
                        enhanced_ambient = min(base_ambient + 0.15, 0.8)
                        enhanced_diffuse = min(base_diffuse + 0.2, 1.0)
                        volume_property.SetAmbient(enhanced_ambient)
                        volume_property.SetDiffuse(enhanced_diffuse)
                except Exception as e:
                    print(f"警告：无法设置混合模式优化: {e}")
                
                print("✅ VTK底层API双标量独立控制已实现（六步优化完整实现）")
            except Exception as e:
                print(f"警告：VTK底层API设置失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("错误：体积数据无效，无法添加渲染")

    # 第六步优化：添加方向光
    if use_env_lighting:
        print("\n正在应用第六步优化：方向光设置...")
        try:
            scene_center = combined_volume.center
            
            main_light = pv.Light(
                position=(scene_center[0] + 15, scene_center[1] + 15, scene_center[2] + 25),
                focal_point=scene_center,
                color="white",
                intensity=0.85
            )
            plotter.add_light(main_light)
            
            side_light = pv.Light(
                position=(scene_center[0] - 15, scene_center[1] - 15, scene_center[2] + 25),
                focal_point=scene_center,
                color="lightblue",
                intensity=0.45
            )
            plotter.add_light(side_light)
            
            print(f"✅ 已添加方向光（主光源强度=0.85，侧光强度=0.45）")
        except Exception as e:
            print(f"⚠️  无法添加方向光: {e}")

    # ========================================
    # 矢量场优化：模式1 - 弯曲箭头（三维流动感优化）
    # ========================================
    def get_neighbors(sample_points, target_idx, k=5):
        """获取目标采样点的k个空间最近邻（含自身）"""
        target_point = sample_points[target_idx]
        distances = np.linalg.norm(sample_points - target_point, axis=1)
        neighbor_indices = np.argsort(distances)[:k]
        return neighbor_indices

    def smooth_velocity_field(sample_points, velocities, sigma=1.0):
        """高斯卷积平滑速度场"""
        if not SCIPY_AVAILABLE:
            return velocities
        smoothed_vel = np.zeros_like(velocities)
        for i in range(3):
            smoothed_vel[:, i] = gaussian_filter1d(velocities[:, i], sigma=sigma)
        return smoothed_vel

    def create_bent_arrows(sample_points, velocities, speeds, arrow_scale=60.0, 
                          k_neighbors=4, spline_degree=3, max_bend_factor=0.2):
        """生成三维弯曲箭头"""
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

    # ========================================
    # 矢量场优化：模式2 - 三维流线（全局流动趋势优化）
    # ========================================
    def create_3d_streamlines(grid, sample_points, velocity_scalars, 
                          streamline_length=50.0, step_size=0.5, n_seeds=400):
        """生成三维流线（优化版本）"""
        if not SCIPY_INTEGRATE_AVAILABLE:
            print("⚠️  scipy.integrate不可用，无法生成三维流线")
            return None
        
        # 优化种子点生成
        min_effective_speed = 0.01
        speeds = np.linalg.norm(velocity_scalars, axis=1)
        high_speed_mask = speeds > min_effective_speed
        
        if np.any(high_speed_mask):
            high_coords = sample_points[high_speed_mask]
            n_high = min(n_seeds // 2, len(high_coords))
            if n_high > 0:
                high_idx = np.random.choice(len(high_coords), size=n_high, replace=False)
                high_seeds = high_coords[high_idx]
                
                low_coords = sample_points[~high_speed_mask]
                n_low = min(n_seeds - n_high, len(low_coords))
                if n_low > 0:
                    low_idx = np.random.choice(len(low_coords), size=n_low, replace=(len(low_coords) < n_low))
                    low_seeds = low_coords[low_idx]
                    seed_points_coords = np.vstack([high_seeds, low_seeds])
                else:
                    seed_points_coords = high_seeds
            else:
                seed_points_coords = sample_points[np.random.choice(len(sample_points), size=min(n_seeds, len(sample_points)), replace=False)]
        else:
            seed_points_coords = sample_points[np.random.choice(len(sample_points), size=min(n_seeds, len(sample_points)), replace=False)]
        
        seed_points = pv.PolyData(seed_points_coords)
        print(f"    种子点数量: {seed_points.n_points}")
        
        streamlines = grid.streamlines_from_source(
            source=seed_points,
            vectors='velocity',
            integration_direction='both',
            initial_step_length=0.1,
            terminal_speed=1e-3,
            max_steps=2000
        )
        
        if streamlines.n_points > 0:
            if 'velocity' in streamlines.array_names:
                streamline_speed = np.linalg.norm(streamlines['velocity'], axis=1)
            else:
                streamline_speed = np.ones(streamlines.n_points) * np.mean(speeds)
            streamlines['speed'] = streamline_speed
        
        return streamlines

    # 速度箭头可视化（独立不透明层）
    # 使用传入的 vector_mode 参数，不再交互式输入
    print(f"\n✅ 使用矢量场模式: {vector_mode} ({'弯曲箭头' if vector_mode == 1 else '三维流线' if vector_mode == 2 else '直线箭头'})")
    
    # 根据选择的模式执行相应的可视化
    if vector_mode == 1:
        # 模式1 - 弯曲箭头
        print("\n" + "="*60)
        print("矢量场优化：模式1 - 弯曲箭头（三维流动感优化）")
        print("="*60)
        
        speed_max = np.max(sample_speeds)
        if speed_max > 0:
            arrow_scale_factor = 50.0 / speed_max
        else:
            arrow_scale_factor = 1.0
        
        use_bent_arrows = SCIPY_AVAILABLE
        if use_bent_arrows:
            try:
                bent_arrows = create_bent_arrows(
                    sample_points=sample_points_coords,
                    velocities=sample_velocities,
                    speeds=sample_speeds,
                    arrow_scale=60.0,
                    k_neighbors=4,
                    spline_degree=3,
                    max_bend_factor=0.3
                )
                
                if bent_arrows is not None and bent_arrows.n_points > 0:
                    arrows = bent_arrows
                    print("✅ 已使用弯曲箭头（模式1优化）")
                else:
                    use_bent_arrows = False
                    print("⚠️  弯曲箭头生成失败，回退到直线箭头")
            except Exception as e:
                use_bent_arrows = False
                print(f"⚠️  弯曲箭头生成异常: {e}")
        else:
            use_bent_arrows = False
            print("⚠️  SciPy不可用，无法生成弯曲箭头，将使用直线箭头")
        
        if not use_bent_arrows:
            arrows = sample_points.glyph(
                orient='velocity',
                scale='speed',
                factor=arrow_scale_factor
            )
            print("✅ 已使用直线箭头（传统模式）")
        
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
        
        try:
            arrow_property = arrow_actor.GetProperty()
            arrow_property.SetOpacity(1.0)
            if hasattr(arrow_property, 'SetRenderLinesAsTubes'):
                arrow_property.SetRenderLinesAsTubes(True)
            if hasattr(arrow_property, 'SetLineWidth'):
                arrow_property.SetLineWidth(4.5)
            if hasattr(arrow_property, 'SetDepthWrite'):
                arrow_property.SetDepthWrite(False)
            print("✅ 已调整箭头渲染属性，确保不被体积遮挡")
        except Exception as e:
            print(f"警告：无法调整箭头渲染属性: {e}")
        
        if use_bent_arrows:
            print(f"✅ 弯曲箭头已添加（模式1优化）")
        else:
            print(f"速度箭头已添加")
    
    elif vector_mode == 2:
        # 模式2 - 三维流线
        if not SCIPY_INTEGRATE_AVAILABLE:
            print("⚠️  scipy.integrate不可用，仍尝试使用PyVista内置流线生成")
        print("\n" + "="*60)
        print("矢量场优化：模式2 - 三维流线（全局流动趋势优化）")
        print("="*60)
        
        try:
            streamlines = create_3d_streamlines(
                grid=atmosphere_grid,
                sample_points=sample_points_coords,
                velocity_scalars=sample_velocities,
                n_seeds=400
            )
            
            if streamlines is not None and streamlines.n_points > 0:
                streamline_actor = plotter.add_mesh(
                    streamlines,
                    scalars='speed',
                    cmap='cool',
                    line_width=2.0,
                    opacity=0.8,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': '流线速度 (Speed)'},
                    pickable=True
                )
                
                try:
                    streamline_property = streamline_actor.GetProperty()
                    streamline_property.SetOpacity(0.8)
                    if hasattr(streamline_property, 'SetLineWidth'):
                        streamline_property.SetLineWidth(2.0)
                    if hasattr(streamline_property, 'SetSpecular'):
                        streamline_property.SetSpecular(0.5)
                    if hasattr(streamline_property, 'SetSpecularPower'):
                        streamline_property.SetSpecularPower(10)
                    if hasattr(streamline_property, 'SetDepthWrite'):
                        streamline_property.SetDepthWrite(False)
                    print("✅ 三维流线已添加（模式2：PyVista内置流线）")
                except Exception as e:
                    print(f"警告：无法调整流线渲染属性: {e}")
            else:
                print("⚠️  未能生成有效流线")
        except Exception as e:
            print(f"⚠️  模式2流线生成异常: {e}")
            import traceback
            traceback.print_exc()
    
    if vector_mode == 3:
        # 模式3 - 传统直线箭头
        print("\n" + "="*60)
        print("矢量场可视化：模式3 - 传统直线箭头")
        print("="*60)
        
        speed_max = np.max(sample_speeds)
        if speed_max > 0:
            arrow_scale_factor = 50.0 / speed_max
        else:
            arrow_scale_factor = 1.0
        
        arrows = sample_points.glyph(
            orient='velocity',
            scale='speed',
            factor=arrow_scale_factor
        )
        
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
        
        try:
            arrow_property = arrow_actor.GetProperty()
            arrow_property.SetOpacity(1.0)
            if hasattr(arrow_property, 'SetRenderLinesAsTubes'):
                arrow_property.SetRenderLinesAsTubes(True)
            if hasattr(arrow_property, 'SetLineWidth'):
                arrow_property.SetLineWidth(4.5)
            if hasattr(arrow_property, 'SetDepthWrite'):
                arrow_property.SetDepthWrite(False)
            print("✅ 已调整箭头渲染属性，确保不被体积遮挡")
        except Exception as e:
            print(f"警告：无法调整箭头渲染属性: {e}")
        
        print(f"✅ 直线箭头已添加（模式3）")
        print(f"   采样点数={sample_points.n_points}，箭头数={arrows.n_points}")
        print(f"   速度范围: [{np.min(sample_speeds):.4f}, {np.max(sample_speeds):.4f}]")
        print(f"   箭头缩放因子: {arrow_scale_factor:.4f}")
    else:
        print("警告：采样点为空或速度全为0，无法添加箭头")

    # 打印体积渲染信息
    print(f"\n" + "="*60)
    print("体积渲染信息")
    print("="*60)
    print(f"组合体积点数: {combined_volume.n_points}")
    print(f"水含量数据范围: [{water_min_val:.6f}, {water_max_val:.6f}]")
    print(f"温度数据范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")
    print(f"透明度映射: 水含量[{water_min_val:.6f}, {water_max_val:.6f}] -> 透明度[0.0, 0.3]（策略17：反幂函数映射，温和突出高水含量）")
    print(f"网格尺寸: nx={atm_nx}, ny={atm_ny}, nz={atm_nz}")

    # 设置相机位置，确保3D立方体正确显示
    plotter.add_axes()
    
    # 计算场景中心和范围
    scene_center = combined_volume.center
    bounds = combined_volume.bounds
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    max_range = max(x_range, y_range, z_range)
    
    print(f"\n✅ 场景信息:")
    print(f"   场景中心: ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})")
    print(f"   场景范围: X={x_range:.2f}, Y={y_range:.2f}, Z={z_range:.2f}")
    print(f"   场景边界: X=[{bounds[0]:.2f}, {bounds[1]:.2f}], Y=[{bounds[2]:.2f}, {bounds[3]:.2f}], Z=[{bounds[4]:.2f}, {bounds[5]:.2f}]")
    
    # 设置相机位置，从等轴视角观察立方体
    camera_distance = max_range * 2.5
    plotter.camera_position = [
        (scene_center[0] + camera_distance, 
         scene_center[1] + camera_distance, 
         scene_center[2] + camera_distance),  # 相机位置
        scene_center,  # 焦点（场景中心）
        (0, 0, 1)  # 上方向
    ]
    
    # 设置等轴视角（isometric view）
    plotter.camera.azimuth = 45
    plotter.camera.elevation = 30
    
    print(f"✅ 已设置相机位置（等轴视角）")
    print(f"   相机距离: {camera_distance:.2f}")
    print(f"   方位角: 45°, 仰角: 30°")
    print(f"\n✅ 3D立方体可视化已准备就绪")
    
    if return_image:
        # 先执行一次渲染，再截图，避免“Nothing to screenshot”
        plotter.show(auto_close=False, interactive=False)
        img = plotter.screenshot(return_img=True)
        plotter.close()
        if img is not None:
            buf = io.BytesIO()
            iio.imwrite(buf, img, format='png')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        return None
    else:
        plotter.show()

# ----------------------------
# 主程序入口（用于直接运行脚本）
# ----------------------------
if __name__ == "__main__":
    # 默认参数
    lat_start, lat_end = 10, 40
    lon_start, lon_end = 100, 130
    time_step = 0
    layer_min = 0
    layer_max = 50
    data_quality = -6
    scale_xy = 25
    atmosphere_nz = 20
    vector_mode = 1
    
    # 调用主函数
    visualize_atmosphere_3d(
        lon_min=lon_start,
        lon_max=lon_end,
        lat_min=lat_start,
        lat_max=lat_end,
        time_step=time_step,
        layer_min=layer_min,
        layer_max=layer_max,
        data_quality=data_quality,
        scale_xy=scale_xy,
        atmosphere_nz=atmosphere_nz,
        vector_mode=vector_mode
    )