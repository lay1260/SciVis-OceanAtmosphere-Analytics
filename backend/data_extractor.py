"""
大气海洋数据提取工具
支持从 OpenVisus 数据源提取任意经纬范围、时间点、层数的数据
"""

import numpy as np
import OpenVisus as ov
from netCDF4 import Dataset
import os
from typing import Tuple, List, Optional, Dict
import json
from datetime import datetime

# ============================================================
# 0. 变量定义
# ============================================================

# 大气变量（GEOS数据集，立方体网格，6个面，每个面1440x1440x51）
ATMOSPHERE_VARIABLES = [
    'P', 'U', 'V', 'W', 'T', 'H', 
    'CO', 'CO2', 'QI', 'QL', 'RI', 'RL', 
    'DELP', 'DTHDT', 'DTHDTCN', 'FCLD'
]

# 海洋变量（LLC2160数据集，经纬度网格，8640x6480x90）
# 注意：变量名是 Theta 和 Salt（首字母大写）
OCEAN_VARIABLES = ['U', 'V', 'W', 'Theta', 'Salt']


# ============================================================
# 1. 立方体网格到经纬度转换
# ============================================================

def cubed_sphere_index_to_lonlat(face: int, i: int, j: int, N: int = 1440) -> Tuple[float, float]:
    """
    将 cubed-sphere 索引 (face, i, j) 转为 (lon, lat)
    
    Args:
        face: 立方体面编号 (0~5)
        i, j: 立方体面上的索引 (0~N-1)
        N: 每个面的边长，默认1440 (C1440网格)
    
    Returns:
        (lon, lat): 经纬度坐标（度）
    """
    # 单面局部坐标 [-1, 1]
    xi = (2 * i + 1) / N - 1
    eta = (2 * j + 1) / N - 1
    
    # gnomonic cubed-sphere -> xyz
    if face == 0:
        X, Y, Z = 1.0, xi, -eta
    elif face == 1:
        X, Y, Z = -xi, 1.0, -eta
    elif face == 2:
        X, Y, Z = -1.0, -xi, -eta
    elif face == 3:
        X, Y, Z = xi, -1.0, -eta
    elif face == 4:
        X, Y, Z = eta, xi, 1.0
    elif face == 5:
        X, Y, Z = -eta, xi, -1.0
    else:
        raise ValueError(f"face 必须是 0~5，当前值: {face}")
    
    # xyz -> lon/lat (弧度 -> 度)
    lon = np.arctan2(Y, X) * 180.0 / np.pi
    lat = np.arctan2(Z, np.sqrt(X**2 + Y**2)) * 180.0 / np.pi
    
    return lon, lat


def get_all_face_coordinates(N: int = 1440) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    获取所有6个面的经纬度坐标网格（使用向量化计算）
    
    Args:
        N: 每个面的边长
    
    Returns:
        dict: {face: (lon_grid, lat_grid)} 每个面的经纬度网格
    """
    coords = {}
    for face in range(6):
        i = np.arange(N)
        j = np.arange(N)
        ii, jj = np.meshgrid(i, j, indexing='ij')
        
        # 向量化计算局部坐标
        xi = (2 * ii + 1) / N - 1
        eta = (2 * jj + 1) / N - 1
        
        # 根据face计算xyz坐标
        if face == 0:
            X, Y, Z = 1.0, xi, -eta
        elif face == 1:
            X, Y, Z = -xi, 1.0, -eta
        elif face == 2:
            X, Y, Z = -1.0, -xi, -eta
        elif face == 3:
            X, Y, Z = xi, -1.0, -eta
        elif face == 4:
            X, Y, Z = eta, xi, 1.0
        elif face == 5:
            X, Y, Z = -eta, xi, -1.0
        
        # xyz -> lon/lat (向量化)
        lon_grid = np.arctan2(Y, X) * 180.0 / np.pi
        lat_grid = np.arctan2(Z, np.sqrt(X**2 + Y**2)) * 180.0 / np.pi
        
        coords[face] = (lon_grid, lat_grid)
    return coords


# ============================================================
# 2. 数据加载器
# ============================================================

def load_field(variable: str, face: int):
    """
    加载指定变量和面的 OpenVisus 数据集
    
    Args:
        variable: 变量名 (如 'u', 'v', 'p', 'theta', 'salt' 等)
        face: 立方体面编号 (0~5)
    
    Returns:
        OpenVisus Dataset 对象
    """
    url = (
        f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
        f"nasa/nsdf/climate3/dyamond/GEOS/GEOS_{variable.upper()}/"
        f"{variable.lower()}_face_{face}_depth_52_time_0_10269.idx"
    )
    return ov.LoadDataset(url)


# ============================================================
# 3. 经纬范围筛选
# ============================================================

def find_points_in_range(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在经纬度网格中找到指定范围内的点
    
    Args:
        lon_grid: 经度网格 (N, M)
        lat_grid: 纬度网格 (N, M)
        lon_min, lon_max: 经度范围
        lat_min, lat_max: 纬度范围
    
    Returns:
        (i_indices, j_indices): 满足条件的点的索引
    """
    # 处理经度跨越180/-180度的情况
    if lon_min > lon_max:
        # 跨越日期变更线
        mask = ((lon_grid >= lon_min) | (lon_grid <= lon_max)) & \
               (lat_grid >= lat_min) & (lat_grid <= lat_max)
    else:
        mask = (lon_grid >= lon_min) & (lon_grid <= lon_max) & \
               (lat_grid >= lat_min) & (lat_grid <= lat_max)
    
    i_indices, j_indices = np.where(mask)
    return i_indices, j_indices


# ============================================================
# 4. 数据提取主函数
# ============================================================

def extract_data(
    variables: List[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    time_step: int,
    layer_min: Optional[int] = None,
    layer_max: Optional[int] = None,
    quality: int = -9,
    N: int = 1440
) -> Dict[str, np.ndarray]:
    """
    提取指定经纬范围、时间点、层数的数据
    
    Args:
        variables: 变量名列表 (如 ['u', 'v', 'p'])
        lon_min, lon_max: 经度范围
        lat_min, lat_max: 纬度范围
        time_step: 时间步索引
        layers: 层数列表（正数=大气层，负数=海洋层，按空间分布排列）
        quality: 数据质量等级，默认-9
        N: 网格分辨率，默认1440
    
    Returns:
        dict: {variable: data_array} 每个变量的数据数组
    """
    # 确定层数范围
    # 根据数据信息：
    # - 大气数据：51层（索引0-50）
    # - 海洋数据：90层（索引0-89）
    # 如果未指定，需要根据变量类型确定默认层数
    if layer_min is None or layer_max is None:
        # 默认全部层
        # 需要先读取一个样本来确定实际层数
        sample_db = None
        for var in variables:
            try:
                if var.upper() in ATMOSPHERE_VARIABLES:
                    sample_db = load_field(var, 0)
                elif var.lower() in OCEAN_VARIABLES:
                    # 海洋数据使用不同的URL格式
                    base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
                    if var.lower() in ["theta", "w"]:
                        ocn_url = base_url + f"mit_output/llc2160_{var.lower()}/llc2160_{var.lower()}.idx"
                    elif var.lower() == "u":
                        ocn_url = base_url + "mit_output/llc2160_arco/visus.idx"
                    else:
                        ocn_url = base_url + f"mit_output/llc2160_{var.lower()}/{var.lower()}_llc2160_x_y_depth.idx"
                    sample_db = ov.LoadDataset(ocn_url)
                if sample_db:
                    break
            except:
                continue
        
        if sample_db:
            # 读取一个样本来确定层数
            try:
                sample_data = sample_db.read(time=0, quality=quality)
                total_layers = sample_data.shape[2] if len(sample_data.shape) == 3 else 1
                if layer_min is None:
                    layer_min = 0 if total_layers > 0 else 0
                if layer_max is None:
                    layer_max = total_layers - 1 if total_layers > 0 else 0
            except:
                # 如果读取失败，使用默认值
                # 根据数据信息：大气51层（0-50），海洋90层（0-89）
                if layer_min is None:
                    layer_min = 0
                if layer_max is None:
                    # 需要根据变量类型确定默认层数
                    # 暂时使用大气层数（51层），如果用户选择了海洋变量，会在后面调整
                    layer_max = 50  # 默认大气51层（0-50）
        else:
            if layer_min is None:
                layer_min = 0
            if layer_max is None:
                layer_max = 50  # 默认大气51层（0-50）
    
    # 检查是否有海洋变量和大气变量
    has_ocean_vars = any(v in OCEAN_VARIABLES for v in variables)
    has_atm_vars = any(v in ATMOSPHERE_VARIABLES for v in variables)
    
    # 分别确定大气和海洋的层数范围
    # 大气数据：51层（0-50）
    # 海洋数据：90层（0-89）
    if layer_min is None or layer_max is None:
        if has_ocean_vars and not has_atm_vars:
            # 只有海洋变量，默认使用海洋层数（0-89）
            if layer_min is None:
                layer_min = 0
            if layer_max is None:
                layer_max = 89
        elif has_atm_vars and not has_ocean_vars:
            # 只有大气变量，默认使用大气层数（0-50）
            if layer_min is None:
                layer_min = 0
            if layer_max is None:
                layer_max = 50
        else:
            # 混合变量，默认使用全部层数（0-89，包括大气和海洋）
            if layer_min is None:
                layer_min = 0
            if layer_max is None:
                layer_max = 89
    
    # 为大气和海洋分别生成层数列表
    # 大气层数范围：限制在0-50
    atm_layer_min = max(0, layer_min) if layer_min is not None else 0
    atm_layer_max = min(50, layer_max) if layer_max is not None else 50
    atm_layers = list(range(atm_layer_min, atm_layer_max + 1)) if atm_layer_max >= atm_layer_min else []
    
    # 海洋层数范围：限制在0-89
    ocn_layer_min = max(0, layer_min) if layer_min is not None else 0
    ocn_layer_max = min(89, layer_max) if layer_max is not None else 89
    ocn_layers = list(range(ocn_layer_min, ocn_layer_max + 1)) if ocn_layer_max >= ocn_layer_min else []
    
    # 保存层数范围（用于后续处理）
    actual_layer_min = layer_min
    actual_layer_max = layer_max
    
    print(f"\n{'='*60}")
    print(f"开始提取数据")
    print(f"  经纬范围: [{lon_min:.2f}, {lon_max:.2f}] × [{lat_min:.2f}, {lat_max:.2f}]")
    print(f"  时间步: {time_step}")
    if has_atm_vars:
        print(f"  大气层数范围: [{atm_layer_min}, {atm_layer_max}] (共 {len(atm_layers)} 层)")
    if has_ocean_vars:
        print(f"  海洋层数范围: [{ocn_layer_min}, {ocn_layer_max}] (共 {len(ocn_layers)} 层)")
    print(f"  变量: {variables}")
    print(f"{'='*60}\n")
    
    # 分离大气和海洋变量
    # 注意：U, V, W 同时存在于大气和海洋变量中
    # 我们假设用户输入的变量名已经正确区分（大气用大写，海洋也用大写但会明确指定）
    # 实际上，由于变量名完全相同，我们需要根据用户的选择来判断
    # 但为了简化，我们按照变量列表的顺序匹配：先匹配大气变量，再匹配海洋变量
    atm_vars = []
    ocn_vars = []
    unclassified = []
    
    for v in variables:
        # 先检查是否在大气变量列表中
        if v in ATMOSPHERE_VARIABLES:
            atm_vars.append(v)
        # 再检查是否在海洋变量列表中（但不在大气变量列表中，避免重复）
        elif v in OCEAN_VARIABLES:
            ocn_vars.append(v)
        else:
            unclassified.append(v)
    
    # 检查是否有未分类的变量
    if unclassified:
        print(f"  警告: 以下变量未分类，将被跳过: {unclassified}")
    
    # 检查U, V, W是否同时被分类为大气和海洋变量
    # 如果用户同时选择了大气和海洋的U, V, W，我们需要警告用户
    common_vars = set(atm_vars) & set(ocn_vars)
    if common_vars:
        print(f"  警告: 以下变量同时存在于大气和海洋数据中: {common_vars}")
        print(f"  这些变量将被同时提取（大气和海洋版本）")
    
    # 存储所有面的数据
    all_data = {var: [] for var in variables}
    all_coords = {'lon': [], 'lat': []}
    
    # ============================================================
    # 处理大气数据（使用立方体网格映射）
    # ============================================================
    if atm_vars:
        print(f"\n处理大气数据（{len(atm_vars)} 个变量）...")
        # 遍历所有6个面
        for face in range(6):
            print(f"处理 Face {face}...")
            
            try:
                # 获取该面的经纬度坐标
                lon_grid, lat_grid = get_all_face_coordinates(N)[face]
                
                # 找到范围内的点
                i_indices, j_indices = find_points_in_range(
                    lon_grid, lat_grid, lon_min, lon_max, lat_min, lat_max
                )
                
                if len(i_indices) == 0:
                    print(f"  Face {face}: 无数据点在范围内")
                    continue
                
                print(f"  Face {face}: 找到 {len(i_indices)} 个数据点")
                
                # 先保存坐标（基于立方体网格的经纬度坐标）
                # 注意：坐标应该基于立方体网格，而不是数据形状
                for idx in range(len(i_indices)):
                    i, j = i_indices[idx], j_indices[idx]
                    # 确保索引在立方体网格范围内（N x N）
                    if i < N and j < N:
                        all_coords['lon'].append(lon_grid[i, j])
                        all_coords['lat'].append(lat_grid[i, j])
                
                # 读取每个大气变量的数据
                for var in atm_vars:
                    try:
                        db = load_field(var, face)
                        timesteps = db.getTimesteps()
                        
                        if time_step >= len(timesteps):
                            print(f"  警告: 时间步 {time_step} 超出范围 (最大: {len(timesteps)-1})")
                            continue
                        
                        t = timesteps[time_step]
                        data_raw = db.read(time=t, quality=quality)
                        
                        # 检查数据是否为空
                        if data_raw is None or data_raw.size == 0:
                            print(f"  警告: 变量 {var} 数据为空，跳过")
                            continue
                        
                        # 检查数据形状：参考detect14month2.py，数据应该是 (y, x, z) 格式
                        if len(data_raw.shape) != 3:
                            print(f"  警告: 变量 {var} 数据形状异常: {data_raw.shape}")
                            continue
                        
                        # 参考detect14month2.py的读取方式，直接使用数据，不进行转置
                        # 数据形状应该是 (1440, 1440, 52) 或 (1440, 1440, 51)
                        ny, nx, nz = data_raw.shape
                        print(f"    数据形状: {data_raw.shape} (y={ny}, x={nx}, z={nz})")
                        
                        # 验证数据形状是否符合预期（大气数据应该是1440x1440x51或1440x1440x52）
                        if ny != 1440 or nx != 1440:
                            print(f"    警告: 数据形状与预期不符，预期 (1440, 1440, 51或52)，实际 {data_raw.shape}")
                        if nz not in [51, 52]:
                            print(f"    警告: 深度维度与预期不符，预期51或52层，实际{nz}层")
                        
                        # 如果数据有52层，只使用前51层（索引0-50）
                        if nz == 52:
                            data_raw = data_raw[:, :, :51]
                            nz = 51
                        
                        # 提取指定层的数据（只使用大气层数范围）
                        n_points = len(i_indices)
                        n_layers = len(atm_layers)
                        var_data_points = []
                        
                        if n_layers == 0:
                            print(f"    警告: 变量 {var} 没有有效的层数范围（大气层数范围: {atm_layer_min}-{atm_layer_max}）")
                            continue
                        
                        # 对每个点，提取所有层的数据
                        # 参考detect14month2.py：直接使用索引访问 data_raw[:, :, layer]
                        # 立方体网格索引 (i, j) 应该直接对应数据索引，因为都是1440x1440
                        valid_point_count = 0
                        for point_idx in range(n_points):
                            i_cube, j_cube = i_indices[point_idx], j_indices[point_idx]
                            
                            # 确保索引在有效范围内（立方体网格是NxN，数据是1440x1440）
                            # 如果N=1440，直接使用；否则需要映射
                            if N == 1440 and ny == 1440 and nx == 1440:
                                # 数据形状与立方体网格相同，直接使用
                                i_data, j_data = i_cube, j_cube
                            else:
                                # 数据形状与立方体网格不同，需要索引映射
                                i_data = int(i_cube * ny / N) if N > 0 else 0
                                j_data = int(j_cube * nx / N) if N > 0 else 0
                                i_data = min(i_data, ny - 1)
                                j_data = min(j_data, nx - 1)
                            
                            # 确保索引在有效范围内
                            if i_data >= ny or j_data >= nx or i_data < 0 or j_data < 0:
                                continue
                            
                            valid_point_count += 1
                            point_layers_data = []
                            
                            # 只提取大气层数范围内的数据
                            # 参考detect14month2.py：使用 data_raw[:, :, layer] 访问
                            for layer in atm_layers:
                                # 大气层（从顶部开始，0是最上层）
                                if 0 <= layer < nz:
                                    try:
                                        layer_value = float(data_raw[i_data, j_data, layer])
                                        # 检查是否为NaN或无效值
                                        if np.isnan(layer_value) or np.isinf(layer_value):
                                            layer_value = np.nan
                                    except (IndexError, ValueError) as e:
                                        layer_value = np.nan
                                else:
                                    layer_value = np.nan
                                
                                point_layers_data.append(layer_value)
                            
                            var_data_points.append(point_layers_data)
                        
                        # 存储数据
                        if var_data_points:
                            all_data[var].extend(var_data_points)
                            print(f"    变量 {var}: 成功提取 {len(var_data_points)} 个数据点")
                    
                    except Exception as e:
                        error_msg = str(e)
                        if "empty content" in error_msg.lower():
                            print(f"  警告: 变量 {var} 数据为空或不存在，跳过")
                        else:
                            print(f"  错误: 读取变量 {var} 失败: {e}")
                        continue
            
            except Exception as e:
                print(f"  错误: 处理 Face {face} 失败: {e}")
                continue
    
    # ============================================================
    # 处理海洋数据（直接使用经纬度索引，不需要立方体网格映射）
    # ============================================================
    if ocn_vars:
        print(f"\n处理海洋数据（{len(ocn_vars)} 个变量）...")
        
        for var in ocn_vars:
            try:
                # 加载海洋数据集
                # 根据数据信息，海洋变量是：U, V, W, Theta, Salt
                # 注意：变量名是 Theta 和 Salt（首字母大写），但URL中可能使用小写
                base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
                var_lower = var.lower()
                
                # 根据变量名构建URL（参考ocean1.py等文件中的URL格式）
                if var == "Theta" or var_lower == "theta":
                    ocn_url = base_url + "mit_output/llc2160_theta/llc2160_theta.idx"
                elif var_lower == "w":
                    ocn_url = base_url + "mit_output/llc2160_w/llc2160_w.idx"
                elif var_lower == "u":
                    ocn_url = base_url + "mit_output/llc2160_arco/visus.idx"
                elif var == "Salt" or var_lower == "salt":
                    ocn_url = base_url + "mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"
                elif var_lower == "v":
                    ocn_url = base_url + "mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
                else:
                    # 默认格式
                    ocn_url = base_url + f"mit_output/llc2160_{var_lower}/{var_lower}_llc2160_x_y_depth.idx"
                
                db = ov.LoadDataset(ocn_url)
                timesteps = db.getTimesteps()
                
                if time_step >= len(timesteps):
                    print(f"  警告: 时间步 {time_step} 超出范围 (最大: {len(timesteps)-1})")
                    continue
                
                t = timesteps[time_step]
                data_full = db.read(time=t, quality=quality)
                
                # 检查数据是否为空
                if data_full is None or data_full.size == 0:
                    print(f"  警告: 变量 {var} 数据为空，跳过")
                    continue
                
                # 海洋数据形状: (lat_dim, lon_dim, depth_dim)
                # 参考velocity_3D_vector_optimized.py：直接使用 data_full[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end, :nz]
                if len(data_full.shape) != 3:
                    print(f"  警告: 变量 {var} 数据形状异常: {data_full.shape}")
                    continue
                
                lat_dim, lon_dim, depth_dim = data_full.shape
                print(f"    数据形状: {data_full.shape} (lat={lat_dim}, lon={lon_dim}, depth={depth_dim})")
                
                # 验证数据形状是否符合预期（根据实际数据可能有所不同）
                if depth_dim != 90:
                    print(f"    警告: 深度维度与预期不符，预期90层，实际{depth_dim}层")
                
                # 计算经纬度索引范围
                # 纬度: -90 到 90
                lat_idx_start = max(0, int(lat_dim * (lat_min + 90) / 180))
                lat_idx_end = min(lat_dim, int(lat_dim * (lat_max + 90) / 180) + 1)
                
                # 经度: -180 到 180
                if lon_min > lon_max:
                    # 跨越日期变更线
                    lon_idx_start1 = max(0, int(lon_dim * (lon_min + 180) / 360))
                    lon_idx_end1 = lon_dim
                    lon_idx_start2 = 0
                    lon_idx_end2 = min(lon_dim, int(lon_dim * (lon_max + 180) / 360) + 1)
                    # 需要处理两个区域
                    lon_ranges = [(lon_idx_start1, lon_idx_end1), (lon_idx_start2, lon_idx_end2)]
                else:
                    lon_idx_start = max(0, int(lon_dim * (lon_min + 180) / 360))
                    lon_idx_end = min(lon_dim, int(lon_dim * (lon_max + 180) / 360) + 1)
                    lon_ranges = [(lon_idx_start, lon_idx_end)]
                
                var_data_points = []
                var_coords_lon = []
                var_coords_lat = []
                
                # 提取数据
                for lon_start_idx, lon_end_idx in lon_ranges:
                    data_region = data_full[lat_idx_start:lat_idx_end, lon_start_idx:lon_end_idx, :]
                    
                    # 生成该区域的经纬度网格
                    lat_indices = np.arange(lat_idx_start, lat_idx_end)
                    lon_indices = np.arange(lon_start_idx, lon_end_idx)
                    
                    for lat_idx in lat_indices:
                        for lon_idx in lon_indices:
                            # 计算实际经纬度
                            actual_lat = (lat_idx / lat_dim) * 180 - 90
                            actual_lon = (lon_idx / lon_dim) * 360 - 180
                            
                            # 检查是否在范围内
                            if (lat_min <= actual_lat <= lat_max) and \
                               ((lon_min <= actual_lon <= lon_max) or (lon_min > lon_max and (actual_lon >= lon_min or actual_lon <= lon_max))):
                                
                                point_layers_data = []
                                
                                # 只提取海洋层数范围内的数据
                                # 参考velocity_3D_vector_optimized.py：直接使用 data_full[lat_idx, lon_idx, layer]
                                for layer in ocn_layers:
                                    # 海洋层（从顶部开始，0是最上层）
                                    if 0 <= layer < depth_dim:
                                        try:
                                            layer_value = float(data_full[lat_idx, lon_idx, layer])
                                            # 检查是否为NaN或无效值
                                            if np.isnan(layer_value) or np.isinf(layer_value):
                                                layer_value = np.nan
                                        except (IndexError, ValueError) as e:
                                            layer_value = np.nan
                                    else:
                                        layer_value = np.nan
                                
                                    point_layers_data.append(layer_value)
                                
                                var_data_points.append(point_layers_data)
                                var_coords_lon.append(actual_lon)
                                var_coords_lat.append(actual_lat)
                
                # 存储数据
                if var_data_points:
                    all_data[var].extend(var_data_points)
                    # 只保存一次坐标（使用第一个海洋变量）
                    if var == ocn_vars[0]:
                        all_coords['lon'].extend(var_coords_lon)
                        all_coords['lat'].extend(var_coords_lat)
                
                print(f"  海洋变量 {var}: 提取了 {len(var_data_points)} 个数据点")
            
            except Exception as e:
                print(f"  错误: 读取海洋变量 {var} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 转换为numpy数组
    result = {}
    
    # 先处理坐标数据（每个点只保存一次，不重复）
    # 注意：大气数据和海洋数据可能有不同的坐标点，需要合并去重
    if all_coords['lon']:
        # 转换为numpy数组并去重（基于经纬度）
        # 使用字典去重（保留第一次出现的坐标）
        unique_coords = {}
        coord_order = []  # 保持顺序
        
        for lon, lat in zip(all_coords['lon'], all_coords['lat']):
            coord_key = (round(lon, 6), round(lat, 6))  # 保留6位小数精度
            if coord_key not in unique_coords:
                unique_coords[coord_key] = (lon, lat)
                coord_order.append(coord_key)
        
        unique_lons, unique_lats = zip(*[unique_coords[k] for k in coord_order])
        result['lon'] = np.array(unique_lons)
        result['lat'] = np.array(unique_lats)
        print(f"  坐标去重: 原始 {len(all_coords['lon'])} 个点 -> 去重后 {len(result['lon'])} 个点")
    else:
        result['lon'] = np.array([])
        result['lat'] = np.array([])
    
    # 合并所有面的数据
    # 注意：需要确保每个变量的数据点数与坐标点数匹配
    for var in variables:
        if all_data[var] and len(all_data[var]) > 0:
            # all_data[var] 是列表，每个元素是一个点的所有层数据（列表）
            # 直接转换为numpy数组
            result[var] = np.array(all_data[var])
        else:
            result[var] = np.array([])
    
    # 检查数据点数与坐标点数的匹配
    if len(result['lon']) > 0:
        n_coords = len(result['lon'])
        print(f"\n数据点数检查:")
        for var in variables:
            if len(result[var]) > 0:
                n_data_points = result[var].shape[0]
                if n_data_points != n_coords:
                    print(f"  警告: 变量 {var} 的数据点数 ({n_data_points}) 与坐标点数 ({n_coords}) 不匹配")
                else:
                    print(f"  ✓ 变量 {var}: {n_data_points} 个数据点，与坐标点数匹配")
    
    # 保存层信息
    # 合并大气和海洋层数
    all_layers = []
    if has_atm_vars and len(atm_layers) > 0:
        all_layers.extend(atm_layers)
    if has_ocean_vars and len(ocn_layers) > 0:
        all_layers.extend(ocn_layers)
    
    if len(all_layers) > 0:
        result['layers'] = np.array(all_layers)
        result['layer_names'] = [f"layer_{l}" for l in all_layers]
        result['layer_min'] = actual_layer_min
        result['layer_max'] = actual_layer_max
        if has_atm_vars:
            result['atm_layers'] = np.array(atm_layers)
        if has_ocean_vars:
            result['ocn_layers'] = np.array(ocn_layers)
    else:
        result['layers'] = np.array([])
        result['layer_names'] = []
        result['layer_min'] = None
        result['layer_max'] = None
    
    print(f"\n提取完成:")
    for var in variables:
        if len(result[var]) > 0:
            print(f"  {var}: shape={result[var].shape}")
    if len(result['lon']) > 0:
        print(f"  坐标点数: {len(result['lon'])}")
        if has_atm_vars:
            print(f"  大气层数: {len(atm_layers)} ({atm_layers[:5]}...{atm_layers[-5:] if len(atm_layers) > 10 else atm_layers})")
        if has_ocean_vars:
            print(f"  海洋层数: {len(ocn_layers)} ({ocn_layers[:5]}...{ocn_layers[-5:] if len(ocn_layers) > 10 else ocn_layers})")
    
    return result


# ============================================================
# 5. 数据保存
# ============================================================

def save_data(
    data: Dict[str, np.ndarray],
    save_path: str,
    format: str = 'npz'
):
    """
    保存提取的数据到本地文件
    
    Args:
        data: 数据字典
        save_path: 保存路径
        format: 保存格式 ('npz' 或 'nc')
    """
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if format == 'npz':
        # 保存为 NumPy 压缩格式
        np.savez_compressed(save_path, **data)
        print(f"\n数据已保存到: {save_path}")
        print(f"  文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    elif format == 'nc':
        # 保存为 NetCDF 格式
        nc = Dataset(save_path, 'w', format='NETCDF4')
        
        # 创建维度
        n_points = len(data['lon'])
        nc.createDimension('points', n_points)
        
        # 创建坐标变量
        lon_var = nc.createVariable('lon', 'f8', ('points',))
        lat_var = nc.createVariable('lat', 'f8', ('points',))
        lon_var[:] = data['lon']
        lat_var[:] = data['lat']
        lon_var.units = 'degrees_east'
        lat_var.units = 'degrees_north'
        
        # 创建层维度（如果有多层数据）
        if 'layers' in data and len(data['layers']) > 0:
            n_layers = len(data['layers'])
            nc.createDimension('layers', n_layers)
            layers_var = nc.createVariable('layers', 'i4', ('layers',))
            layers_var[:] = data['layers']
            layers_var.long_name = 'Layer indices (positive=atmosphere, negative=ocean)'
        
        # 创建数据变量
        for var_name, var_data in data.items():
            if var_name not in ['lon', 'lat', 'layers', 'layer_names'] and len(var_data) > 0:
                if var_data.ndim == 1:
                    # 一维数据（单层）
                    var = nc.createVariable(var_name, 'f4', ('points',))
                    var[:] = var_data
                elif var_data.ndim == 2:
                    # 二维数据（多层）
                    var = nc.createVariable(var_name, 'f4', ('points', 'layers'))
                    var[:] = var_data
                var.long_name = f'{var_name} variable'
        
        nc.close()
        print(f"\n数据已保存到: {save_path} (NetCDF格式)")
        print(f"  文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    else:
        raise ValueError(f"不支持的保存格式: {format}，请使用 'npz' 或 'nc'")


# ============================================================
# 6. 主程序
# ============================================================

def main():
    """主程序：交互式数据提取"""
    print("\n" + "="*60)
    print("大气海洋数据提取工具")
    print("="*60)
    
    # 1. 输入经纬范围
    print("\n【1】请输入经纬范围（左下角和右上角）:")
    lon_min = float(input("  左下角经度 (lon_min): "))
    lat_min = float(input("  左下角纬度 (lat_min): "))
    lon_max = float(input("  右上角经度 (lon_max): "))
    lat_max = float(input("  右上角纬度 (lat_max): "))
    
    # 2. 输入时间点
    print("\n【2】请输入时间点:")
    time_step = int(input("  时间步索引 (time_step): "))
    
    # 3. 输入层数范围
    print("\n【3】请输入层数范围（默认全部层）:")
    print("  层数说明: 正数=大气层（从顶部开始，0是最上层），负数=海洋层（从底部开始，-1是最底层）")
    layer_min_input = input("  层数下界 (layer_min，直接回车使用默认全部): ").strip()
    layer_max_input = input("  层数上界 (layer_max，直接回车使用默认全部): ").strip()
    
    layer_min = int(layer_min_input) if layer_min_input else None
    layer_max = int(layer_max_input) if layer_max_input else None
    
    # 4. 输入变量（默认全部）
    print("\n【4】请输入要提取的变量（默认全部）:")
    print(f"  大气变量 (GEOS, 立方体网格, 51层): {', '.join(ATMOSPHERE_VARIABLES)}")
    print(f"  海洋变量 (LLC2160, 经纬度网格, 90层): {', '.join(OCEAN_VARIABLES)}")
    print(f"  注意: U, V, W 同时存在于大气和海洋数据中，需要根据上下文区分")
    vars_input = input("  变量列表 (用逗号分隔，直接回车提取全部): ").strip()
    
    if vars_input:
        variables = [v.strip() for v in vars_input.split(',')]
    else:
        # 默认提取全部变量
        variables = ATMOSPHERE_VARIABLES + OCEAN_VARIABLES
        print(f"  使用默认: 提取全部变量 ({len(variables)} 个)")
    
    # 5. 是否保存
    print("\n【5】是否保存数据到本地?")
    save_choice = input("  (y/n，默认y): ").strip().lower()
    save_data_flag = save_choice in ['y', 'yes', '是', '']  # 空字符串也视为yes
    
    save_path = None
    if save_data_flag:
        save_path_input = input("  保存路径 (直接回车使用默认: extracted_data_YYYYMMDD_HHMMSS.npz): ").strip()
        if not save_path_input:
            # 生成默认文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"extracted_data_{timestamp}.npz"
        else:
            save_path = save_path_input
    
    # 6. 提取数据
    print("\n开始提取数据...")
    try:
        data = extract_data(
            variables=variables,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            time_step=time_step,
            layer_min=layer_min,
            layer_max=layer_max
        )
        
        # 7. 保存数据
        if save_data_flag and save_path:
            # 根据文件扩展名确定格式
            if save_path.endswith('.nc'):
                save_data(data, save_path, format='nc')
            else:
                save_data(data, save_path, format='npz')
        
        print("\n" + "="*60)
        print("数据提取完成！")
        print("="*60)
        
        # 显示数据摘要
        print("\n数据摘要:")
        for var in variables:
            if var in data and len(data[var]) > 0:
                print(f"  {var}: shape={data[var].shape}, "
                      f"min={np.min(data[var]):.4f}, max={np.max(data[var]):.4f}")
        
    except Exception as e:
        print(f"\n错误: 数据提取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

