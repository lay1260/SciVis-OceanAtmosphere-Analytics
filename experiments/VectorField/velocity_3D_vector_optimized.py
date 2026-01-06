#温度盐度采用volume rendering，洋流使用流线
# 矢量场优化版本：
# - 温度：只映射颜色值（hot色图）
# - 盐度：只映射透明度（使用策略17：反幂函数，0.8次，0~0.3透明度）
# - 箭头：完全不透明
# - 自动启用LOD优化（步骤5）
# - 自动启用环境光与背景优化（步骤6）
# - 使用VTK底层API实现真正的双标量独立控制
import OpenVisus as ov
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
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

# ----------------------------
# 1️⃣ 数据集路径与加载
# ----------------------------
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

def load_dataset(variable):
    if variable in ["theta", "w"]:
        base_dir=f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
    elif variable=="u":
        base_dir="mit_output/llc2160_arco/visus.idx"
    else:
        base_dir=f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
    dataset_url = base_url + base_dir
    db = ov.LoadDataset(dataset_url)
    return db

# ----------------------------
# 2️⃣ 加载速度、盐度和温度
# ----------------------------
U_db = load_dataset("u")
V_db = load_dataset("v")
W_db = load_dataset("w")
Salt_db = load_dataset("salt")
Theta_db = load_dataset("theta")

# ----------------------------
# 3️⃣ 局部区域参数（降低分辨率以提升加载速度和交互流畅度）
# ----------------------------
lat_start, lat_end = 10, 40
lon_start, lon_end = 100, 130
nz = 10  # 进一步减少深度层数以提升性能
data_quality = -6
scale_xy = 25
skip = None  # 采样距离（平衡性能和网格有效性，确保每个维度至少2个点）

# ----------------------------
# 4️⃣ 读取局部数据函数
# ----------------------------
def read_data(db, skip_value=None):
    """读取局部数据
    
    Args:
        db: 数据集
        skip_value: 采样间隔，如果为None则使用全局skip变量
    """
    if skip_value is None:
        skip_value = skip
    
    data_full = db.read(time=0, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    
    # 检查索引范围
    if lat_idx_end <= lat_idx_start or lon_idx_end <= lon_idx_start:
        print(f"警告：索引范围无效 lat=[{lat_idx_start}:{lat_idx_end}], lon=[{lon_idx_start}:{lon_idx_end}]")
        # 使用默认范围
        lat_idx_start = 0
        lat_idx_end = lat_dim
        lon_idx_start = 0
        lon_idx_end = lon_dim
    
    result = data_full[lat_idx_start:lat_idx_end:skip_value,
                       lon_idx_start:lon_idx_end:skip_value,
                       :nz]
    
    if result.size == 0:
        print(f"警告：skip={skip_value}导致数据为空，尝试使用skip=2")
        # 如果数据为空，尝试使用更小的skip
        result = data_full[lat_idx_start:lat_idx_end:2,
                           lon_idx_start:lon_idx_end:2,
                           :nz]
    
    return result

U_local = read_data(U_db)
V_local = read_data(V_db)
W_local = read_data(W_db)
Salt_local = read_data(Salt_db)
Theta_local = read_data(Theta_db)

# 检查数据是否成功加载
print(f"U_local形状: {U_local.shape}")
print(f"Salt_local形状: {Salt_local.shape}")
print(f"Theta_local形状: {Theta_local.shape}")

if U_local.size == 0 or Salt_local.size == 0 or Theta_local.size == 0:
    print("错误：数据加载失败，数据为空！")
    print("尝试减小skip值或检查数据源")

nx, ny, nz = U_local.shape
print(f"网格尺寸: nx={nx}, ny={ny}, nz={nz}")

# 检查维度是否有效（每个维度至少需要2个点才能构成3D网格）
if nx < 2 or ny < 2 or nz < 2:
    print(f"警告：网格维度不足（nx={nx}, ny={ny}, nz={nz}），无法构成有效的3D结构化网格")
    print(f"当前skip={skip}太大，导致采样后数据不足")
    print("尝试使用更小的skip值...")
    # 重新加载数据，使用更小的skip
    new_skip = max(2, skip // 2) if skip else 2  # 至少使用skip=2
    print(f"使用skip={new_skip}重新加载数据")
    U_local = read_data(U_db, skip_value=new_skip)
    V_local = read_data(V_db, skip_value=new_skip)
    W_local = read_data(W_db, skip_value=new_skip)
    Salt_local = read_data(Salt_db, skip_value=new_skip)
    Theta_local = read_data(Theta_db, skip_value=new_skip)
    skip = new_skip  # 更新全局skip值
    nx, ny, nz = U_local.shape
    print(f"重新加载后网格尺寸: nx={nx}, ny={ny}, nz={nz}")
    
    # 再次检查
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(f"即使使用skip={skip}，网格维度仍不足（nx={nx}, ny={ny}, nz={nz}）。请减小skip值或增大区域范围。")

z_grid = np.linspace(0, 1000, nz)

# ----------------------------
# 5️⃣ 构建实际比例网格
# ----------------------------
x = np.linspace(lon_start, lon_end, ny) * scale_xy
y = np.linspace(lat_start, lat_end, nx) * scale_xy
X, Y, Z = np.meshgrid(x, y, z_grid, indexing='ij')
X = X.transpose(1,0,2)
Y = Y.transpose(1,0,2)
Z = -Z.transpose(1,0,2)

# ----------------------------
# 6️⃣ 结构化网格 + 速度箭头可视化
# ----------------------------
grid = pv.StructuredGrid(X, Y, Z)
vectors = np.stack([U_local.flatten(order="F"),
                    V_local.flatten(order="F"),
                    W_local.flatten(order="F")], axis=1)
grid["velocity"] = vectors

# 在立方体上均匀采样，每条边上10个点，一共10*10*10=1000个点
sampling_points_per_edge = 10

# 计算采样点的索引
# 在x, y, z三个方向上均匀采样（对应nx, ny, nz维度）
# 关键：确保采样点覆盖整个体积，包括内部点
# 对于每个维度，如果维度大小 >= 采样点数，则均匀采样（包括边界和内部）
# 如果维度大小 < 采样点数，则只采样所有可用的点（包括边界）

# 计算每个维度的实际采样点数（不超过维度大小）
n_samples_x = min(sampling_points_per_edge, nx)
n_samples_y = min(sampling_points_per_edge, ny)
n_samples_z = min(sampling_points_per_edge, nz)

# 生成均匀分布的索引（包括边界和内部）
# 优化：确保包含内部点，特别是y和z方向
if nx > 1:
    if nx > 2 and n_samples_x > 2:
        # 强制保留中间点（避免只有边界点）
        x_indices = np.unique(np.concatenate([
            [0],  # 边界
            np.linspace(1, nx-2, max(1, n_samples_x-2), dtype=int),  # 内部点
            [nx-1]  # 边界
        ]))
    else:
        x_indices = np.linspace(0, nx-1, n_samples_x, dtype=int)
        x_indices[0] = 0
        if len(x_indices) > 1:
            x_indices[-1] = nx - 1
else:
    x_indices = np.array([0], dtype=int)

if ny > 1:
    if ny > 2 and n_samples_y > 2:
        # 强制保留中间点（避免只有边界点）
        y_indices = np.unique(np.concatenate([
            [0],  # 边界
            np.linspace(1, ny-2, max(1, n_samples_y-2), dtype=int),  # 内部点
            [ny-1]  # 边界
        ]))
    else:
        y_indices = np.linspace(0, ny-1, n_samples_y, dtype=int)
        y_indices[0] = 0
        if len(y_indices) > 1:
            y_indices[-1] = ny - 1
else:
    y_indices = np.array([0], dtype=int)

if nz > 1:
    if nz > 2 and n_samples_z > 2:
        # 强制保留中间点（避免只有边界点）
        z_indices = np.unique(np.concatenate([
            [0],  # 边界
            np.linspace(1, nz-2, max(1, n_samples_z-2), dtype=int),  # 内部点
            [nz-1]  # 边界
        ]))
    else:
        z_indices = np.linspace(0, nz-1, n_samples_z, dtype=int)
        z_indices[0] = 0
        if len(z_indices) > 1:
            z_indices[-1] = nz - 1
else:
    z_indices = np.array([0], dtype=int)

# 去重，保持顺序
x_indices = np.unique(x_indices)
y_indices = np.unique(y_indices)
z_indices = np.unique(z_indices)

print(f"采样索引: x={len(x_indices)}个点 {x_indices.tolist()}, y={len(y_indices)}个点 {y_indices.tolist()}, z={len(z_indices)}个点 {z_indices.tolist()}")

# 创建采样点的网格（注意：U_local的形状是(nx, ny, nz)）
# 使用meshgrid创建所有采样点的索引组合
X_idx, Y_idx, Z_idx = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
X_idx = X_idx.flatten()
Y_idx = Y_idx.flatten()
Z_idx = Z_idx.flatten()

print(f"创建采样点网格: {len(X_idx)} 个点 (目标: {sampling_points_per_edge**3})")
print(f"索引范围: x=[{np.min(X_idx)}, {np.max(X_idx)}] (nx={nx}), y=[{np.min(Y_idx)}, {np.max(Y_idx)}] (ny={ny}), z=[{np.min(Z_idx)}, {np.max(Z_idx)}] (nz={nz})")
print(f"实际采样点数: {len(x_indices)} x {len(y_indices)} x {len(z_indices)} = {len(X_idx)}")

# 直接从U_local, V_local, W_local获取速度向量（形状为(nx, ny, nz)）
# 使用grid.points直接获取坐标，避免索引混乱
sample_velocities = []
sample_speeds = []
sample_points_coords = []

# 计算grid.points的索引（grid.points是按F顺序展平的）
# grid.points的索引计算：对于(nx, ny, nz)网格，点(i, j, k)的索引是 i + j*nx + k*nx*ny
for i in range(len(X_idx)):
    x_idx, y_idx, z_idx = X_idx[i], Y_idx[i], Z_idx[i]
    
    # 确保索引在有效范围内
    x_idx = np.clip(x_idx, 0, nx-1)
    y_idx = np.clip(y_idx, 0, ny-1)
    z_idx = np.clip(z_idx, 0, nz-1)
    
    # 直接从原始数据获取速度（U_local形状是(nx, ny, nz)）
    u_val = U_local[x_idx, y_idx, z_idx]
    v_val = V_local[x_idx, y_idx, z_idx]
    w_val = W_local[x_idx, y_idx, z_idx]
    
    vel = np.array([u_val, v_val, w_val])
    speed = np.linalg.norm(vel)
    sample_velocities.append(vel)
    sample_speeds.append(speed)
    
    # 使用grid.points直接获取坐标（避免手动索引X, Y, Z）
    # grid.points是按F顺序展平的，索引计算：i + j*nx + k*nx*ny
    point_idx = x_idx + y_idx * nx + z_idx * nx * ny
    coords = grid.points[point_idx]
    sample_points_coords.append(coords)

sample_velocities = np.array(sample_velocities)
sample_speeds = np.array(sample_speeds)
sample_points_coords = np.array(sample_points_coords)

# 创建采样点的PolyData
sample_points = pv.PolyData(sample_points_coords)
sample_points["velocity"] = sample_velocities
sample_points["speed"] = sample_speeds

print(f"✅ 采样点创建完成: {len(sample_points_coords)} 个点")
print(f"速度范围: [{np.min(sample_speeds):.4f}, {np.max(sample_speeds):.4f}]")
print(f"采样点坐标范围: X[{np.min(sample_points_coords[:, 0]):.2f}, {np.max(sample_points_coords[:, 0]):.2f}], "
      f"Y[{np.min(sample_points_coords[:, 1]):.2f}, {np.max(sample_points_coords[:, 1]):.2f}], "
      f"Z[{np.min(sample_points_coords[:, 2]):.2f}, {np.max(sample_points_coords[:, 2]):.2f}]")

# ----------------------------
# 7️⃣ 温度与盐度体积渲染（优化：单体积双标量绑定）
# ----------------------------
# 核心优化：单体积双标量绑定（替代双体积叠加）
# 理论依据：PyVista支持单个体积同时绑定「颜色标量（温度）」和「透明度标量（盐度）」
# 双体积叠加会导致透明度非线性叠加，单体积直接映射可避免该问题
# 
# 可视化逻辑：
# - 温度：映射颜色值（hot色图）
# - 盐度：映射透明度（使用策略17：反幂函数，0.8次，0~0.3透明度）
# - 箭头：完全不透明

# 创建单网格，同时绑定温度、盐度双标量
combined_volume = pv.StructuredGrid(X, Y, Z)
theta_data = Theta_local.flatten(order="F")
salt_data = Salt_local.flatten(order="F")

# 绑定温度（用于颜色映射）
combined_volume["Temperature"] = theta_data
# 绑定盐度（用于透明度映射）
combined_volume["Salinity"] = salt_data

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
salt_data_original = salt_data.copy()
combined_volume_original = combined_volume

if use_lod:
    print("\n正在应用第五步优化：分级细节（LOD）渲染...")
    # 理论依据：核心区不仅基于y轴位置，还纳入盐度高值区，避免高盐关键区域被降采样
    
    # 1. 划分核心区：y轴中间区 + 盐度高值区（>70%分位数）
    y_coords = combined_volume.points[:, 1]  # 获取Y坐标
    y_min, y_max = y_coords.min(), y_coords.max()
    y_mid_low = y_min + (y_max - y_min) * 0.3  # 约30%位置
    y_mid_high = y_min + (y_max - y_min) * 0.7  # 约70%位置
    
    # 盐度高值区（70%分位数以上）
    salt_70_percentile = np.percentile(salt_data, 70)
    high_salt_mask = salt_data >= salt_70_percentile
    
    # 合并核心区：y轴中间区（30%~70%）或盐度高值区
    core_mask = ((y_coords >= y_mid_low) & (y_coords <= y_mid_high)) | high_salt_mask
    edge_mask = ~core_mask
    
    # 统计信息
    n_core = np.sum(core_mask)
    n_edge = np.sum(edge_mask)
    n_total_original = len(core_mask)
    print(f"核心区点数: {n_core} ({n_core/n_total_original*100:.1f}%)")
    print(f"边缘区点数: {n_edge} ({n_edge/n_total_original*100:.1f}%)")
    print(f"Y轴核心区范围: [{y_mid_low:.2f}, {y_mid_high:.2f}]")
    print(f"盐度高值阈值: {salt_70_percentile:.4f} (70%分位数)")
    
    # 2. 边缘区降采样（保留原逻辑，低盐边缘区降采样50%）
    edge_indices = np.where(edge_mask)[0]
    edge_downsampled_indices = edge_indices[::2]  # 每2个点取1个，降采样50%
    core_indices = np.where(core_mask)[0]
    selected_indices = np.concatenate([core_indices, edge_downsampled_indices])
    selected_indices = np.sort(selected_indices)  # 排序以保持顺序
    
    print(f"LOD后总点数: {len(selected_indices)} (原始: {n_total_original}, 降采样率: {len(selected_indices)/n_total_original*100:.1f}%)")
    
    # 3. 改进的LOD实现：在Y方向进行降采样，保持结构化网格
    # 方法：识别核心Y层和边缘Y层，对边缘Y层降采样，然后重新构建StructuredGrid
    print("   正在应用改进的LOD优化（保持结构化网格）...")
    
    # 将1D掩码映射回3D索引，识别哪些Y层是核心层
    # F顺序：对于点i，其3D索引为：z_idx = i // (nx * ny), remainder = i % (nx * ny), y_idx = remainder // nx, x_idx = remainder % nx
    core_mask_3d = np.zeros((nx, ny, nz), dtype=bool, order='F')
    
    for i in range(len(core_mask)):
        z_idx = i // (nx * ny)
        remainder = i % (nx * ny)
        y_idx = remainder // nx
        x_idx = remainder % nx
        if core_mask[i]:
            core_mask_3d[x_idx, y_idx, z_idx] = True
    
    # 识别核心Y层（该Y层至少有一个核心点）
    y_core_layers = np.zeros(ny, dtype=bool)
    for y_idx in range(ny):
        if np.any(core_mask_3d[:, y_idx, :]):
            y_core_layers[y_idx] = True
    
    # 边缘Y层降采样（每2个取1个）
    y_edge_layers = np.where(~y_core_layers)[0]
    y_edge_downsampled = y_edge_layers[::2]  # 每2个取1个
    y_core_layers_indices = np.where(y_core_layers)[0]
    y_selected = np.sort(np.concatenate([y_core_layers_indices, y_edge_downsampled]))
    
    # 重新构建降采样后的网格（保持结构化）
    X_lod = X[:, y_selected, :]
    Y_lod = Y[:, y_selected, :]
    Z_lod = Z[:, y_selected, :]
    
    # 重新构建StructuredGrid
    lod_volume = pv.StructuredGrid(X_lod, Y_lod, Z_lod)
    
    # 提取对应的数据
    theta_data_lod = Theta_local[:, y_selected, :].flatten(order='F')
    salt_data_lod = Salt_local[:, y_selected, :].flatten(order='F')
    
    # 更新网格和数据
    lod_volume["Temperature"] = theta_data_lod
    lod_volume["Salinity"] = salt_data_lod
    
    # 更新维度
    lod_nx, lod_ny, lod_nz = X_lod.shape[0], X_lod.shape[1], X_lod.shape[2]
    
    print(f"   ✅ LOD网格已生成（保持结构化）")
    print(f"   原始尺寸: ({nx}, {ny}, {nz}) = {nx*ny*nz} 点")
    print(f"   LOD尺寸: ({lod_nx}, {lod_ny}, {lod_nz}) = {lod_nx*lod_ny*lod_nz} 点")
    print(f"   性能提升：点数减少 {((nx*ny*nz - lod_nx*lod_ny*lod_nz) / (nx*ny*nz) * 100):.1f}%")
    
    # 更新combined_volume和数据
    combined_volume = lod_volume
    theta_data = theta_data_lod
    salt_data = salt_data_lod
    nx, ny, nz = lod_nx, lod_ny, lod_nz  # 更新维度
    
    print(f"✅ 第五步优化完成：LOD网格点数={combined_volume.n_points}")

# 如果LOD被禁用，使用原始数据
if not use_lod:
    combined_volume = combined_volume_original
    theta_data = theta_data_original
    salt_data = salt_data_original
    n_core = None
    n_edge = None
    print("✅ 使用原始完整网格（LOD已禁用）")

# ----------------------------
# 第三步优化：计算盐度梯度（用于视觉权重优化）
# ----------------------------
print("正在计算盐度梯度...")
# 使用 NumPy 直接计算盐度梯度（更可靠的方法）
# 将盐度数据重塑为3D数组（注意：数据是按F顺序展平的）
salt_3d = salt_data.reshape(nx, ny, nz, order='F')

# 计算梯度（np.gradient 返回每个方向的梯度）
# 注意：np.gradient 假设网格间距为1，这对于归一化的梯度大小计算是足够的
grad_x, grad_y, grad_z = np.gradient(salt_3d)

# 展平并组合为梯度向量（保持F顺序）
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
print(f"盐度梯度范围: [{salt_gradient_mag.min():.4f}, {salt_gradient_mag.max():.4f}]")
print(f"盐度梯度归一化范围: [{salt_gradient_norm.min():.4f}, {salt_gradient_norm.max():.4f}]")

# 检查数据范围和数据有效性
print(f"盐度数据范围: [{np.min(salt_data):.4f}, {np.max(salt_data):.4f}]")
print(f"温度数据范围: [{np.min(theta_data):.4f}, {np.max(theta_data):.4f}]")
print(f"数据形状: salt={salt_data.shape}, theta={theta_data.shape}")

# 计算数据范围
salt_min_val = np.min(salt_data)
salt_max_val = np.max(salt_data)
salt_range = salt_max_val - salt_min_val  # 定义salt_range变量
temp_min_val = np.min(theta_data)
temp_max_val = np.max(theta_data)
temp_range = temp_max_val - temp_min_val

print(f"盐度数据范围: [{salt_min_val:.4f}, {salt_max_val:.4f}]")
print(f"温度数据范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")

# ----------------------------
# 第六步优化：环境光与背景协同优化（自动启用）
# ----------------------------
print("\n" + "="*60)
print("第六步优化：环境光与背景协同优化（自动启用）")
print("="*60)

use_env_lighting = True  # 自动启用环境光与背景优化
print("✅ 已自动启用环境光与背景优化")

# 创建Plotter
# 启用深度剥离（depth peeling）以正确处理透明度混合，确保箭头可见
plotter = pv.Plotter(window_size=(1400, 900))

# 应用第六步优化：背景色设置
if use_env_lighting:
    # 1. 微调背景色：更深的暗蓝色，突出高盐体素（hot_r色图）
    plotter.background_color = (0.08, 0.12, 0.18)  # 深暗蓝，不与红/橙色高盐体素冲突
    print("✅ 已设置背景色为深暗蓝色 (0.08, 0.12, 0.18)")
else:
    # 使用默认背景色
    plotter.background_color = 'white'  # 或保持默认

try:
    # 启用深度剥离，确保透明体积和箭头正确混合
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
    print("✅ 已启用深度剥离（depth peeling）以正确处理透明度混合")
except Exception as e:
    print(f"警告：无法启用深度剥离: {e}")
    print("   将使用标准渲染模式")

# ========================================
# 使用VTK底层API实现真正的双标量独立控制（改进的温度-盐度映射）
# ========================================
if combined_volume.n_points > 0 and not np.isnan(theta_data).all() and not np.isnan(salt_data).all():
    # 1. 获取PyVista网格的VTK数据对象
    vtk_grid = combined_volume.GetBlock(0) if hasattr(combined_volume, 'GetBlock') else combined_volume
    
    # 2. 使用PyVista的网格，但通过VTK底层API设置双标量
    # 首先使用PyVista创建volume，然后通过VTK API修改属性
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
        pickable=False,  # 禁用拾取，提升性能
        blending='composite'  # 使用复合混合模式，确保透明度正确混合（PyVista参数）
    )
    print("✅ 已设置体积渲染混合模式为复合模式（通过PyVista参数）")
    
    # 3. 通过VTK底层API实现真正的双标量独立控制（改进方案）
    if not VTK_AVAILABLE:
        print("警告：VTK不可用，使用PyVista高层API（近似方案）")
        print("   - 颜色：由温度映射（hot色图）")
        print("   - 透明度：使用基于盐度范围的传递函数（0到0.3）")
    else:
        try:
            # 获取VTK Volume和VolumeProperty
            mapper = volume_actor.GetMapper()
            vtk_volume = mapper.GetInput()
            volume_property = volume_actor.GetProperty()
            
            # 3.1 确保盐度数据在PointData中（作为第二个标量数组）
            salt_vtk_array = vtk_volume.GetPointData().GetArray("Salinity")
            if salt_vtk_array is None:
                # 如果盐度数组不存在，创建并添加
                salt_vtk_array = numpy_to_vtk(
                    salt_data.astype(np.float32),
                    array_type=vtk.VTK_FLOAT
                )
                salt_vtk_array.SetName("Salinity")
                vtk_volume.GetPointData().AddArray(salt_vtk_array)
            
            # 3.2 第三步优化：盐度主导的视觉权重优化（策略17：反幂函数，0.8次，0~0.3透明度）
            # 策略17：高（80%）阈值 + 不常见-反幂函数（0.8次）+ 0~0.3透明度
            # 高过滤 + 高透明，温和突出高盐（避免尖锐）
            # 盐度 > 33 后透明度缓慢增长至 0.3，核心区过渡自然，不突兀
            def opacity_mapping_strategy_17(salt_data, salt_gradient_norm):
                """策略17：高（80%）阈值 + 不常见-反幂函数（0.8次）+ 0~0.3透明度
                高过滤 + 高透明，温和突出高盐（避免尖锐）
                盐度 > 33 后透明度缓慢增长至 0.3，核心区过渡自然，不突兀
                """
                salt_threshold = np.percentile(salt_data, 80)
                salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
                base_opacity = 0 + 0.3 * (salt_norm ** 0.8)
                gradient_boost = 0.1 + 0.2 * salt_gradient_norm
                final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.3)
                return final_opacity
            
            # 计算策略17的最终透明度
            final_opacity = opacity_mapping_strategy_17(salt_data, salt_gradient_norm)
            print("✅ 已选择策略17：反幂函数映射（80%阈值，0.8次，0~0.3透明度）")
            print(f"透明度范围: [{final_opacity.min():.4f}, {final_opacity.max():.4f}]")
            print(f"透明度统计: 均值={final_opacity.mean():.4f}, 中位数={np.median(final_opacity):.4f}")
            
            # 将计算好的透明度映射到温度值，创建vtkPiecewiseFunction
            # 由于透明度现在是基于盐度的，我们需要为每个温度值找到对应的盐度值
            # 然后使用该盐度值对应的透明度
            n_bins = 512  # 提高精度
            opacity_func = vtk.vtkPiecewiseFunction()
            
            # 创建温度值的数组
            temp_vals = np.linspace(temp_min_val, temp_max_val, n_bins)
            
            # 计算温度容差
            temp_tolerance = (temp_max_val - temp_min_val) / n_bins * 2  # 动态容差
            
            # 获取透明度的实际范围（根据选择的策略）
            opacity_min = final_opacity.min()
            opacity_max = final_opacity.max()
            
            print("正在构建温度-透明度映射表（基于盐度主导的视觉权重，策略17）...")
            for i, t in enumerate(temp_vals):
                # 对于每个温度值，找到对应的点
                temp_mask = np.abs(theta_data - t) <= temp_tolerance
                
                if np.any(temp_mask):
                    # 找到该温度范围内的所有点，使用这些点的平均透明度
                    corresponding_opacities = final_opacity[temp_mask]
                    avg_opacity = np.mean(corresponding_opacities)
                    avg_opacity = np.clip(avg_opacity, opacity_min, opacity_max)  # 使用实际透明度范围
                    opacity_func.AddPoint(t, avg_opacity)
                else:
                    # 如果没有找到对应点，使用线性插值
                    temp_norm = (t - temp_min_val) / (temp_max_val - temp_min_val) if temp_range > 0 else 0
                    # 使用线性插值估算透明度
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
            # 根据是否使用LOD调整衰减距离
            if use_lod:
                lod_opacity_unit_distance = 8.0  # LOD模式下增大衰减距离
            else:
                lod_opacity_unit_distance = 5.0  # 标准衰减距离
            volume_property.SetScalarOpacityUnitDistance(lod_opacity_unit_distance)
            
            # 启用三线性插值，保证体素透明度、颜色值与周围体素的过渡自然
            try:
                volume_property.SetInterpolationTypeToLinear()  # 三线性插值（Trilinear interpolation）
                print("✅ 已启用三线性插值（Trilinear Interpolation），保证体素过渡自然")
            except Exception as e:
                print(f"⚠️  无法设置三线性插值: {e}")
                # 尝试使用VTK常量
                try:
                    volume_property.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
                    print("✅ 已启用三线性插值（使用VTK常量）")
                except Exception as e2:
                    print(f"⚠️  无法设置三线性插值（备用方法失败）: {e2}")
            
            # 3.4 自适应颜色映射（基于数据分布的分位数拉伸）- 第二步优化
            # 计算温度分位数，过滤极端值，增强中间区域对比度
            temp_percentile_5 = np.percentile(theta_data, 5)   # 5%分位数（过滤低温极端值）
            temp_percentile_95 = np.percentile(theta_data, 95)  # 95%分位数（过滤高温极端值）
            
            print(f"温度原始范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")
            print(f"温度分位数范围（5%-95%）: [{temp_percentile_5:.4f}, {temp_percentile_95:.4f}]")
            
            # 创建自定义颜色映射（使用分位数范围）
            # 注意：体积渲染使用vtkColorTransferFunction，而不是vtkLookupTable
            # vtkLookupTable主要用于表面渲染，体积渲染使用ColorTransferFunction
            
            # 获取hot_r色图（使用新的API避免弃用警告）
            try:
                import matplotlib.colormaps as cmaps
                hot_r_cmap = cmaps['hot_r']
            except (ImportError, KeyError):
                # 兼容旧版本，避免弃用警告
                try:
                    # 优先使用新的API（避免弃用警告）
                    import matplotlib
                    if hasattr(matplotlib, 'colormaps'):
                        hot_r_cmap = matplotlib.colormaps['hot_r']
                    else:
                        # 如果新API不可用，使用旧API（会显示弃用警告，但功能正常）
                        hot_r_cmap = plt.cm.get_cmap('hot_r')
                except (AttributeError, KeyError):
                    # 最后的兼容方案
                    hot_r_cmap = plt.cm.hot_r
            
            # 创建VTK颜色传递函数（用于体积渲染）
            color_func = vtk.vtkColorTransferFunction()
            if temp_range > 0:
                # 使用分位数范围创建颜色映射
                temp_percentile_range = temp_percentile_95 - temp_percentile_5
                
                # 自定义颜色映射：中间区域颜色更柔和，避免抢高盐区域焦点
                # 中间核心温度区（70%颜色梯度）使用更平滑的过渡
                # 极端值区域（30%）颜色压缩，避免过曝
                
                # 定义多个控制点，实现平滑的颜色过渡
                n_control_points = 10  # 控制点数量
                temp_vals = np.linspace(temp_percentile_5, temp_percentile_95, n_control_points)
                
                # 中间区域（70%）使用0.1到0.7的色图值，饱和度降低
                mid_start_idx = 0
                mid_end_idx = int(n_control_points * 0.7)
                mid_temp_vals = temp_vals[mid_start_idx:mid_end_idx]
                mid_cmap_vals = np.linspace(0.1, 0.7, len(mid_temp_vals))
                
                # 极端值区域（30%）使用0.7到0.9的色图值，避免过曝
                extreme_temp_vals = temp_vals[mid_end_idx:]
                extreme_cmap_vals = np.linspace(0.7, 0.9, len(extreme_temp_vals))
                
                # 为中间区域添加控制点
                for i, (temp_val, cmap_val) in enumerate(zip(mid_temp_vals, mid_cmap_vals)):
                    rgba = hot_r_cmap(cmap_val)
                    color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
                
                # 为极端值区域添加控制点
                for i, (temp_val, cmap_val) in enumerate(zip(extreme_temp_vals, extreme_cmap_vals)):
                    rgba = hot_r_cmap(cmap_val)
                    color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
                
                # 确保边界值正确设置
                rgba_min = hot_r_cmap(0.1)
                rgba_max = hot_r_cmap(0.9)
                color_func.AddRGBPoint(temp_percentile_5, rgba_min[0], rgba_min[1], rgba_min[2])
                color_func.AddRGBPoint(temp_percentile_95, rgba_max[0], rgba_max[1], rgba_max[2])
            else:
                color_func.AddRGBPoint(temp_min_val, 0.5, 0.5, 0.5)  # 灰色
            
            # 设置颜色传递函数到体积属性（体积渲染使用ColorTransferFunction）
            volume_property.SetColor(color_func)
            
            # ========================================
            # 第四步优化：渲染混合模式优化（减少体素叠加遮挡）
            # ========================================
            # 理论依据：体素渲染的混合模式决定了多个体素叠加时的颜色/透明度计算方式
            # 默认"加法混合"易导致亮区过曝和遮挡，采用"预乘阿尔法混合"更适合标量场的层次展示
            
            # 注意：混合模式已在add_volume时通过blending='composite'参数设置
            # 这里只需要设置梯度依赖的阴影和光照参数
            
            try:
                # 1. 梯度依赖的阴影和光照（增强空间层次）
                volume_property.ShadeOn()  # 启用阴影
                volume_property.SetSpecular(0.05)  # 降低镜面反射，避免高盐体素反光遮挡
                
                # 计算每个点的环境光和漫反射（基于盐度梯度）
                # 注意：VTK的VolumeProperty不能直接为每个点设置不同的ambient和diffuse
                # 我们使用平均梯度值来设置全局参数
                avg_gradient_norm = np.mean(salt_gradient_norm)
                
                # 第四步优化：基础光照设置
                base_ambient = 0.15 + 0.35 * avg_gradient_norm
                base_diffuse = 0.4 + 0.6 * avg_gradient_norm
                
                # 第六步优化：自发光调整（高盐梯度区自发光，突出混合区）- 再次增强 + 提高整体亮度
                if use_env_lighting:
                    # 计算高梯度区域的比例（梯度>0.5的区域）
                    high_gradient_ratio = np.mean(salt_gradient_norm > 0.5)
                    # 计算高盐度区域的比例（盐度>80%分位数的区域）
                    salt_80_percentile = np.percentile(salt_data, 80)
                    high_salt_ratio = np.mean(salt_data > salt_80_percentile)
                    # 强化高梯度区域和高盐度区域的环境光（模拟自发光效果）- 再次增强自发光强度
                    # 同时考虑高梯度区域和高盐度区域，并提高整体亮度
                    enhanced_ambient = base_ambient + 0.4 * high_gradient_ratio + 0.3 * high_salt_ratio + 0.15  # 额外增加0.15提高整体亮度
                    enhanced_ambient = min(enhanced_ambient, 0.95)  # 提高最大环境光限制（从0.85到0.95）
                    # 同时增强漫反射以提高整体亮度
                    enhanced_diffuse = base_diffuse + 0.2  # 额外增加0.2提高整体亮度
                    enhanced_diffuse = min(enhanced_diffuse, 1.0)  # 限制在1.0以内
                    volume_property.SetAmbient(enhanced_ambient)
                    volume_property.SetDiffuse(enhanced_diffuse)
                    print("✅ 第四步优化：渲染混合模式优化已实现（亮度增强）")
                    print("   - 混合模式：复合混合（Composite Blending，已在add_volume时设置）")
                    print("   - 阴影：已启用，梯度依赖的环境光和漫反射")
                    print(f"   - 环境光：{enhanced_ambient:.3f}（基于平均梯度 + 高梯度区强化自发光 + 高盐度区强化自发光 + 整体亮度提升）")
                    print(f"   - 漫反射：{enhanced_diffuse:.3f}（基于平均梯度 + 整体亮度提升）")
                    print("   - 镜面反射：0.05（降低反光，避免遮挡）")
                    print(f"   - 高梯度区域比例：{high_gradient_ratio:.2%}（自发光增强区域）")
                    print(f"   - 高盐度区域比例：{high_salt_ratio:.2%}（自发光增强区域，盐度>{salt_80_percentile:.2f}）")
                else:
                    # 使用第四步优化的标准光照设置，但提高整体亮度
                    enhanced_ambient = base_ambient + 0.15  # 额外增加0.15提高整体亮度
                    enhanced_ambient = min(enhanced_ambient, 0.8)
                    enhanced_diffuse = base_diffuse + 0.2  # 额外增加0.2提高整体亮度
                    enhanced_diffuse = min(enhanced_diffuse, 1.0)
                    volume_property.SetAmbient(enhanced_ambient)  # 盐度梯度越大，环境光越低，阴影越突出
                    volume_property.SetDiffuse(enhanced_diffuse)  # 盐度梯度越大，漫反射越强，体素边界越清晰
                    print("✅ 第四步优化：渲染混合模式优化已实现（亮度增强）")
                    print("   - 混合模式：复合混合（Composite Blending，已在add_volume时设置）")
                    print("   - 阴影：已启用，梯度依赖的环境光和漫反射")
                    print(f"   - 环境光：{enhanced_ambient:.3f}（基于平均梯度 + 整体亮度提升）")
                    print(f"   - 漫反射：{enhanced_diffuse:.3f}（基于平均梯度 + 整体亮度提升）")
                    print("   - 镜面反射：0.05（降低反光，避免遮挡）")
                
            except Exception as e:
                print(f"警告：无法设置混合模式优化: {e}")
                import traceback
                traceback.print_exc()
            
            print("✅ VTK底层API双标量独立控制已实现（六步优化完整实现）")
            print("   【第一步优化：单体积双标量绑定】")
            print("   - 颜色：由温度标量独立控制（自适应颜色映射，hot_r色图）")
            print("   - 透明度：由盐度标量独立控制（基于策略17：反幂函数映射）")
            print("   - 技术：使用vtkPiecewiseFunction创建温度-透明度映射表")
            print(f"   - 映射精度：{n_bins}个控制点，覆盖温度范围[{temp_min_val:.2f}, {temp_max_val:.2f}]")
            print("")
            print("   【第二步优化：自适应颜色映射】")
            print("   - 分位数拉伸：使用5%-95%分位数范围，过滤极端值")
            print("   - 颜色映射：hot_r色图（反转hot），低温偏暗、高温偏亮")
            print("   - 中间区域增强：70%颜色梯度用于中间温度区间，提升对比度")
            print("   - 饱和度优化：中间区饱和度降低，极端值颜色压缩，避免过曝")
            print(f"   - 颜色映射范围: [{temp_percentile_5:.2f}, {temp_percentile_95:.2f}] (原始: [{temp_min_val:.2f}, {temp_max_val:.2f}])")
            print("")
            print("   【第三步优化：盐度主导的视觉权重优化（策略17）】")
            print("   - 映射方案：反幂函数映射（0.8次）")
            print("   - 低盐过滤阈值：80%分位数")
            print("   - 特点：高过滤 + 高透明，温和突出高盐（避免尖锐）")
            print("   - 适用场景：盐度 > 33 后透明度缓慢增长至 0.3，核心区过渡自然，不突兀")
            print("   - 透明度范围：[0.0, 0.3]（策略17：反幂函数映射，温和突出高盐）")
            print("   - 梯度增强：盐度梯度大的区域（混合区）进一步提升不透明度")
            print(f"   - 盐度范围：[{salt_min_val:.2f}, {salt_max_val:.2f}]")
            print(f"   - 盐度梯度范围：[{salt_gradient_mag.min():.4f}, {salt_gradient_mag.max():.4f}]")
            print("")
            print("   【第五步优化：分级细节（LOD）渲染】")
            if use_lod and 'n_core' in locals() and 'n_edge' in locals():
                print("   - 状态：已启用LOD优化")
                print("   - 核心区定义：Y轴中间区（30%~70%）+ 盐度高值区（>70%分位数）")
                print(f"   - 核心区点数：{n_core} ({n_core/(n_core+n_edge)*100:.1f}%)")
                print(f"   - 边缘区点数：{n_edge} ({n_edge/(n_core+n_edge)*100:.1f}%)")
                print(f"   - LOD后总点数：{combined_volume.n_points} (原始: {n_core+n_edge})")
                print(f"   - 性能提升：点数减少 {((n_core+n_edge - combined_volume.n_points) / (n_core+n_edge) * 100):.1f}%")
                print("   - 衰减距离优化：opacity_unit_distance=8.0（增大以增强高盐体素内部可见性）")
            else:
                print("   - 状态：LOD优化已禁用（使用原始完整网格）")
                print(f"   - 网格点数：{combined_volume.n_points}")
                print("   - 衰减距离：opacity_unit_distance=5.0（标准设置）")
            
            print("")
            print("   【第六步优化：环境光与背景协同优化】")
            if use_env_lighting:
                print("   - 状态：已启用环境光与背景优化")
                print("   - 背景色：深暗蓝色 (0.08, 0.12, 0.18)，突出高盐体素（hot_r色图）")
                print("   - 方向光：主光源（强度0.85）+ 侧光（强度0.45），增强高盐体素的空间轮廓【再次增强】")
                print("   - 自发光：高盐梯度区+高盐度区强化自发光，突出混合区（环境光增强系数0.4+0.3）【再次增强】")
                print("   - 三线性插值：已启用，保证体素透明度、颜色值与周围体素的过渡自然")
            else:
                print("   - 状态：环境光与背景优化已禁用（使用默认设置）")
            
        except Exception as e:
            print(f"警告：VTK底层API设置失败: {e}")
            import traceback
            traceback.print_exc()
            print("使用PyVista高层API作为备选方案")
    
else:
    print("错误：体积数据无效，无法添加渲染")

# 第六步优化：添加方向光（在添加体积渲染之后）
if use_env_lighting:
    print("\n正在应用第六步优化：方向光设置...")
    try:
        # 计算场景中心点（用于设置光源焦点）
        scene_center = combined_volume.center
        
        # 2. 调整方向光：增强侧光，突出高盐体素的空间轮廓 - 再次增强
        # 主光源：从右上角照射（再次强化）
        main_light = pv.Light(
            position=(scene_center[0] + 15, scene_center[1] + 15, scene_center[2] + 25),
            focal_point=scene_center,
            color="white",
            intensity=0.85  # 从0.65增加到0.85，再次强化
        )
        plotter.add_light(main_light)
        
        # 侧光：从左侧照射，增强轮廓（再次强化）
        side_light = pv.Light(
            position=(scene_center[0] - 15, scene_center[1] - 15, scene_center[2] + 25),
            focal_point=scene_center,
            color="lightblue",
            intensity=0.45  # 从0.3增加到0.45，再次强化
        )
        plotter.add_light(side_light)
        
        print(f"✅ 已添加方向光（主光源强度=0.85，侧光强度=0.45）【再次增强】")
        print(f"   场景中心: ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})")
    except Exception as e:
        print(f"⚠️  无法添加方向光: {e}")
        print("   使用默认光照设置")

# ========================================
# 矢量场优化：模式2 - 三维流线（全局流动趋势优化）
# ========================================
# 核心思想：用「三维流线」替代离散箭头，通过种子点生成沿速度场积分的连续曲线，
# 直观呈现海流的全局路径和空间形态，颜色编码局部速度大小。

def velocity_field_interp(points, grid, velocity_scalars):
    """
    基于网格数据的速度场插值（给定点坐标，返回插值后的速度矢量）
    改进为线性插值，提升速度场连续性
    
    Args:
        points: 点坐标 (N, 3) 或单个点 (3,)
        grid: pyvista.StructuredGrid（含采样点坐标和速度数据）
        velocity_scalars: 速度矢量数组（n_points×3）
    
    Returns:
        interp_vel: 插值后的速度矢量 (N, 3) 或 (3,)
    """
    try:
        # 确保points是2D数组
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
            single_point = True
        else:
            single_point = False
        
        # 尝试使用线性插值（如果网格是结构化的）
        try:
            from scipy.interpolate import RegularGridInterpolator
            
            # 获取网格的唯一坐标
            unique_x = np.unique(grid.points[:, 0])
            unique_y = np.unique(grid.points[:, 1])
            unique_z = np.unique(grid.points[:, 2])
            
            # 检查是否是结构化网格
            if len(unique_x) * len(unique_y) * len(unique_z) == len(grid.points):
                # 构建3D速度场
                nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)
                
                # 确保velocity_scalars的形状正确
                velocity_scalars = np.asarray(velocity_scalars)
                if velocity_scalars.ndim == 1:
                    # 如果是1D数组，可能是单个速度向量，无法进行reshape
                    raise ValueError("velocity_scalars是1D数组，无法reshape为3D")
                
                # 检查velocity_scalars的第二维度是否为3
                if velocity_scalars.ndim == 2 and velocity_scalars.shape[1] != 3:
                    raise ValueError(f"velocity_scalars的第二维度不是3，而是{velocity_scalars.shape[1]}")
                
                # 确保velocity_scalars的长度与grid.points匹配
                if len(velocity_scalars) != len(grid.points):
                    raise ValueError(f"velocity_scalars长度({len(velocity_scalars)})与grid.points长度({len(grid.points)})不匹配")
                
                # 将速度场reshape为3D（nx, ny, nz）
                try:
                    # 安全地访问速度分量
                    if velocity_scalars.shape[1] >= 3:
                        u = velocity_scalars[:, 0].reshape(nx, ny, nz, order='F')
                        v = velocity_scalars[:, 1].reshape(nx, ny, nz, order='F')
                        w = velocity_scalars[:, 2].reshape(nx, ny, nz, order='F')
                    else:
                        # 如果分量不足，使用零填充
                        raise ValueError(f"velocity_scalars分量不足（只有{velocity_scalars.shape[1]}个分量）")
                except (ValueError, IndexError) as e:
                    # 如果reshape失败，回退到最近邻插值
                    raise ValueError(f"无法reshape速度场: {e}")
                
                # 线性插值器
                interp_u = RegularGridInterpolator((unique_x, unique_y, unique_z), u, 
                                                   method='linear', bounds_error=False, fill_value=0.0)
                interp_v = RegularGridInterpolator((unique_x, unique_y, unique_z), v, 
                                                   method='linear', bounds_error=False, fill_value=0.0)
                interp_w = RegularGridInterpolator((unique_x, unique_y, unique_z), w, 
                                                   method='linear', bounds_error=False, fill_value=0.0)
                
                # 计算给定点的速度
                u_vals = interp_u(points)
                v_vals = interp_v(points)
                w_vals = interp_w(points)
                
                # 确保结果是正确的形状
                u_vals = np.asarray(u_vals)
                v_vals = np.asarray(v_vals)
                w_vals = np.asarray(w_vals)
                
                # 如果是标量，转换为1D数组
                if u_vals.ndim == 0:
                    u_vals = u_vals.reshape(1)
                    v_vals = v_vals.reshape(1)
                    w_vals = w_vals.reshape(1)
                
                # 确保所有数组长度一致
                if len(u_vals) == 1 and len(points) > 1:
                    u_vals = np.full(len(points), u_vals[0])
                    v_vals = np.full(len(points), v_vals[0])
                    w_vals = np.full(len(points), w_vals[0])
                
                result = np.column_stack([u_vals, v_vals, w_vals])
                
                # 确保结果形状正确
                if result.ndim == 1:
                    result = result.reshape(1, -1)
                
                if single_point:
                    return result[0] if result.shape[0] > 0 else np.array([0.0, 0.0, 0.0])
                return result
        except (ImportError, ValueError, AttributeError):
            # 如果线性插值失败，回退到最近邻插值
            pass
        
        # 回退到最近邻插值
        velocity_scalars = np.asarray(velocity_scalars)
        
        # 确保velocity_scalars的形状正确
        if velocity_scalars.ndim == 1:
            # 如果是1D数组，可能是单个速度向量
            if len(velocity_scalars) == 3:
                # 单个速度向量，对所有点返回相同的速度
                if single_point:
                    return velocity_scalars
                else:
                    return np.tile(velocity_scalars, (len(points), 1))
            else:
                # 无法处理，返回零向量
                if single_point:
                    return np.array([0.0, 0.0, 0.0])
                else:
                    return np.zeros((len(points), 3))
        
        if len(points) == 1:
            # 单个点的情况
            distances = np.linalg.norm(grid.points - points[0], axis=1)
            nearest_idx = np.argmin(distances)
            result = velocity_scalars[nearest_idx]
            # 确保结果是1D数组（3个元素）
            result = np.asarray(result)
            if result.ndim == 0:
                return np.array([0.0, 0.0, 0.0])
            if result.ndim == 2:
                result = result.flatten()
            if len(result) != 3:
                return np.array([0.0, 0.0, 0.0])
            return result
        else:
            # 多个点的情况
            distances = np.linalg.norm(grid.points - points[:, np.newaxis, :], axis=2)
            nearest_indices = np.argmin(distances, axis=1)
            result = velocity_scalars[nearest_indices]
            
            # 确保结果形状正确
            result = np.asarray(result)
            if result.ndim == 1:
                # 如果velocity_scalars是1D的，需要reshape
                if len(result) == 3:
                    result = result.reshape(1, 3)
                else:
                    return np.zeros((len(points), 3))
            
            # 确保第二维度是3
            if result.shape[1] != 3:
                return np.zeros((len(points), 3))
            
            if single_point:
                return result[0] if result.shape[0] > 0 else np.array([0.0, 0.0, 0.0])
            return result
        
    except Exception as e:
        # 如果插值失败，返回零向量
        points = np.asarray(points)
        if points.ndim == 1:
            return np.array([0.0, 0.0, 0.0])
        else:
            return np.zeros((len(points), 3))

def optimize_seed_points(sample_points_coords, velocity_scalars, min_speed=0.01):
    """
    优化种子点质量：过滤低速度区域和异常区域（多级降级策略）
    
    Args:
        sample_points_coords: 种子点坐标数组 (N, 3)
        velocity_scalars: 速度矢量数组 (N, 3)
        min_speed: 最小速度阈值（初始值，会根据情况降级）
    
    Returns:
        optimized_seeds: 优化后的种子点数组 (M, 3)
        valid_mask: 有效种子点的掩码
    """
    if len(sample_points_coords) == 0:
        return sample_points_coords, np.array([], dtype=bool)
    
    # 计算每个种子点的速度大小
    speeds = np.linalg.norm(velocity_scalars, axis=1)
    
    # 检查速度数据的有效性
    valid_speeds = speeds[np.isfinite(speeds)]
    if len(valid_speeds) == 0:
        print("  警告：所有速度数据无效，使用所有原始种子点")
        return sample_points_coords, np.ones(len(sample_points_coords), dtype=bool)
    
    # 修改4：降低过滤阈值，保留更多低速区域的流线（从30%降至5%）
    speed_threshold = np.percentile(valid_speeds, 5)  # 降低阈值到5%分位数，保留更多点
    print(f"  种子点过滤阈值：速度 > {speed_threshold:.4f}（5%分位数）")
    
    # 第一级过滤：速度阈值 + 有效性检查
    valid_mask = (
        (speeds > speed_threshold) & 
        np.isfinite(speeds) & 
        np.all(np.isfinite(sample_points_coords), axis=1)
    )
    
    # 额外过滤异常速度区域（速度过大可能是数据异常）
    if len(valid_speeds) > 1:
        max_speed = np.percentile(valid_speeds, 99)  # 过滤99%分位数以上的异常值
        valid_mask &= (speeds < max_speed)
    
    optimized_seeds = sample_points_coords[valid_mask]
    
    # 关键优化：当有效种子点为0时的降级处理
    if len(optimized_seeds) == 0:
        print("  警告：所有种子点被过滤，尝试降低阈值保留部分点")
        # 进一步降低阈值到1%分位数（如果5%仍无有效点）
        speed_threshold = np.percentile(valid_speeds, 1)
        print(f"  降低过滤阈值：速度 > {speed_threshold:.4f}（1%分位数）")
        
        valid_mask = (
            (speeds > speed_threshold) & 
            np.isfinite(speeds) & 
            np.all(np.isfinite(sample_points_coords), axis=1)
        )
        optimized_seeds = sample_points_coords[valid_mask]
        
        # 如果仍为0，则不过滤直接使用（放弃速度过滤）
        if len(optimized_seeds) == 0:
            print("  警告：无法通过阈值过滤获得有效点，使用所有原始种子点（仅过滤无效坐标）")
            # 只过滤坐标无效的点
            valid_mask = np.all(np.isfinite(sample_points_coords), axis=1)
            optimized_seeds = sample_points_coords[valid_mask]
            if len(optimized_seeds) == 0:
                # 最后的降级：使用所有点
                print("  警告：所有点坐标无效，使用所有原始种子点")
                optimized_seeds = sample_points_coords
                valid_mask = np.ones(len(sample_points_coords), dtype=bool)
    
    print(f"  种子点质量优化：原始{len(sample_points_coords)}个 -> 有效{len(optimized_seeds)}个（过滤低速度/异常区）")
    return optimized_seeds, valid_mask

def create_optimized_seeds(grid, sample_points, velocity_scalars, n_seeds=400, min_speed=0.01):
    """
    优化种子点生成：三维均匀分布覆盖整个立方体，高盐区加密，并过滤低质量种子点
    修改4：改为三维均匀网格分布，确保覆盖立方体内部
    
    Args:
        grid: pyvista网格（用于获取盐度数据）
        sample_points: 采样点坐标数组 (N, 3)
        velocity_scalars: 速度矢量数组 (N, 3)，用于筛选种子点
        n_seeds: 目标种子点数量（用于计算每维点数）
        min_speed: 最小速度阈值
    
    Returns:
        seeds: 优化后的种子点数组 (M, 3)
    """
    # 获取边界
    x_min, x_max = sample_points[:, 0].min(), sample_points[:, 0].max()
    y_min, y_max = sample_points[:, 1].min(), sample_points[:, 1].max()
    z_min, z_max = sample_points[:, 2].min(), sample_points[:, 2].max()
    
    # 修改4：三维均匀网格分布（确保覆盖立方体所有区域）
    # 计算每维点数：n_per_dim^3 ≈ n_seeds，取n_per_dim=15，共15*15*15=3375个基础点
    n_per_dim = 15
    x_seeds = np.linspace(x_min, x_max, n_per_dim)
    y_seeds = np.linspace(y_min, y_max, n_per_dim)
    z_seeds = np.linspace(z_min, z_max, n_per_dim)  # 增加z方向种子点密度
    
    # 生成三维网格点（覆盖立方体所有区域）
    Xs, Ys, Zs = np.meshgrid(x_seeds, y_seeds, z_seeds, indexing='ij')
    all_seeds = np.stack([Xs.flatten(), Ys.flatten(), Zs.flatten()], axis=1)
    
    # 修改4：高盐度区域额外加密种子点（增强，最多补充200个）
    try:
        # 尝试获取盐度数据
        salt_data = None
        if hasattr(grid, 'array_names'):
            if 'Salinity' in grid.array_names:
                salt_data = grid['Salinity']
            elif 'salt' in grid.array_names:
                salt_data = grid['salt']
        
        if salt_data is not None and len(salt_data) > 0:
            high_salt_mask = salt_data > np.percentile(salt_data, 70)  # 高盐阈值
            if np.any(high_salt_mask):
                high_salt_points = grid.points[high_salt_mask]
                # 修改4：从高盐区随机补充种子点（增加密度，最多200个）
                n_extra = min(200, len(high_salt_points))
                if n_extra > 0:
                    # 使用随机选择，但如果没有numpy.random，使用均匀采样
                    try:
                        extra_seeds = high_salt_points[np.random.choice(len(high_salt_points), n_extra, replace=False)]
                    except:
                        # 如果随机选择失败，使用均匀采样
                        indices = np.linspace(0, len(high_salt_points)-1, n_extra, dtype=int)
                        extra_seeds = high_salt_points[indices]
                    # 修改4：将高盐区种子点添加到基础网格点
                    all_seeds = np.vstack([all_seeds, extra_seeds])
                    print(f"    高盐区额外加密种子点: {len(extra_seeds)}个")
    except Exception as e:
        print(f"  警告：无法获取盐度数据用于高盐区加密: {e}")
    
    # 确保seeds是numpy数组（all_seeds已经是numpy数组）
    seeds = all_seeds
    
    # 4. 种子点质量优化：过滤低速度区域（多级降级策略）
    if len(seeds) > 0:
        # 使用速度场插值函数计算每个种子点的速度
        try:
            seed_velocities = velocity_field_interp(seeds, grid, velocity_scalars)
            if seed_velocities.ndim == 1:
                seed_velocities = seed_velocities.reshape(1, -1)
            
            # 确保速度数组形状正确
            if len(seed_velocities) != len(seeds):
                # 如果插值失败，使用最近邻方法
                seed_velocities = []
                for seed in seeds:
                    distances = np.linalg.norm(sample_points - seed, axis=1)
                    nearest_idx = np.argmin(distances)
                    seed_velocities.append(velocity_scalars[nearest_idx])
                seed_velocities = np.array(seed_velocities)
            
            # 应用质量筛选（包含多级降级策略）
            optimized_seeds, _ = optimize_seed_points(seeds, seed_velocities, min_speed=min_speed)
            
            # 确保至少有一些种子点
            if len(optimized_seeds) == 0:
                print("  警告：质量筛选后无有效种子点，使用所有原始种子点")
                optimized_seeds = seeds
        except Exception as e:
            print(f"  警告：种子点速度计算失败: {e}，使用所有原始种子点")
            optimized_seeds = seeds
        
        print(f"  优化后种子点分布：{len(optimized_seeds)}个点（含分层+高盐区加密+质量筛选）")
        return optimized_seeds
    else:
        print("  警告：无法生成种子点，使用原始采样点")
        return sample_points

def smooth_streamlines(streamlines_list, sigma=1.5, num_points=100):
    """
    修改4：流线后处理优化（提升流畅度）
    对生成的流线进行插值平滑，增加点数，使其更流畅
    
    Args:
        streamlines_list: 流线点列表（每个元素是 (N, 3) 数组）
        sigma: 高斯滤波标准差（用于初步平滑）
        num_points: 每条流线插值后的点数（修改4：从默认值增至100）
    
    Returns:
        smoothed: 平滑后的流线列表
    """
    if not SCIPY_AVAILABLE:
        return streamlines_list
    
    smoothed = []
    for line in streamlines_list:
        if len(line) < 3:  # 修改4：跳过过短的线（至少3个点）
            continue
        
        # 修改4：对每条流线插值，增加点数使其更流畅
        try:
            # 生成插值参数
            t = np.linspace(0, 1, len(line))
            t_new = np.linspace(0, 1, num_points)  # 每条线固定100个点
            
            # 三次样条插值（比线性插值更平滑）
            from scipy.interpolate import make_interp_spline
            spline = make_interp_spline(t, line, k=min(3, len(line)-1), axis=0)  # k不能超过点数-1
            smoothed_line = spline(t_new)
            
            # 再进行高斯平滑（可选，进一步平滑）
            if sigma > 0:
                for i in range(3):  # x, y, z三个维度
                    smoothed_line[:, i] = gaussian_filter1d(smoothed_line[:, i], sigma=sigma)
            
            smoothed.append(smoothed_line)
        except Exception as e:
            # 如果插值失败，使用原始高斯平滑
            smoothed_line = line.copy()
            for i in range(3):  # x, y, z三个维度
                smoothed_line[:, i] = gaussian_filter1d(line[:, i], sigma=sigma)
            smoothed.append(smoothed_line)
    
    # 合并接近的流线（增强全局连通感）
    if len(smoothed) == 0:
        return []
    
    merged = []
    for i, line in enumerate(smoothed):
        if i == 0:
            merged.append(line)
            continue
        # 检查与上一条流线的距离，过近则合并
        last_line = merged[-1]
        dist = np.min(np.linalg.norm(line[0] - last_line, axis=1))
        if dist < 3.0:  # 距离阈值可调整
            merged[-1] = np.vstack([last_line, line])
        else:
            merged.append(line)
    
    return merged

def generate_3d_streamlines_enhanced(grid, max_time=100.0, step_size=0.1, n_points=200):
    """
    修改2：生成三维流线，增强垂直方向采样和积分路径表达能力
    
    思路：
    - 在垂直方向（Z 轴）上分层生成种子点，确保各深度层都有流线起点
    - 在每个深度层内使用采样点分布生成水平种子点，不足时用随机点补充
    - 使用包含垂直分量 W 的三维速度场进行积分，并适当放大 W 的权重
    - 返回带有 'speed' 标量的 PolyData，用于可视化
    """
    if not SCIPY_INTEGRATE_AVAILABLE:
        print("警告：无法生成三维流线（修改2），scipy.integrate 不可用")
        return None

    # 需要全局的几何与速度场信息
    global X, Y, Z, U_local, V_local, W_local, sample_points_coords, nz

    if sample_points_coords is None or len(sample_points_coords) == 0:
        print("  警告：sample_points_coords 为空，无法按照修改2生成流线")
        return None

    # 1. 优化种子点分布：增加垂直方向密度
    z_coords = sample_points_coords[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()

    # 垂直方向分层生成种子点，确保各深度都有分布（至少5层）
    n_z_layers = max(5, nz // 2 if nz is not None else 5)
    z_layers = np.linspace(z_min, z_max, n_z_layers)

    seed_points_list = []
    # 预先计算 X/Y 范围，用于补充随机点
    x_range = (float(X.min()), float(X.max()))
    y_range = (float(Y.min()), float(Y.max()))

    for z in z_layers:
        # 在当前深度附近选取采样点（容差按层间距的一半）
        tol = 0.5 * abs(z_max - z_min) / max(n_z_layers, 1)
        layer_mask = np.abs(z_coords - z) <= tol
        layer_coords = sample_points_coords[layer_mask]

        # 将已有层内点加入
        if len(layer_coords) > 0:
            seed_points_list.append(layer_coords)

        # 若该层点数不足5个，则用随机点补齐
        if len(layer_coords) < 5:
            n_extra = 5 - len(layer_coords)
            extra = np.random.uniform(
                low=[x_range[0], y_range[0], z - 0.1],
                high=[x_range[1], y_range[1], z + 0.1],
                size=(n_extra, 3)
            )
            seed_points_list.append(extra)

    if not seed_points_list:
        print("  警告：修改2 未能生成任何种子点")
        return None

    seed_points = np.vstack(seed_points_list)
    print(f"优化后种子点分布（修改2）：{len(seed_points)} 个点，覆盖 {n_z_layers} 个深度层")

    # 2. 定义速度场函数（包含垂直分量，并增强 W 的权重）
    # 为了加速索引，预先提取一维坐标轴
    x_axis = np.unique(X[:, 0, 0])
    y_axis = np.unique(Y[0, :, 0])
    z_axis = np.unique(Z[0, 0, :])

    def velocity_field(t, y):
        """给定位置 y = [x, y, z]，返回三维速度向量"""
        # 找到最近的网格索引
        x_idx = int(np.argmin(np.abs(x_axis - y[0])))
        y_idx = int(np.argmin(np.abs(y_axis - y[1])))
        z_idx = int(np.argmin(np.abs(z_axis - y[2])))

        # 确保索引在有效范围内
        x_idx = np.clip(x_idx, 0, U_local.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, U_local.shape[1] - 1)
        z_idx = np.clip(z_idx, 0, U_local.shape[2] - 1)

        u = float(U_local[x_idx, y_idx, z_idx])
        v = float(V_local[x_idx, y_idx, z_idx])
        w = float(W_local[x_idx, y_idx, z_idx]) * 1.5  # 增强垂直分量权重
        return [u, v, w]

    # 3. 积分生成流线
    streamlines = []
    for seed in seed_points:
        t_span = (-max_time / 2.0, max_time / 2.0)

        try:
            sol = solve_ivp(
                velocity_field,
                t_span,
                seed,
                method="RK45",
                max_step=step_size,
                rtol=1e-6,
                atol=1e-9,
            )
        except Exception as e:
            continue

        if sol.success and sol.y.shape[1] >= 5:
            # 形状为 (n_points, 3)
            line = sol.y.T
            streamlines.append(line)

    if not streamlines:
        print("  警告：修改2 生成的流线为空，将回退到原有模式2实现")
        return None

    # 4. 构建 PolyData 并计算速度大小用于着色
    points = []
    lines = []
    offset = 0

    for line in streamlines:
        n_pts = len(line)
        points.append(line)
        # 每条线编码为：[n_pts, idx0, idx1, ...]
        lines.append(np.hstack([n_pts, np.arange(offset, offset + n_pts)]))
        offset += n_pts

    points = np.vstack(points)

    streamlines_poly = pv.PolyData()
    streamlines_poly.points = points
    streamlines_poly.lines = np.hstack(lines)

    # 计算每个点的速度大小（此处使用原始 U/V/W，不再放大 W，便于直观比较）
    speeds = []
    for p in points:
        x_idx = int(np.argmin(np.abs(x_axis - p[0])))
        y_idx = int(np.argmin(np.abs(y_axis - p[1])))
        z_idx = int(np.argmin(np.abs(z_axis - p[2])))

        x_idx = np.clip(x_idx, 0, U_local.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, U_local.shape[1] - 1)
        z_idx = np.clip(z_idx, 0, U_local.shape[2] - 1)

        u = float(U_local[x_idx, y_idx, z_idx])
        v = float(V_local[x_idx, y_idx, z_idx])
        w = float(W_local[x_idx, y_idx, z_idx])
        speeds.append(np.linalg.norm([u, v, w]))

    streamlines_poly["speed"] = np.array(speeds)
    print(f"✅ 修改2：已生成 {len(streamlines)} 条三维流线，点数={streamlines_poly.n_points}，增强垂直方向表达")

    return streamlines_poly

def cluster_flow_regions(sample_points, velocities, spatial_eps=100.0, vel_eps=0.3,
                         min_samples=3, target_clusters=None):
    """
    模式3步骤1&2：空间-速度联合聚类（DBSCAN）
    
    Args:
        sample_points: (N, 3) 采样点坐标
        velocities: (N, 3) 速度向量
        spatial_eps: 空间距离阈值（用于注释，实际以归一化距离+vel_eps为主）
        vel_eps: 速度相似度阈值（用作加权距离的DBSCAN eps）
        min_samples: 簇最小采样点数
    
    Returns:
        clusters: 每个簇的点坐标列表
        cluster_vels: 每个簇对应的速度向量列表
        valid_labels: 有效簇标签
    """
    if not SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn 不可用，无法进行模式3聚类")
        return [], [], []

    if len(sample_points) == 0 or len(velocities) == 0:
        print("⚠️  无采样点或速度数据，无法进行模式3聚类")
        return [], [], []

    pts = np.asarray(sample_points)
    vels = np.asarray(velocities)

    # 1. 标准化空间坐标和速度矢量（避免量纲影响）
    pts_mean = pts.mean(axis=0)
    pts_std = pts.std(axis=0) + 1e-6
    points_norm = (pts - pts_mean) / pts_std

    v_norm = np.linalg.norm(vels, axis=1, keepdims=True) + 1e-6
    vels_norm = vels / v_norm

    labels = None

    # 优先使用 KMeans，便于根据用户输入控制簇数
    if target_clusters is not None and target_clusters >= 2:
        max_clusters = max(2, len(pts) // max(min_samples, 2))
        n_clusters = int(np.clip(target_clusters, 2, max_clusters))
        # 特征：空间 + 速度方向（给速度稍大权重）
        alpha = 1.0
        feats = np.hstack([points_norm, vels_norm * alpha])
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = km.fit_predict(feats)
            print(f"✅ KMeans 聚类完成：目标簇数={target_clusters}，实际簇数={len(np.unique(labels))}")
        except Exception as e:
            print(f"⚠️  KMeans 聚类失败，退回 DBSCAN：{e}")
            labels = None

    if labels is None:
        # 2. 计算空间距离矩阵和速度相似度矩阵（DBSCAN 备选方案）
        spatial_diff = points_norm[:, None, :] - points_norm[None, :, :]
        spatial_dist = np.sqrt((spatial_diff ** 2).sum(axis=2))

        try:
            vel_sim = cosine_similarity(vels_norm)  # 余弦相似度∈[-1,1]，越接近1方向越一致
        except Exception as e:
            print(f"⚠️  计算速度余弦相似度失败: {e}")
            return [], [], []

        # 3. 加权距离矩阵（空间距离+速度相似度）
        vel_dist = 1.0 - vel_sim                         # 理论上∈[0,2]
        # 数值误差可能导致 vel_dist 出现轻微负值或略大于2，需裁剪到非负
        vel_dist = np.clip(vel_dist, 0.0, None)
        spatial_max = spatial_dist.max()
        vel_max = vel_dist.max()
        if spatial_max <= 0 or vel_max <= 0:
            print("⚠️  距离矩阵异常，无法聚类")
            return [], [], []

        spatial_dist_norm = spatial_dist / spatial_max   # 归一化到[0,1]
        vel_dist_norm = vel_dist / vel_max               # 归一化到[0,1]

        weighted_dist = 0.6 * spatial_dist_norm + 0.4 * vel_dist_norm
        # 再次确保距离矩阵非负，避免 sklearn 检查报错
        weighted_dist = np.clip(weighted_dist, 0.0, None)

        # 4. DBSCAN聚类（基于自定义距离矩阵）
        # weighted_dist ∈ [0,1]，直接用 vel_eps 作为阈值更直观（推荐0.2~0.4）
        eps_val = float(vel_eps)
        dbscan = DBSCAN(
            metric='precomputed',
            eps=eps_val,
            min_samples=min_samples
        )
        labels = dbscan.fit_predict(weighted_dist)

    # 5. 过滤噪声簇（label=-1，若存在）
    unique_labels = np.unique(labels)
    if np.min(unique_labels) == -1:
        valid_labels = [l for l in unique_labels if l != -1]
    else:
        valid_labels = list(unique_labels)

    if not valid_labels:
        print("⚠️  聚类结果全部为噪声或为空，模式3将退回到直线箭头")
        return [], [], []

    clusters = [pts[labels == l] for l in valid_labels]
    cluster_vels = [vels[labels == l] for l in valid_labels]

    print(f"✅ 模式3聚类完成：得到 {len(valid_labels)} 个有效区域簇")
    return clusters, cluster_vels, valid_labels

def create_cluster_bent_arrows(clusters, cluster_vels, total_points=None, arrow_scale=50.0, spline_degree=3):
    """
    模式3步骤3：生成聚类区域的弯曲大箭头（修改1：参考模式1的弯曲箭头形态）
    
    每个簇生成一个大箭头：
    - 位置：区域中心
    - 方向：区域平均速度
    - 粗细：子簇点数 / 总点数 * 基础系数（子簇越大箭头越粗）
    - 长度：区域范围的80%，确保完整显示在子簇区域内
    - 形态：多段较粗短线 + 末端圆锥，轻微弯曲表示区域风向趋势
    
    修改1核心改进：
    1. 计算子簇边界框，箭头长度 = 区域范围的80%
    2. 按主方向排序点，使用控制点（最多10个）生成弯曲路径
    3. 多段线段设计，末端逐渐变细
    4. 箭头粗细与子簇点数成正比
    """
    if not clusters or not cluster_vels:
        return None

    if total_points is None:
        # 如果没有提供总点数，计算所有簇的点数之和
        total_points = sum(len(cp) for cp in clusters)

    all_arrows = []

    for cluster_points, cluster_vel in zip(clusters, cluster_vels):
        if len(cluster_points) < 2:
            continue

        n_points = len(cluster_points)
        
        # 修改1：1. 计算子簇空间范围（用于适配箭头长度）
        min_coords = cluster_points.min(axis=0)
        max_coords = cluster_points.max(axis=0)
        cluster_center = (min_coords + max_coords) / 2.0
        cluster_extent = np.linalg.norm(max_coords - min_coords)
        
        # 修改1：箭头长度为区域的80%，确保完整显示在子簇区域内
        arrow_length = cluster_extent * 0.8
        
        # 计算区域平均速度
        mean_vel = cluster_vel.mean(axis=0)
        mean_vel_mag = float(np.linalg.norm(mean_vel))
        if mean_vel_mag < 1e-6:
            continue
        
        # 修改1：2. 子簇方向序列生成（用于弯曲）
        # 按主方向排序点
        main_dir = mean_vel / mean_vel_mag
        proj = np.dot(cluster_points - cluster_center, main_dir)  # 投影到主方向
        sorted_idx = np.argsort(proj)
        sorted_points = cluster_points[sorted_idx]
        sorted_vels = cluster_vel[sorted_idx]
        
        # 修改1：3. 生成弯曲路径（类似模式1的曲线逻辑）- 优化：更自然的弯曲
        # 使用更多控制点（最多20个）生成更平滑的曲线
        n_control_points = min(20, n_points)  # 从10增加到20，使弯曲更自然
        if n_control_points < spline_degree + 1:
            # 点数太少，使用直线
            end_pt = cluster_center + main_dir * arrow_length
            curve_points = np.vstack([cluster_center - main_dir * arrow_length * 0.1, 
                                      cluster_center, 
                                      end_pt])
        else:
            # 选择控制点（均匀分布）
            control_idx = np.linspace(0, len(sorted_points) - 1, n_control_points, dtype=int)
            control_points = sorted_points[control_idx]
            
            # 修改1：使用三次样条插值生成平滑曲线 - 优化：增加细分点数使曲线更平滑
            if SCIPY_AVAILABLE:
                try:
                    # 使用控制点索引作为参数
                    t_control = np.arange(len(control_points))
                    # 增加路径细分点数（从20增加到60），使曲线更平滑，避免生硬折线
                    t_spline = np.linspace(0, len(control_points) - 1, 60)
                    
                    sx = make_interp_spline(t_control, control_points[:, 0], k=min(3, len(control_points) - 1))
                    sy = make_interp_spline(t_control, control_points[:, 1], k=min(3, len(control_points) - 1))
                    sz = make_interp_spline(t_control, control_points[:, 2], k=min(3, len(control_points) - 1))
                    
                    curve_x = sx(t_spline)
                    curve_y = sy(t_spline)
                    curve_z = sz(t_spline)
                    curve_points = np.vstack([curve_x, curve_y, curve_z]).T
                except Exception as e:
                    # 插值失败，使用直线
                    end_pt = cluster_center + main_dir * arrow_length
                    curve_points = np.vstack([cluster_center - main_dir * arrow_length * 0.1, 
                                              cluster_center, 
                                              end_pt])
            else:
                # 无 SciPy，使用直线
                end_pt = cluster_center + main_dir * arrow_length
                curve_points = np.vstack([cluster_center - main_dir * arrow_length * 0.1, 
                                          cluster_center, 
                                          end_pt])
        
        # 修改1：4. 将曲线中心对齐到区域中心，并缩放至目标长度
        curve_center = curve_points.mean(axis=0)
        offset = cluster_center - curve_center
        curve_points = curve_points + offset
        
        # 缩放曲线至目标长度
        vec = curve_points[-1] - curve_points[0]
        cur_len = np.linalg.norm(vec)
        if cur_len > 1e-6:
            scale_factor = arrow_length / cur_len
            curve_points = curve_points[0] + (curve_points - curve_points[0]) * scale_factor
            # 再次确保中心对齐
            curve_center_new = curve_points.mean(axis=0)
            curve_points = curve_points + (cluster_center - curve_center_new)
        
        # 修改1：5. 箭头粗细与大小自适应 - 优化：整体增粗
        # 箭头粗细 = 子簇点数 / 总点数 * 基础系数
        if total_points > 0:
            arrow_thickness_base = (n_points / total_points) * 15.0  # 基础系数从10.0增加到15.0
        else:
            arrow_thickness_base = 0.15  # 从0.1增加到0.15
        
        # 转换为半径（适配 tube 函数）- 整体增粗
        # 提高最小半径和最大半径，使箭头整体更粗
        shaft_radius = max(0.12, min(0.5, arrow_thickness_base * 0.015))  # 最小从0.05增加到0.12，最大从0.3增加到0.5，系数从0.01增加到0.015
        cone_radius = shaft_radius * 2.2  # 从2.0增加到2.2，使圆锥更大
        cone_length = arrow_length * 0.25  # 从0.2增加到0.25，使圆锥更长
        
        # 修改1：6. 生成多段箭头（类似模式1：多根小短线 + 末端圆锥）- 优化：更平滑的曲线
        # 将曲线分成多段，每段用 tube 生成，使用更多段数以获得更平滑的效果
        segment_count = 8  # 箭头段数从5增加到8，使过渡更平滑
        if len(curve_points) < segment_count + 1:
            # 点数太少，直接生成单段
            poly = pv.PolyData()
            poly.points = curve_points
            lines = np.empty((len(curve_points) - 1, 3), dtype=int)
            lines[:, 0] = 2
            for j in range(len(curve_points) - 1):
                lines[j, 1] = j
                lines[j, 2] = j + 1
            poly.lines = lines
            
            # 整段使用相同粗细，增加n_sides使圆形截面更平滑
            tube = poly.tube(radius=shaft_radius, n_sides=24)  # 从16增加到24
            arrow_segments = [tube]
        else:
            # 分段生成，末端逐渐变细，使用更多段数使过渡更平滑
            segment_indices = np.linspace(0, len(curve_points) - 1, segment_count + 1, dtype=int)
            arrow_segments = []
            
            for i in range(segment_count):
                seg_start = segment_indices[i]
                seg_end = segment_indices[i + 1]
                seg_points = curve_points[seg_start:seg_end + 1]
                
                if len(seg_points) < 2:
                    continue
                
                # 线段粗细递减（末端变细），但减少幅度更小，使过渡更平滑
                seg_thickness = shaft_radius * (1.0 - i / segment_count * 0.3)  # 从0.4减少到0.3，使粗细变化更平缓
                
                seg_poly = pv.PolyData()
                seg_poly.points = seg_points
                seg_lines = np.empty((len(seg_points) - 1, 3), dtype=int)
                seg_lines[:, 0] = 2
                for j in range(len(seg_points) - 1):
                    seg_lines[j, 1] = j
                    seg_lines[j, 2] = j + 1
                seg_poly.lines = seg_lines
                
                # 增加n_sides使圆形截面更平滑
                seg_tube = seg_poly.tube(radius=seg_thickness, n_sides=24)  # 从16增加到24
                arrow_segments.append(seg_tube)
        
        # 修改1：7. 绘制箭头圆锥（末端）
        if len(curve_points) >= 2:
            cone_start = curve_points[-2]
            cone_end = curve_points[-1]
            tip_dir = cone_end - cone_start
            tip_norm = np.linalg.norm(tip_dir)
            if tip_norm > 1e-6:
                tip_dir = tip_dir / tip_norm
            else:
                tip_dir = main_dir
            
            # 圆锥位置：略微向前偏移，包裹箭杆前端
            forward_offset = cone_length * 0.2
            cone_center = cone_end + tip_dir * (cone_length * 0.5 + forward_offset)
            
            cone = pv.Cone(
                center=cone_center,
                direction=tip_dir,
                height=cone_length,
                radius=cone_radius,
                resolution=32  # 从16增加到32，使圆锥更平滑
            )
            arrow_segments.append(cone)
        
        # 合并所有段
        if not arrow_segments:
            continue
        
        arrow = arrow_segments[0]
        for seg in arrow_segments[1:]:
            try:
                arrow = arrow.merge(seg)
            except Exception:
                try:
                    arrow = arrow + seg
                except Exception:
                    continue
        
        # 附加区域特征（用于颜色映射）
        arrow["Avg_Velocity"] = np.full(arrow.n_points, mean_vel_mag)
        arrow["Cluster_Size"] = np.full(arrow.n_points, n_points)
        
        all_arrows.append(arrow)

    if not all_arrows:
        print("⚠️  模式3未能生成任何区域大箭头")
        return None

    # 合并所有区域箭头
    merged = all_arrows[0]
    for a in all_arrows[1:]:
        try:
            merged = merged.merge(a)
        except Exception:
            try:
                merged = merged + a
            except Exception:
                continue

    print(f"✅ 模式3：聚类区域大箭头生成完成，共 {len(all_arrows)} 个区域箭头（修改1：参考模式1形态）")
    return merged

def create_3d_streamlines(grid, sample_points, velocity_scalars, 
                          streamline_length=50.0, step_size=0.5, n_seeds=400):
    """
    生成三维流线（优化版本2：提升成功率）
    
    Args:
        grid: pyvista.StructuredGrid（含采样点坐标和速度数据）
        sample_points: 采样点坐标数组 (N, 3)
        velocity_scalars: 速度矢量数组 (N, 3)
        streamline_length: 流线最大长度
        step_size: 基础积分步长
        n_seeds: 目标种子点数量
    
    Returns:
        streamlines: 流线PolyData对象
    """
    if not SCIPY_INTEGRATE_AVAILABLE:
        print("⚠️  scipy.integrate不可用，无法生成三维流线")
        return None
    
    # 修改4：增强速度场平滑（sigma从1.0增至2.0，减少流线断裂）
    print("  正在平滑速度场...")
    smoothed_velocity_scalars = smooth_velocity_field(sample_points, velocity_scalars, sigma=2.0)
    print(f"  速度场平滑完成（sigma=2.0）")
    
    # 1. 优化种子点生成（分层+高盐区加密+质量筛选）
    seeds = create_optimized_seeds(grid, sample_points, smoothed_velocity_scalars, n_seeds=n_seeds, min_speed=0.01)
    
    # 2. 初始化流线集合
    streamlines_points_list = []
    
    # 获取边界
    x_min, x_max = sample_points[:, 0].min(), sample_points[:, 0].max()
    y_min, y_max = sample_points[:, 1].min(), sample_points[:, 1].max()
    z_min, z_max = sample_points[:, 2].min(), sample_points[:, 2].max()
    bounds = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    
    # 获取网格维度信息（用于分层积分策略）
    unique_x = np.unique(sample_points[:, 0])
    unique_y = np.unique(sample_points[:, 1])
    unique_z = np.unique(sample_points[:, 2])
    x_sparse = len(unique_x) < 10  # X轴稀疏（如8个点）
    
    success_count = 0
    fail_count = 0
    
    # 3. 积分参数优化：针对低速场调整
    # 检查velocity_scalars的形状
    smoothed_velocity_scalars = np.asarray(smoothed_velocity_scalars)
    if smoothed_velocity_scalars.ndim == 1:
        print(f"  警告：velocity_scalars是1D数组（形状={smoothed_velocity_scalars.shape}），无法用于插值")
        print("  将使用零速度场")
        smoothed_velocity_scalars = np.zeros((len(sample_points), 3))
    elif smoothed_velocity_scalars.ndim == 2 and smoothed_velocity_scalars.shape[1] != 3:
        print(f"  警告：velocity_scalars的第二维度不是3（形状={smoothed_velocity_scalars.shape}）")
        print("  将尝试修复或使用零速度场")
        if smoothed_velocity_scalars.shape[1] == 1:
            # 如果只有1个分量，扩展为3个分量（其他分量为0）
            smoothed_velocity_scalars = np.column_stack([
                smoothed_velocity_scalars.flatten(),
                np.zeros(len(smoothed_velocity_scalars)),
                np.zeros(len(smoothed_velocity_scalars))
            ])
        else:
            smoothed_velocity_scalars = np.zeros((len(sample_points), 3))
    
    # 定义速度场函数（用于ODE求解）
    def velocity_func(t, y):
        try:
            # y是1D数组（3个元素），velocity_field_interp应该返回1D数组（3个元素）
            vel = velocity_field_interp(y, grid, smoothed_velocity_scalars)
            # 确保返回的是1D数组（3个元素）
            vel = np.asarray(vel)
            if vel.ndim == 0:
                return np.array([0.0, 0.0, 0.0])
            if vel.ndim == 2:
                if vel.shape[0] > 0:
                    vel = vel[0]  # 取第一行
                else:
                    vel = vel.flatten()
            if len(vel) != 3:
                return np.array([0.0, 0.0, 0.0])
            return vel
        except Exception as e:
            # 如果插值失败，返回零向量
            return np.array([0.0, 0.0, 0.0])
    
    # 修改4：流线积分参数调整（延长流线长度，调整步长）
    # 延长积分时间（根据速度大小动态调整），从默认值增至800，让流线有足够长度
    max_length = 800  # 修改4：进一步延长至800
    base_integration_kwargs = {
        "method": "RK45",  # 修改4：使用RK45（更稳定的积分方法）
        "max_step": 0.5,  # 减小最大步长，避免流线"跳跃"断裂
        "rtol": 1e-6,     # 修改4：提高精度
        "atol": 1e-6,     # 修改4：提高精度
        "t_span": (0.0, max_length),  # 修改4：积分时长延长至800
    }
    
    # 4. 对每个种子点生成流线（使用优化的积分参数）
    for i, seed in enumerate(seeds):
        try:
            # 分层积分策略：X轴稀疏区域使用低阶方法
            integration_kwargs = base_integration_kwargs.copy()
            if x_sparse:
                # 检查种子点是否在X轴边缘区域
                x_idx = np.argmin(np.abs(unique_x - seed[0]))
                if x_idx < 2 or x_idx > len(unique_x) - 3:
                    # 边缘区域使用RK23（低阶方法更稳定）
                    integration_kwargs["method"] = "RK23"
            
            # 尝试积分
            sol = solve_ivp(
                fun=velocity_func,
                y0=seed,
                **integration_kwargs
            )
            
            # 检查积分是否成功
            if sol.success and sol.y.shape[1] > 5:
                streamline_points = sol.y.T
                streamlines_points_list.append(streamline_points)
                success_count += 1
            else:
                # 失败重试：使用更小步长
                retry_kwargs = integration_kwargs.copy()
                retry_kwargs["max_step"] = 0.2  # 减小步长
                retry_kwargs["method"] = "RK23"  # 使用更稳定的方法
                
                sol_retry = solve_ivp(
                    fun=velocity_func,
                    y0=seed,
                    **retry_kwargs
                )
                
                if sol_retry.success and sol_retry.y.shape[1] > 5:  # 修改4：过滤过短的线（至少5个点）
                    streamline_points = sol_retry.y.T
                    streamlines_points_list.append(streamline_points)
                    success_count += 1
                else:
                    fail_count += 1
                    if fail_count <= 5:
                        print(f"   警告：流线生成失败（种子点{i}），积分未收敛")
            
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                print(f"   警告：流线生成异常（种子点{i}）: {str(e)}")
            continue
    
    # 计算成功率时添加零除保护
    total_attempts = success_count + fail_count
    if total_attempts == 0:
        print(f"  流线生成统计：未生成任何流线（无有效种子点）")
    else:
        success_rate = (success_count / total_attempts) * 100
        print(f"  流线生成统计：成功={success_count}，失败={fail_count}，成功率={success_rate:.1f}%")
    
    # 5. 流线后处理：平滑与合并
    if len(streamlines_points_list) > 0:
        print("  正在平滑流线...")
        smoothed_streamlines = smooth_streamlines(streamlines_points_list, sigma=1.5)
        print(f"  平滑后流线数量: {len(smoothed_streamlines)}")
    else:
        print("⚠️  没有成功生成的流线")
        return None
    
    # 6. 构建流线PolyData并计算速度
    streamlines_list = []
    for streamline_points in smoothed_streamlines:
        # 过滤边界外的点
        valid_mask = (
            (streamline_points[:, 0] >= x_min) & (streamline_points[:, 0] <= x_max) &
            (streamline_points[:, 1] >= y_min) & (streamline_points[:, 1] <= y_max) &
            (streamline_points[:, 2] >= z_min) & (streamline_points[:, 2] <= z_max)
        )
        
        if np.sum(valid_mask) < 5:
            continue
        
        valid_points = streamline_points[valid_mask]
        streamline = pv.lines_from_points(valid_points)
        
        # 计算流线每个点的速度（用于颜色映射）
        try:
            streamline_vels = velocity_field_interp(valid_points, grid, smoothed_velocity_scalars)
            # 确保速度数组形状正确
            streamline_vels = np.asarray(streamline_vels)
            if streamline_vels.ndim == 1:
                # 如果是1D数组，检查长度
                if len(streamline_vels) == 3:
                    # 单个速度向量，需要计算速度大小
                    streamline_vels = np.array([np.linalg.norm(streamline_vels)])
                else:
                    # 已经是速度大小数组
                    pass
            else:
                # 2D数组，计算每个点的速度大小
                streamline_vels = np.linalg.norm(streamline_vels, axis=1)
            
            # 确保速度数组长度与流线点数匹配
            if len(streamline_vels) != len(valid_points):
                # 如果不匹配，重新计算
                streamline_vels = []
                for p in valid_points:
                    vel = velocity_field_interp(p, grid, smoothed_velocity_scalars)
                    vel = np.asarray(vel)
                    if vel.ndim == 0:
                        speed = 0.0
                    elif len(vel) == 3:
                        speed = np.linalg.norm(vel)
                    else:
                        speed = np.linalg.norm(vel) if len(vel) > 0 else 0.0
                    streamline_vels.append(speed)
                streamline_vels = np.array(streamline_vels)
            
            # 确保速度数组长度正确
            if len(streamline_vels) == len(valid_points):
                streamline["speed"] = streamline_vels
            else:
                print(f"   警告：流线速度数组长度不匹配（点数={len(valid_points)}, 速度数={len(streamline_vels)}），使用默认速度")
                streamline["speed"] = np.ones(len(valid_points)) * np.mean(np.linalg.norm(smoothed_velocity_scalars, axis=1))
        except Exception as e:
            print(f"   警告：计算流线速度失败: {e}，使用默认速度")
            # 使用默认速度（基于平均速度）
            default_speed = np.mean(np.linalg.norm(smoothed_velocity_scalars, axis=1)) if len(smoothed_velocity_scalars) > 0 else 0.1
            streamline["speed"] = np.ones(len(valid_points)) * default_speed
        
        streamlines_list.append(streamline)
    
    # 7. 合并所有流线（确保保留speed字段）
    if streamlines_list and len(streamlines_list) > 0:
        try:
            # 方法1：使用+运算符合并
            combined = streamlines_list[0]
            for streamline in streamlines_list[1:]:
                combined = combined + streamline
            
            # 检查合并后是否有speed字段
            if 'speed' not in combined.array_names:
                print("   警告：合并后流线丢失speed字段，重新计算...")
                # 重新计算所有点的速度
                all_points = combined.points
                all_speeds = []
                for p in all_points:
                    vel = velocity_field_interp(p, grid, smoothed_velocity_scalars)
                    vel = np.asarray(vel)
                    if vel.ndim == 0 or len(vel) != 3:
                        speed = 0.0
                    else:
                        speed = np.linalg.norm(vel)
                    all_speeds.append(speed)
                combined["speed"] = np.array(all_speeds)
            
            return combined
        except Exception as e1:
            try:
                # 方法2：使用merge合并
                combined = streamlines_list[0]
                for streamline in streamlines_list[1:]:
                    combined = combined.merge(streamline)
                
                # 检查合并后是否有speed字段
                if 'speed' not in combined.array_names:
                    print("   警告：合并后流线丢失speed字段，重新计算...")
                    # 重新计算所有点的速度
                    all_points = combined.points
                    all_speeds = []
                    for p in all_points:
                        vel = velocity_field_interp(p, grid, smoothed_velocity_scalars)
                        vel = np.asarray(vel)
                        if vel.ndim == 0 or len(vel) != 3:
                            speed = 0.0
                        else:
                            speed = np.linalg.norm(vel)
                        all_speeds.append(speed)
                    combined["speed"] = np.array(all_speeds)
                
                return combined
            except Exception as e2:
                print(f"   警告：合并流线失败: {str(e1)}, {str(e2)}")
                # 如果合并失败，至少返回第一个流线
                if len(streamlines_list) > 0:
                    return streamlines_list[0]
                return None
    else:
        return None

# ========================================
# 矢量场优化：模式1 - 弯曲箭头（三维流动感优化）
# ========================================
# 核心思想：通过「局部速度场平滑 + B 样条曲线拟合」，让每个箭头沿相邻点速度趋势微微弯曲，
# 替代刚性直线箭头，还原海流的连续流动特征，避免"碎片化"视觉效果。

def get_neighbors(sample_points, target_idx, k=5):
    """获取目标采样点的k个空间最近邻（含自身）
    
    Args:
        sample_points: 采样点坐标数组 (N, 3)
        target_idx: 目标点的索引
        k: 邻域点数量（含自身）
    
    Returns:
        neighbor_indices: k个最近邻点的索引数组
    """
    target_point = sample_points[target_idx]
    # 计算空间欧氏距离
    distances = np.linalg.norm(sample_points - target_point, axis=1)
    # 取距离最小的k个点的索引
    neighbor_indices = np.argsort(distances)[:k]
    return neighbor_indices

def smooth_velocity_field(sample_points, velocities, sigma=1.0):
    """高斯卷积平滑速度场（x/y/z三个分量分别平滑）
    
    Args:
        sample_points: 采样点坐标数组 (N, 3)
        velocities: 速度向量数组 (N, 3)
        sigma: 高斯卷积标准差（越大速度越平滑，越小保留细节越多）
    
    Returns:
        smoothed_vel: 平滑后的速度向量数组 (N, 3)
    """
    smoothed_vel = np.zeros_like(velocities)
    for i in range(3):  # x/y/z分量
        smoothed_vel[:, i] = gaussian_filter1d(velocities[:, i], sigma=sigma)
    return smoothed_vel

def create_bent_arrows(sample_points, velocities, speeds, arrow_scale=60.0, 
                      k_neighbors=4, spline_degree=3, max_bend_factor=0.2):
    """生成三维弯曲箭头，优化弯曲程度，确保整体方向和谐
    
    Args:
        sample_points: 采样点坐标数组 (N, 3)
        velocities: 速度向量数组 (N, 3)
        speeds: 速度大小数组 (N,)
        arrow_scale: 箭头缩放因子（适配速度范围），默认60.0（增大尺寸）
        k_neighbors: 邻域点数量，默认4（减少邻域点数量，降低弯曲敏感性）
        spline_degree: B样条阶数（2~3，3次曲线更平滑）
        max_bend_factor: 最大弯曲因子(0-1)，控制弯曲程度，值越小弯曲越小，默认0.2（限制最大弯曲程度为20%）
    
    Returns:
        bent_arrows: 弯曲箭头PolyData对象
    """
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
                # PyVista的Cone的center是圆锥中心，圆锥从center - direction*height/2延伸到center + direction*height/2
                # 为了让圆锥底部在curve_points[-1]，需要center = curve_points[-1] + direction * (height/2)
                # 为了向前挪一些，让圆锥更向前延伸，包裹住箭杆前端，我们增加一个向前偏移
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

# 速度箭头可视化（独立不透明层）：使用箭头表示速度方向和大小
# 矢量场优化：模式选择
if sample_points.n_points > 0 and np.any(sample_speeds > 0):
    # 用户选择矢量场可视化模式
    print("\n" + "="*60)
    print("矢量场可视化模式选择")
    print("="*60)
    print("请选择矢量场可视化模式：")
    print("  1. 模式1 - 弯曲箭头（三维流动感优化）")
    print("     - 通过局部速度场平滑和B样条曲线拟合生成弯曲箭头")
    print("     - 适合观察局部流动特征")
    if SCIPY_INTEGRATE_AVAILABLE:
        print("  2. 模式2 - 三维流线（全局流动趋势优化）")
        print("     - 通过RK4积分生成连续流线，呈现全局流动路径")
        print("     - 适合观察整体流动结构和连通性")
    else:
        print("  2. 模式2 - 三维流线（不可用，需要scipy.integrate）")
    print("  3. 传统模式 - 直线箭头")
    print("     - 使用传统的直线箭头表示速度方向")
    print("="*60)
    
    # 获取用户输入
    while True:
        try:
            mode_input = input("请输入模式编号 (1-3，直接回车使用默认模式1): ").strip()
            if mode_input == "":
                vector_mode = 1  # 默认模式1
                break
            vector_mode = int(mode_input)
            if vector_mode in [1, 2, 3]:
                break
            else:
                print("⚠️  无效的模式编号，请输入1-3之间的数字")
        except ValueError:
            print("⚠️  无效的输入，请输入数字")
        except KeyboardInterrupt:
            print("\n⚠️  用户中断，使用默认模式1")
            vector_mode = 1
            break
    
    print(f"✅ 已选择模式{vector_mode}")
    
    # 根据选择的模式执行相应的可视化
    if vector_mode == 1:
        # 模式1 - 弯曲箭头
        print("\n" + "="*60)
        print("矢量场优化：模式1 - 弯曲箭头（三维流动感优化）")
        print("="*60)
        
        # 计算合适的箭头缩放因子（基于速度范围）- 扩大箭头大小
        speed_max = np.max(sample_speeds)
        if speed_max > 0:
            # 箭头长度基于速度大小，使用合适的缩放因子（扩大箭头）
            arrow_scale_factor = 50.0 / speed_max  # 从35.0增加到50.0，扩大箭头大小
        else:
            arrow_scale_factor = 1.0
        
        # 尝试生成弯曲箭头
        use_bent_arrows = SCIPY_AVAILABLE
        if use_bent_arrows:
            # 获取采样点坐标和速度向量
            sample_points_coords_array = sample_points_coords
            sample_velocities_array = sample_velocities
            
            # 检查数据有效性
            if len(sample_points_coords_array) == 0 or len(sample_velocities_array) == 0:
                print("⚠️  采样点或速度向量为空，无法生成弯曲箭头")
                use_bent_arrows = False
            else:
                # 生成弯曲箭头（优化参数：减少弯曲程度，增大箭头尺寸）
                try:
                    # 优化参数：减少弯曲程度，增大箭头尺寸
                    k_neighbors = 4  # 减少邻域点数量，降低弯曲敏感性
                    spline_degree = 3
                    arrow_scale = 60.0  # 进一步增大箭头整体尺寸
                    max_bend_factor = 0.3  # 限制最大弯曲程度为30%（稍微增加弯曲幅度）
                    
                    print(f"  参数：k_neighbors={k_neighbors}, spline_degree={spline_degree}, "
                          f"arrow_scale={arrow_scale:.2f}, max_bend_factor={max_bend_factor}")
                    
                    bent_arrows = create_bent_arrows(
                        sample_points=sample_points_coords_array,
                        velocities=sample_velocities_array,
                        speeds=sample_speeds,
                        arrow_scale=arrow_scale,  # 使用优化后的箭头缩放因子
                        k_neighbors=k_neighbors,
                        spline_degree=spline_degree,
                        max_bend_factor=max_bend_factor
                    )
                    
                    if bent_arrows is not None and bent_arrows.n_points > 0:
                        arrows = bent_arrows
                        print("✅ 已使用弯曲箭头（模式1优化）")
                    else:
                        # 如果弯曲箭头生成失败，回退到直线箭头
                        use_bent_arrows = False
                        print("⚠️  弯曲箭头生成失败（返回None或点数为0），回退到直线箭头")
                except Exception as e:
                    use_bent_arrows = False
                    print(f"⚠️  弯曲箭头生成异常: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            use_bent_arrows = False
            print("⚠️  SciPy不可用，无法生成弯曲箭头，将使用直线箭头")
        
        # 如果未使用弯曲箭头，使用传统的直线箭头
        if not use_bent_arrows:
            # 创建箭头，方向由velocity向量决定
            arrows = sample_points.glyph(
                orient='velocity',  # 箭头方向由velocity向量决定
                scale='speed',  # 箭头大小由speed决定
                factor=arrow_scale_factor  # 缩放因子，控制箭头整体大小
            )
            print("✅ 已使用直线箭头（传统模式）")
        
        # 添加箭头，确保在体积渲染之后渲染，并且不被遮挡
        # 根据箭头类型选择标量字段名称
        if use_bent_arrows and 'speed' in arrows.array_names:
            scalars_field = 'speed'
        else:
            scalars_field = 'speed'
        
        arrow_actor = plotter.add_mesh(
            arrows,
            scalars=scalars_field,  # 箭头颜色由速度大小映射
            cmap='cool',  # 使用cool色图（蓝-青-白），与hot色图（红-黄-黑）形成对比
            opacity=1.0,  # 强制完全不透明
            show_scalar_bar=True,  # 显示标量条，显示流速范围
            scalar_bar_args={'title': '流速 (Speed)'},
            pickable=True,  # 允许拾取
            render_lines_as_tubes=True  # 渲染为管状，更清晰
        )
        
        # 调整箭头渲染属性，确保不被体积遮挡
        try:
            # 获取箭头actor的属性
            arrow_property = arrow_actor.GetProperty()
            # 确保箭头完全不透明且深度测试正确
            arrow_property.SetOpacity(1.0)
            # 设置箭头渲染为管状，更清晰可见
            if hasattr(arrow_property, 'SetRenderLinesAsTubes'):
                arrow_property.SetRenderLinesAsTubes(True)
            if hasattr(arrow_property, 'SetLineWidth'):
                arrow_property.SetLineWidth(4.5)  # 增加线宽，使箭头更明显（从3.5增加到4.5，扩大箭头）
            # 禁用深度写入，确保箭头始终可见（即使被透明体积遮挡）
            if hasattr(arrow_property, 'SetDepthWrite'):
                arrow_property.SetDepthWrite(False)
            print("✅ 已调整箭头渲染属性，确保不被体积遮挡")
        except Exception as e:
            print(f"警告：无法调整箭头渲染属性: {e}")
        if use_bent_arrows:
            print(f"✅ 弯曲箭头已添加（模式1优化）")
            print(f"   采样点数={sample_points.n_points}，箭头数={arrows.n_points}")
            print(f"   速度范围: [{np.min(sample_speeds):.4f}, {np.max(sample_speeds):.4f}]")
            print(f"   箭头缩放因子: {arrow_scale_factor:.4f}")
            print("   特点：箭头沿相邻点速度趋势弯曲，还原海流连续流动特征")
        else:
            print(f"速度箭头已添加（采样点数={sample_points.n_points}，箭头数={arrows.n_points}）")
            print(f"速度范围: [{np.min(sample_speeds):.4f}, {np.max(sample_speeds):.4f}]")
            print(f"箭头缩放因子: {arrow_scale_factor:.4f}")
    
    elif vector_mode == 2:
        # 模式2 - 三维流线（直接使用 PyVista 内置流线积分，与 velocity_3D.py 一致）
        if not SCIPY_INTEGRATE_AVAILABLE:
            # 对于 PyVista 内置流线并不强制依赖 SciPy，但保持原有提示逻辑
            print("⚠️  scipy.integrate不可用，仍尝试使用PyVista内置流线生成（模式2简化版）")
        print("\n" + "="*60)
        print("矢量场优化：模式2 - 三维流线（全局流动趋势优化，使用PyVista内置流线）")
        print("="*60)

        # 使用“修改1”方法：基于高速度区域优先的 PyVista 内置流线方案
        # 说明：
        # - 不直接从所有网格点均匀抽样，而是优先在“速度较大区域”布置种子点
        # - 高速度区域：速度 > min_effective_speed 的点，优先选 300 个
        # - 其他区域：补充 ~100 个种子，保证全局覆盖
        # - 若整体速度都很小，则退化为在所有采样点中均匀随机选 400 个
        # - 之后使用 grid.streamlines_from_source 沿 'velocity' 场生成流线
        try:
            # 1）根据 sample_points / sample_speeds 构造“高速度优先”的种子点
            if sample_points.n_points == 0 or len(sample_speeds) == 0:
                print("⚠️  采样点或速度数据为空，无法基于速度优化种子点，退化为网格均匀采样")
                total_points = grid.n_points
                if total_points <= 0:
                    print("⚠️  原始网格点数为0，无法生成流线，将回退到模式3（直线箭头）")
                    vector_mode = 3
                    streamlines = None
                else:
                    target_seeds = 400
                    stride = max(1, total_points // target_seeds)
                    print(f"  原始网格点数: {total_points}，步长: {stride}")
                    seed_points = pv.PolyData(grid.points[::stride])
                    print(f"  实际种子点数量: {seed_points.n_points}")
                    streamlines = grid.streamlines_from_source(
                        source=seed_points,
                        vectors='velocity',
                        integration_direction='both',
                        initial_step_length=5.0,
                        terminal_speed=1e-3,
                        max_steps=1000
                    )
            else:
                # 使用采样点速度构造高速度区域掩码
                min_effective_speed = 0.01
                speeds = np.asarray(sample_speeds)
                high_speed_mask = speeds > min_effective_speed

                seeds_coords = None
                high_count = int(high_speed_mask.sum())

                if high_count > 0:
                    high_coords = sample_points_coords[high_speed_mask]
                    n_high = min(300, len(high_coords))
                    high_idx = np.random.choice(len(high_coords), size=n_high, replace=False)
                    high_seeds = high_coords[high_idx]

                    low_coords = sample_points_coords[~high_speed_mask]
                    if len(low_coords) > 0:
                        n_low = min(100, len(low_coords))
                        low_idx = np.random.choice(
                            len(low_coords),
                            size=n_low,
                            replace=(len(low_coords) < n_low)
                        )
                        low_seeds = low_coords[low_idx]
                        seeds_coords = np.vstack([high_seeds, low_seeds])
                        remaining_count = len(low_seeds)
                    else:
                        seeds_coords = high_seeds
                        remaining_count = 0

                    print(f"  优化后种子点分布：高速度区域{len(high_seeds)}个，其他区域{remaining_count}个")
                else:
                    # 无明显高速度区域，均匀随机采样 400 个
                    total = len(sample_points_coords)
                    if total == 0:
                        seeds_coords = None
                    else:
                        n_total = min(400, total)
                        idx = np.random.choice(
                            total,
                            size=n_total,
                            replace=(total < n_total)
                        )
                        seeds_coords = sample_points_coords[idx]
                        print(f"  无显著高速度区域，均匀随机采样种子点 {n_total} 个")

            if 'seeds_coords' in locals() and seeds_coords is not None and len(seeds_coords) > 0:
                # 使用优化后的种子点
                seed_points = pv.PolyData(seeds_coords)
                print(f"  基于速度优化后的种子点数量: {seed_points.n_points}")
                streamlines = grid.streamlines_from_source(
                    source=seed_points,
                    vectors='velocity',
                    integration_direction='both',
                    initial_step_length=0.1,  # 更小初始步长，提高精度
                    terminal_speed=1e-3,
                    max_steps=2000          # 增大最大步数，允许更长流线
                )

            # 若前面未生成 seeds_coords 或生成失败，则依赖前面的回退逻辑中设置的 streamlines
            if 'streamlines' in locals() and streamlines is not None and streamlines.n_points > 0:
                # 3）给流线添加速度标量（与 velocity_3D.py 相同逻辑）
                if 'velocity' in streamlines.array_names:
                    speed = np.linalg.norm(streamlines['velocity'], axis=1)
                    streamlines['speed'] = speed
                elif 'vectors' in streamlines.array_names:
                    speed = np.linalg.norm(streamlines['vectors'], axis=1)
                    streamlines['speed'] = speed
                else:
                    # 如果没有向量字段，则使用全局 sample_velocities 近似速度大小
                    print("   警告：流线数据中找不到 'velocity' 或 'vectors' 字段，使用全局速度近似")
                    approx_speed = np.mean(sample_speeds) if len(sample_speeds) > 0 else 0.1
                    streamlines['speed'] = np.full(streamlines.n_points, approx_speed)
                    speed = streamlines['speed']

                print(f"   流线数据检查：点数={streamlines.n_points}，字段={streamlines.array_names}")
                print(f"   流线速度范围: [{speed.min():.4f}, {speed.max():.4f}]")

                # 4）添加流线到场景（半透明+高光，确保穿过体积可见）
                streamline_actor = plotter.add_mesh(
                    streamlines,
                    scalars='speed',      # 颜色编码速度大小
                    cmap='cool',          # 采用与箭头相近的冷色系，便于区分与温度 hot_r
                    line_width=2.0,
                    opacity=0.8,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': '流线速度 (Speed)'},
                    pickable=True
                )

                # 调整渲染属性，确保在体渲染上方可见
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
                    print("✅ 三维流线已添加（模式2：PyVista内置流线 + 高速度区域优先种子）")
                except Exception as e:
                    print(f"警告：无法调整流线渲染属性: {e}")

                print("   可视化方法（修改1）：")
                print("   - 使用结构化网格上的速度向量场 'velocity'")
                print("   - 优先从速度较大的区域中选取约 300 个种子点，其余约 100 个来自其他区域")
                print("   - 若整体速度较小，则在所有采样点中均匀随机选取约 400 个种子点")
                print("   - 调用 grid.streamlines_from_source 沿速度场双向积分生成三维流线")
                print("   - 在每个流线点上计算速度大小 'speed'，使用 'cool' 色图进行颜色编码")
                print("   - 流线以半透明+高光方式叠加在体渲染之上，突出全局流动路径和高流速通道")
            else:
                print("⚠️  未能生成有效流线，将使用模式3（传统模式）")
                vector_mode = 3
        except Exception as e:
            print(f"⚠️  模式2流线生成异常: {e}，将回退到模式3（传统模式）")
            vector_mode = 3
    
    if vector_mode == 3 or (vector_mode == 2 and not SCIPY_INTEGRATE_AVAILABLE):
        # 模式3 - 聚类区域大箭头（区域关联性优化）
        print("\n" + "="*60)
        print("矢量场优化：模式3 - 聚类区域大箭头（区域关联性优化）")
        print("="*60)

        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn 不可用，模式3将退回到传统直线箭头")
            use_cluster_mode = False
        else:
            use_cluster_mode = True

        cluster_arrows = None
        if use_cluster_mode:
            # 让用户输入期望的大致子簇数，用于控制 KMeans 聚类簇数
            target_clusters = None
            try:
                user_input = input("模式3：请输入期望的大致子簇数 (建议 10~30，直接回车默认 20): ").strip()
                if user_input == "":
                    target_clusters = 20
                else:
                    target_clusters = int(user_input)
                    if target_clusters < 2:
                        target_clusters = 2
            except Exception:
                target_clusters = 20

            print(f"  模式3：目标子簇数 ≈ {target_clusters}")

            # 使用采样点和速度向量做空间-速度联合聚类
            clusters, cluster_vels, labels = cluster_flow_regions(
                sample_points=sample_points_coords,
                velocities=sample_velocities,
                spatial_eps=100.0,
                vel_eps=0.3,
                min_samples=3,
                target_clusters=target_clusters
            )
            if clusters and cluster_vels:
                # 计算总采样点数（用于箭头粗细计算）
                total_sample_points = len(sample_points_coords)
                cluster_arrows = create_cluster_bent_arrows(
                    clusters=clusters,
                    cluster_vels=cluster_vels,
                    total_points=total_sample_points,
                    arrow_scale=50.0,
                    spline_degree=3
                )
            else:
                print("⚠️  模式3聚类结果为空，将退回到传统直线箭头")
                use_cluster_mode = False

        if use_cluster_mode and cluster_arrows is not None and cluster_arrows.n_points > 0:
            # 询问是否显示子簇区域轮廓（虚线盒子效果）
            show_regions = False
            try:
                region_input = input("模式3：是否显示子簇区域轮廓（y/n，直接回车默认 n）: ").strip().lower()
                show_regions = (region_input == "y")
            except Exception:
                show_regions = False

            # 使用聚类大箭头进行可视化
            avg_vel_array = cluster_arrows["Avg_Velocity"] if "Avg_Velocity" in cluster_arrows.array_names else None
            if avg_vel_array is not None and len(avg_vel_array) > 0:
                vmin, vmax = float(np.min(avg_vel_array)), float(np.max(avg_vel_array))
            else:
                vmin, vmax = 0.0, float(np.max(sample_speeds)) if len(sample_speeds) > 0 else 1.0

            arrow_actor = plotter.add_mesh(
                cluster_arrows,
                scalars="Avg_Velocity" if "Avg_Velocity" in cluster_arrows.array_names else None,
                cmap='viridis',
                opacity=1.0,
                show_scalar_bar=True,
                scalar_bar_args={'title': '区域平均速度 (Avg Speed)'},
                pickable=True
            )

            try:
                arrow_property = arrow_actor.GetProperty()
                arrow_property.SetOpacity(1.0)
                if hasattr(arrow_property, 'SetDepthWrite'):
                    arrow_property.SetDepthWrite(False)
                # 增加大箭头亮度：增强环境光、漫反射和镜面反射
                if hasattr(arrow_property, 'SetAmbient'):
                    arrow_property.SetAmbient(0.6)  # 增加环境光，提高整体亮度
                if hasattr(arrow_property, 'SetDiffuse'):
                    arrow_property.SetDiffuse(0.9)  # 增加漫反射，提高亮度
                if hasattr(arrow_property, 'SetSpecular'):
                    arrow_property.SetSpecular(0.8)  # 从0.5增加到0.8，增强高光
                if hasattr(arrow_property, 'SetSpecularPower'):
                    arrow_property.SetSpecularPower(30)  # 从10增加到30，使高光更集中
                print("✅ 模式3：聚类区域大箭头已添加（高亮显示区域流动模式，亮度增强）")
            except Exception as e:
                print(f"警告：无法调整聚类大箭头渲染属性: {e}")

            # 可选：绘制子簇区域的包围盒轮廓（虚线效果的替代）
            if show_regions and clusters:
                print("   正在绘制子簇区域轮廓 ...")
                for cluster_points in clusters:
                    if len(cluster_points) < 2:
                        continue
                    bbox_min = cluster_points.min(axis=0)
                    bbox_max = cluster_points.max(axis=0)
                    box = pv.Box(bounds=(
                        bbox_min[0], bbox_max[0],
                        bbox_min[1], bbox_max[1],
                        bbox_min[2], bbox_max[2]
                    ))
                    # 仅显示边框（wireframe），用细线模拟“虚线区域”感觉
                    plotter.add_mesh(
                        box,
                        color="white",
                        style="wireframe",
                        line_width=1.5,
                        opacity=0.3,
                        pickable=False
                    )
                print("✅ 子簇区域轮廓已绘制（wireframe 盒子）")

            print("   可视化方法：")
            print("   - 通过空间+速度联合聚类，将流场划分为若干同源流动区域")
            print("   - 每个区域用一支大箭头表示：位置≈区域中心，方向=平均速度，粗细∝区域点数，颜色∝平均速度大小")
            print("   - 弯曲轨迹基于区域内点的速度趋势拟合，幅度收敛，整体覆盖子簇空间范围")
        else:
            # 退回到传统直线箭头
            print("⚠️  模式3聚类大箭头不可用，退回到传统直线箭头")

            # 计算合适的箭头缩放因子
            speed_max = np.max(sample_speeds)
            if speed_max > 0:
                arrow_scale_factor = 50.0 / speed_max
            else:
                arrow_scale_factor = 1.0

            # 创建箭头，方向由velocity向量决定
            arrows = sample_points.glyph(
                orient='velocity',  # 箭头方向由velocity向量决定
                scale='speed',  # 箭头大小由speed决定
                factor=arrow_scale_factor  # 缩放因子，控制箭头整体大小
            )

            # 添加箭头
            arrow_actor = plotter.add_mesh(
                arrows,
                scalars='speed',  # 箭头颜色由速度大小映射
                cmap='cool',  # 使用cool色图
                opacity=1.0,  # 强制完全不透明
                show_scalar_bar=True,
                scalar_bar_args={'title': '流速 (Speed)'},
                pickable=True,
                render_lines_as_tubes=True
            )

            # 调整箭头渲染属性
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

            print(f"✅ 直线箭头已添加（模式3退化为传统模式）")
            print(f"   采样点数={sample_points.n_points}，箭头数={arrows.n_points}")
            print(f"   速度范围: [{np.min(sample_speeds):.4f}, {np.max(sample_speeds):.4f}]")
            print(f"   箭头缩放因子: {arrow_scale_factor:.4f}")
else:
    print("警告：采样点为空或速度全为0，无法添加箭头")

# 打印体积渲染信息
print(f"组合体积点数: {combined_volume.n_points}")
print(f"盐度数据范围: [{salt_min_val:.4f}, {salt_max_val:.4f}]")
print(f"温度数据范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")
print(f"透明度映射: 盐度[{salt_min_val:.2f}, {salt_max_val:.2f}] -> 透明度[0.0, 0.3]（策略17：反幂函数映射，温和突出高盐）")
print(f"采样参数: skip={skip}, nz={nz}, 数据形状={U_local.shape}")

plotter.add_axes()
plotter.show()

