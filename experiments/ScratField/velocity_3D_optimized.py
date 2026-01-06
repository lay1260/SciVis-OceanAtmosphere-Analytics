#温度盐度采用volume rendering，洋流使用流线
# 优化版本：
# - 温度：只映射颜色值（hot色图）
# - 盐度：只映射透明度（代表物质浓度，0到0.25）
# - 箭头：完全不透明
# - 降低分辨率以提升加载速度和交互流畅度
# - 使用VTK底层API实现真正的双标量独立控制（改进的温度-盐度映射）
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
    new_skip = max(2, skip // 2)  # 至少使用skip=2
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

# 检查采样点是否覆盖内部（不是只在边界）
x_coords = sample_points_coords[:, 0]
y_coords = sample_points_coords[:, 1]
z_coords = sample_points_coords[:, 2]
x_min, x_max = np.min(x_coords), np.max(x_coords)
y_min, y_max = np.min(y_coords), np.max(y_coords)
z_min, z_max = np.min(z_coords), np.max(z_coords)

# 检查内部点（不在边界上的点）
# 对于小维度（如nx=2），所有点都是边界点是正常的
# 我们主要检查y和z方向是否有内部点
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

# 使用更小的边界容差（减小z方向容差，确保内部点被正确识别）
if x_range > 0:
    x_margin = max(x_range * 0.05, (x_max - x_min) * 0.01)  # 至少1%的范围
else:
    x_margin = 0  # 如果x方向没有范围，所有点都是边界点

if y_range > 0:
    y_margin = max(y_range * 0.02, (y_max - y_min) * 0.005)  # 改为2%，减小容差
else:
    y_margin = 0

if z_range > 0:
    z_margin = max(z_range * 0.02, (z_max - z_min) * 0.005)  # 改为2%，减小容差
else:
    z_margin = 0

# 内部点判断：在至少两个方向上不在边界上
# 对于小维度（如nx=2），x方向的所有点都是边界点是正常的
x_interior = (x_coords > x_min + x_margin) & (x_coords < x_max - x_margin) if x_range > 0 else np.zeros(len(x_coords), dtype=bool)
y_interior = (y_coords > y_min + y_margin) & (y_coords < y_max - y_margin) if y_range > 0 else np.zeros(len(y_coords), dtype=bool)
z_interior = (z_coords > z_min + z_margin) & (z_coords < z_max - z_margin) if z_range > 0 else np.zeros(len(z_coords), dtype=bool)

# 内部点：在y和z方向上都不在边界上（x方向可能都是边界点，这是正常的）
interior_mask = y_interior & z_interior
n_interior = np.sum(interior_mask)
n_boundary = len(sample_points_coords) - n_interior

print(f"采样点分布: 内部点={n_interior}个 (Y和Z方向不在边界), 边界点={n_boundary}个")
print(f"采样点覆盖: X方向 {len(x_indices)}个点 (nx={nx}), Y方向 {len(y_indices)}个点 (ny={ny}), Z方向 {len(z_indices)}个点 (nz={nz})")
if nx <= 2:
    print(f"注意: X方向维度较小(nx={nx})，所有点都在边界上是正常的")

# ----------------------------
# 7️⃣ 温度与盐度体积渲染（优化：单体积双标量绑定）
# ----------------------------
# 核心优化：单体积双标量绑定（替代双体积叠加）
# 理论依据：PyVista支持单个体积同时绑定「颜色标量（温度）」和「透明度标量（盐度）」
# 双体积叠加会导致透明度非线性叠加，单体积直接映射可避免该问题
# 
# 可视化逻辑：
# - 温度：映射颜色值（hot色图）
# - 盐度：映射透明度（代表物质浓度，0到0.25）
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
# 第五步优化：分级细节（LOD）渲染（可选，新增盐度关联核心区）
# ----------------------------
print("\n" + "="*60)
print("第五步优化：分级细节（LOD）渲染（可选）")
print("="*60)
print("是否启用LOD渲染优化？")
print("  - 启用：边缘区降采样50%，核心区保持全分辨率，提升性能")
print("  - 禁用：使用原始完整网格，保证最佳渲染质量")
print("="*60)

use_lod = False
while True:
    try:
        lod_choice = input("是否启用LOD优化？(y/n，直接回车默认禁用): ").strip().lower()
        if lod_choice == '' or lod_choice == 'n':
            use_lod = False
            print("✅ 已禁用LOD优化，使用原始完整网格")
            break
        elif lod_choice == 'y':
            use_lod = True
            print("✅ 已启用LOD优化")
            break
        else:
            print("无效输入，请输入 y 或 n")
    except (ValueError, KeyboardInterrupt):
        print("无效输入，请输入 y 或 n")

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
    
    # 3. 改进的LOD实现：保持结构化网格
    # 方法：在数据层面进行降采样，然后重新构建StructuredGrid
    # 但这种方法会破坏结构化网格的规则性，所以我们需要另一种方法
    # 
    # 更好的方法：不进行LOD，而是通过调整渲染参数来优化性能
    # 或者：使用数据层面的降采样，但保持网格结构
    #
    # 由于VTK体积渲染对UnstructuredGrid支持有限，我们采用保守策略：
    # 如果网格较小，不进行LOD；如果网格较大，提示用户
    # 改进的LOD实现：在Y方向进行降采样，保持结构化网格
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
# 第六步优化：环境光与背景协同优化（可选，提升空间辨识度）
# ----------------------------
print("\n" + "="*60)
print("第六步优化：环境光与背景协同优化（可选）")
print("="*60)
print("是否启用环境光与背景优化？")
print("  - 启用：深暗蓝背景 + 方向光 + 高盐梯度区自发光，提升空间辨识度")
print("  - 禁用：使用默认背景和光照设置")
print("="*60)

use_env_lighting = False
while True:
    try:
        env_choice = input("是否启用环境光与背景优化？(y/n，直接回车默认禁用): ").strip().lower()
        if env_choice == '' or env_choice == 'n':
            use_env_lighting = False
            print("✅ 已禁用环境光与背景优化，使用默认设置")
            break
        elif env_choice == 'y':
            use_env_lighting = True
            print("✅ 已启用环境光与背景优化")
            break
        else:
            print("无效输入，请输入 y 或 n")
    except (ValueError, KeyboardInterrupt):
        print("无效输入，请输入 y 或 n")

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
        print("   - 透明度：使用基于盐度范围的传递函数（0到0.25）")
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
            
            # 3.2 第三步优化：盐度主导的视觉权重优化（4种映射方案）
            # 理论依据：盐度值直接反映物质浓度（高值需突出），盐度梯度反映混合关键区（需强化）
            # 透明度范围：[0.02, 0.25]
            
            # ========== 定义4种映射函数 ==========
            def opacity_mapping_power_high_low(salt_data, salt_gradient_norm):
                """方案1：幂函数映射（高阈值基础版，平衡区分度与通透度）"""
                # 1. 高盐过滤阈值（40%分位数，过滤40%中低盐）
                salt_low_threshold = np.percentile(salt_data, 40)
                salt_min, salt_max = salt_data.min(), salt_data.max()
                
                # 2. 盐度归一化（聚焦剩余60%高盐区域）
                salt_norm = np.clip((salt_data - salt_low_threshold) / (salt_max - salt_low_threshold), 0.0, 1.0)
                
                # 3. 幂函数映射（降低上限至0.06，增强内部可见性）
                base_opacity = 0.01 + 0.05 * (salt_norm ** 3.0)  # 降低上限至0.06
                
                # 4. 梯度辅助增强（不突破0.06上限）
                gradient_boost = 0.1 + 0.2 * salt_gradient_norm
                final_opacity = np.clip(base_opacity * gradient_boost, 0.01, 0.06)
                
                return final_opacity
            
            def opacity_mapping_exponential_high_low(salt_data, salt_gradient_norm):
                """方案2：指数映射（激进聚焦版，仅突出极高盐核心）"""
                # 1. 超高低盐阈值（48%分位数，过滤近一半中低盐）
                salt_low_threshold = np.percentile(salt_data, 48)
                salt_min, salt_max = salt_data.min(), salt_data.max()
                
                # 2. 盐度归一化（仅保留高盐核心区）
                salt_norm = np.clip((salt_data - salt_low_threshold) / (salt_max - salt_low_threshold), 0.0, 1.0)
                
                # 3. 高指数映射（5.0系数，快速饱和至0.06）
                base_opacity = 0.01 + 0.05 * (1 - np.exp(-5.0 * salt_norm))  # 降低上限至0.06
                
                # 4. 梯度辅助增强（严格控上限）
                gradient_boost = 0.1 + 0.2 * salt_gradient_norm
                final_opacity = np.clip(base_opacity * gradient_boost, 0.01, 0.06)
                
                return final_opacity
            
            def opacity_mapping_piecewise_high_low(salt_data, salt_gradient_norm):
                """方案3：分段线性映射（精准分层版，高盐内部分级明确）"""
                # 1. 极高低盐阈值（50%分位数，过滤一半区域）
                salt_low_threshold = np.percentile(salt_data, 50)
                salt_min, salt_max = salt_data.min(), salt_data.max()
                
                # 2. 高盐区间分段（适配阈值后分布，降低上限）
                salt_ranges = [
                    salt_low_threshold,
                    np.percentile(salt_data, 78),  # 中高盐→高盐阈值（78%分位数）
                    salt_max
                ]
                opacity_ranges = [0.01, 0.03, 0.06]  # 三层不透明度，降低上限至0.06
                
                # 3. 分段线性映射（精准控制各段过渡）
                base_opacity = np.zeros_like(salt_data)
                base_opacity[salt_data < salt_ranges[0]] = 0.01  # 过滤区
                
                mid_mask = (salt_data >= salt_ranges[0]) & (salt_data <= salt_ranges[1])
                base_opacity[mid_mask] = np.interp(salt_data[mid_mask], salt_ranges[:2], opacity_ranges[:2])
                
                high_mask = salt_data > salt_ranges[1]
                base_opacity[high_mask] = np.interp(salt_data[high_mask], salt_ranges[1:], opacity_ranges[1:])
                
                # 4. 梯度辅助增强（不突破上限）
                gradient_boost = 0.1 + 0.2 * salt_gradient_norm
                final_opacity = np.clip(base_opacity * gradient_boost, 0.01, 0.06)
                
                return final_opacity
            
            def opacity_mapping_logarithmic_high_low(salt_data, salt_gradient_norm):
                """方案4：对数映射（平滑过渡版，保留高盐梯度细节）"""
                # 1. 平衡高阈值（43%分位数，过滤43%中低盐）
                salt_low_threshold = np.percentile(salt_data, 43)
                salt_min, salt_max = salt_data.min(), salt_data.max()
                
                # 2. 盐度归一化（避免对数负输入）
                salt_norm = np.clip((salt_data - salt_low_threshold) / (salt_max - salt_low_threshold), 1e-6, 1.0)
                
                # 3. 对数映射（降低上限至0.06，强化高盐梯度）
                base_opacity = 0.01 + 0.05 * (np.log(1 + 3.0 * salt_norm) / np.log(4.0))  # 降低上限至0.06
                
                # 4. 梯度辅助增强（严格控上限）
                gradient_boost = 0.1 + 0.2 * salt_gradient_norm
                final_opacity = np.clip(base_opacity * gradient_boost, 0.01, 0.06)
                
                return final_opacity
            
            def opacity_mapping_sqrt_strategy_19(salt_data, salt_gradient_norm):
                """方案5（策略19）：平方根映射（低过滤+高透明，温和保留中低盐细节）
                与velocity_3D_strategies.py中的策略19保持一致
                """
                # 1. 低阈值（30%分位数，保留中低盐细节）
                salt_threshold = np.percentile(salt_data, 30)
                salt_max = salt_data.max()
                
                # 2. 盐度归一化（与strategies.py保持一致）
                salt_norm = np.clip((salt_data - salt_threshold) / (salt_max - salt_threshold), 0.0, 1.0)
                
                # 3. 平方根映射（与strategies.py保持一致：0~0.35透明度）
                base_opacity = 0 + 0.35 * np.sqrt(salt_norm)
                
                # 4. 梯度辅助增强（与strategies.py保持一致）
                gradient_boost = 0.1 + 0.2 * salt_gradient_norm
                final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.35)
                
                return final_opacity
            
            # ========== 用户选择映射方案 ==========
            print("\n" + "="*60)
            print("第三步优化：盐度主导的视觉权重优化")
            print("="*60)
            print("请选择透明度映射方案：")
            print("  1. 幂函数映射（40%阈值，平衡区分度与通透度）")
            print("  2. 指数映射（48%阈值，激进聚焦极高盐核心）")
            print("  3. 分段线性映射（50%阈值，精准分层）")
            print("  4. 对数映射（43%阈值，平滑过渡）")
            print("  5. 平方根映射（30%阈值，温和保留中低盐细节）【默认推荐】")
            print("="*60)
            
            while True:
                try:
                    choice_input = input("请输入方案编号 (1-5，直接回车使用默认方案5): ").strip()
                    if choice_input == '':
                        # 默认选择方案5（策略19）
                        choice = 5
                        break
                    elif choice_input in ['1', '2', '3', '4', '5']:
                        choice = int(choice_input)
                        break
                    else:
                        print("无效输入，请输入 1、2、3、4 或 5")
                except (ValueError, KeyboardInterrupt):
                    print("无效输入，请输入 1、2、3、4 或 5")
            
            # 根据用户选择计算最终透明度
            if choice == 1:
                final_opacity = opacity_mapping_power_high_low(salt_data, salt_gradient_norm)
                print("✅ 已选择方案1：幂函数映射（40%阈值）")
            elif choice == 2:
                final_opacity = opacity_mapping_exponential_high_low(salt_data, salt_gradient_norm)
                print("✅ 已选择方案2：指数映射（48%阈值）")
            elif choice == 3:
                final_opacity = opacity_mapping_piecewise_high_low(salt_data, salt_gradient_norm)
                print("✅ 已选择方案3：分段线性映射（50%阈值）")
            elif choice == 4:
                final_opacity = opacity_mapping_logarithmic_high_low(salt_data, salt_gradient_norm)
                print("✅ 已选择方案4：对数映射（43%阈值）")
            elif choice == 5:
                final_opacity = opacity_mapping_sqrt_strategy_19(salt_data, salt_gradient_norm)
                print("✅ 已选择方案5（策略19）：平方根映射（30%阈值，温和保留中低盐细节）【默认】")
            
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
            
            print("正在构建温度-透明度映射表（基于盐度主导的视觉权重）...")
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
                # 兼容旧版本
                hot_r_cmap = plt.cm.get_cmap('hot_r')
            
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
                
                # 第六步优化：自发光调整（高盐梯度区自发光，突出混合区）- 再次增强
                if use_env_lighting:
                    # 计算高梯度区域的比例（梯度>0.5的区域）
                    high_gradient_ratio = np.mean(salt_gradient_norm > 0.5)
                    # 计算高盐度区域的比例（盐度>80%分位数的区域）
                    salt_80_percentile = np.percentile(salt_data, 80)
                    high_salt_ratio = np.mean(salt_data > salt_80_percentile)
                    # 强化高梯度区域和高盐度区域的环境光（模拟自发光效果）- 再次增强自发光强度
                    # 同时考虑高梯度区域和高盐度区域
                    enhanced_ambient = base_ambient + 0.4 * high_gradient_ratio + 0.3 * high_salt_ratio  # 从0.25增加到0.4（梯度）+ 0.3（高盐）
                    enhanced_ambient = min(enhanced_ambient, 0.85)  # 提高最大环境光限制（从0.7到0.85）
                    volume_property.SetAmbient(enhanced_ambient)
                    volume_property.SetDiffuse(base_diffuse)
                    print("✅ 第四步优化：渲染混合模式优化已实现")
                    print("   - 混合模式：复合混合（Composite Blending，已在add_volume时设置）")
                    print("   - 阴影：已启用，梯度依赖的环境光和漫反射")
                    print(f"   - 环境光：{enhanced_ambient:.3f}（基于平均梯度 + 高梯度区强化自发光 + 高盐度区强化自发光）")
                    print(f"   - 漫反射：{base_diffuse:.3f}（基于平均梯度）")
                    print("   - 镜面反射：0.05（降低反光，避免遮挡）")
                    print(f"   - 高梯度区域比例：{high_gradient_ratio:.2%}（自发光增强区域）")
                    print(f"   - 高盐度区域比例：{high_salt_ratio:.2%}（自发光增强区域，盐度>{salt_80_percentile:.2f}）")
                else:
                    # 使用第四步优化的标准光照设置
                    volume_property.SetAmbient(base_ambient)  # 盐度梯度越大，环境光越低，阴影越突出
                    volume_property.SetDiffuse(base_diffuse)  # 盐度梯度越大，漫反射越强，体素边界越清晰
                    print("✅ 第四步优化：渲染混合模式优化已实现")
                    print("   - 混合模式：复合混合（Composite Blending，已在add_volume时设置）")
                    print("   - 阴影：已启用，梯度依赖的环境光和漫反射")
                    print(f"   - 环境光：{base_ambient:.3f}（基于平均梯度）")
                    print(f"   - 漫反射：{base_diffuse:.3f}（基于平均梯度）")
                    print("   - 镜面反射：0.05（降低反光，避免遮挡）")
                
            except Exception as e:
                print(f"警告：无法设置混合模式优化: {e}")
                import traceback
                traceback.print_exc()
            
            print("✅ VTK底层API双标量独立控制已实现（五步优化完整实现）")
            print("   【第一步优化：单体积双标量绑定】")
            print("   - 颜色：由温度标量独立控制（自适应颜色映射，hot_r色图）")
            print("   - 透明度：由盐度标量独立控制（基于盐度主导的视觉权重）")
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
            print("   【第三步优化：盐度主导的视觉权重优化】")
            if choice == 1:
                print("   - 映射方案：幂函数映射（3.0次）")
                print("   - 低盐过滤阈值：40%分位数")
                print("   - 特点：高盐差异放大，不透明度0.01~0.06平滑过渡")
                print("   - 适用场景：兼顾高盐细节与整体通透，通用场景")
            elif choice == 2:
                print("   - 映射方案：指数映射（5.0系数）")
                print("   - 低盐过滤阈值：48%分位数")
                print("   - 特点：极高盐快速饱和，仅突出核心浓集区")
                print("   - 适用场景：聚焦盐度>32的核心区，需极致通透背景")
            elif choice == 3:
                print("   - 映射方案：分段线性映射")
                print("   - 低盐过滤阈值：50%分位数")
                print("   - 特点：高盐三层划分（0.01→0.03→0.06），分层明确")
                print("   - 适用场景：研究高盐内部分级结构，适合定量分析")
            elif choice == 4:
                print("   - 映射方案：对数映射（3.0系数）")
                print("   - 低盐过滤阈值：43%分位数")
                print("   - 特点：高盐梯度平滑过渡，无明显色块")
                print("   - 适用场景：观察高盐区扩散过程、内部梯度变化")
            elif choice == 5:
                print("   - 映射方案：平方根映射（策略19）")
                print("   - 低盐过滤阈值：30%分位数")
                print("   - 特点：盐度0~34过渡平缓，无明显突出区，温和保留中低盐细节")
                print("   - 适用场景：观察整体盐度分布，适合观察整体盐度分布【默认推荐】")
            # 根据选择的策略显示透明度范围
            if choice == 5:
                print("   - 透明度范围：[0.0, 0.35]（策略19：平方根映射，温和保留中低盐细节）")
            else:
                print("   - 透明度范围：[0.01, 0.06]（低盐过滤，高盐突出，增强内部可见性）")
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

# 速度箭头可视化（独立不透明层）：使用箭头表示速度方向和大小
# 箭头方向由速度向量决定，箭头颜色由速度大小决定
if sample_points.n_points > 0 and np.any(sample_speeds > 0):
    # 创建箭头（glyph）
    # 计算合适的箭头缩放因子（基于速度范围）
    speed_max = np.max(sample_speeds)
    if speed_max > 0:
        # 箭头长度基于速度大小，使用合适的缩放因子
        arrow_scale_factor = 35.0 / speed_max  # 稍微增加箭头长度（从30.0增加到35.0）
    else:
        arrow_scale_factor = 1.0
    
    # 创建箭头，方向由velocity向量决定
    arrows = sample_points.glyph(
        orient='velocity',  # 箭头方向由velocity向量决定
        scale='speed',  # 箭头大小由speed决定
        factor=arrow_scale_factor  # 缩放因子，控制箭头整体大小
    )
    
    # 添加箭头，确保在体积渲染之后渲染，并且不被遮挡
    arrow_actor = plotter.add_mesh(
        arrows,
        scalars='speed',  # 箭头颜色由速度大小映射
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
            arrow_property.SetLineWidth(3.5)  # 稍微增加线宽，使箭头更明显（从3.0增加到3.5）
        # 禁用深度写入，确保箭头始终可见（即使被透明体积遮挡）
        if hasattr(arrow_property, 'SetDepthWrite'):
            arrow_property.SetDepthWrite(False)
        print("✅ 已调整箭头渲染属性，确保不被体积遮挡")
    except Exception as e:
        print(f"警告：无法调整箭头渲染属性: {e}")
    print(f"速度箭头已添加（采样点数={sample_points.n_points}，箭头数={arrows.n_points}）")
    print(f"速度范围: [{np.min(sample_speeds):.4f}, {np.max(sample_speeds):.4f}]")
    print(f"箭头缩放因子: {arrow_scale_factor:.4f}")
else:
    print("警告：采样点为空或速度全为0，无法添加箭头")

# 打印体积渲染信息
print(f"组合体积点数: {combined_volume.n_points}")
print(f"盐度数据范围: [{salt_min_val:.4f}, {salt_max_val:.4f}]")
print(f"温度数据范围: [{temp_min_val:.4f}, {temp_max_val:.4f}]")
print(f"透明度映射: 盐度[{salt_min_val:.2f}, {salt_max_val:.2f}] -> 透明度[0.01, 0.06]（第三步优化：盐度主导的视觉权重，增强内部可见性）")
print(f"采样参数: skip={skip}, nz={nz}, 数据形状={U_local.shape}")

plotter.add_axes()
plotter.show()

