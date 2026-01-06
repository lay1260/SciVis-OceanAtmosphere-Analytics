# 盐度主导视觉权重优化：20种策略组合可视化
# 本脚本实现前5种策略，依次显示每种策略的立方体可视化效果
# 策略说明：覆盖 过滤阈值（低/中/高）× 映射函数（基础/不常见）× 透明度范围（灵活区间）
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
# 3️⃣ 局部区域参数
# ----------------------------
lat_start, lat_end = 10, 40
lon_start, lon_end = 100, 130
nz = 10
data_quality = -6
scale_xy = 25
skip = None  # 采样距离

# ----------------------------
# 4️⃣ 读取局部数据函数
# ----------------------------
def read_data(db, skip_value=None):
    """读取局部数据"""
    if skip_value is None:
        skip_value = skip
    
    data_full = db.read(time=0, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    
    if lat_idx_end <= lat_idx_start or lon_idx_end <= lon_idx_start:
        lat_idx_start = 0
        lat_idx_end = lat_dim
        lon_idx_start = 0
        lon_idx_end = lon_dim
    
    result = data_full[lat_idx_start:lat_idx_end:skip_value,
                       lon_idx_start:lon_idx_end:skip_value,
                       :nz]
    
    if result.size == 0:
        result = data_full[lat_idx_start:lat_idx_end:2,
                           lon_idx_start:lon_idx_end:2,
                           :nz]
    
    return result

U_local = read_data(U_db)
V_local = read_data(V_db)
W_local = read_data(W_db)
Salt_local = read_data(Salt_db)
Theta_local = read_data(Theta_db)

print(f"U_local形状: {U_local.shape}")
print(f"Salt_local形状: {Salt_local.shape}")
print(f"Theta_local形状: {Theta_local.shape}")

nx, ny, nz = U_local.shape
print(f"网格尺寸: nx={nx}, ny={ny}, nz={nz}")

# 检查维度
if nx < 2 or ny < 2 or nz < 2:
    print(f"警告：网格维度不足，尝试使用更小的skip值...")
    new_skip = max(2, skip // 2)
    U_local = read_data(U_db, skip_value=new_skip)
    V_local = read_data(V_db, skip_value=new_skip)
    W_local = read_data(W_db, skip_value=new_skip)
    Salt_local = read_data(Salt_db, skip_value=new_skip)
    Theta_local = read_data(Theta_db, skip_value=new_skip)
    skip = new_skip
    nx, ny, nz = U_local.shape
    print(f"重新加载后网格尺寸: nx={nx}, ny={ny}, nz={nz}")

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
# 6️⃣ 结构化网格 + 速度箭头采样点准备
# ----------------------------
grid = pv.StructuredGrid(X, Y, Z)
vectors = np.stack([U_local.flatten(order="F"),
                    V_local.flatten(order="F"),
                    W_local.flatten(order="F")], axis=1)
grid["velocity"] = vectors

# 在立方体上均匀采样，每条边上10个点，一共10*10*10=1000个点
sampling_points_per_edge = 10

# 计算采样点的索引
# 计算每个维度的实际采样点数（不超过维度大小）
n_samples_x = min(sampling_points_per_edge, nx)
n_samples_y = min(sampling_points_per_edge, ny)
n_samples_z = min(sampling_points_per_edge, nz)

# 生成均匀分布的索引（包括边界和内部）
if nx > 1:
    if nx > 2 and n_samples_x > 2:
        x_indices = np.unique(np.concatenate([
            [0],
            np.linspace(1, nx-2, max(1, n_samples_x-2), dtype=int),
            [nx-1]
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
        y_indices = np.unique(np.concatenate([
            [0],
            np.linspace(1, ny-2, max(1, n_samples_y-2), dtype=int),
            [ny-1]
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
        z_indices = np.unique(np.concatenate([
            [0],
            np.linspace(1, nz-2, max(1, n_samples_z-2), dtype=int),
            [nz-1]
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

# 创建采样点的网格
X_idx, Y_idx, Z_idx = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
X_idx = X_idx.flatten()
Y_idx = Y_idx.flatten()
Z_idx = Z_idx.flatten()

# 获取速度向量和采样点坐标
sample_velocities = []
sample_speeds = []
sample_points_coords = []

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
    sample_velocities.append(vel)
    sample_speeds.append(speed)
    
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

# ----------------------------
# 7️⃣ 准备温度盐度数据
# ----------------------------
theta_data = Theta_local.flatten(order="F")
salt_data = Salt_local.flatten(order="F")

# 计算盐度梯度
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

print(f"盐度梯度范围: [{salt_gradient_mag.min():.4f}, {salt_gradient_mag.max():.4f}]")
print(f"盐度数据范围: [{salt_data.min():.4f}, {salt_data.max():.4f}]")
print(f"温度数据范围: [{theta_data.min():.4f}, {theta_data.max():.4f}]")

# 计算数据范围
salt_min_val = np.min(salt_data)
salt_max_val = np.max(salt_data)
temp_min_val = np.min(theta_data)
temp_max_val = np.max(theta_data)

# ----------------------------
# 8️⃣ 定义20种透明度映射策略函数
# ----------------------------

def opacity_strategy_1(salt_data, salt_gradient_norm):
    """策略1：低（20%）阈值 + 基础-线性 + 0.02~0.25透明度
    需保留中低盐细节，不刻意放大差异
    """
    salt_threshold = np.percentile(salt_data, 20)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0.02 + 0.23 * salt_norm
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.25)
    return final_opacity

def opacity_strategy_2(salt_data, salt_gradient_norm):
    """策略2：低（30%）阈值 + 基础-幂函数（2.0次）+ 0~0.3透明度
    低过滤 + 平衡透明，兼顾细节与突出
    """
    salt_threshold = np.percentile(salt_data, 30)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.3 * (salt_norm ** 2.0)
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.3)
    return final_opacity

def opacity_strategy_3(salt_data, salt_gradient_norm):
    """策略3：低（25%）阈值 + 基础-指数（3.0系数）+ 0.02~0.2透明度
    低过滤 + 通透，避免中低盐叠加变浊
    """
    salt_threshold = np.percentile(salt_data, 25)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0.02 + 0.18 * (1 - np.exp(-3.0 * salt_norm))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.2)
    return final_opacity

def opacity_strategy_4(salt_data, salt_gradient_norm):
    """策略4：低（30%）阈值 + 基础-对数（2.5系数）+ 0~0.35透明度
    低过滤 + 高透明上限，突出中高盐梯度
    """
    salt_threshold = np.percentile(salt_data, 30)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 1e-6, 1.0)
    base_opacity = 0 + 0.35 * (np.log(1 + 2.5 * salt_norm) / np.log(3.5))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.35)
    return final_opacity

def opacity_strategy_5(salt_data, salt_gradient_norm):
    """策略5：中（40%）阈值 + 不常见-双曲正切（tanh，4.0系数）+ 0.02~0.25透明度
    中过滤 + 平衡透明，强化中间盐度过渡
    """
    salt_threshold = np.percentile(salt_data, 40)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0.02 + 0.23 * (np.tanh(4.0 * salt_norm) + 1) / 2
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.25)
    return final_opacity

def opacity_strategy_6(salt_data, salt_gradient_norm):
    """策略6：中（50%）阈值 + 不常见-Sigmoid（5.0系数）+ 0~0.4透明度
    中过滤 + 高透明上限，聚焦中高盐核心
    """
    salt_threshold = np.percentile(salt_data, 50)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.4 / (1 + np.exp(-5.0 * (salt_norm - 0.5)))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.4)
    return final_opacity

def opacity_strategy_7(salt_data, salt_gradient_norm):
    """策略7：中（45%）阈值 + 基础-幂函数（3.0次）+ 0.02~0.3透明度
    中过滤 + 高透明，放大高盐差异
    """
    salt_threshold = np.percentile(salt_data, 45)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0.02 + 0.28 * (salt_norm ** 3.0)
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.3)
    return final_opacity

def opacity_strategy_8(salt_data, salt_gradient_norm):
    """策略8：中（50%）阈值 + 不常见-反函数（1.0系数）+ 0~0.25透明度
    中过滤 + 通透，反向突出高盐（避免过曝）
    """
    salt_threshold = np.percentile(salt_data, 50)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.25 * (1 - 1 / (1 + 2.0 * salt_norm))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.25)
    return final_opacity

def opacity_strategy_9(salt_data, salt_gradient_norm):
    """策略9：中（42%）阈值 + 基础-线性分段（3段）+ 0.02~0.25透明度
    中过滤 + 精准分层，适合定量分析
    """
    salt_threshold = np.percentile(salt_data, 42)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    
    # 分段线性映射：[0→0.02, 0.3→0.15, 1.0→0.25]
    base_opacity = np.zeros_like(salt_data)
    base_opacity[salt_norm < 0.3] = np.interp(salt_norm[salt_norm < 0.3], [0, 0.3], [0.02, 0.15])
    base_opacity[salt_norm >= 0.3] = np.interp(salt_norm[salt_norm >= 0.3], [0.3, 1.0], [0.15, 0.25])
    
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.25)
    return final_opacity

def opacity_strategy_10(salt_data, salt_gradient_norm):
    """策略10：中（48%）阈值 + 不常见-平方根倒数（1.0系数）+ 0~0.35透明度
    中过滤 + 高透明，温和突出高盐
    """
    salt_threshold = np.percentile(salt_data, 48)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.35 * (1 - np.sqrt(1 / (1 + 3.0 * salt_norm)))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.35)
    return final_opacity

def opacity_strategy_11(salt_data, salt_gradient_norm):
    """策略11：高（70%）阈值 + 基础-指数（6.0系数）+ 0~0.2透明度
    高过滤 + 极致通透，仅突出顶级高盐
    """
    salt_threshold = np.percentile(salt_data, 70)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.2 * (1 - np.exp(-6.0 * salt_norm))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.2)
    return final_opacity

def opacity_strategy_12(salt_data, salt_gradient_norm):
    """策略12：高（75%）阈值 + 不常见-tanh + 幂函数（组合）+ 0.02~0.3透明度
    高过滤 + 高透明，双重放大高盐差异
    """
    salt_threshold = np.percentile(salt_data, 75)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0.02 + 0.28 * (np.tanh(3.0 * salt_norm ** 2) + 1) / 2
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.3)
    return final_opacity

def opacity_strategy_13(salt_data, salt_gradient_norm):
    """策略13：高（80%）阈值 + 基础-幂函数（4.0次）+ 0~0.25透明度
    高过滤 + 通透，聚焦极少数顶级高盐点
    """
    salt_threshold = np.percentile(salt_data, 80)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.25 * (salt_norm ** 4.0)
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.25)
    return final_opacity

def opacity_strategy_14(salt_data, salt_gradient_norm):
    """策略14：高（72%）阈值 + 不常见-对数 + 指数（组合）+ 0.02~0.25透明度
    高过滤 + 平衡透明，保留高盐梯度
    """
    salt_threshold = np.percentile(salt_data, 72)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 1e-6, 1.0)
    base_opacity = 0.02 + 0.23 * np.log(1 + 3.0 * (1 - np.exp(-4.0 * salt_norm)))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.25)
    return final_opacity

def opacity_strategy_15(salt_data, salt_gradient_norm):
    """策略15：高（78%）阈值 + 基础-线性 + 0~0.2透明度
    高过滤 + 线性基准，对比非线性效果
    """
    salt_threshold = np.percentile(salt_data, 78)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.2 * salt_norm
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.2)
    return final_opacity

def opacity_strategy_16(salt_data, salt_gradient_norm):
    """策略16：低（20%）阈值 + 不常见-Sigmoid（3.0系数）+ 0.02~0.3透明度
    低过滤 + 高透明，突出中高盐混合区
    """
    salt_threshold = np.percentile(salt_data, 20)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0.02 + 0.28 / (1 + np.exp(-3.0 * (salt_norm - 0.6)))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.3)
    return final_opacity

def opacity_strategy_17(salt_data, salt_gradient_norm):
    """策略17：高（80%）阈值 + 不常见-反幂函数（0.8次）+ 0~0.3透明度
    高过滤 + 高透明，温和突出高盐（避免尖锐）
    """
    salt_threshold = np.percentile(salt_data, 80)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.3 * (salt_norm ** 0.8)
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.3)
    return final_opacity

def opacity_strategy_18(salt_data, salt_gradient_norm):
    """策略18：中（45%）阈值 + 不常见-双曲正弦（sinh，2.0系数）+ 0.02~0.25透明度
    中过滤 + 平衡透明，强化高盐区细节
    """
    salt_threshold = np.percentile(salt_data, 45)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0.02 + 0.23 * (np.sinh(2.0 * salt_norm) / np.sinh(2.0))
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.25)
    return final_opacity

def opacity_strategy_19(salt_data, salt_gradient_norm):
    """策略19：低（30%）阈值 + 不常见-平方根（1.0系数）+ 0~0.35透明度
    低过滤 + 高透明，温和保留中低盐细节
    """
    salt_threshold = np.percentile(salt_data, 30)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    base_opacity = 0 + 0.35 * np.sqrt(salt_norm)
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.0, 0.35)
    return final_opacity

def opacity_strategy_20(salt_data, salt_gradient_norm):
    """策略20：高（75%）阈值 + 基础-指数分段（2段）+ 0.02~0.3透明度
    高过滤 + 高透明，精准聚焦极高盐核心
    """
    salt_threshold = np.percentile(salt_data, 75)
    salt_norm = np.clip((salt_data - salt_threshold) / (salt_data.max() - salt_threshold), 0.0, 1.0)
    
    # 分段指数：salt_norm<0.5（系数 3.0），salt_norm≥0.5（系数 6.0）
    base_opacity = np.zeros_like(salt_data)
    low_mask = salt_norm < 0.5
    high_mask = salt_norm >= 0.5
    
    base_opacity[low_mask] = 0.02 + 0.28 * (1 - np.exp(-3.0 * salt_norm[low_mask]))
    # 高段需要从低段的终点开始
    low_end_value = 0.02 + 0.28 * (1 - np.exp(-3.0 * 0.5))
    high_range = 0.3 - low_end_value
    base_opacity[high_mask] = low_end_value + high_range * (1 - np.exp(-6.0 * (salt_norm[high_mask] - 0.5)))
    
    gradient_boost = 0.1 + 0.2 * salt_gradient_norm
    final_opacity = np.clip(base_opacity * gradient_boost, 0.02, 0.3)
    return final_opacity

# 策略函数列表（所有20种策略）
strategy_functions = [
    opacity_strategy_1,
    opacity_strategy_2,
    opacity_strategy_3,
    opacity_strategy_4,
    opacity_strategy_5,
    opacity_strategy_6,
    opacity_strategy_7,
    opacity_strategy_8,
    opacity_strategy_9,
    opacity_strategy_10,
    opacity_strategy_11,
    opacity_strategy_12,
    opacity_strategy_13,
    opacity_strategy_14,
    opacity_strategy_15,
    opacity_strategy_16,
    opacity_strategy_17,
    opacity_strategy_18,
    opacity_strategy_19,
    opacity_strategy_20
]

# 策略描述列表（所有20种策略）
strategy_descriptions = [
    "策略1：低（20%）阈值 + 基础-线性 + 0.02~0.25透明度",
    "策略2：低（30%）阈值 + 基础-幂函数（2.0次）+ 0~0.3透明度",
    "策略3：低（25%）阈值 + 基础-指数（3.0系数）+ 0.02~0.2透明度",
    "策略4：低（30%）阈值 + 基础-对数（2.5系数）+ 0~0.35透明度",
    "策略5：中（40%）阈值 + 不常见-双曲正切（tanh，4.0系数）+ 0.02~0.25透明度",
    "策略6：中（50%）阈值 + 不常见-Sigmoid（5.0系数）+ 0~0.4透明度",
    "策略7：中（45%）阈值 + 基础-幂函数（3.0次）+ 0.02~0.3透明度",
    "策略8：中（50%）阈值 + 不常见-反函数（1.0系数）+ 0~0.25透明度",
    "策略9：中（42%）阈值 + 基础-线性分段（3段）+ 0.02~0.25透明度",
    "策略10：中（48%）阈值 + 不常见-平方根倒数（1.0系数）+ 0~0.35透明度",
    "策略11：高（70%）阈值 + 基础-指数（6.0系数）+ 0~0.2透明度",
    "策略12：高（75%）阈值 + 不常见-tanh+幂函数（组合）+ 0.02~0.3透明度",
    "策略13：高（80%）阈值 + 基础-幂函数（4.0次）+ 0~0.25透明度",
    "策略14：高（72%）阈值 + 不常见-对数+指数（组合）+ 0.02~0.25透明度",
    "策略15：高（78%）阈值 + 基础-线性 + 0~0.2透明度",
    "策略16：低（20%）阈值 + 不常见-Sigmoid（3.0系数）+ 0.02~0.3透明度",
    "策略17：高（80%）阈值 + 不常见-反幂函数（0.8次）+ 0~0.3透明度",
    "策略18：中（45%）阈值 + 不常见-双曲正弦（sinh，2.0系数）+ 0.02~0.25透明度",
    "策略19：低（30%）阈值 + 不常见-平方根（1.0系数）+ 0~0.35透明度",
    "策略20：高（75%）阈值 + 基础-指数分段（2段）+ 0.02~0.3透明度"
]

# ----------------------------
# 9️⃣ 依次显示20种策略的可视化
# ----------------------------
print("\n" + "="*60)
print("开始依次显示20种透明度映射策略的可视化效果")
print("="*60)

for strategy_idx, (strategy_func, description) in enumerate(zip(strategy_functions, strategy_descriptions), 1):
    print(f"\n正在渲染 {description}...")
    
    # 计算该策略的透明度
    final_opacity = strategy_func(salt_data, salt_gradient_norm)
    
    print(f"透明度范围: [{final_opacity.min():.4f}, {final_opacity.max():.4f}]")
    print(f"透明度统计: 均值={final_opacity.mean():.4f}, 中位数={np.median(final_opacity):.4f}")
    
    # 创建组合体积
    combined_volume = pv.StructuredGrid(X, Y, Z)
    combined_volume["Temperature"] = theta_data
    combined_volume["Salinity"] = salt_data
    
    # 创建Plotter
    plotter = pv.Plotter(window_size=(1400, 900), title=f"{description}")
    
    try:
        plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        print("✅ 已启用深度剥离")
    except Exception as e:
        print(f"警告：无法启用深度剥离: {e}")
    
    # 使用VTK底层API实现双标量独立控制
    if combined_volume.n_points > 0 and not np.isnan(theta_data).all() and not np.isnan(salt_data).all():
        if VTK_AVAILABLE:
            try:
                # 添加体积
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
                    pickable=False,
                    blending='composite'
                )
                
                # 获取VTK Volume和VolumeProperty
                mapper = volume_actor.GetMapper()
                vtk_volume = mapper.GetInput()
                volume_property = volume_actor.GetProperty()
                
                # 确保盐度数据在PointData中
                salt_vtk_array = vtk_volume.GetPointData().GetArray("Salinity")
                if salt_vtk_array is None:
                    salt_vtk_array = numpy_to_vtk(
                        salt_data.astype(np.float32),
                        array_type=vtk.VTK_FLOAT
                    )
                    salt_vtk_array.SetName("Salinity")
                    vtk_volume.GetPointData().AddArray(salt_vtk_array)
                
                # 创建透明度传递函数（基于温度值，但使用计算好的透明度）
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
                
                # 设置透明度函数
                volume_property.SetScalarOpacity(opacity_func)
                volume_property.SetScalarOpacityUnitDistance(5.0)
                
                # 自适应颜色映射（基于分位数）
                temp_percentile_5 = np.percentile(theta_data, 5)
                temp_percentile_95 = np.percentile(theta_data, 95)
                
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
                
                print(f"✅ 策略{strategy_idx}体积渲染设置完成")
                print("   【第一步优化：单体积双标量绑定】")
                print("   - 颜色：由温度标量独立控制（自适应颜色映射，hot_r色图）")
                print("   - 透明度：由盐度标量独立控制（基于策略函数）")
                print("   【第二步优化：自适应颜色映射】")
                print(f"   - 分位数拉伸：使用5%-95%分位数范围 [{temp_percentile_5:.2f}, {temp_percentile_95:.2f}]")
                print("   - 颜色映射：hot_r色图，中间区域增强对比度")
                
            except Exception as e:
                print(f"警告：VTK底层API设置失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("警告：VTK不可用，使用PyVista高层API")
    
    # 速度箭头可视化（独立不透明层）：使用箭头表示速度方向和大小
    # 箭头方向由速度向量决定，箭头颜色由速度大小决定
    if sample_points.n_points > 0 and np.any(sample_speeds > 0):
        # 创建箭头（glyph）
        speed_max = np.max(sample_speeds)
        if speed_max > 0:
            arrow_scale_factor = 35.0 / speed_max  # 稍微增加箭头长度
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
            arrow_property = arrow_actor.GetProperty()
            arrow_property.SetOpacity(1.0)
            if hasattr(arrow_property, 'SetRenderLinesAsTubes'):
                arrow_property.SetRenderLinesAsTubes(True)
            if hasattr(arrow_property, 'SetLineWidth'):
                arrow_property.SetLineWidth(3.5)  # 稍微增加线宽，使箭头更明显
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
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 显示
    print(f"显示策略{strategy_idx}的可视化窗口...")
    print("关闭窗口后将继续下一个策略...")
    plotter.show()
    
    print(f"✅ 策略{strategy_idx}显示完成\n")

print("="*60)
print("所有20种策略的可视化已完成！")
print("="*60)

