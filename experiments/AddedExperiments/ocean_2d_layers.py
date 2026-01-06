# 2D Ocean Plane Visualization: 6 Depth Layers Combined in One Figure
# - Scalar Field: Temperature mapped to color (hot colormap), Salinity mapped to transparency (linear, 0~1)
# - Vector Field: Bent arrows visualize horizontal velocity (U and V combined)
# - Control reading density via quality parameter (-4), balancing resolution and memory
import OpenVisus as ov
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os

# 矢量场优化：模式1 - 弯曲箭头所需依赖
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告：SciPy不可用，将使用直线箭头（模式1优化需要SciPy）")

# 矢量场优化：模式2 - 流线所需依赖
try:
    from scipy.integrate import solve_ivp
    SCIPY_INTEGRATE_AVAILABLE = True
except ImportError:
    SCIPY_INTEGRATE_AVAILABLE = False
    print("警告：scipy.integrate不可用，模式2（流线）将不可用")

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
print("Loading datasets...")
U_db = load_dataset("u")
V_db = load_dataset("v")
Salt_db = load_dataset("salt")
Theta_db = load_dataset("theta")

# ----------------------------
# 3️⃣ 局部区域参数
# ----------------------------
lat_start, lat_end = 10, 40
lon_start, lon_end = 100, 130
nz = 10  # 深度层数
data_quality = -4  # 降低读取密度（-4比0低，但仍保持较高分辨率）
scale_xy = 25
# 不再使用skip，但通过quality参数控制分辨率

# ----------------------------
# 4️⃣ 读取局部数据函数（全量数据，不采样）
# ----------------------------
def read_data(db):
    """读取局部数据（全量，不采样）"""
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
    
    # 不使用skip，读取全量数据
    result = data_full[lat_idx_start:lat_idx_end,
                       lon_idx_start:lon_idx_end,
                       :nz]
    
    return result

print("Reading data...")
U_local = read_data(U_db)
V_local = read_data(V_db)
Salt_local = read_data(Salt_db)
Theta_local = read_data(Theta_db)

nx, ny, nz = U_local.shape
print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")

# ----------------------------
# 5️⃣ 构建2D网格坐标
# ----------------------------
x = np.linspace(lon_start, lon_end, ny) * scale_xy
y = np.linspace(lat_start, lat_end, nx) * scale_xy
X_2d, Y_2d = np.meshgrid(x, y, indexing='ij')
X_2d = X_2d.transpose(1, 0)
Y_2d = Y_2d.transpose(1, 0)

# ----------------------------
# 6️⃣ 盐度透明度映射函数（线性映射，0~1）
# ----------------------------
def opacity_mapping_linear(salt_data):
    """线性透明度映射：盐度范围映射到0~1透明度"""
    salt_min = salt_data.min()
    salt_max = salt_data.max()
    if salt_max > salt_min:
        # 线性归一化到0~1
        opacity = (salt_data - salt_min) / (salt_max - salt_min)
    else:
        opacity = np.ones_like(salt_data) * 0.5  # 如果盐度值相同，使用中等透明度
    return np.clip(opacity, 0.0, 1.0)

# ----------------------------
# 7️⃣ 2D弯曲箭头生成函数
# ----------------------------
def create_2d_bent_arrows(x_coords, y_coords, u_vel, v_vel, speeds, arrow_scale=50.0, k_neighbors=4):
    """生成2D弯曲箭头（改进：确保longitude和latitude方向均匀采样）"""
    if not SCIPY_AVAILABLE:
        return None
    
    # 获取坐标范围和唯一值
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 改进：在longitude和latitude方向均匀采样，确保密度适中且覆盖完整
    # 目标：每个方向约12-15个采样点，总约144-225个箭头（降低密度）
    n_x_samples = 12  # longitude方向采样点数（降低密度）
    n_y_samples = 12  # latitude方向采样点数（降低密度）
    
    # 生成均匀网格采样点（确保覆盖整个范围，包括latitude 700-1000）
    x_samples = np.linspace(x_min, x_max, n_x_samples)
    y_samples = np.linspace(y_min, y_max, n_y_samples)  # 确保覆盖整个latitude范围
    X_samples, Y_samples = np.meshgrid(x_samples, y_samples)
    sample_points_2d = np.column_stack([X_samples.flatten(), Y_samples.flatten()])
    
    # 对每个采样点，找到最近网格点的速度
    sample_vels = []
    sample_speeds = []
    valid_sample_points = []
    
    # 确保速度数据形状正确（u_vel和v_vel应该是(nx, ny)形状）
    # x_coords和y_coords是(nx, ny)形状
    nx, ny = x_coords.shape
    if u_vel.shape != (nx, ny):
        # 如果形状不匹配，尝试转置
        if u_vel.shape == (ny, nx):
            u_vel = u_vel.T
            v_vel = v_vel.T
            speeds = speeds.T
        else:
            print(f"    警告：速度数据形状不匹配 u_vel.shape={u_vel.shape}, x_coords.shape={x_coords.shape}")
    
    for sp in sample_points_2d:
        # 找到最近的网格点索引（在x_coords和y_coords中）
        # x_coords是(nx, ny)，第一维是latitude，第二维是longitude
        # 需要找到对应的索引
        x_dist = np.abs(x_coords - sp[0])
        y_dist = np.abs(y_coords - sp[1])
        total_dist = x_dist + y_dist
        min_idx = np.unravel_index(np.argmin(total_dist), total_dist.shape)
        y_idx, x_idx = min_idx  # y_idx是latitude索引，x_idx是longitude索引
        
        # 确保索引在有效范围内
        y_idx = np.clip(y_idx, 0, nx-1)
        x_idx = np.clip(x_idx, 0, ny-1)
        
        # 获取对应的速度
        u_val = u_vel[y_idx, x_idx]
        v_val = v_vel[y_idx, x_idx]
        speed_val = speeds[y_idx, x_idx]
        
        # 过滤速度过小的点
        if speed_val > np.percentile(speeds.flatten(), 5):  # 只保留速度大于5%分位数的点
            valid_sample_points.append(sp)
            sample_vels.append([u_val, v_val])
            sample_speeds.append(speed_val)
    
    if len(valid_sample_points) == 0:
        return None
    
    sample_points = np.array(valid_sample_points)
    sample_vels = np.array(sample_vels)
    sample_speeds = np.array(sample_speeds)
    
    arrows = []
    speed_max = np.max(sample_speeds) if len(sample_speeds) > 0 else 1.0
    
    # 增大箭头缩放因子
    arrow_scale_factor = arrow_scale * 1.5  # 增大箭头大小
    
    for i in range(len(sample_points)):
        try:
            current_point = sample_points[i]
            current_vel = sample_vels[i]
            speed = sample_speeds[i]
            
            if speed < 0.01 * speed_max:
                continue
            
            # 获取邻域点
            distances = np.linalg.norm(sample_points - current_point, axis=1)
            neighbor_indices = np.argsort(distances)[:k_neighbors]
            neighbor_points = sample_points[neighbor_indices]
            neighbor_vels = sample_vels[neighbor_indices]
            
            # 平滑速度场
            smoothed_vels = []
            for j in range(len(neighbor_vels)):
                weights = np.exp(-distances[neighbor_indices[j]]**2 / (2 * 1.0**2))
                smoothed_vels.append(neighbor_vels[j] * weights)
            smoothed_vels = np.array(smoothed_vels)
            avg_vel = np.mean(smoothed_vels, axis=0)
            
            # 生成曲线点
            num_points = 5
            total_length = speed * arrow_scale_factor / speed_max  # 使用增大的缩放因子
            curve_points = []
            current_pos = current_point.copy()
            
            for j in range(num_points):
                t = j / (num_points - 1) if num_points > 1 else 0
                dir_vec = (1-t) * current_vel + t * avg_vel
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-6:
                    dir_vec = dir_vec / dir_norm
                else:
                    dir_vec = current_vel / (np.linalg.norm(current_vel) + 1e-6)
                
                step = dir_vec * (total_length / (num_points - 1)) if num_points > 1 else dir_vec * total_length
                current_pos = current_pos + step
                curve_points.append(current_pos.copy())
            
            # 创建2D箭头（使用matplotlib的quiver）
            if len(curve_points) >= 2:
                # 计算箭头方向
                arrow_dir = curve_points[-1] - curve_points[0]
                arrow_norm = np.linalg.norm(arrow_dir)
                if arrow_norm > 1e-6:
                    arrows.append({
                        'pos': current_point,
                        'dir': arrow_dir / arrow_norm,
                        'length': arrow_norm,
                        'speed': speed
                    })
        except Exception:
            continue
    
    print(f"    Generated {len(arrows)} arrows (uniformly distributed in longitude and latitude directions)")
    return arrows

# ----------------------------
# 8️⃣ 2D流线生成函数（参考velocity_3D.py，使用PyVista的streamlines_from_source）
# ----------------------------
def create_2d_streamlines(x_coords, y_coords, u_vel, v_vel, n_seeds=100):
    """生成2D流线（参考velocity_3D.py，使用PyVista的streamlines_from_source）"""
    try:
        # 1. 创建2D结构化网格（参考velocity_3D.py）
        # 添加一个虚拟的z维度（z=0），创建3D网格以便使用PyVista的流线功能
        z_coords = np.zeros_like(x_coords)
        
        # 创建3D网格（z维度为0）
        grid_2d = pv.StructuredGrid(x_coords, y_coords, z_coords)
        
        # 2. 添加速度向量（需要3D向量，W分量为0）
        # 确保速度数据形状正确
        if u_vel.shape != x_coords.shape:
            # 如果形状不匹配，尝试转置
            u_vel = u_vel.T if u_vel.T.shape == x_coords.shape else u_vel
            v_vel = v_vel.T if v_vel.T.shape == y_coords.shape else v_vel
        
        # 展平速度数据
        u_flat = u_vel.flatten(order='F')
        v_flat = v_vel.flatten(order='F')
        w_flat = np.zeros_like(u_flat)  # W分量为0（2D流线）
        
        # 创建3D速度向量
        vectors = np.stack([u_flat, v_flat, w_flat], axis=1)
        grid_2d["velocity"] = vectors
        
        # 3. 生成种子点（参考velocity_3D.py，使用均匀采样）
        # 种子点间隔（根据网格大小自适应）
        stride = max(1, int(np.sqrt(grid_2d.n_points / n_seeds)))
        seed_points = pv.PolyData(grid_2d.points[::stride])
        
        print(f"    Seed points: {seed_points.n_points} (stride: {stride})")
        
        # 4. 生成流线（参考velocity_3D.py的参数）
        streamlines = grid_2d.streamlines_from_source(
            source=seed_points,
            vectors='velocity',
            integration_direction='both',  # 双向积分，确保连贯性
            initial_step_length=2.0,  # 适中的初始步长
            terminal_speed=1e-3,
            max_steps=2000  # 增加最大步数，确保流线足够长
        )
        
        # 5. 给流线添加速度标量（参考velocity_3D.py）
        if 'velocity' in streamlines.array_names:
            speed = np.linalg.norm(streamlines['velocity'], axis=1)
            streamlines['speed'] = speed
        elif 'vectors' in streamlines.array_names:
            speed = np.linalg.norm(streamlines['vectors'], axis=1)
            streamlines['speed'] = speed
        
        # 6. 转换为2D点列表（移除z坐标）
        streamlines_2d = []
        if streamlines.n_points > 0:
            # 获取流线点（移除z坐标）
            points_3d = streamlines.points
            points_2d = points_3d[:, :2]  # 只取x和y坐标
            
            # 根据流线的lines信息分割成单独的流线
            if hasattr(streamlines, 'lines') and len(streamlines.lines) > 0:
                lines = streamlines.lines
                offset = 0
                i = 0
                while i < len(lines):
                    if i < len(lines):
                        n_points_in_line = lines[i]
                        if n_points_in_line > 0 and offset + n_points_in_line <= len(points_2d):
                            line_points = points_2d[offset:offset+n_points_in_line]
                            if len(line_points) > 5:  # 只保留足够长的流线
                                streamlines_2d.append(line_points)
                            offset += n_points_in_line
                            i += n_points_in_line + 1
                        else:
                            i += 1
                    else:
                        break
            else:
                # 如果没有lines信息，将所有点作为一条流线
                if len(points_2d) > 5:
                    streamlines_2d.append(points_2d)
        
        print(f"    Successfully generated {len(streamlines_2d)} streamlines (using PyVista streamlines_from_source)")
        return streamlines_2d
        
    except Exception as e:
        print(f"    Warning: PyVista streamline generation failed: {e}")
        return None

# ----------------------------
# 9️⃣ 可视化单层子图函数（用于6层合并显示）
# ----------------------------
def visualize_single_layer_subplot(ax, layer_idx, u_layer, v_layer, salt_layer, theta_layer, 
                                   speed_layer, X_2d, Y_2d):
    """在子图中可视化单层海洋平面（只显示弯曲箭头模式，所有文字为英文）"""
    # 计算透明度（线性映射，0~1）
    opacity_layer = opacity_mapping_linear(salt_layer)
    
    # 创建RGBA图像（温度颜色 + 盐度透明度）
    temp_norm = mcolors.Normalize(vmin=theta_layer.min(), vmax=theta_layer.max())
    temp_colors = plt.cm.hot_r(temp_norm(theta_layer))
    temp_colors[:, :, 3] = opacity_layer  # 设置alpha通道（盐度映射）
    
    # 使用imshow显示带透明度的图像，使用'bicubic'插值确保过渡更平滑
    ax.imshow(temp_colors, extent=[X_2d.min(), X_2d.max(), Y_2d.min(), Y_2d.max()], 
              origin='lower', aspect='auto', interpolation='bicubic', 
              filternorm=True, filterrad=4.0)
    
    # 添加弯曲箭头
    arrows = create_2d_bent_arrows(X_2d, Y_2d, u_layer, v_layer, speed_layer, arrow_scale=40.0)
    if arrows:
        for arrow in arrows:
            pos = arrow['pos']
            dir_vec = arrow['dir']
            length = arrow['length']
            speed = arrow['speed']
            
            # 绘制箭头（增大箭头大小）
            ax.arrow(pos[0], pos[1], dir_vec[0]*length*0.9, dir_vec[1]*length*0.9,
                    head_width=length*0.25, head_length=length*0.3,
                    fc='cyan', ec='cyan', alpha=0.8, linewidth=2.0)
    
    # 设置标签和标题（英文）
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title(f'Layer {layer_idx+1} (Depth Index {layer_idx})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条（使用ScalarMappable）
    sm = plt.cm.ScalarMappable(cmap='hot_r', norm=temp_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature', fontsize=9)

# ----------------------------
# 🔟 可视化所有6层（集中在一张图中）
# ----------------------------
def visualize_all_layers(output_dir="ocean_2d_layers_output"):
    """可视化所有6层，集中在一张图中（2x3布局）"""
    print("\n" + "="*60)
    print("Generating 6-layer ocean plane visualization (all in one figure)")
    print("="*60)
    
    # 选择6个深度层（均匀分布）
    layer_indices = np.linspace(0, nz-1, 6, dtype=int)
    print(f"Selected depth layer indices: {layer_indices}")
    
    # 创建图形（2x3布局，6个子图）
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()  # 展平为1D数组便于索引
    
    # 为每一层创建子图
    for i, layer_idx in enumerate(layer_indices):
        print(f"\nProcessing Layer {layer_idx+1} (Depth Index {layer_idx})...")
        
        # 提取该层数据
        u_layer = U_local[:, :, layer_idx]
        v_layer = V_local[:, :, layer_idx]
        salt_layer = Salt_local[:, :, layer_idx]
        theta_layer = Theta_local[:, :, layer_idx]
        
        # 计算水平速度大小
        speed_layer = np.sqrt(u_layer**2 + v_layer**2)
        
        print(f"  Temperature range: [{theta_layer.min():.4f}, {theta_layer.max():.4f}]")
        print(f"  Salinity range: [{salt_layer.min():.4f}, {salt_layer.max():.4f}]")
        print(f"  Speed range: [{speed_layer.min():.4f}, {speed_layer.max():.4f}]")
        
        # 在对应的子图中可视化
        visualize_single_layer_subplot(
            ax=axes[i],
            layer_idx=layer_idx,
            u_layer=u_layer,
            v_layer=v_layer,
            salt_layer=salt_layer,
            theta_layer=theta_layer,
            speed_layer=speed_layer,
            X_2d=X_2d,
            Y_2d=Y_2d
        )
    
    # 添加总标题（英文）
    fig.suptitle('Ocean Plane Visualization: 6 Depth Layers\n(Temperature: Color, Salinity: Transparency, Velocity: Bent Arrows)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 为总标题留出空间
    
    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ocean_6_layers_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved combined image: {output_path}")
    plt.close()

# ----------------------------
# 🔟 主程序：可视化6层（集中在一张图中）
# ----------------------------
# 创建输出目录
output_dir = "ocean_2d_layers_output"
os.makedirs(output_dir, exist_ok=True)

# 可视化所有6层，集中在一张图中
visualize_all_layers(output_dir=output_dir)

print(f"\n{'='*60}")
print(f"✅ All 6 layers visualization completed. Saved to: {output_dir}")
print(f"{'='*60}")

