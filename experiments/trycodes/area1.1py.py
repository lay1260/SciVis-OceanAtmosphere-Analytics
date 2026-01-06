import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
import OpenVisus as ov

# -----------------------
# 参数设置
# -----------------------
# 区域范围
lat_min, lat_max = 0, 30
lon_min, lon_max = 105, 135
layers = 8  # 上层8层
target_lat = np.linspace(lat_min, lat_max, 60)
target_lon = np.linspace(lon_min, lon_max, 120)

# GEOS 变量 & face
geos_var = 'u'  # 东风分量
geos_face = 0
geos_quality = -6  # 快速原型

# MITgcm 变量
mit_var = 'u'
mit_quality = -9

# -----------------------
# 1. 加载 GEOS 数据
# -----------------------
geos_url = f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{geos_var.upper()}/{geos_var.lower()}_face_{geos_face}_depth_52_time_0_10269.idx"
db_geos = ov.LoadDataset(geos_url)

# 读取第0个时间步
geos_data = db_geos.read(time=0, quality=geos_quality)
print("GEOS shape:", geos_data.shape)  # (depth, lat, lon)

# 只保留表面 + 上层8层
geos_data = geos_data[-layers:, :, :]  # shape: (8, lat, lon)

# -----------------------
# 2. 加载 MITgcm 数据
# -----------------------
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
if mit_var=="theta" or mit_var=="w":
    base_dir=f"mit_output/llc2160_{mit_var}/llc2160_{mit_var}.idx"
elif mit_var=="u":
    base_dir= "mit_output/llc2160_arco/visus.idx"
else:
    base_dir=f"mit_output/llc2160_{mit_var}/{mit_var}_llc2160_x_y_depth.idx"
mit_url = base_url + base_dir

db_mit = ov.LoadDataset(mit_url)
mit_data = db_mit.read(time=0, quality=mit_quality)
print("MITgcm shape:", mit_data.shape)  # (depth, lat, lon)

# -----------------------
# 3. 区域裁剪
# -----------------------
# GEOS经纬度 (假设均匀)
geos_lat = np.linspace(-90, 90, geos_data.shape[1])
geos_lon = np.linspace(-180, 180, geos_data.shape[2])

# 找到索引
lat_idx = np.where((geos_lat >= lat_min) & (geos_lat <= lat_max))[0]
lon_idx = np.where((geos_lon >= lon_min) & (geos_lon <= lon_max))[0]

geos_region = geos_data[:, lat_idx[:, None], lon_idx]  # shape: (layers, lat, lon)

# MITgcm经纬度 (假设均匀)
mit_lat = np.linspace(-90, 90, mit_data.shape[1])
mit_lon = np.linspace(-180, 180, mit_data.shape[2])

lat_mit_idx = np.where((mit_lat >= lat_min) & (mit_lat <= lat_max))[0]
lon_mit_idx = np.where((mit_lon >= lon_min) & (mit_lon <= lon_max))[0]

mit_region = mit_data[:, lat_mit_idx[:, None], lon_mit_idx]

# -----------------------
# 4. 网格重投影到统一 target_lat / target_lon
# -----------------------
# GEOS插值
geos_interp = np.zeros((layers, len(target_lat), len(target_lon)))
lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing='ij')
points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)

for k in range(layers):
    interp_func = RegularGridInterpolator(
        (geos_lat[lat_idx], geos_lon[lon_idx]),
        geos_region[k],
        bounds_error=False,
        fill_value=np.nan
    )
    geos_interp[k] = interp_func(points).reshape(len(target_lat), len(target_lon))

# MITgcm插值
mit_interp = np.zeros_like(geos_interp)
for k in range(layers):
    interp_func = RegularGridInterpolator(
        (mit_lat[lat_mit_idx], mit_lon[lon_mit_idx]),
        mit_region[k],
        bounds_error=False,
        fill_value=np.nan
    )
    mit_interp[k] = interp_func(points).reshape(len(target_lat), len(target_lon))

# -----------------------
# 5. PyVista 3D 可视化
# -----------------------
# 创建 StructuredGrid
X, Y, Z = np.meshgrid(target_lon, target_lat, np.arange(layers))
grid = pv.StructuredGrid(X, Y, Z)

# 添加数据
grid["u_geos"] = geos_interp.ravel()
grid["u_mit"] = mit_interp.ravel()

# 可视化
p = pv.Plotter()
p.add_mesh(grid, scalars="u_geos", cmap="coolwarm", show_edges=False)
p.add_arrows(grid.points, np.c_[geos_interp.ravel(), np.zeros_like(geos_interp.ravel()), np.zeros_like(geos_interp.ravel())], mag=1)
p.show()
