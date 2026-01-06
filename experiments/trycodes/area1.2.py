import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
import OpenVisus as ov
import time

# -----------------------
# 参数设置
# -----------------------
lat_min, lat_max = 0, 30
lon_min, lon_max = 105, 135
layers = 8  # 上层8层
target_lat = np.linspace(lat_min, lat_max, 60)
target_lon = np.linspace(lon_min, lon_max, 120)

geos_var = 'u'
geos_face = 0
geos_quality = -6

mit_var = 'u'
mit_quality = -9

time_steps = 3  # 示例3帧，可按实际需要增加
time_interval = 6  # 每6小时一帧

# -----------------------
# 1. 加载 GEOS 数据
# -----------------------
geos_url = f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{geos_var.upper()}/{geos_var.lower()}_face_{geos_face}_depth_52_time_0_10269.idx"
db_geos = ov.LoadDataset(geos_url)

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

# -----------------------
# 3. 网格坐标
# -----------------------
X, Y, Z = np.meshgrid(target_lon, target_lat, np.arange(layers))
grid = pv.StructuredGrid(X, Y, Z)

# 创建绘图器
plotter = pv.Plotter()
plotter.add_axes()
plotter.add_text("DYAMOND Animation", font_size=14)

# -----------------------
# 4. 动态帧循环
# -----------------------
for t in range(time_steps):
    print(f"Loading timestep {t} ...")

    # GEOS 数据
    geos_data = db_geos.read(time=t, quality=geos_quality)[-layers:, :, :]
    geos_lat = np.linspace(-90, 90, geos_data.shape[1])
    geos_lon = np.linspace(-180, 180, geos_data.shape[2])
    lat_idx = np.where((geos_lat >= lat_min) & (geos_lat <= lat_max))[0]
    lon_idx = np.where((geos_lon >= lon_min) & (geos_lon <= lon_max))[0]
    geos_region = geos_data[:, lat_idx[:, None], lon_idx]

    # 插值到统一网格
    geos_interp = np.zeros((layers, len(target_lat), len(target_lon)))
    points = np.stack(np.meshgrid(target_lat, target_lon, indexing='ij'), axis=-1).reshape(-1,2)
    for k in range(layers):
        interp_func = RegularGridInterpolator(
            (geos_lat[lat_idx], geos_lon[lon_idx]),
            geos_region[k],
            bounds_error=False,
            fill_value=np.nan
        )
        geos_interp[k] = interp_func(points).reshape(len(target_lat), len(target_lon))

    # MITgcm 数据
    mit_data = db_mit.read(time=t, quality=mit_quality)
    mit_lat = np.linspace(-90, 90, mit_data.shape[1])
    mit_lon = np.linspace(-180, 180, mit_data.shape[2])
    lat_mit_idx = np.where((mit_lat >= lat_min) & (mit_lat <= lat_max))[0]
    lon_mit_idx = np.where((mit_lon >= lon_min) & (mit_lon <= lon_max))[0]
    mit_region = mit_data[-layers:, lat_mit_idx[:, None], lon_mit_idx]

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
    # 更新网格数据（新版 PyVista 直接用 grid["name"]）
    # -----------------------
    grid["u_geos"] = geos_interp.ravel()
    grid["u_mit"] = mit_interp.ravel()

    # 清空旧渲染
    plotter.clear()
    plotter.add_mesh(grid, scalars="u_geos", cmap="coolwarm", show_edges=False)

    # 简单箭头示意（大气风场）
    vectors = np.zeros((grid.n_points, 3))
    vectors[:,0] = geos_interp.ravel()  # u 分量
    plotter.add_arrows(grid.points, vectors, mag=1)

    plotter.render()
    time.sleep(0.5)

plotter.show()
