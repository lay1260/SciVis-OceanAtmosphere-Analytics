import numpy as np
import pyvista as pv
import OpenVisus as ov

# ----------------------------
# 配置参数
# ----------------------------
# 区域经纬度裁剪 (GEOS 和 LLC2160)
lon_min, lon_max = 105, 135
lat_min, lat_max = 0, 30

# 动态展示时间步间隔
time_start = 0
time_end = 10  # 示例只取前10个时间步，可根据实际修改
time_step = 1

# 数据分辨率 (OpenVisus quality参数，-15粗, 0全分辨率)
atm_quality = -6
ocn_quality = -9

# ----------------------------
# 加载 GEOS 大气数据（示例 U 风场）
# ----------------------------
atm_variable = "U"
atm_face = 0
atm_url = f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{atm_variable}/{atm_variable.lower()}_face_{atm_face}_depth_52_time_0_10269.idx"
atm_db = ov.LoadDataset(atm_url)

print(f"Atmosphere dataset loaded: {atm_db.getField().name}")
print(f"Timesteps: {len(atm_db.getTimesteps())}")

# ----------------------------
# 加载 LLC2160 海洋数据（示例盐度）
# ----------------------------
ocn_variable = "salt"
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
ocn_dir = f"mit_output/llc2160_{ocn_variable}/{ocn_variable}_llc2160_x_y_depth.idx"
ocn_url = base_url + ocn_dir
ocn_db = ov.LoadDataset(ocn_url)

print(f"Ocean dataset loaded: {ocn_db.getField().name}")
print(f"Timesteps: {len(ocn_db.getTimesteps())}")

# ----------------------------
# 初始化 PyVista Plotter
# ----------------------------
plotter = pv.Plotter()
plotter.open_gif("dyamond_dynamic.gif")  # 可以保存动画

# ----------------------------
# 计算坐标轴缩放比例
# ----------------------------
# 计算实际数据范围
x_range = lon_max - lon_min
y_range = lat_max - lat_min
z_range = 50 - (-200)  # 大气顶部到底部的总高度

# 找到最大范围，让所有轴都缩放到这个范围
max_range = max(x_range, y_range, z_range)

# 计算每个轴需要的缩放比例
x_scale = max_range / x_range
y_scale = max_range / y_range
z_scale = max_range / z_range

# ----------------------------
# 循环时间步渲染
# ----------------------------
for t in range(time_start, time_end, time_step):
    # ----- 读取大气数据 -----
    atm_data = atm_db.read(time=t, quality=atm_quality)
    # 简单降采样，按需
    atm_data_ds = atm_data[::5, ::5, ::5]  # 水平+垂直降采样
    nx, ny, nz = atm_data_ds.shape
    x = np.linspace(lon_min, lon_max, nx)
    y = np.linspace(lat_min, lat_max, ny)
    z = np.linspace(0, 50, nz)  # 假设层高度单位
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # ----- 读取海洋数据 -----
    ocn_data = ocn_db.read(time=t, quality=ocn_quality)
    # 降采样
    ocn_data_ds = ocn_data[::4, ::4, ::5]
    nx_o, ny_o, nz_o = ocn_data_ds.shape
    x_o = np.linspace(lon_min, lon_max, nx_o)
    y_o = np.linspace(lat_min, lat_max, ny_o)
    z_o = np.linspace(0, -200, nz_o)  # 假设深度，海水向下为负
    Xo, Yo, Zo = np.meshgrid(x_o, y_o, z_o, indexing="ij")

    # ----- 构建 PyVista 网格 -----
    atm_grid = pv.StructuredGrid(X, Y, Z)
    atm_grid["U"] = atm_data_ds.flatten(order="F")

    ocn_grid = pv.StructuredGrid(Xo, Yo, Zo)
    ocn_grid["salt"] = ocn_data_ds.flatten(order="F")

    # ----- 渲染 -----
    plotter.clear()
    plotter.add_mesh(ocn_grid, scalars="salt", opacity=0.5, cmap="viridis")
    plotter.add_mesh(atm_grid, scalars="U", opacity=0.5, cmap="coolwarm")

    plotter.add_axes()
    plotter.add_text(f"Timestep: {t}", font_size=20)

    # 设置坐标轴缩放比例，使xyz轴长度一致
    plotter.set_scale(x_scale, y_scale, z_scale)

    plotter.view_isometric()
    plotter.write_frame()  # 保存每帧到 GIF

plotter.close()
print("Animation saved as dyamond_dynamic.gif")