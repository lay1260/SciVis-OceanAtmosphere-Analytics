import OpenVisus as ov
import numpy as np
import pyvista as pv
import time

# ----------------------------
# 1️⃣ 在线加载数据集
# ----------------------------
variable = "salt"  # 可选 u,v,w,theta,salt
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

if variable in ["theta", "w"]:
    base_dir = f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
elif variable == "u":
    base_dir = "mit_output/llc2160_arco/visus.idx"
else:
    base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"

dataset_url = base_url + base_dir
db = ov.LoadDataset(dataset_url)

print(f'Dimensions: {db.getLogicBox()[1]}')
print(f'Total Timesteps: {len(db.getTimesteps())}')
print(f'Field: {db.getField().name}')

# ----------------------------
# 2️⃣ 局部区域参数
# ----------------------------
lat_start, lat_end = 0, 40
lon_start, lon_end = 100, 140
nz = 8  # 前8层
data_quality = -9  # 在线读取分辨率
scale_xy = 25  # X/Y 缩放系数，使比例与深度匹配


# ----------------------------
# 3️⃣ 读取局部数据函数
# ----------------------------
def read_region(time_index=0):
    data_full = db.read(time=time_index, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape

    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)

    data_local = data_full[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end, :nz]
    print(f"Loaded region shape: {data_local.shape}")
    return data_local


# ----------------------------
# 4️⃣ 构建 PyVista StructuredGrid
# ----------------------------
def build_grid(data_local, scale_xy=1.0):
    nx, ny, nz_grid = data_local.shape
    x = np.linspace(lon_start, lon_end, ny) * scale_xy
    y = np.linspace(lat_start, lat_end, nx) * scale_xy
    z = np.linspace(0, 1000, nz_grid)  # 不放大深度

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = Z.transpose(1, 0, 2)

    grid = pv.StructuredGrid(X, Y, -Z)
    grid["value"] = data_local.flatten(order="F")
    return grid


# ----------------------------
# 5️⃣ 初始化主窗口
# ----------------------------
data0 = read_region(0)
grid = build_grid(data0, scale_xy=scale_xy)

plotter = pv.Plotter()
vol = plotter.add_volume(grid, cmap="viridis", opacity="sigmoid")

# ----------------------------
# 6️⃣ 鼠标点击两点生成截面
# ----------------------------
clicked_points = []


def click_callback(point, _):  # 第二个参数忽略
    clicked_points.append(point)
    if len(clicked_points) == 2:
        p1, p2 = clicked_points
        line = pv.Line(p1, p2, resolution=100)
        slice_data = grid.sample_along_line(line)
        plotter.add_mesh(slice_data, scalars="value", cmap="viridis", line_width=3)
        clicked_points.clear()


plotter.enable_point_picking(callback=click_callback, show_message=True, use_picker=True)


# ----------------------------
# 7️⃣ 时间播放功能
# ----------------------------
def update_time(time_index):
    data_local = read_region(time_index)
    grid["value"] = data_local.flatten(order="F")
    plotter.update_scalars(grid["value"])
    plotter.render()


def play_time(num_steps=10, delay=0.5):
    for t in range(num_steps):
        update_time(t)
        time.sleep(delay)


# ----------------------------
# 8️⃣ 展示
# ----------------------------
plotter.show()
