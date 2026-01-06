#温度使用体渲染，盐度使用切片，流线在温度高值区被遮盖
import OpenVisus as ov
import numpy as np
import pyvista as pv

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
nz = 20
data_quality = -6
scale_xy = 25
skip = 2  # XY 下采样

# ----------------------------
# 4️⃣ 读取局部数据函数
# ----------------------------
def read_data(db):
    data_full = db.read(time=0, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    return data_full[lat_idx_start:lat_idx_end:skip,
                     lon_idx_start:lon_idx_end:skip,
                     :nz]

U_local = read_data(U_db)
V_local = read_data(V_db)
W_local = read_data(W_db)
Salt_local = read_data(Salt_db)
Theta_local = read_data(Theta_db)

nx, ny, nz = U_local.shape
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
# 6️⃣ 结构化网格 + 流线
# ----------------------------
grid = pv.StructuredGrid(X, Y, Z)
vectors = np.stack([U_local.flatten(order="F"),
                    V_local.flatten(order="F"),
                    W_local.flatten(order="F")], axis=1)
grid["velocity"] = vectors

seed_points = pv.PolyData(grid.points[::10])
streamlines = grid.streamlines_from_source(
    source=seed_points,
    vectors='velocity',
    integration_direction='both',
    initial_step_length=5,
    terminal_speed=1e-3,
    max_steps=1000
)
speed = np.linalg.norm(streamlines['velocity'], axis=1)
streamlines['speed'] = speed
streamline_tube = streamlines.tube(radius=2)  # 增加可见性

# ----------------------------
# 7️⃣ 盐度切片 + 温度体积渲染
# ----------------------------
# 温度体积
theta_volume = pv.StructuredGrid(X, Y, Z)
theta_volume["theta"] = Theta_local.flatten(order="F")

# 创建 Plotter
plotter = pv.Plotter(window_size=(1400, 900))

# 温度体积 GPU 渲染
plotter.add_volume(theta_volume, scalars="theta", cmap="hot", opacity="sigmoid")

# 盐度切片显示
for k in range(0, nz, 2):  # 每隔两层
    slice_salt = pv.StructuredGrid(X[:,:,k], Y[:,:,k], Z[:,:,k])
    slice_salt["salt"] = Salt_local[:,:,k].flatten(order="F")
    plotter.add_mesh(slice_salt, scalars="salt", cmap="Blues", opacity=0.1)

# 流线
plotter.add_mesh(streamline_tube, scalars='speed', cmap='cool', opacity=0.8)

plotter.add_axes()
plotter.show()
