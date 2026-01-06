#盐度切片
import OpenVisus as ov
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

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

# 速度与盐度
U_db = load_dataset("u")
V_db = load_dataset("v")
W_db = load_dataset("w")
Salt_db = load_dataset("salt")

# ----------------------------
# 2️⃣ 局部区域参数
# ----------------------------
lat_start, lat_end = 10, 40
lon_start, lon_end = 100, 130
nz = 20
data_quality = -6
scale_xy = 25

# ----------------------------
# 3️⃣ 读取局部数据函数
# ----------------------------
def read_data(db):
    data_full = db.read(time=0, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    return data_full[lat_idx_start:lat_idx_end,
                     lon_idx_start:lon_idx_end,
                     :nz]

U_local = read_data(U_db)
V_local = read_data(V_db)
W_local = read_data(W_db)
Salt_local = read_data(Salt_db)
nx, ny, nz = U_local.shape
z_grid = np.linspace(0, 1000, nz)

# ----------------------------
# 4️⃣ 坐标网格
# ----------------------------
x = np.linspace(lon_start, lon_end, ny) * scale_xy
y = np.linspace(lat_start, lat_end, nx) * scale_xy
X, Y, Z = np.meshgrid(x, y, z_grid, indexing='ij')
X = X.transpose(1, 0, 2)
Y = Y.transpose(1, 0, 2)
Z = -Z.transpose(1, 0, 2)

# ----------------------------
# 5️⃣ PyVista 3D流线 + 多层盐度
# ----------------------------
grid = pv.StructuredGrid(X, Y, Z)
vectors = np.stack([U_local.flatten(order="F"),
                    V_local.flatten(order="F"),
                    W_local.flatten(order="F")], axis=1)
grid["velocity"] = vectors

# 种子点间隔
seed_points = pv.PolyData(grid.points[::10])

# 生成流线 (不使用 return_source)
streamlines = grid.streamlines_from_source(
    source=seed_points,
    vectors='velocity',
    integration_direction='both',
    initial_step_length=5,
    terminal_speed=1e-3,
    max_steps=1000
)

# 给流线添加速度标量
if 'velocity' in streamlines.array_names:
    speed = np.linalg.norm(streamlines['velocity'], axis=1)
    streamlines['speed'] = speed
elif 'vectors' in streamlines.array_names:
    speed = np.linalg.norm(streamlines['vectors'], axis=1)
    streamlines['speed'] = speed

# 叠加盐度层 (多层)
plotter = pv.Plotter(window_size=(1000, 700))
for k in range(nz):
    surface_salt = pv.StructuredGrid(
        X[:, :, k],
        Y[:, :, k],
        Z[:, :, k]
    )
    surface_salt["salt"] = Salt_local[:, :, k].flatten(order="F")
    plotter.add_mesh(surface_salt, scalars="salt", cmap="Blues", opacity=0.1)

# 添加流线
plotter.add_mesh(streamlines, scalars='speed', cmap='cool', line_width=2, opacity=0.8)
plotter.add_axes()
plotter.show()

