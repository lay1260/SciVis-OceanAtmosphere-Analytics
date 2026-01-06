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
lat_start, lat_end = 0, 40
lon_start, lon_end = 100, 140
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
# 5️⃣ PyVista 3D流线 + 盐度背景
# ----------------------------
grid = pv.StructuredGrid(X, Y, Z)
vectors = np.stack([U_local.flatten(order="F"),
                    V_local.flatten(order="F"),
                    W_local.flatten(order="F")], axis=1)
grid["velocity"] = vectors

# 种子点密集
seed_points = pv.PolyData(grid.points[::5])

streamlines = grid.streamlines_from_source(
    source=seed_points,
    vectors='velocity',
    integration_direction='both',
    max_time=500,
    initial_step_length=5,
    terminal_speed=1e-3
)

# 海表盐度层 (第0层)
surface_salt = pv.StructuredGrid(
    X[:, :, 0],
    Y[:, :, 0],
    Z[:, :, 0]
)
surface_salt["salt"] = Salt_local[:, :, 0].flatten(order="F")

plotter = pv.Plotter(window_size=(1000, 700))
plotter.add_mesh(surface_salt, scalars="salt", cmap="Blues", opacity=0.6)  # 半透明
plotter.add_mesh(streamlines, color="cyan", opacity=0.6)
plotter.add_axes()
plotter.show()

# ----------------------------
# 6️⃣ 二维俯视图交互选点 + 沿线剖面
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
c = ax.imshow(U_local[:, :, 0],
              extent=[x[0], x[-1], y[0], y[-1]],
              origin='lower', cmap='Blues')
plt.colorbar(c, ax=ax, label='U velocity (m/s)')
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Top-down view (layer 0)")

# 比例尺
scalebar_len = 100
ax.hlines(y[0]+2, x[0]+2, x[0]+2+scalebar_len, colors='k', linewidth=3)
ax.text(x[0]+2, y[0]+5, f"{scalebar_len} km", color='k')

# 经纬线
lon_ticks = np.arange(lon_start, lon_end+1, 20)
lat_ticks = np.arange(lat_start, lat_end+1, 20)
ax.set_xticks(lon_ticks*scale_xy)
ax.set_yticks(lat_ticks*scale_xy)
ax.set_xticklabels([f"{lon}°E" for lon in lon_ticks])
ax.set_yticklabels([f"{lat}°N" for lat in lat_ticks])
ax.grid(True, color='white', linestyle='--', linewidth=0.5)

clicked_points = []

def onclick(event):
    if event.inaxes != ax:
        return
    clicked_points.append((event.xdata, event.ydata))
    ax.plot(event.xdata, event.ydata, 'yo', markersize=8)
    fig.canvas.draw()

    if len(clicked_points) == 2:
        (x1, y1), (x2, y2) = clicked_points
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        fig.canvas.draw()

        # 沿线速度剖面
        n_samples = 200
        line_x = np.linspace(x1, x2, n_samples)
        line_y = np.linspace(y1, y2, n_samples)
        ix = np.clip(np.round((line_x/scale_xy-lon_start)*(ny-1)/(lon_end-lon_start)).astype(int), 0, ny-1)
        iy = np.clip(np.round((line_y/scale_xy-lat_start)*(nx-1)/(lat_end-lat_start)).astype(int), 0, nx-1)

        speed_line = np.zeros((nz, n_samples))
        for k in range(nz):
            u_line = U_local[iy, ix, k]
            v_line = V_local[iy, ix, k]
            w_line = W_local[iy, ix, k]
            speed_line[k, :] = np.sqrt(u_line**2 + v_line**2 + w_line**2)

        plt.figure(figsize=(8,4))
        plt.imshow(speed_line, extent=[0,1,-z_grid[-1],0], aspect='auto', cmap='viridis')
        plt.colorbar(label='Speed (m/s)')
        plt.xlabel("Normalized distance along line")
        plt.ylabel("Depth (m)")
        plt.title("Velocity magnitude cross-section along line")
        plt.show()

        clicked_points.clear()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
