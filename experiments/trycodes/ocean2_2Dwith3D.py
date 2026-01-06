import OpenVisus as ov
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ 数据集加载
# ----------------------------
variable = "salt"
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
dataset_url = base_url + base_dir
db = ov.LoadDataset(dataset_url)

# ----------------------------
# 2️⃣ 区域参数
# ----------------------------
lat_start, lat_end = 0, 40
lon_start, lon_end = 100, 140
nz = 40
data_quality = -6
scale_xy = 25

# ----------------------------
# 3️⃣ 读取局部数据
# ----------------------------
def read_region(time_index=0):
    data_full = db.read(time=time_index, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    return data_full[lat_idx_start:lat_idx_end,
                     lon_idx_start:lon_idx_end,
                     :nz]

data_local = read_region(0)
nx, ny, nz = data_local.shape
z_grid = np.linspace(0, 1000, nz)
x = np.linspace(lon_start, lon_end, ny) * scale_xy
y = np.linspace(lat_start, lat_end, nx) * scale_xy
X, Y = np.meshgrid(x, y)

# ----------------------------
# 4️⃣ 构建 PyVista StructuredGrid
# ----------------------------
def build_grid(data_local):
    nx, ny, nz_grid = data_local.shape
    z = np.linspace(0, 1000, nz_grid)
    X3d, Y3d, Z3d = np.meshgrid(x, y, z, indexing='ij')
    X3d = X3d.transpose(1, 0, 2)
    Y3d = Y3d.transpose(1, 0, 2)
    Z3d = Z3d.transpose(1, 0, 2)
    grid = pv.StructuredGrid(X3d, Y3d, -Z3d)
    grid["value"] = data_local.flatten(order="F")
    return grid

grid = build_grid(data_local)

# ----------------------------
# 5️⃣ PyVista 3D 可视化
# ----------------------------
plotter = pv.Plotter(window_size=(800, 600))
opacity_tf = [0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.35]
vol = plotter.add_volume(grid, cmap="Blues", opacity=opacity_tf, shade=True)

# ----------------------------
# 6️⃣ 二维俯视图 + 交互
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
c = ax.imshow(data_local[:, :, 0], extent=[x[0], x[-1], y[0], y[-1]],
              origin='lower', cmap='Blues')
plt.colorbar(c, ax=ax, label='Salt (g/kg)')
ax.set_xlabel("Longitude (scaled)")
ax.set_ylabel("Latitude (scaled)")
ax.set_title("Top-down view (layer 0)")
# 比例尺
scalebar_len = 100
ax.hlines(y[0]+2, x[0]+2, x[0]+2+scalebar_len, colors='k', linewidth=3)
ax.text(x[0]+2, y[0]+5, f"{scalebar_len} km", color='k')

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

        # 绘制剖面
        n_samples = 200
        line_x = np.linspace(x1, x2, n_samples)
        line_y = np.linspace(y1, y2, n_samples)
        ix = np.clip(np.round((line_x / scale_xy - lon_start) * (ny-1)/(lon_end-lon_start)).astype(int), 0, ny-1)
        iy = np.clip(np.round((line_y / scale_xy - lat_start) * (nx-1)/(lat_end-lat_start)).astype(int), 0, nx-1)
        salt_along_line = np.zeros((nz, n_samples))
        for k in range(nz):
            salt_along_line[k, :] = data_local[iy, ix, k]

        plt.figure(figsize=(8, 4))
        plt.imshow(salt_along_line, extent=[0, 1, -z_grid[-1], 0],
                   aspect='auto', cmap='Blues')
        plt.colorbar(label='Salt (g/kg)')
        plt.xlabel('Normalized distance along line')
        plt.ylabel('Depth (m)')
        plt.title('Salt cross-section along line')
        plt.show()

        clicked_points.clear()

fig.canvas.mpl_connect('button_press_event', onclick)

# ----------------------------
# 7️⃣ 显示 PyVista 3D 窗口
# ----------------------------
plotter.show()
plt.show()
