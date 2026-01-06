import OpenVisus as ov
import numpy as np
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

U_db = load_dataset("u")
V_db = load_dataset("v")
W_db = load_dataset("w")

# ----------------------------
# 2️⃣ 局部区域参数
# ----------------------------
lat_start, lat_end = 0, 40
lon_start, lon_end = 100, 140
nz = 20
data_quality = -6
scale_xy = 25

# ----------------------------
# 3️⃣ 读取局部速度数据函数
# ----------------------------
def read_velocity(db):
    data_full = db.read(time=0, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    return data_full[lat_idx_start:lat_idx_end,
                     lon_idx_start:lon_idx_end,
                     :nz]

U_local = read_velocity(U_db)
V_local = read_velocity(V_db)
W_local = read_velocity(W_db)
nx, ny, nz = U_local.shape
z_grid = np.linspace(0, 1000, nz)

# ----------------------------
# 4️⃣ 坐标网格
# ----------------------------
x = np.linspace(lon_start, lon_end, ny) * scale_xy
y = np.linspace(lat_start, lat_end, nx) * scale_xy

# ----------------------------
# 5️⃣ 二维俯视图选择两点
# ----------------------------
fig, ax = plt.subplots(figsize=(6,6))
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

        # ----------------------------
        # 6️⃣ 计算截面坐标
        # ----------------------------
        n_samples = 200
        line_x = np.linspace(x1, x2, n_samples)
        line_y = np.linspace(y1, y2, n_samples)
        ix = np.clip(np.round((line_x/scale_xy-lon_start)*(ny-1)/(lon_end-lon_start)).astype(int), 0, ny-1)
        iy = np.clip(np.round((line_y/scale_xy-lat_start)*(nx-1)/(lat_end-lat_start)).astype(int), 0, nx-1)

        # 截面速度向量（沿截面方向 + 深度）
        # x沿线方向，z沿深度
        speed_x = np.zeros((nz, n_samples))
        speed_z = np.zeros((nz, n_samples))
        for k in range(nz):
            # 沿线方向速度 = 投影到连线方向
            dx = x2 - x1
            dy = y2 - y1
            dist = np.hypot(dx, dy)
            speed_x[k,:] = (U_local[iy, ix, k]*dx + V_local[iy, ix, k]*dy)/dist
            speed_z[k,:] = -W_local[iy, ix, k]  # 深度方向向下为正

        # ----------------------------
        # 7️⃣ 绘制截面流线
        # ----------------------------
        X, Z = np.meshgrid(np.arange(n_samples), z_grid)
        plt.figure(figsize=(10,4))
        plt.streamplot(np.arange(n_samples), z_grid, speed_x, speed_z,
                       density=2.0, color=np.sqrt(speed_x**2+speed_z**2),
                       cmap='viridis', arrowsize=1.5)
        plt.gca().invert_yaxis()
        plt.colorbar(label='Velocity magnitude (m/s)')
        plt.xlabel('Distance along line')
        plt.ylabel('Depth (m)')
        plt.title('Velocity streamlines along cross-section')
        plt.show()

        clicked_points.clear()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
