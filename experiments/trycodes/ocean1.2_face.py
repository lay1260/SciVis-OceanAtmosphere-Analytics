import OpenVisus as ov
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ 在线加载数据集
# ----------------------------
variable = "salt"
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

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
scale_xy = 25


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
    z = np.linspace(0, 1000, nz_grid)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = Z.transpose(1, 0, 2)

    grid = pv.StructuredGrid(X, Y, -Z)
    grid["value"] = data_local.flatten(order="F")
    return grid, x, y, z


# ----------------------------
# 5️⃣ 初始化
# ----------------------------
data0 = read_region(0)
grid, x_grid, y_grid, z_grid = build_grid(data0, scale_xy=scale_xy)

plotter = pv.Plotter()
vol = plotter.add_volume(grid, cmap="viridis", opacity="sigmoid")

# ----------------------------
# 6️⃣ 点击两点生成线-深度截面
# ----------------------------
clicked_points = []


def click_callback(point, _):
    clicked_points.append(point)
    if len(clicked_points) == 2:
        p1, p2 = clicked_points

        # 生成 100 个采样点沿线
        line_x = np.linspace(p1[0], p2[0], 100)
        line_y = np.linspace(p1[1], p2[1], 100)
        nx, ny, nz = data0.shape
        salt_along_line = np.zeros((nz, len(line_x)))

        # 将 line_x,line_y 映射回数组索引
        ix = np.clip(np.round((line_x / scale_xy - lon_start) * (ny - 1) / (lon_end - lon_start)).astype(int), 0,
                     ny - 1)
        iy = np.clip(np.round((line_y / scale_xy - lat_start) * (nx - 1) / (lat_end - lat_start)).astype(int), 0,
                     nx - 1)

        for k in range(nz):
            salt_along_line[k, :] = data0[iy, ix, k]

        # 绘制线-深度截面
        plt.figure(figsize=(8, 4))
        plt.imshow(salt_along_line, extent=[0, 1, -z_grid[-1], 0], aspect='auto', cmap='viridis')
        plt.colorbar(label='Salt (g/kg)')
        plt.xlabel('Normalized distance along line')
        plt.ylabel('Depth (m)')
        plt.title('Salt cross-section along line')
        plt.show()

        clicked_points.clear()


plotter.enable_point_picking(callback=click_callback, show_message=True, use_picker=True)

# ----------------------------
# 7️⃣ 展示
# ----------------------------
plotter.show()
