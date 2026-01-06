import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#----------------------------------------------------------------
# 读取时间-空间立方体
#----------------------------------------------------------------
cube = np.load("TC_time_space_cube.npy")  # shape: (T, 18, 36)

# 求每个格子的发生次数
freq = cube.sum(axis=0)   # shape: (18, 36)

#----------------------------------------------------------------
# 构造网格的真实经纬度坐标（10° 分辨率）
#----------------------------------------------------------------
lat_edges = np.linspace(-90, 90, 19)
lon_edges = np.linspace(-180, 180, 37)

lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)

#----------------------------------------------------------------
# 画图
#----------------------------------------------------------------
plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

# 海岸线、陆地
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)

# 台风频次热力图
c = ax.pcolormesh(
    lon_grid, lat_grid, freq,
    cmap="hot_r",
    transform=ccrs.PlateCarree()
)

# colorbar
cb = plt.colorbar(c, orientation="vertical", pad=0.02, shrink=0.7)
cb.set_label("Typhoon Frequency (10° grid)", fontsize=12)

plt.title("Global Typhoon Frequency Heatmap (10°×10°)", fontsize=16)
plt.show()
