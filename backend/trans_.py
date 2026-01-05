import pandas as pd
import numpy as np

# C1440 网格参数
N = 180

# cubed-sphere 索引 -> lon/lat 函数（向量化版本）
def cubed_sphere_index_to_lonlat_vectorized(faces, i_arr, j_arr, N=1440):
    xi = (2*i_arr + 1)/N - 1
    eta = (2*j_arr + 1)/N - 1

    X = np.zeros_like(xi, dtype=float)
    Y = np.zeros_like(xi, dtype=float)
    Z = np.zeros_like(xi, dtype=float)

    # face=0
    mask = (faces==0)
    X[mask] = 1.0
    Y[mask] = xi[mask]
    Z[mask] = -eta[mask]

    # face=1
    mask = (faces==1)
    X[mask] = -xi[mask]
    Y[mask] = 1.0
    Z[mask] = -eta[mask]

    # face=2
    mask = (faces==2)
    X[mask] = -1.0
    Y[mask] = -xi[mask]
    Z[mask] = -eta[mask]

    # face=3
    mask = (faces==3)
    X[mask] = xi[mask]
    Y[mask] = -1.0
    Z[mask] = -eta[mask]

    # face=4
    mask = (faces==4)
    X[mask] = eta[mask]
    Y[mask] = xi[mask]
    Z[mask] = 1.0

    # face=5
    mask = (faces==5)
    X[mask] = -eta[mask]
    Y[mask] = xi[mask]
    Z[mask] = -1.0

    lon = np.arctan2(Y, X) * 180.0 / np.pi
    lat = np.arctan2(Z, np.sqrt(X**2 + Y**2)) * 180.0 / np.pi

    return lon, lat

# 读取 CSV
df = pd.read_csv("tc_events_june_5days1.csv")

# 前四列假设为 time, face, center_x, center_y
face_arr = df.iloc[:,1].astype(int).values
i_arr = df.iloc[:,2].astype(int).values
j_arr = df.iloc[:,3].astype(int).values

# 向量化计算 lon/lat
lon_arr, lat_arr = cubed_sphere_index_to_lonlat_vectorized(face_arr, i_arr, j_arr, N)

# 添加到 DataFrame
df['lon'] = lon_arr
df['lat'] = lat_arr

# 输出 CSV
df.to_csv("LL_tc_events_june_5days1.csv", index=False)

print("转换完成，已生成 LL_tc_events_june_5days1.csv")
