import numpy as np
import OpenVisus as ov
from scipy import ndimage

# ==============================
# 参数
# ==============================
FACE = 0
TIME = 100
STRIDE = 4   # ✅ 你控制的下采样率（4 → 1440→360）

LOW_LEVEL = slice(0, 3)
MID_LEVEL = slice(5, 10)

WIND_THRESHOLD = 15.0
WARM_CORE_THRESHOLD = 1.0
PRESSURE_PERCENTILE = 10


# ==============================
# GEOS 读取（稳定版本）
# ==============================
def read_geos(variable):
    url = (
        "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
        f"nasa/nsdf/climate3/dyamond/GEOS/GEOS_{variable.upper()}/"
        f"{variable.lower()}_face_{FACE}_depth_52_time_0_10269.idx"
    )

    db = ov.LoadDataset(url)

    # ✅ 必须 full resolution
    raw = db.read(time=TIME, quality=0)   # (z, y, x)

    # ✅ 统一顺序
    data = np.transpose(raw, (1, 2, 0))   # (y, x, z)

    # ✅ 手动下采样（关键）
    data = data[::STRIDE, ::STRIDE, :]

    return data


# ==============================
# 1️⃣ 读取数据
# ==============================
print("Loading GEOS data...")

U = read_geos('U')
V = read_geos('V')
T = read_geos('T')
P = read_geos('P')

print("Shapes:", U.shape, V.shape, T.shape, P.shape)
# ✅ 必须完全一致，如：(360, 360, 52)


# ==============================
# 2️⃣ 低层风速
# ==============================
U_low = U[:, :, LOW_LEVEL]
V_low = V[:, :, LOW_LEVEL]

wind = np.sqrt(U_low**2 + V_low**2)
max_wind = wind.max(axis=2)


# ==============================
# 3️⃣ 暖心
# ==============================
T_low = T[:, :, LOW_LEVEL].mean(axis=2)
T_mid = T[:, :, MID_LEVEL].mean(axis=2)
warm_core = T_mid - T_low


# ==============================
# 4️⃣ 低压
# ==============================
P_low = P[:, :, LOW_LEVEL].mean(axis=2)
p_thresh = np.percentile(P_low, PRESSURE_PERCENTILE)
low_pressure = P_low < p_thresh


# ==============================
# 5️⃣ 台风候选
# ==============================
tc_mask = (
    (max_wind > WIND_THRESHOLD) &
    (warm_core > WARM_CORE_THRESHOLD) &
    (low_pressure)
)

print("Candidate points:", tc_mask.sum())


# ==============================
# 6️⃣ 连通区域
# ==============================
labels, n = ndimage.label(tc_mask)
print("Detected TC systems:", n)


# ==============================
# 7️⃣ 输出
# ==============================
for lab in range(1, n + 1):
    mask = labels == lab
    if mask.sum() < 30:
        continue

    y, x = np.where(mask)
    print(
        f"TC center=({x.mean():.1f},{y.mean():.1f}) | "
        f"minP={P_low[mask].min():.1f} | "
        f"maxV={max_wind[mask].max():.1f}"
    )
