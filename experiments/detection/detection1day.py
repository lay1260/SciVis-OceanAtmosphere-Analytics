import numpy as np
import OpenVisus as ov
from scipy import ndimage
import csv

# ==============================
# 参数
# ==============================
FACES = [0]                 # 先从 face 0 开始，后面可以加到 [0,1,2,3,4,5]
TIME_START = 4380
TIME_END = 4500
TIME_STEP=6
STRIDE = 4

LOW_LEVEL = slice(0, 4)
MID_LEVEL = slice(10, 20)

WIND_THRESHOLD = 12.0
WARM_CORE_THRESHOLD = 0.5
LOW_PRESSURE_THRESHOLD = 99500  # Pa

OUTPUT_FILE = "tc_events.csv"


# ==============================
# GEOS 稳定读取
# ==============================
def read_geos(variable, face, time):
    url = (
        "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
        f"nasa/nsdf/climate3/dyamond/GEOS/GEOS_{variable.upper()}/"
        f"{variable.lower()}_face_{face}_depth_52_time_0_10269.idx"
    )

    db = ov.LoadDataset(url)
    raw = db.read(time=time, quality=0)          # (z,y,x)
    data = np.transpose(raw, (1, 2, 0))          # (y,x,z)
    data = data[::STRIDE, ::STRIDE, :]            # 手动降分辨率

    return data


# ==============================
# 台风检测（单 time, 单 face）
# ==============================
from scipy.ndimage import gaussian_filter

def detect_tc_one_step(face, time):
    U = read_geos('U', face, time)
    V = read_geos('V', face, time)

    # ======================
    # 1️⃣ 低层最大风速
    # ======================
    wind = np.sqrt(
        U[:, :, LOW_LEVEL]**2 + V[:, :, LOW_LEVEL]**2
    )
    max_wind = wind.max(axis=2)

    # ======================
    # 2️⃣ 背景风（平滑）
    # ======================
    wind_bg = gaussian_filter(max_wind, sigma=3)

    # ======================
    # 3️⃣ 台风候选判据（关键）
    # ======================
    tc_mask = (
        (max_wind > WIND_THRESHOLD) &
        (max_wind > wind_bg + 5.0)
    )

    labels, n = ndimage.label(tc_mask)

    events = []

    for lab in range(1, n + 1):
        mask = labels == lab
        if mask.sum() < 20:
            continue

        y, x = np.where(mask)

        events.append({
            "time": time,
            "face": face,
            "center_x": float(x.mean()),
            "center_y": float(y.mean()),
            "max_wind": float(max_wind[mask].max()),
            "area": int(mask.sum())
        })

    # ======================
    # 调试输出
    # ======================
    print(
        f"time {time} face {face} | "
        f"maxWind={max_wind.max():.1f} m/s | "
        f"bgWind={wind_bg.max():.1f} m/s | "
        f"TC_pixels={tc_mask.sum()}"
    )

    return events


# ==============================
# 主程序：遍历时间，写文件
# ==============================
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "time", "face",
            "center_x", "center_y",
            "max_wind", "area"
        ]
    )

    writer.writeheader()

    for time in range(TIME_START, TIME_END, TIME_STEP):
        print(f"\nProcessing time = {time}")

        for face in FACES:
            events = detect_tc_one_step(face, time)

            for ev in events:
                writer.writerow(ev)

            print(f"  face {face}: {len(events)} TC candidates")

print(f"\n✅ Detection finished. Results saved to {OUTPUT_FILE}")
