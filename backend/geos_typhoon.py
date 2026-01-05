"""
使用 GEOS 风场（分 face 的 idx）粗略识别台风中心。
基于涡度 + （可选）风速阈值 + 近邻合并，供快速测试。

注意：
- URL 模板仅覆盖单个 face；如需全球需遍历 6 个 face。
- 层位索引需根据数据实际垂直层定义调整。
- 阈值、半径需按分辨率和需求调节。
"""

import OpenVisus as ov
import numpy as np
import json
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from skimage.measure import label, regionprops

# -------------------- 配置区 --------------------

# GEOS face URL 模板
GEOS_FACE_URL = (
    "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
    "nasa/nsdf/climate3/dyamond/GEOS/GEOS_{var_u}/"
    "{var_l}_face_{face}_depth_52_time_0_10269.idx"
)

# 选择的 face（0-5）
FACE = 0

# 选择的层位索引：需根据 GEOS 垂直层定义调整
# 这里假设使用第 850hPa 附近层（示例：层 40），未确认请根据元数据修改
LEVEL_INDEX = 40

# 降采样质量（OpenVisus quality，负数=降分辨率）
DATA_QUALITY = -9

# 阈值与半径（物理）
VORTICITY_THRESHOLD = 5e-5
WIND_THRESHOLD = 20.0  # m/s，可设为 None 跳过风速筛选
MERGE_DISTANCE_KM = 150.0
MSLP_RADIUS_KM = 300.0  # 若使用风速/涡度筛选时可忽略

# 网格尺度（米）。若已知分辨率，用实际值；否则可用经纬度换算后更新。
DX_M = 7000.0
DY_M = 7000.0

# 多面/批量参数（类似示例脚本）
BATCH_FACES = list(range(6))
BATCH_QUALITY = -9
BATCH_TIME_STEP_INTERVAL = 6   # 每6小时
BATCH_DAYS_TO_TEST = 7
HOURS_PER_DAY = 24
REG_VORT_THRESHOLD = 8e-5
REG_P_LOW_THRESHOLD = -100
REG_MIN_REGION_SIZE = 3
REG_STORE_RES = 10
REG_DETECT_RES = 5

# 风速主导的台风检测参数（简化版）
WIND_FACES = list(range(6))  # 遍历全部6个face
WIND_TIME_STEP = 6
WIND_STRIDE = 4
WIND_LOW_LEVEL_SLICE = slice(0, 4)
WIND_THRESHOLD = 12.0
WIND_BG_ADD = 5.0
WIND_MIN_AREA = 20
WIND_SIGMA = 3
WIND_OUT_FILE = "tc_events.csv"
WIND_READ_QUALITY = -6  # 风速检测读取质量（负值=降分辨率以提速）

# ------------------------------------------------


def get_geos_face_url(variable="u", face=0):
    return GEOS_FACE_URL.format(var_u=variable.upper(), var_l=variable.lower(), face=face)


def load_geos_face(variable="u", face=0):
    url = get_geos_face_url(variable, face)
    return ov.LoadDataset(url)


def get_timesteps(face=0):
    """
    返回指定 face 的时间步列表（整数索引）。
    """
    db_u = load_geos_face("u", face)
    return list(map(int, db_u.getTimesteps()))


# ---------------------- 区域检测（5度网格 → 10度方块） ---------------------- #

def resample_to_grid(data, target_res=REG_DETECT_RES):
    """
    将 (x,y,z) 取最上层并重采样到 target_res(度) 的规则网格。
    """
    nx, ny, nz = data.shape
    lat_bins = int(180 / target_res)
    lon_bins = int(360 / target_res)
    grid = np.zeros((lat_bins, lon_bins))
    lat_idx = np.linspace(0, nx - 1, lat_bins).astype(int)
    lon_idx = np.linspace(0, ny - 1, lon_bins).astype(int)
    for i, ii in enumerate(lat_idx):
        for j, jj in enumerate(lon_idx):
            grid[i, j] = data[ii, jj, 0]
    return grid


def calc_vorticity(u, v):
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    vort = dv_dx - du_dy
    return gaussian_filter(vort, sigma=1)


def bin_5deg_to_10deg(lat_bin, lon_bin):
    lat10 = lat_bin // 2
    lon10 = lon_bin // 2
    lat_min = lat10 * REG_STORE_RES - 90
    lat_max = lat_min + REG_STORE_RES
    lon_min = lon10 * REG_STORE_RES - 180
    lon_max = lon_min + REG_STORE_RES
    return lat_min, lat_max, lon_min, lon_max


def detect_tropical_cyclones_single(t, db_u, db_v, db_p):
    u_raw = db_u.read(time=t, quality=BATCH_QUALITY)
    v_raw = db_v.read(time=t, quality=BATCH_QUALITY)
    p_raw = db_p.read(time=t, quality=BATCH_QUALITY)

    u = resample_to_grid(u_raw)
    v = resample_to_grid(v_raw)
    p = resample_to_grid(p_raw)

    vort = calc_vorticity(u, v)
    mask_vort = np.abs(vort) > REG_VORT_THRESHOLD

    p_anom = p - np.mean(p)
    mask_p = p_anom < REG_P_LOW_THRESHOLD

    mask = mask_vort & mask_p
    lbl = label(mask)
    props = regionprops(lbl)
    cubes = []
    for region in props:
        if region.area < REG_MIN_REGION_SIZE:
            continue
        cy, cx = region.centroid
        lat_min, lat_max, lon_min, lon_max = bin_5deg_to_10deg(int(cy), int(cx))
        cubes.append({
            "time_step": int(t),
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max
        })
    return cubes


def process_face(face, days=BATCH_DAYS_TO_TEST, interval=BATCH_TIME_STEP_INTERVAL):
    print(f"Processing face {face} ...")
    db_u = load_geos_face("u", face)
    db_v = load_geos_face("v", face)
    db_p = load_geos_face("p", face)
    timesteps = db_u.getTimesteps()
    max_timestep = min(len(timesteps), days * HOURS_PER_DAY)
    results_face = []
    for idx in range(0, max_timestep, interval):
        t = timesteps[idx]
        cubes = detect_tropical_cyclones_single(t, db_u, db_v, db_p)
        results_face.extend(cubes)
        if idx % 24 == 0:
            print(f"[Face {face}] processed timestep {t}")
    print(f"Face {face} done. Detected {len(results_face)} cubes.")
    return results_face


# ---------------------- 风速主导的快速检测（类似用户脚本） ---------------------- #

def read_geos(variable, face, time, quality=WIND_READ_QUALITY, stride=WIND_STRIDE, db=None):
    """
    读取 GEOS u/v 数据，并在 (y,x) 维度上做 stride 下采样。
    返回形状 (y, x, z)。
    """
    if db is None:
        url = (
            "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
            f"nasa/nsdf/climate3/dyamond/GEOS/GEOS_{variable.upper()}/"
            f"{variable.lower()}_face_{face}_depth_52_time_0_10269.idx"
        )
        db = ov.LoadDataset(url)
    raw = db.read(time=time, quality=quality)      # (z,y,x)
    data = np.transpose(raw, (1, 2, 0))            # (y,x,z)
    data = data[::stride, ::stride, :]             # 手动降分辨率
    return data


def detect_tc_wind(face, time, db_u=None, db_v=None):
    """
    仅用风速阈值 + 背景风对比的快速候选检测。
    返回 list of dicts：{time, face, center_x, center_y, max_wind, area}
    """
    U = read_geos("U", face, time, db=db_u)
    V = read_geos("V", face, time, db=db_v)

    wind = np.sqrt(U[:, :, WIND_LOW_LEVEL_SLICE] ** 2 + V[:, :, WIND_LOW_LEVEL_SLICE] ** 2)
    max_wind = wind.max(axis=2)

    wind_bg = gaussian_filter(max_wind, sigma=WIND_SIGMA)

    tc_mask = (max_wind > WIND_THRESHOLD) & (max_wind > wind_bg + WIND_BG_ADD)
    labels, n = ndimage.label(tc_mask)

    events = []
    for lab in range(1, n + 1):
        mask = labels == lab
        if mask.sum() < WIND_MIN_AREA:
            continue
        y, x = np.where(mask)
        events.append({
            "time": int(time),
            "face": int(face),
            "center_x": int(round(x.mean())),
            "center_y": int(round(y.mean())),
            "max_wind": int(round(float(max_wind[mask].max()))),
            "area": int(mask.sum())
        })

    print(
        f"time {time} face {face} | "
        f"maxWind={max_wind.max():.1f} m/s | "
        f"bgWind={wind_bg.max():.1f} m/s | "
        f"TC_pixels={tc_mask.sum()}"
    )
    return events


def slice_uv(db_u, db_v, time_step, level_index, quality):
    u_data = db_u.read(time=time_step, quality=quality)
    v_data = db_v.read(time=time_step, quality=quality)
    if u_data.ndim == 3:
        u = u_data[:, :, level_index]
        v = v_data[:, :, level_index]
    elif u_data.ndim == 2:
        u, v = u_data, v_data
    else:
        raise ValueError(f"Unexpected dims u={u_data.shape}, v={v_data.shape}")
    return u, v


def merge_close(points, dx_m, dy_m, merge_km):
    if not points:
        return points
    used = [False] * len(points)
    merged = []
    for ia, a in enumerate(points):
        if used[ia]:
            continue
        group = [a]
        used[ia] = True
        for ib in range(ia + 1, len(points)):
            if used[ib]:
                continue
            b = points[ib]
            di = (a[0] - b[0]) * dy_m
            dj = (a[1] - b[1]) * dx_m
            dist_km = (di * di + dj * dj) ** 0.5 / 1000.0
            if dist_km <= merge_km:
                group.append(b)
                used[ib] = True
        mi = int(round(sum(p[0] for p in group) / len(group)))
        mj = int(round(sum(p[1] for p in group) / len(group)))
        merged.append((mi, mj))
    return merged


def detect_typhoon_centers_geos(
    time_step,
    face=FACE,
    level_index=LEVEL_INDEX,
    quality=DATA_QUALITY,
    vorticity_threshold=VORTICITY_THRESHOLD,
    wind_threshold=WIND_THRESHOLD,
    merge_distance_km=MERGE_DISTANCE_KM,
):
    """
    返回 [(row, col, time_step)] 列表
    """
    # 加载 u/v
    db_u = load_geos_face("u", face)
    db_v = load_geos_face("v", face)

    # 取指定层
    u, v = slice_uv(db_u, db_v, time_step, level_index, quality)

    # 涡度
    dv_dx = np.gradient(v, DX_M, axis=1)
    du_dy = np.gradient(u, DY_M, axis=0)
    vort = dv_dx - du_dy

    mask = vort > vorticity_threshold

    # 叠加风速条件（可选）
    if wind_threshold is not None:
        wind = np.sqrt(u * u + v * v)
        mask &= wind > wind_threshold

    cand = np.where(mask)
    centers = list(zip(cand[0].tolist(), cand[1].tolist()))

    centers = merge_close(centers, DX_M, DY_M, merge_distance_km)
    centers = [(i, j, int(time_step)) for i, j in centers]
    return centers


if __name__ == "__main__":
    """
    用法示例：
    1) 单个时间步：
       python geos_typhoon.py 0

    2) 时间范围：
       python geos_typhoon.py 0 10   # 遍历 time_step 0-10

    3) 生成 6 face、7 天、每 6 小时一次的 10°x10° 方块 JSON：
       python geos_typhoon.py cubes

    4) 风速主导的快速检测并写 CSV：
       python geos_typhoon.py windcsv <start> <end> <step> [outfile]
       # faces 使用 WIND_FACES 配置，默认 face 0，步长/阈值见顶部配置
    """
    import sys

    if len(sys.argv) == 1:
        print("用法: python geos_typhoon.py <start> [end] | cubes | windcsv <start> <end> <step> [outfile]")
        sys.exit(0)

    # 批量方块模式
    if sys.argv[1].lower() == "cubes":
        faces = BATCH_FACES
        pool = Pool(processes=min(3, cpu_count()))
        results_all_faces = pool.map(process_face, faces)
        pool.close()
        pool.join()

        merged = {}
        for face_list in results_all_faces:
            for cube in face_list:
                key = (cube["time_step"], cube["lat_min"], cube["lat_max"], cube["lon_min"], cube["lon_max"])
                merged[key] = cube
        merged_list = list(merged.values())
        print(f"Total unique cubes after merge: {len(merged_list)}")
        out_file = "TC_time_space_cube_6faces_1week.json"
        with open(out_file, "w") as f:
            json.dump(merged_list, f, indent=2)
        print(f"Saved → {out_file}")
        sys.exit(0)

    # 风速 CSV 模式
    if sys.argv[1].lower() == "windcsv":
        if len(sys.argv) < 5:
            print("用法: python geos_typhoon.py windcsv <start> <end> <step> [outfile]")
            sys.exit(1)
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        step = int(sys.argv[4])
        out_file = sys.argv[5] if len(sys.argv) > 5 else WIND_OUT_FILE

        # 预加载各 face 的 u/v 数据集，避免重复 LoadDataset
        face_db = {}
        for face in WIND_FACES:
            face_db[face] = (load_geos_face("u", face), load_geos_face("v", face))

        with open(out_file, "w", newline="") as f:
            fieldnames = ["time", "face", "center_x", "center_y", "max_wind", "area"]
            f.write(",".join(fieldnames) + "\n")
            for t in range(start, end, step):
                print(f"\nProcessing time = {t}")
                for face in WIND_FACES:
                    db_u, db_v = face_db[face]
                    events = detect_tc_wind(face, t, db_u=db_u, db_v=db_v)
                    for ev in events:
                        row = [ev[k] for k in fieldnames]
                        f.write(",".join(map(str, row)) + "\n")
                    print(f"  face {face}: {len(events)} TC candidates")
        print(f"\nSaved → {out_file}")
        sys.exit(0)

    # 单时间步/时间范围模式
    start = int(sys.argv[1])
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start

    for ts in range(start, end + 1):
        res = detect_typhoon_centers_geos(ts)
        print(f"time_step={ts}, centers={len(res)}")
        for c in res:
            print("  ", c)

