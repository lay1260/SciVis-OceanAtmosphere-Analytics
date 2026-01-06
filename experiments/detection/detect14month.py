import numpy as np
import OpenVisus as ov
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
import json
from multiprocessing import Pool, cpu_count

# ---------------------------------------------
# 参数设置
# ---------------------------------------------
FACES = list(range(6))
QUALITY = -9
TIME_STEP_INTERVAL = 6  # 每6小时检测一次
DETECT_RES = 5
STORE_RES = 10

VORT_THRESHOLD = 8e-5
P_LOW_THRESHOLD = -100
MIN_REGION_SIZE = 3


# ---------------------------------------------
# 数据加载器
# ---------------------------------------------
def load_field(variable, face):
    url = (
        f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
        f"nasa/nsdf/climate3/dyamond/GEOS/GEOS_{variable.upper()}/"
        f"{variable.lower()}_face_{face}_depth_52_time_0_10269.idx"
    )
    return ov.LoadDataset(url)


# ---------------------------------------------
# 网格重采样与涡度计算
# ---------------------------------------------
def resample_to_grid(data, target_res=5):
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
    lat_min = lat10 * STORE_RES - 90
    lat_max = lat_min + STORE_RES
    lon_min = lon10 * STORE_RES - 180
    lon_max = lon_min + STORE_RES
    return lat_min, lat_max, lon_min, lon_max


# ---------------------------------------------
# 检测单个 timestep
# ---------------------------------------------
def detect_tropical_cyclones_single(t, db_u, db_v, db_p):
    u_raw = db_u.read(time=t, quality=QUALITY)
    v_raw = db_v.read(time=t, quality=QUALITY)
    p_raw = db_p.read(time=t, quality=QUALITY)

    u = resample_to_grid(u_raw)
    v = resample_to_grid(v_raw)
    p = resample_to_grid(p_raw)

    vort = calc_vorticity(u, v)
    mask_vort = np.abs(vort) > VORT_THRESHOLD
    p_anom = p - np.mean(p)
    mask_p = p_anom < P_LOW_THRESHOLD
    mask = mask_vort & mask_p

    lbl = label(mask)
    props = regionprops(lbl)

    cubes = []
    for region in props:
        if region.area < MIN_REGION_SIZE:
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


# ---------------------------------------------
# 多进程处理每个 face
# ---------------------------------------------
def process_face(face):
    print(f"Processing face {face} ...")
    db_u = load_field("u", face)
    db_v = load_field("v", face)
    db_p = load_field("p", face)
    timesteps = db_u.getTimesteps()
    results_face = []
    for idx in range(0, len(timesteps), TIME_STEP_INTERVAL):
        t = timesteps[idx]
        cubes = detect_tropical_cyclones_single(t, db_u, db_v, db_p)
        results_face.extend(cubes)
        if idx % 168 == 0: #24*7一周输出一次
            print(f"[Face {face}] processed timestep {t}")
    print(f"Face {face} done. Detected {len(results_face)} cubes.")
    return results_face


# ---------------------------------------------
# 主程序
# ---------------------------------------------
if __name__ == "__main__":
    pool = Pool(processes=min(len(FACES), cpu_count()))
    results_all_faces = pool.map(process_face, FACES)
    pool.close()
    pool.join()

    # 合并并去重（相同 time + lat/lon box）
    merged = {}
    for face_list in results_all_faces:
        for cube in face_list:
            key = (cube["time_step"], cube["lat_min"], cube["lat_max"], cube["lon_min"], cube["lon_max"])
            merged[key] = cube

    merged_list = list(merged.values())
    print(f"Total unique cubes after merge: {len(merged_list)}")

    # 保存
    with open("TC_time_space_cube_6faces.json", "w") as f:
        json.dump(merged_list, f, indent=2)

    print("Saved → TC_time_space_cube_6faces.json")
