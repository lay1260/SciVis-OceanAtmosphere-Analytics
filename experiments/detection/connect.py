import json
import numpy as np

#-------------------------------------------------
# 加载一周检测结果
#-------------------------------------------------
with open("TC_time_space_cube_6faces_1week_3cores.json") as f:
    detections = json.load(f)

# 转成列表（以防不是）
detections = list(detections)

#-------------------------------------------------
# 预处理：将lat/lon转换为10°网格索引
#-------------------------------------------------
def lat_to_idx(lat):
    return (lat + 90) // 10  # -90→0

def lon_to_idx(lon):
    return (lon + 180) // 10  # -180→0

for d in detections:
    d["lat_idx"] = int(lat_to_idx(d["lat_min"]))
    d["lon_idx"] = int(lon_to_idx(d["lon_min"]))

#-------------------------------------------------
# 合并连续点为台风轨迹
#-------------------------------------------------
tracks = []
track_id = 1

# 按时间排序
detections_sorted = sorted(detections, key=lambda x: x["time_step"])

# 用于记录每个点是否已归类
used = set()

def is_neighbor(d1, d2):
    """判断是否属于同一个台风（时间连续 + 空间相邻）"""
    if abs(d1["time_step"] - d2["time_step"]) > 12:  # 超过12小时不连续
        return False
    if abs(d1["lat_idx"] - d2["lat_idx"]) > 1:
        return False
    if abs(d1["lon_idx"] - d2["lon_idx"]) > 1:
        return False
    return True

for i, d in enumerate(detections_sorted):
    if i in used:
        continue

    # 启动一个新的轨迹
    current_track = {
        "track_id": track_id,
        "start_time": d["time_step"],
        "end_time": d["time_step"],
        "trajectory": [d],
    }
    used.add(i)

    # 继续向后找连接点
    last = d
    for j in range(i+1, len(detections_sorted)):
        if j in used:
            continue
        nxt = detections_sorted[j]
        if is_neighbor(last, nxt):
            current_track["trajectory"].append(nxt)
            current_track["end_time"] = nxt["time_step"]
            used.add(j)
            last = nxt  # 继续往后连

    tracks.append(current_track)
    track_id += 1

#-------------------------------------------------
# 构建时间-空间立方体 (t × lat × lon)
#-------------------------------------------------
# 获取时间范围
times = sorted(list({d["time_step"] for d in detections}))
time_to_idx = {t: i for i, t in enumerate(times)}

T = len(times)
LAT = 18   # -90~90 (10°格)
LON = 36   # -180~180 (10°格)

cube = np.zeros((T, LAT, LON), dtype=np.uint8)

for d in detections_sorted:
    ti = time_to_idx[d["time_step"]]
    cube[ti, d["lat_idx"], d["lon_idx"]] = 1

#-------------------------------------------------
# 保存结果
#-------------------------------------------------
with open("TC_tracks.json", "w") as f:
    json.dump(tracks, f, indent=2)

np.save("TC_time_space_cube.npy", cube)

print("✔ 已生成 TC_tracks.json（台风轨迹）")
print("✔ 已生成 TC_time_space_cube.npy（时间-空间矩阵）")
