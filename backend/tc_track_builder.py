"""
读取风速检测生成的 CSV (time,face,center_x,center_y,max_wind,area)，
按时间顺序将同一面上相近的检测关联成“台风事件”轨迹，输出 JSON。

改进点：
- 支持最小点数/持续时间过滤；
- 输出轨迹的范围 bbox、最大/平均风速、累计面积；
- 参数可通过 CLI 或默认常量调节。

简化假设：
- 不跨 face 关联（每个 face 内独立跟踪）。
- 使用像素坐标距离（受 stride/降采样影响）。
"""

import csv
import json
import sys
import math
from collections import defaultdict

# 默认参数
DEFAULT_DIST_THRESH = 80.0     # 像素距离阈值
DEFAULT_MAX_GAP = 3            # 允许的时间步缺口
DEFAULT_MIN_POINTS = 2         # 轨迹最少点数
DEFAULT_MIN_DURATION = 1       # 轨迹最少持续步数


def load_events(csv_path):
    events = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "time": int(float(row["time"])),
                "face": int(float(row["face"])),
                "x": float(row["center_x"]),
                "y": float(row["center_y"]),
                "max_wind": float(row["max_wind"]),
                "area": int(float(row["area"])),
            })
    return events


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def build_tracks(
    events,
    dist_thresh=DEFAULT_DIST_THRESH,
    max_gap_steps=DEFAULT_MAX_GAP,
    min_points=DEFAULT_MIN_POINTS,
    min_duration=DEFAULT_MIN_DURATION,
):
    """
    简单最近邻关联：同一 face 内，按时间排序。
    dist_thresh: 像素距离阈值，超过则新建轨迹
    max_gap_steps: 允许的时间间隔（time 步数）最大缺口
    min_points: 轨迹最少点数
    min_duration: 轨迹最少持续步数（end_time - start_time + 1）
    """
    tracks = []
    # face 分组
    events_by_face = defaultdict(list)
    for ev in events:
        events_by_face[ev["face"]].append(ev)

    for face, face_events in events_by_face.items():
        face_events.sort(key=lambda e: e["time"])
        active = []
        current_time = None
        for ev in face_events:
            t = ev["time"]
            # 清理过久未更新的轨迹
            active = [tr for tr in active if (t - tr["last_time"]) <= max_gap_steps]
            # 找最近轨迹
            best_idx = -1
            best_d = 1e18
            for idx, tr in enumerate(active):
                d = dist(ev, tr["last"])
                if d < best_d:
                    best_d = d
                    best_idx = idx
            if best_idx >= 0 and best_d <= dist_thresh:
                # 追加到该轨迹
                tr = active[best_idx]
                tr["points"].append(ev)
                tr["last"] = ev
                tr["last_time"] = t
            else:
                # 新轨迹
                active.append({
                    "face": face,
                    "points": [ev],
                    "last": ev,
                    "last_time": t,
                })
        # 结束剩余轨迹
        tracks.extend(active)

    # 计算元信息
    out_tracks = []
    for idx, tr in enumerate(tracks):
        pts = tr["points"]
        times = [p["time"] for p in pts]
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        start_t = min(times)
        end_t = max(times)
        duration = end_t - start_t + 1
        if len(pts) < min_points or duration < min_duration:
            continue
        max_wind = max(p["max_wind"] for p in pts)
        mean_wind = sum(p["max_wind"] for p in pts) / len(pts)
        total_area = sum(p["area"] for p in pts)
        out_tracks.append({
            "id": idx + 1,
            "face": tr["face"],
            "start_time": start_t,
            "end_time": end_t,
            "duration_steps": duration,
            "bbox": {
                "min_x": min(xs), "max_x": max(xs),
                "min_y": min(ys), "max_y": max(ys),
            },
            "max_wind": max_wind,
            "mean_wind": mean_wind,
            "total_area": total_area,
            "total_points": len(pts),
            "points": pts,
        })
    return out_tracks


def main():
    if len(sys.argv) < 3:
        print("用法: python tc_track_builder.py <input_csv> <output_json> [dist_thresh] [max_gap_steps] [min_points] [min_duration]")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_json = sys.argv[2]
    dist_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DIST_THRESH
    max_gap = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_MAX_GAP
    min_points = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_MIN_POINTS
    min_duration = int(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_MIN_DURATION

    events = load_events(input_csv)
    tracks = build_tracks(
        events,
        dist_thresh=dist_thresh,
        max_gap_steps=max_gap,
        min_points=min_points,
        min_duration=min_duration,
    )

    with open(output_json, "w") as f:
        json.dump(tracks, f, indent=2)
    print(f"Tracks: {len(tracks)}, saved → {output_json}")


if __name__ == "__main__":
    main()

