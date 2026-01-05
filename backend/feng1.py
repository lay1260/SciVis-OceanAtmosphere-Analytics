# 1. 读取DYAMOND数据
import argparse
import os
import time

import OpenVisus as ov
import numpy as np
import matplotlib.pyplot as plt

# 设置参数
DATASET_PRESETS = {
    "salt_llc2160": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
    "salt_llc1080_subset": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc1080_salt/salt_llc1080_x_y_depth.idx",
}

parser = argparse.ArgumentParser(description="DYAMOND 台风识别")
parser.add_argument(
    "--dataset-url",
    default=None,
    help="OpenVisus 数据集 URL 或本地 .idx 路径（默认使用环境变量 FENG_DATASET_URL 或示例盐度数据）",
)
parser.add_argument(
    "--dataset-preset",
    choices=list(DATASET_PRESETS.keys()),
    help="使用内置子索引（体量更小的示例数据）",
)
args = parser.parse_args()

DEFAULT_DATASET_URL = (
    os.environ.get("FENG_DATASET_URL")
    or DATASET_PRESETS["salt_llc2160"]
)

if args.dataset_url:
    field = args.dataset_url
    preset_label = "custom-url"
elif args.dataset_preset:
    field = DATASET_PRESETS[args.dataset_preset]
    preset_label = args.dataset_preset
else:
    field = DEFAULT_DATASET_URL
    preset_label = "default"

print(f"当前数据集预设: {preset_label}", flush=True)
print("Dataset URL:", field, flush=True)
print("正在加载 OpenVisus 数据集，请稍候（首次会下载大量数据）...", flush=True)

# 提升 OpenVisus 日志级别，便于查看下载进度
try:
    ov.SetLoggingLevel("Info")
except AttributeError:
    print("[Warning] 当前 OpenVisus 版本不支持 SetLoggingLevel，跳过日志级别设置。", flush=True)

# 加载数据集
start_time = time.time()
db = ov.LoadDataset(field)
elapsed = time.time() - start_time
print(f"数据集加载完成，耗时 {elapsed:.2f} 秒，开始分析。", flush=True)

# 2. 台风识别核心代码
def detect_typhoon_centers(data, time_step):
    # 提取MSLP和850hPa涡度
    print(f"[Progress] 正在读取 time={time_step} 的 MSLP 数据...", flush=True)
    start_time = time.time()
    mslp = db.read(time=time_step, z=[89, 90])  # 近海面气压
    elapsed = time.time() - start_time
    mslp_size_mb = mslp.nbytes / (1024 * 1024)
    print(f"[Progress] MSLP 读取完成: 形状={mslp.shape}, 大小={mslp_size_mb:.2f} MB, 耗时={elapsed:.2f}秒, 范围=[{mslp.min():.2f}, {mslp.max():.2f}]", flush=True)
    
    print(f"[Progress] 正在读取 time={time_step} 的 850hPa 风场...", flush=True)
    start_time = time.time()
    u, v = db.read(time=time_step, z=[80, 81])  # 850hPa风场
    elapsed = time.time() - start_time
    u_size_mb = u.nbytes / (1024 * 1024)
    v_size_mb = v.nbytes / (1024 * 1024)
    print(f"[Progress] 风场读取完成: u形状={u.shape}, {u_size_mb:.2f} MB; v形状={v.shape}, {v_size_mb:.2f} MB, 总耗时={elapsed:.2f}秒", flush=True)
    print("[Progress] 数据读取完毕，开始计算涡度...", flush=True)
    
    # 计算涡度
    dx = 7000  # 网格分辨率(大气7km)
    dy = 7000
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    vorticity = dv_dx - du_dy
    
    # 识别涡度极大值点(台风种子)
    threshold = 5e-5  # 涡度阈值
    max_vort_indices = np.where(vorticity > threshold)
    
    # 检查每个种子点是否对应MSLP极小值
    centers = []
    for i, j in zip(*max_vort_indices):
        # 检查该点周围278km范围内MSLP是否有<1015hPa的极小值
        radius = 278000  # 278km
        y_min = max(0, i - radius//dy)
        y_max = min(mslp.shape[0], i + radius//dy)
        x_min = max(0, j - radius//dx)
        x_max = min(mslp.shape[1], j + radius//dx)
        
        local_mslp = mslp[y_min:y_max, x_min:x_max]
        if np.min(local_mslp) < 1015:
            centers.append((i, j, time_step))
    
    print(f"[Progress] 共检测到 {len(centers)} 个候选台风中心。", flush=True)
    return centers, mslp, vorticity


def visualize_typhoon_centers(time_step=0, save_path=None):
    """
    对检测结果进行可视化：
    - 左图：MSLP分布及台风中心
    - 右图：850hPa涡度场及台风中心
    """
    centers, mslp, vorticity = detect_typhoon_centers(db, time_step)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 绘制MSLP
    ax_mslp = axes[0]
    mslp_plot = ax_mslp.imshow(mslp[0], cmap='coolwarm', origin='lower')
    ax_mslp.set_title(f'MSLP 分布 (time={time_step})')
    ax_mslp.set_xlabel('经度网格 index')
    ax_mslp.set_ylabel('纬度网格 index')
    fig.colorbar(mslp_plot, ax=ax_mslp, fraction=0.046, pad=0.04, label='hPa')

    # 绘制涡度
    ax_vor = axes[1]
    vor_plot = ax_vor.imshow(vorticity, cmap='magma', origin='lower')
    ax_vor.set_title(f'850hPa 涡度 (time={time_step})')
    ax_vor.set_xlabel('经度网格 index')
    ax_vor.set_ylabel('纬度网格 index')
    fig.colorbar(vor_plot, ax=ax_vor, fraction=0.046, pad=0.04, label='s⁻¹')

    # 叠加台风中心
    if centers:
        y_coords = [c[0] for c in centers]
        x_coords = [c[1] for c in centers]
        for ax in axes:
            ax.scatter(x_coords, y_coords, s=30, c='lime', edgecolors='black', linewidths=0.5, label='台风中心')
            ax.legend(loc='upper right')
    else:
        for ax in axes:
            ax.text(0.5, 0.5, '未检测到台风中心', transform=ax.transAxes,
                    ha='center', va='center', color='white',
                    bbox=dict(facecolor='black', alpha=0.5))

    fig.suptitle('DYAMOND 台风识别结果', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    visualize_typhoon_centers(time_step=0)