# 检查时间序列数据脚本
# 用于验证不同时间帧的数据是否不同
import OpenVisus as ov
import numpy as np

# ----------------------------
# 数据集路径与加载
# ----------------------------
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

def load_dataset(variable):
    if variable in ["theta", "w"]:
        base_dir=f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
    elif variable=="u":
        base_dir="mit_output/llc2160_arco/visus.idx"
    else:
        base_dir=f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
    dataset_url = base_url + base_dir
    db = ov.LoadDataset(dataset_url)
    return db

# ----------------------------
# 局部区域参数（与ocean_3d_animation.py保持一致）
# ----------------------------
lat_start, lat_end = 10, 40
lon_start, lon_end = 100, 130
nz = 10
data_quality = -6
skip = None

# ----------------------------
# 读取局部数据函数
# ----------------------------
def read_data_at_time(db, time_step, skip_value=None):
    """读取指定时间步的局部数据"""
    if skip_value is None:
        skip_value = skip
    
    data_full = db.read(time=time_step, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    
    if lat_idx_end <= lat_idx_start or lon_idx_end <= lon_idx_start:
        lat_idx_start = 0
        lat_idx_end = lat_dim
        lon_idx_start = 0
        lon_idx_end = lon_dim
    
    result = data_full[lat_idx_start:lat_idx_end:skip_value,
                       lon_idx_start:lon_idx_end:skip_value,
                       :nz]
    
    if result.size == 0:
        result = data_full[lat_idx_start:lat_idx_end:2,
                           lon_idx_start:lon_idx_end:2,
                           :nz]
    
    return result

# ----------------------------
# 加载数据集
# ----------------------------
print("正在加载数据集...")
U_db = load_dataset("u")
V_db = load_dataset("v")
W_db = load_dataset("w")
Salt_db = load_dataset("salt")
Theta_db = load_dataset("theta")
print("✅ 数据集加载完成\n")

# ----------------------------
# 检查时间序列数据
# ----------------------------
# 使用与ocean_3d_animation.py相同的时间步设置
TIME_START = 0
TIME_END = 90
TIME_STEP = 10
time_steps = list(range(TIME_START, TIME_END + 1, TIME_STEP))

print(f"检查时间步: {time_steps}")
print(f"共 {len(time_steps)} 个时间帧\n")
print("="*80)

# 存储每个时间步的数据
time_series_data = {
    'time_steps': time_steps,
    'U': [],
    'V': [],
    'W': [],
    'Salt': [],
    'Theta': []
}

# 加载所有时间步的数据
for i, t in enumerate(time_steps):
    print(f"\n加载时间步 {t} ({i+1}/{len(time_steps)})...")
    try:
        U_local = read_data_at_time(U_db, t)
        V_local = read_data_at_time(V_db, t)
        W_local = read_data_at_time(W_db, t)
        Salt_local = read_data_at_time(Salt_db, t)
        Theta_local = read_data_at_time(Theta_db, t)
        
        time_series_data['U'].append(U_local)
        time_series_data['V'].append(V_local)
        time_series_data['W'].append(W_local)
        time_series_data['Salt'].append(Salt_local)
        time_series_data['Theta'].append(Theta_local)
        
        print(f"  数据形状: U={U_local.shape}, V={V_local.shape}, W={W_local.shape}")
        print(f"            Salt={Salt_local.shape}, Theta={Theta_local.shape}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("数据统计信息对比")
print("="*80)

# 对每个字段进行统计
for field_name in ['U', 'V', 'W', 'Salt', 'Theta']:
    print(f"\n【{field_name} 字段】")
    print("-" * 80)
    
    if len(time_series_data[field_name]) == 0:
        print("  ⚠️  无数据")
        continue
    
    # 计算每个时间步的统计信息
    stats = []
    for i, data in enumerate(time_series_data[field_name]):
        if data is None or data.size == 0:
            stats.append(None)
            continue
        
        stat = {
            'time_step': time_steps[i],
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'shape': data.shape
        }
        stats.append(stat)
    
    # 打印统计信息
    print(f"{'时间步':<10} {'最小值':<15} {'最大值':<15} {'均值':<15} {'标准差':<15} {'形状':<20}")
    print("-" * 80)
    for stat in stats:
        if stat is None:
            print(f"{'N/A':<10} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<20}")
        else:
            print(f"{stat['time_step']:<10} {stat['min']:<15.6f} {stat['max']:<15.6f} "
                  f"{stat['mean']:<15.6f} {stat['std']:<15.6f} {str(stat['shape']):<20}")
    
    # 检查数据是否相同
    print("\n  数据差异检查:")
    if len(stats) > 1:
        # 比较第一个时间步和其他时间步
        base_stat = stats[0]
        if base_stat is not None:
            all_same = True
            for i in range(1, len(stats)):
                if stats[i] is None:
                    print(f"    ⚠️  时间步 {time_steps[i]}: 数据为空")
                    all_same = False
                    continue
                
                # 比较统计信息
                min_diff = abs(stats[i]['min'] - base_stat['min'])
                max_diff = abs(stats[i]['max'] - base_stat['max'])
                mean_diff = abs(stats[i]['mean'] - base_stat['mean'])
                
                if min_diff < 1e-6 and max_diff < 1e-6 and mean_diff < 1e-6:
                    print(f"    ⚠️  时间步 {time_steps[i]}: 与时间步 {base_stat['time_step']} 完全相同（差异 < 1e-6）")
                    all_same = False
                else:
                    print(f"    ✅ 时间步 {time_steps[i]}: 与时间步 {base_stat['time_step']} 不同")
                    print(f"       最小值差异: {min_diff:.6f}, 最大值差异: {max_diff:.6f}, 均值差异: {mean_diff:.6f}")
            
            if all_same:
                print(f"    ✅ 所有时间步数据都不同")
        else:
            print(f"    ⚠️  基准时间步数据为空")
    else:
        print(f"    ⚠️  只有一个时间步，无法比较")

# ----------------------------
# 详细数据差异分析
# ----------------------------
print("\n" + "="*80)
print("详细数据差异分析（逐元素比较）")
print("="*80)

for field_name in ['U', 'V', 'W', 'Salt', 'Theta']:
    print(f"\n【{field_name} 字段 - 逐元素差异】")
    print("-" * 80)
    
    if len(time_series_data[field_name]) < 2:
        print("  ⚠️  数据不足，无法进行差异分析")
        continue
    
    # 使用第一个时间步作为基准
    base_data = time_series_data[field_name][0]
    if base_data is None or base_data.size == 0:
        print("  ⚠️  基准数据为空")
        continue
    
    print(f"  基准时间步: {time_steps[0]}, 数据形状: {base_data.shape}")
    
    for i in range(1, len(time_series_data[field_name])):
        current_data = time_series_data[field_name][i]
        if current_data is None or current_data.size == 0:
            print(f"  ⚠️  时间步 {time_steps[i]}: 数据为空")
            continue
        
        if current_data.shape != base_data.shape:
            print(f"  ⚠️  时间步 {time_steps[i]}: 数据形状不同 ({current_data.shape} vs {base_data.shape})")
            continue
        
        # 计算逐元素差异
        diff = np.abs(current_data - base_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        num_different = np.sum(diff > 1e-6)
        total_elements = diff.size
        percent_different = (num_different / total_elements) * 100
        
        print(f"  时间步 {time_steps[i]}:")
        print(f"    最大差异: {max_diff:.6f}")
        print(f"    平均差异: {mean_diff:.6f}")
        print(f"    不同元素数: {num_different}/{total_elements} ({percent_different:.2f}%)")
        
        if max_diff < 1e-6:
            print(f"    ⚠️  数据完全相同（最大差异 < 1e-6）")
        elif percent_different < 0.1:
            print(f"    ⚠️  数据几乎相同（只有 {percent_different:.2f}% 的元素不同）")
        else:
            print(f"    ✅ 数据有明显差异")

print("\n" + "="*80)
print("检查完成！")
print("="*80)

