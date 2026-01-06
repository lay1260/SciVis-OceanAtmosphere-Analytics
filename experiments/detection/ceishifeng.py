# 1. 读取DYAMOND数据
import OpenVisus as ov
import numpy as np
import matplotlib.pyplot as plt

# 设置参数
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

# 根据变量类型选择正确的URL格式
def get_dataset_url(variable="salt"):
    """
    获取数据集URL
    
    参数:
        variable: 变量名，可选 "salt", "u", "v", "theta", "w" 等
    """
    if variable in ["theta", "w"]:
        base_dir = f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
    elif variable == "u":
        base_dir = "mit_output/llc2160_arco/visus.idx"
    else:
        base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
    return base_url + base_dir

# 默认使用salt数据集（更稳定）
field = get_dataset_url("salt")

# 加载数据集（全局变量）
# 需要加载多个数据集：u, v 风场和MSLP（如果有）
db_u = None  # u风场数据集
db_v = None  # v风场数据集
db_mslp = None  # MSLP数据集（如果有单独的数据集）
db = None  # 当前使用的数据集

# 2. 台风识别核心代码
def detect_typhoon_centers(time_step, data_quality=-9):
    """
    识别台风中心
    
    参数:
        time_step: 时间步索引
        data_quality: 数据质量/分辨率级别，负数表示降低分辨率（-9较粗糙但快速，-6较精细但慢）
    
    返回:
        centers: 台风中心位置列表 [(行, 列, 时间步), ...]
    """
    global db, db_u, db_v, db_mslp
    import time
    
    # 检查数据集是否加载
    if db is None:
        raise RuntimeError("数据集未加载，请先调用 load_dataset()")
    
    print("  步骤1: 读取MSLP和风场数据...")
    print(f"    使用数据质量级别: {data_quality} (负数=降低分辨率，节省内存)")
    start_read = time.time()
    
    # 根据数据集类型读取数据
    try:
        # 读取风场数据 - 需要从u和v数据集读取
        if db_u is None or db_v is None:
            print("    警告: 未加载u/v风场数据集")
            print("    提示: 请使用 'python ceishi——feng.py uv' 来加载u和v数据集")
            raise ValueError("需要加载u和v风场数据集。请使用: python ceishi——feng.py uv")
        
        # 从u和v数据集读取850hPa风场（使用quality参数降低分辨率）
        print("    从u数据集读取数据（降低分辨率以节省内存）...")
        u_data = db_u.read(time=time_step, quality=data_quality)
        print(f"    u数据形状: {u_data.shape}")
        
        print("    从v数据集读取数据（降低分辨率以节省内存）...")
        v_data = db_v.read(time=time_step, quality=data_quality)
        print(f"    v数据形状: {v_data.shape}")
        
        # 处理3D数据，取850hPa层（通常是z=80或81）
        if len(u_data.shape) == 3:
            if u_data.shape[2] > 80:
                u = u_data[:, :, 80]  # 850hPa层
                v = v_data[:, :, 80]
            else:
                u = u_data[:, :, -1]
                v = v_data[:, :, -1]
        elif len(u_data.shape) == 2:
            u = u_data
            v = v_data
        else:
            raise ValueError(f"意外的风场数据维度: u={u_data.shape}, v={v_data.shape}")
        
        # 尝试读取MSLP（如果有单独的数据集，否则使用u数据集的底层作为近似）
         # 注意：u数据集可能不包含MSLP，这里我们使用一个简化的方法
        print("    注意: 使用风场数据的底层作为MSLP近似（实际应用中需要单独的MSLP数据集）")
        if len(u_data.shape) == 3:
            # 使用最底层作为MSLP的近似
            mslp = u_data[:, :, -1]  # 使用最后一层作为近似
        else:
            # 如果没有深度维度，使用u数据本身（这不是真正的MSLP，但可以用于测试）
            mslp = u.copy()
            print("    警告: 无法获取真正的MSLP数据，使用风场数据作为近似")
            
    except Exception as e:
        print(f"    ✗ 数据读取失败: {e}")
        if "cannot allocate buffer" in str(e).lower():
            print("    错误原因: 内存不足，尝试读取的数据太大")
            print("    解决方案: 使用更低的quality值（如-12）或读取部分区域")
        else:
            print("    提示: 需要加载包含MSLP和风场的数据集")
            print("    请使用: python ceishi——feng.py uv")
        raise
    
    read_time = time.time() - start_read
    print(f"    ✓ 数据读取完成（耗时 {read_time:.1f} 秒）")
    print(f"    MSLP形状: {mslp.shape}, U形状: {u.shape}, V形状: {v.shape}")
    
    print("  步骤2: 计算涡度...")
    start_vort = time.time()
    # 计算涡度
    dx = 7000  # 网格分辨率(大气7km)
    dy = 7000
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    vorticity = dv_dx - du_dy
    vort_time = time.time() - start_vort
    print(f"    ✓ 涡度计算完成（耗时 {vort_time:.1f} 秒）")
    
    print("  步骤3: 识别涡度极大值点...")
    start_detect = time.time()
    # 识别涡度极大值点(台风种子)
    threshold = 5e-5  # 涡度阈值
    max_vort_indices = np.where(vorticity > threshold)
    num_candidates = len(max_vort_indices[0])
    print(f"    找到 {num_candidates} 个候选点")
    
    # 检查每个种子点是否对应MSLP极小值
    centers = []
    for idx, (i, j) in enumerate(zip(*max_vort_indices)):
        if idx % 100 == 0 and idx > 0:
            print(f"    处理进度: {idx}/{num_candidates} ({idx*100//num_candidates}%)")
        
        # 检查该点周围278km范围内MSLP是否有<1015hPa的极小值
        radius = 278000  # 278km
        y_min = max(0, i - int(radius//dy))
        y_max = min(mslp.shape[0], i + int(radius//dy))
        x_min = max(0, j - int(radius//dx))
        x_max = min(mslp.shape[1], j + int(radius//dx))
        
        local_mslp = mslp[y_min:y_max, x_min:x_max]
        if np.min(local_mslp) < 1015:
            # 转换为Python原生int类型，避免JSON序列化问题
            centers.append((int(i), int(j), int(time_step)))
    
    detect_time = time.time() - start_detect
    print(f"    ✓ 识别完成（耗时 {detect_time:.1f} 秒）")
    total_time = time.time() - start_read
    print(f"  总分析时间: {total_time:.1f} 秒")
    
    return centers

def track_typhoon_centers(initial_centers, start_time_step, end_time_step, max_search_radius=500, data_quality=-9):
    """
    追踪台风中心在所有时间步中的位置
    
    参数:
        initial_centers: 初始时间步的台风中心列表 [(行, 列, 时间步), ...]
        start_time_step: 起始时间步
        end_time_step: 结束时间步（包含）
        max_search_radius: 最大搜索半径（网格单位），用于匹配同一台风在不同时间步的位置
        data_quality: 数据质量/分辨率级别
    
    返回:
        tracks: 字典，键为台风ID，值为该台风在所有时间步的位置列表 [(行, 列, 时间步), ...]
    """
    global db_u, db_v
    
    if db_u is None or db_v is None:
        raise RuntimeError("需要加载u和v风场数据集")
    
    import time
    
    # 获取总时间步数
    timesteps = db_u.getTimesteps()
    total_timesteps = len(timesteps)
    end_time_step = min(end_time_step, total_timesteps - 1)
    
    print(f"\n开始追踪 {len(initial_centers)} 个台风中心")
    print(f"时间步范围: {start_time_step} 到 {end_time_step} (共 {end_time_step - start_time_step + 1} 个时间步)")
    print(f"最大搜索半径: {max_search_radius} 网格单位")
    
    # 初始化追踪字典：每个初始台风中心分配一个ID
    tracks = {}
    for idx, center in enumerate(initial_centers):
        typhoon_id = f"台风_{idx+1}"
        tracks[typhoon_id] = [center]  # 初始位置
    
    # 当前时间步的台风中心位置（用于匹配）
    current_centers = {typhoon_id: center for typhoon_id, center in zip(tracks.keys(), initial_centers)}
    
    # 遍历后续时间步
    for time_step in range(start_time_step + 1, end_time_step + 1):
        print(f"\n{'='*60}")
        print(f"正在分析时间步 {time_step}/{end_time_step}...")
        print(f"{'='*60}")
        
        try:
            # 识别当前时间步的所有台风中心
            new_centers = detect_typhoon_centers(time_step, data_quality=data_quality)
            print(f"时间步 {time_step} 找到 {len(new_centers)} 个台风中心")
            
            if len(new_centers) == 0:
                print("  警告: 未找到台风中心，可能所有台风都已消散")
                # 为所有追踪的台风添加None，表示消失
                for typhoon_id in tracks.keys():
                    tracks[typhoon_id].append(None)
                continue
            
            # 匹配当前时间步的台风中心与之前时间步的位置
            matched_new_centers = set()  # 已匹配的新中心索引
            
            for typhoon_id, prev_center in current_centers.items():
                if prev_center is None:
                    # 如果上一个时间步已经消失，跳过
                    tracks[typhoon_id].append(None)
                    continue
                
                prev_i, prev_j, _ = prev_center
                
                # 寻找最接近的台风中心
                best_match = None
                best_distance = float('inf')
                best_idx = -1
                
                for idx, new_center in enumerate(new_centers):
                    if idx in matched_new_centers:
                        continue  # 已被匹配
                    
                    new_i, new_j, _ = new_center
                    # 计算欧氏距离
                    distance = np.sqrt((new_i - prev_i)**2 + (new_j - prev_j)**2)
                    
                    if distance < best_distance and distance <= max_search_radius:
                        best_distance = distance
                        best_match = new_center
                        best_idx = idx
                
                if best_match is not None:
                    # 找到匹配
                    tracks[typhoon_id].append(best_match)
                    current_centers[typhoon_id] = best_match
                    matched_new_centers.add(best_idx)
                    print(f"  {typhoon_id}: ({prev_i}, {prev_j}) -> ({best_match[0]}, {best_match[1]}), 距离: {best_distance:.1f}")
                else:
                    # 未找到匹配，台风可能消散
                    tracks[typhoon_id].append(None)
                    current_centers[typhoon_id] = None
                    print(f"  {typhoon_id}: ({prev_i}, {prev_j}) -> 消散（未找到匹配）")
            
            # 处理未匹配的新台风中心（新形成的台风）
            for idx, new_center in enumerate(new_centers):
                if idx not in matched_new_centers:
                    new_typhoon_id = f"台风_{len(tracks) + 1}"
                    # 为这个新台风创建轨迹，前面时间步都是None
                    tracks[new_typhoon_id] = [None] * (time_step - start_time_step) + [new_center]
                    current_centers[new_typhoon_id] = new_center
                    print(f"  新台风: {new_typhoon_id} 在时间步 {time_step} 形成，位置: ({new_center[0]}, {new_center[1]})")
            
        except Exception as e:
            print(f"  错误: 时间步 {time_step} 分析失败: {e}")
            # 为所有追踪的台风添加None
            for typhoon_id in tracks.keys():
                tracks[typhoon_id].append(None)
    
    return tracks

def load_dataset(dataset_url=None, retry_count=3, load_wind_fields=True):
    """
    加载OpenVisus数据集
    
    参数:
        dataset_url: 数据集URL，如果为None则使用默认URL
        retry_count: 重试次数
        load_wind_fields: 是否同时加载u和v风场数据集
    """
    global db, db_u, db_v, field
    import time
    
    if dataset_url:
        field = dataset_url
    
    print(f"正在加载数据集: {field}")
    print("首次加载可能需要一些时间，请耐心等待...")
    print("提示: 如果失败，可能是网络问题或服务器暂时不可用")
    print("\n预计时间:")
    print("  - 数据集连接: 10-60秒（取决于网络）")
    print("  - 首次加载元数据: 30-120秒（需要下载索引）")
    print("  - 后续加载: 5-30秒（使用缓存）")
    
    last_error = None
    for attempt in range(1, retry_count + 1):
        try:
            print(f"\n尝试 {attempt}/{retry_count}...")
            start_time = time.time()
            
            # 尝试加载主数据集
            db = ov.LoadDataset(field)
            elapsed = time.time() - start_time
            print(f"✓ 主数据集加载成功（耗时 {elapsed:.1f} 秒）")
            
            # 显示数据集信息
            try:
                logic_box = db.getLogicBox()
                timesteps = db.getTimesteps()
                print(f"  维度: {logic_box[1]}")
                print(f"  时间步数: {len(timesteps)}")
            except:
                pass
            
            # 如果需要加载风场数据，且当前数据集不是u/v
            if load_wind_fields and "salt" in field.lower():
                print("\n正在加载u和v风场数据集...")
                try:
                    u_url = get_dataset_url("u")
                    v_url = get_dataset_url("v")
                    
                    print(f"  加载u数据集: {u_url}")
                    db_u = ov.LoadDataset(u_url)
                    print("  ✓ u数据集加载成功")
                    
                    print(f"  加载v数据集: {v_url}")
                    db_v = ov.LoadDataset(v_url)
                    print("  ✓ v数据集加载成功")
                except Exception as e:
                    print(f"  警告: 风场数据集加载失败: {e}")
                    print("  将尝试仅使用当前数据集")
            
            return db
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            print(f"✗ 尝试 {attempt} 失败: {error_msg[:100]}")
            
            if attempt < retry_count:
                import time
                wait_time = attempt * 2  # 递增等待时间
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"\n所有尝试都失败了。")
                print("\n可能的解决方案:")
                print("1. 检查网络连接")
                print("2. 尝试使用其他数据集（如salt）")
                print("3. 检查服务器是否可访问")
                print("4. 稍后重试")
                raise Exception(f"数据集加载失败（已重试{retry_count}次）: {error_msg}")
    
    raise last_error

# 3. 主程序执行
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("台风识别程序（使用OpenVisus）")
    print("=" * 60)
    
    # 检查是否提供了自定义数据集URL或变量名
    dataset_url = None
    load_uv_only = False  # 是否只加载u和v
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        # 如果指定了uv或u+v，只加载风场数据
        if arg in ["uv", "u+v", "wind"]:
            load_uv_only = True
            print(f"\n只加载u和v风场数据集")
        # 如果是URL（包含http），直接使用
        elif arg.startswith("http"):
            dataset_url = sys.argv[1]
            print(f"\n使用自定义数据集URL: {dataset_url}")
        else:
            # 否则当作变量名
            variable = sys.argv[1]
            dataset_url = get_dataset_url(variable)
            print(f"\n使用变量 '{variable}' 的数据集: {dataset_url}")
    else:
        print(f"\n使用默认数据集URL: {field}")
        print("提示: 可以指定变量名，如: python ceishi——feng.py salt")
        print("     或只加载风场: python ceishi——feng.py uv")
        print("     或指定完整URL: python ceishi——feng.py <URL>")
    
    try:
        import time
        total_start = time.time()
        
        # 使用OpenVisus加载数据集
        if load_uv_only:
            # 只加载u和v风场数据集
            print("\n" + "="*60)
            print("加载u和v风场数据集")
            print("="*60)
            
            u_url = get_dataset_url("u")
            v_url = get_dataset_url("v")
            
            print(f"\n加载u数据集: {u_url}")
            db_u = ov.LoadDataset(u_url)
            print("✓ u数据集加载成功")
            
            try:
                logic_box = db_u.getLogicBox()
                timesteps = db_u.getTimesteps()
                print(f"  维度: {logic_box[1]}")
                print(f"  时间步数: {len(timesteps)}")
            except:
                pass
            
            print(f"\n加载v数据集: {v_url}")
            db_v = ov.LoadDataset(v_url)
            print("✓ v数据集加载成功")
            
            try:
                logic_box = db_v.getLogicBox()
                timesteps = db_v.getTimesteps()
                print(f"  维度: {logic_box[1]}")
                print(f"  时间步数: {len(timesteps)}")
            except:
                pass
            
            # 设置db为u数据集（用于兼容性）
            db = db_u
            print("\n✓ u和v风场数据集加载完成")
        else:
            # 如果使用u数据集，自动加载v数据集
            if dataset_url and ("u" in dataset_url.lower() or (len(sys.argv) > 1 and sys.argv[1].lower() == "u")):
                # 只加载了u，需要同时加载v
                print("\n检测到u数据集，自动加载v数据集...")
                v_url = get_dataset_url("v")
                print(f"加载v数据集: {v_url}")
                db_v = ov.LoadDataset(v_url)
                print("✓ v数据集加载成功")
                # 设置db_u为当前db
                db_u = db
            
            # 如果使用salt数据集，自动加载u和v风场数据集
            auto_load_wind = True
            if dataset_url and ("salt" in dataset_url.lower() or (len(sys.argv) > 1 and sys.argv[1].lower() == "salt")):
                auto_load_wind = True
            
            load_dataset(dataset_url, load_wind_fields=auto_load_wind)
        load_time = time.time() - total_start
        print(f"\n数据集加载总耗时: {load_time:.1f} 秒")
        
        # 检查是否启用追踪模式
        enable_tracking = len(sys.argv) > 2 and sys.argv[2].lower() in ["track", "追踪", "t"]
        
        # 获取时间步范围参数
        start_time_step = 0
        end_time_step = 0  # 默认只分析第一个时间步
        
        if len(sys.argv) > 2:
            arg2 = sys.argv[2].lower()
            if arg2 not in ["track", "追踪", "t"]:
                # 尝试解析为时间步范围
                if "-" in arg2:
                    try:
                        start_time_step, end_time_step = map(int, arg2.split("-"))
                        # 如果指定了时间步范围，自动启用追踪模式
                        if end_time_step > start_time_step:
                            enable_tracking = True
                    except:
                        pass
                elif arg2.isdigit():
                    end_time_step = int(arg2)
                    # 如果指定了结束时间步，自动启用追踪模式
                    if end_time_step > start_time_step:
                        enable_tracking = True
        
        # 如果启用追踪但未指定结束时间步，使用所有时间步
        if enable_tracking and end_time_step == 0:
            try:
                timesteps = db_u.getTimesteps() if db_u else db.getTimesteps()
                end_time_step = len(timesteps) - 1
                print(f"\n追踪模式: 将分析所有时间步 (0 到 {end_time_step})")
            except:
                end_time_step = 100  # 默认追踪100个时间步
                print(f"\n追踪模式: 将分析前 {end_time_step + 1} 个时间步")
        elif enable_tracking and end_time_step > start_time_step:
            print(f"\n追踪模式已启用: 将分析时间步 {start_time_step} 到 {end_time_step} (共 {end_time_step - start_time_step + 1} 个时间步)")
        
        # 分析初始时间步
        print(f"\n{'='*60}")
        print(f"正在分析时间步 {start_time_step}...")
        print(f"{'='*60}")
        
        # 调用台风识别函数（使用OpenVisus读取的数据）
        # 使用quality=-9降低分辨率，避免内存不足（可以调整为-6更精细，-12更粗糙）
        data_quality = -9
        print(f"\n使用数据质量级别: {data_quality} (可以调整: -12=最粗糙但最快, -9=平衡, -6=精细但慢)")
        
        analysis_start = time.time()
        initial_centers = detect_typhoon_centers(start_time_step, data_quality=data_quality)
        analysis_time = time.time() - analysis_start
        
        # 输出初始结果
        print(f"\n{'='*60}")
        print("初始识别结果:")
        print(f"{'='*60}")
        print(f"找到 {len(initial_centers)} 个潜在的台风中心")
        
        if len(initial_centers) > 0:
            print("\n初始台风中心位置 (行, 列, 时间步):")
            for idx, center in enumerate(initial_centers, 1):
                i, j, t = center
                print(f"  {idx}. 位置: ({i}, {j}), 时间步: {t}")
        else:
            print("\n未检测到台风中心")
            print("提示: 可以尝试调整阈值参数或检查其他时间步")
        
        # 如果启用追踪且找到了初始台风中心
        if enable_tracking and len(initial_centers) > 0:
            print(f"\n{'='*60}")
            print("开始追踪台风中心...")
            print(f"{'='*60}")
            
            tracking_start = time.time()
            tracks = track_typhoon_centers(
                initial_centers, 
                start_time_step, 
                end_time_step,
                max_search_radius=500,  # 最大搜索半径（网格单位）
                data_quality=data_quality
            )
            tracking_time = time.time() - tracking_start
            
            # 输出追踪结果
            print(f"\n{'='*60}")
            print("追踪结果:")
            print(f"{'='*60}")
            
            for typhoon_id, trajectory in tracks.items():
                valid_positions = [pos for pos in trajectory if pos is not None]
                if len(valid_positions) > 0:
                    print(f"\n{typhoon_id}:")
                    print(f"  存在时间步数: {len(valid_positions)}/{len(trajectory)}")
                    print(f"  轨迹:")
                    for pos in trajectory:
                        if pos is not None:
                            i, j, t = pos
                            print(f"    时间步 {t}: 位置 ({i}, {j})")
                        else:
                            print(f"    消散")
                    
                    # 计算移动距离和速度（如果有多个位置）
                    if len(valid_positions) > 1:
                        total_distance = 0
                        for i in range(1, len(valid_positions)):
                            prev_i, prev_j, _ = valid_positions[i-1]
                            curr_i, curr_j, _ = valid_positions[i]
                            distance = np.sqrt((curr_i - prev_i)**2 + (curr_j - prev_j)**2) * 7  # 转换为km（假设7km分辨率）
                            total_distance += distance
                        print(f"  总移动距离: {total_distance:.1f} km")
            
            total_time = time.time() - total_start
            print(f"\n时间统计:")
            print(f"  数据集加载: {load_time:.1f} 秒")
            print(f"  初始分析: {analysis_time:.1f} 秒")
            print(f"  追踪分析: {tracking_time:.1f} 秒")
            print(f"  总耗时: {total_time:.1f} 秒")
        else:
            # 只分析单个时间步
            total_time = time.time() - total_start
            print(f"\n时间统计:")
            print(f"  数据集加载: {load_time:.1f} 秒")
            print(f"  数据分析: {analysis_time:.1f} 秒")
            print(f"  总耗时: {total_time:.1f} 秒")
            
            if not enable_tracking:
                print(f"\n提示: 使用 'python ceishifeng.py uv track' 来追踪台风中心在所有时间步中的位置")
            
    except ImportError as e:
        print(f"\n错误: OpenVisus未安装")
        print("请安装: pip install OpenVisus")
        print("或在chinatravel环境中安装: conda activate chinatravel && pip install OpenVisus")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()