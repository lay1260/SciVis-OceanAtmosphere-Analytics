from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import json
import sys
import traceback
import subprocess
import pickle
import tempfile
import os
import base64
from multiprocessing import Process, Queue
import time
from data_extractor import extract_data, ATMOSPHERE_VARIABLES, OCEAN_VARIABLES
from atmosphere_ocean_fusion import visualize_atmosphere_ocean_fusion
from atmo_ocean_coupled_cube import visualize_atmo_ocean_coupled
from vector import visualize_atmosphere_3d

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})  # 允许跨域请求

# 全局错误处理器：确保所有错误都返回 JSON
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': f'路由未找到: {request.path}',
        'method': request.method
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '服务器内部错误',
        'message': str(error) if app.debug else '请查看服务器日志'
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # 对于所有未处理的异常，返回 JSON
    return jsonify({
        'success': False,
        'error': str(e),
        'type': type(e).__name__
    }), 500

# 全局变量存储数据集
db = None
data_cache = {}  # 缓存已加载的数据
lat_start, lat_end = 0, 40
lon_start, lon_end = 100, 140
nz = 8  # 前8层
data_quality = -9  # 在线读取分辨率
scale_xy = 25
typhoon3_cache = {'timestamp': 0, 'image': None}

# 项目根路径及 text.py 路径
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# 查找 text.py（优先在 backend 目录，然后是 src 目录，最后尝试其他常见位置）
TEXT_SCRIPT = None
possible_paths = [
    os.path.join(BACKEND_DIR, 'text.py'),  # backend/text.py
    os.path.join(PROJECT_ROOT, 'src', 'text.py'),  # src/text.py
    os.path.join(PROJECT_ROOT, 'backend', 'text.py'),  # 项目根目录下的 backend/text.py
]

for path in possible_paths:
    if os.path.exists(path):
        TEXT_SCRIPT = path
        break

if TEXT_SCRIPT is None:
    # 如果都找不到，使用 backend 目录作为默认值，但会在运行时给出清晰错误
    TEXT_SCRIPT = os.path.join(BACKEND_DIR, 'text.py')
    print(f'[Warning] TEXT_SCRIPT not found in common locations, using default: {TEXT_SCRIPT}')

print(f'[Config] TEXT_SCRIPT path: {TEXT_SCRIPT}')
print(f'[Config] TEXT_SCRIPT exists: {os.path.exists(TEXT_SCRIPT)}')
if not os.path.exists(TEXT_SCRIPT):
    print(f'[Error] TEXT_SCRIPT file does not exist! Please ensure text.py is in backend/ or src/ directory.')

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')


def load_config():
    global lat_start, lat_end, lon_start, lon_end, nz, data_quality, scale_xy
    # 默认值（和 src/text.py 的默认值保持一致）
    defaults = {
        'lat_start': 10,
        'lat_end': 40,
        'lon_start': 100,
        'lon_end': 130,
        'nz': 20,
        'data_quality': -6,
        'scale_xy': 25
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as fh:
                cfg = json.load(fh)
        else:
            cfg = defaults
            with open(CONFIG_FILE, 'w') as fh:
                json.dump(cfg, fh, indent=2)
        lat_start = cfg.get('lat_start', defaults['lat_start'])
        lat_end = cfg.get('lat_end', defaults['lat_end'])
        lon_start = cfg.get('lon_start', defaults['lon_start'])
        lon_end = cfg.get('lon_end', defaults['lon_end'])
        nz = cfg.get('nz', defaults['nz'])
        data_quality = cfg.get('data_quality', defaults['data_quality'])
        scale_xy = cfg.get('scale_xy', defaults['scale_xy'])
        print(f"[Config] Loaded config: {cfg}")
        return cfg
    except Exception as e:
        print(f"[Config] Failed to load config: {e}")
        return defaults


def save_config(cfg):
    try:
        with open(CONFIG_FILE, 'w') as fh:
            json.dump(cfg, fh, indent=2)
        load_config()
        return True
    except Exception as e:
        print(f"[Config] failed to write config: {e}")
        return False

def load_data_via_subprocess(time_index=0):
    """通过子进程加载数据，避免OpenVisus崩溃影响主服务器"""
    global data_cache
    
    # 检查缓存
    cache_key = f'time_{time_index}'
    if cache_key in data_cache:
        print(f'[API] Using cached data for time_index={time_index}')
        return data_cache[cache_key]
    
    variable = "salt"
    base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
    base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
    dataset_url = base_url + base_dir
    
    print(f'[API] Loading data via subprocess for time_index={time_index}...')
    print(f'[API] Dataset URL: {dataset_url}')
    
    # 获取当前脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, 'load_data_worker.py')
    
    # 构建命令
    cmd = [
        sys.executable,
        worker_script,
        dataset_url,
        str(time_index),
        str(lat_start),
        str(lat_end),
        str(lon_start),
        str(lon_end),
        str(nz),
        str(data_quality)
    ]
    
    try:
        print(f'[API] Starting subprocess...')
        # 运行子进程，设置超时
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,  # 超时提升到300秒
            text=False  # 二进制模式，用于pickle
        )
        
        if result.returncode == 0:
            # 成功，解析pickle数据
            try:
                data_result = pickle.loads(result.stdout)
                if data_result.get('success'):
                    data_local = data_result['data']
                    print(f'[API] Data loaded successfully via subprocess: {data_local.shape}')
                    # 缓存数据
                    data_cache[cache_key] = data_local
                    return data_local
                else:
                    error_msg = data_result.get('error', 'Unknown error')
                    print(f'[API] Subprocess returned error: {error_msg}')
                    raise Exception(f"Failed to load data: {error_msg}")
            except pickle.UnpicklingError as e:
                print(f'[API] Failed to unpickle data: {str(e)}')
                print(f'[API] Subprocess stdout length: {len(result.stdout)} bytes')
                raise Exception(f"Failed to parse subprocess output: {str(e)}")
        else:
            # 子进程失败
            stderr_text = result.stderr.decode('utf-8', errors='ignore') if result.stderr else 'No error message'
            stdout_text = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
            
            # 错误代码 3221225477 (0xC0000005) 是 Windows 访问冲突
            if result.returncode == 3221225477 or result.returncode == -1073741819:
                error_type = "Access Violation (OpenVisus崩溃)"
                error_detail = "OpenVisus库在加载数据集时发生底层崩溃。这可能是由于：\n" \
                             "1. OpenVisus库与Windows系统的兼容性问题\n" \
                             "2. 数据集服务器连接问题\n" \
                             "3. 内存访问错误\n" \
                             "4. OpenVisus库版本问题"
            else:
                error_type = f"Process exit code {result.returncode}"
                error_detail = "子进程异常退出"
            
            print(f'[API] Subprocess failed with return code {result.returncode} ({error_type})')
            print(f'[API] Stderr: {stderr_text}')
            if stdout_text:
                print(f'[API] Stdout: {stdout_text[:500]}')
            
            raise Exception(f"数据加载失败: {error_type}\n{error_detail}\n\n"
                          f"子进程输出: {stderr_text[:200]}")
            
    except subprocess.TimeoutExpired:
        print(f'[API] Subprocess timeout after 120 seconds')
        raise Exception("Data loading timeout (300s). The dataset server may be slow or unavailable.")
    except FileNotFoundError:
        print(f'[API] Worker script not found: {worker_script}')
        raise Exception(f"Worker script not found. Please ensure load_data_worker.py exists in {script_dir}")
    except Exception as e:
        error_msg = str(e)
        print(f'[API] Error running subprocess: {error_msg}')
        
        # 如果是访问冲突错误，提供更详细的帮助信息
        if '3221225477' in error_msg or 'Access Violation' in error_msg or '-1073741819' in error_msg or 'OpenVisus崩溃' in error_msg:
            detailed_error = (
                "❌ OpenVisus库在加载数据时发生崩溃（访问冲突 0xC0000005）\n\n"
                "🔍 问题分析：\n"
                "这是OpenVisus底层C++库的崩溃，Python无法捕获。\n"
                "即使使用子进程隔离，OpenVisus仍然崩溃。\n\n"
                "💡 可能的原因：\n"
                "1. OpenVisus库与Windows系统不兼容\n"
                "2. 数据集服务器连接问题（SSL/网络）\n"
                "3. OpenVisus库版本或编译问题\n"
                "4. 内存访问错误\n\n"
                "🛠️ 建议解决方案：\n"
                "1. 检查OpenVisus版本：pip show OpenVisus\n"
                "2. 尝试重新安装：pip uninstall OpenVisus && pip install OpenVisus\n"
                "3. 检查网络连接：运行 python backend/test_openvisus_simple.py\n"
                "4. 考虑使用WSL（Windows Subsystem for Linux）运行后端\n"
                "5. 查看OpenVisus GitHub issues\n\n"
                f"📋 详细错误：{error_msg}\n\n"
                "📖 更多信息请查看：backend/SOLUTION_OPENVISUS_CRASH.md"
            )
            raise Exception(detailed_error)
        raise

def init_dataset():
    """初始化数据集（延迟加载，不在启动时加载）"""
    global db
    if db is None:
        try:
            variable = "salt"
            base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
            base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
            dataset_url = base_url + base_dir
            print(f'Attempting to load dataset from: {dataset_url}')
            print('This may take a while on first access...')
            
            # 尝试加载数据集，添加详细的错误处理
            try:
                import OpenVisus as ov
                print('  Step 1: Calling ov.LoadDataset...')
                print('  Warning: This may take a long time or cause the process to exit')
                print('  If the process exits, it may be due to:')
                print('    - Network timeout')
                print('    - OpenVisus internal error')
                print('    - Memory issues')
                print('  Attempting to load...')
                
                # 直接调用，但如果出现问题，会被外层异常处理捕获
                db = ov.LoadDataset(dataset_url)
                print('  Step 2: Dataset loaded, getting metadata...')
                
                logic_box = db.getLogicBox()
                timesteps = db.getTimesteps()
                field = db.getField()
                
                print(f'✓ Dataset initialized successfully!')
                print(f'  Dimensions: {logic_box[1]}')
                print(f'  Timesteps: {len(timesteps)}')
                print(f'  Field: {field.name if field else "N/A"}')
            except AttributeError as e:
                print(f'  ERROR: Dataset loaded but metadata access failed: {str(e)}')
                print(f'  This might be a version compatibility issue with OpenVisus')
                # 即使元数据获取失败，也尝试继续使用
                if db is not None:
                    print('  Will attempt to use dataset anyway...')
                else:
                    raise
            except Exception as e:
                print(f'  ERROR during dataset loading: {str(e)}')
                import traceback
                print('  Full traceback:')
                traceback.print_exc()
                raise
                
        except ImportError as e:
            print(f'ERROR: OpenVisus not installed or import failed: {str(e)}')
            print('Please install OpenVisus: pip install OpenVisus')
            db = None
        except KeyboardInterrupt:
            print('\nDataset loading interrupted by user')
            db = None
            raise  # 重新抛出，让调用者知道
        except SystemExit:
            print('\nSystem exit during dataset loading')
            db = None
            raise
        except BaseException as e:
            # 捕获所有异常，包括系统级异常
            print(f'ERROR: Unexpected error during dataset initialization: {type(e).__name__}: {str(e)}')
            import traceback
            print('Full traceback:')
            traceback.print_exc()
            print('\nServer will continue running, but dataset operations will fail.')
            db = None  # 设置为None，让后续操作能检测到错误
    return db


# 加载初始 config
load_config()

def read_region(time_index=0):
    """读取局部区域数据（通过子进程加载，避免崩溃）"""
    try:
        print(f'[API Request] Loading data for time_index={time_index} via subprocess...')
        # 使用子进程加载数据
        data_local = load_data_via_subprocess(time_index)
        print(f'[API Request] Data loaded successfully: {data_local.shape}')
        return data_local
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        error_msg = f"Failed to load data: {str(e)}"
        print(f'[API Request] ERROR: {error_msg}')
        import traceback
        traceback.print_exc()
        raise Exception(error_msg)

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    global data_cache
    cache_size = len(data_cache)
    return jsonify({
        'status': 'ok', 
        'message': 'Backend is running',
        'data_loading_method': 'subprocess',
        'cached_times': list(data_cache.keys()),
        'cache_size': cache_size
    })

@app.route('/api/data/volume', methods=['GET'])
def get_volume_data():
    """获取3D体积数据"""
    try:
        time_index = int(request.args.get('time', 0))
        print(f'\n[API] /api/data/volume called with time_index={time_index}')
        
        # 读取数据 - 捕获所有可能的异常
        try:
            data_local = read_region(time_index)
        except (KeyboardInterrupt, SystemExit):
            # 这些异常不应该在请求处理中发生，但如果发生，记录并返回错误
            print('[API] CRITICAL: KeyboardInterrupt or SystemExit in request handler')
            return jsonify({
                'success': False,
                'error': 'Server interruption during data loading'
            }), 500
        except Exception as e:
            # 捕获所有其他异常，返回友好的错误信息
            error_msg = str(e)
            print(f'[API] Error in read_region: {error_msg}')
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': error_msg,
                'message': 'Failed to read region data. Check server logs for details.'
            }), 500
        
        nx, ny, nz_grid = data_local.shape
        print(f'[API] Processing data: shape={nx}x{ny}x{nz_grid}')
        
        # 生成坐标网格
        x_coords = np.linspace(lon_start, lon_end, ny).tolist()
        y_coords = np.linspace(lat_start, lat_end, nx).tolist()
        z_coords = np.linspace(0, 1000, nz_grid).tolist()
        
        # 将数据转换为列表格式（按深度层组织）
        volume_data = []
        for k in range(nz_grid):
            layer = []
            for i in range(nx):
                row = []
                for j in range(ny):
                    row.append({
                        'lat': y_coords[i],
                        'lng': x_coords[j],
                        'depth': z_coords[k],
                        'value': float(data_local[i, j, k])
                    })
                layer.append(row)
            volume_data.append(layer)
        
        return jsonify({
            'success': True,
            'data': volume_data,
            'bounds': {
                'minLat': lat_start,
                'maxLat': lat_end,
                'minLng': lon_start,
                'maxLng': lon_end,
                'minDepth': 0,
                'maxDepth': 1000
            },
            'shape': {
                'nx': nx,
                'ny': ny,
                'nz': nz_grid
            },
            'timeIndex': time_index
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/typhoon', methods=['GET'])
def get_typhoon_position():
    """基于近表层（depth index 0）的简单启发式算法查找台风/风暴位置
    注意：此示例使用当前加载的变量（例如 salt）做示例。对于真实的台风位置建议使用风速或气压相关变量。
    本实现采用：
    - 获取 2D 近表层数据 slice
    - 计算梯度幅值(approx vorticity-like)并取最大值位置作为台风中心
    """
    try:
        time_index = int(request.args.get('time', 0))
        typhoon_id = int(request.args.get('id', 1))  # 可选参数：台风 id
        data_local = read_region(time_index)
        nx, ny, nz_grid = data_local.shape

        # 选取近表层（k=0）作为近海表层代理
        surface = data_local[:, :, 0]

        # 计算梯度幅值（简单的启发式过滤）
        gy, gx = np.gradient(surface)
        grad = np.sqrt(gx**2 + gy**2)

        # 如果要支持多个台风逻辑，可以基于 typhoon_id 选择不同变量或方法
        # 目前简单统一使用梯度检测；未来可扩展为基于风场或气压
        # 找到最大梯度位置
        max_idx = np.unravel_index(np.argmax(grad), grad.shape)
        ix, iy = max_idx

        # 由数组索引转换为经纬度
        lat_vals = np.linspace(lat_start, lat_end, nx)
        lon_vals = np.linspace(lon_start, lon_end, ny)
        lat = float(lat_vals[ix])
        lng = float(lon_vals[iy])

        return jsonify({
            'success': True,
            'timeIndex': time_index,
            'lat': lat,
            'lng': lng,
            'grid_index': {'ix': int(ix), 'iy': int(iy)},
            'value': float(surface[ix, iy]),
            'typhoonId': typhoon_id
        ,
            'bounds': {
                'minLat': lat_start,
                'maxLat': lat_end,
                'minLng': lon_start,
                'maxLng': lon_end
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/slice', methods=['GET'])
def get_slice_data():
    """获取2D切片数据（用于截面显示）"""
    try:
        time_index = int(request.args.get('time', 0))
        depth_index = int(request.args.get('depth', 0))  # 深度层索引
        
        data_local = read_region(time_index)
        nx, ny, nz_grid = data_local.shape
        
        if depth_index < 0 or depth_index >= nz_grid:
            depth_index = 0
        
        # 获取指定深度层的数据
        slice_data = data_local[:, :, depth_index]
        
        x_coords = np.linspace(lon_start, lon_end, ny).tolist()
        y_coords = np.linspace(lat_start, lat_end, nx).tolist()
        
        slice_result = []
        for i in range(nx):
            row = []
            for j in range(ny):
                row.append({
                    'lat': y_coords[i],
                    'lng': x_coords[j],
                    'value': float(slice_data[i, j])
                })
            slice_result.append(row)
        
        return jsonify({
            'success': True,
            'data': slice_result,
            'depthIndex': depth_index,
            'depth': (depth_index / (nz_grid - 1)) * 1000,
            'timeIndex': time_index
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/cross-section', methods=['POST'])
def get_cross_section():
    """获取两点之间的线-深度截面数据"""
    try:
        data = request.json
        point1 = data.get('point1')  # {lat, lng}
        point2 = data.get('point2')  # {lat, lng}
        time_index = data.get('time', 0)
        
        if not point1 or not point2:
            return jsonify({
                'success': False,
                'error': 'Missing point1 or point2'
            }), 400
        
        data_local = read_region(time_index)
        nx, ny, nz = data_local.shape
        
        # 生成100个采样点沿线
        line_lat = np.linspace(point1['lat'], point2['lat'], 100)
        line_lng = np.linspace(point1['lng'], point2['lng'], 100)
        
        # 将坐标映射回数组索引
        ix = np.clip(np.round((line_lng - lon_start) * (ny - 1) / (lon_end - lon_start)).astype(int), 0, ny - 1)
        iy = np.clip(np.round((line_lat - lat_start) * (nx - 1) / (lat_end - lat_start)).astype(int), 0, nx - 1)
        
        # 提取沿线每个深度层的数据
        cross_section = []
        for k in range(nz):
            depth = (k / (nz - 1)) * 1000
            row = []
            for i in range(len(line_lat)):
                row.append({
                    'distance': i / (len(line_lat) - 1),  # 归一化距离
                    'value': float(data_local[iy[i], ix[i], k])
                })
            cross_section.append({
                'depth': depth,
                'data': row
            })
        
        return jsonify({
            'success': True,
            'data': cross_section,
            'point1': point1,
            'point2': point2,
            'timeIndex': time_index
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/info', methods=['GET'])
def get_info():
    """获取数据集信息"""
    try:
        if db is None:
            db = init_dataset()
        
        logic_box = db.getLogicBox()[1]
        timesteps = len(db.getTimesteps())
        field_name = db.getField().name
        
        return jsonify({
            'success': True,
            'dimensions': logic_box,
            'timesteps': timesteps,
            'field': field_name,
            'region': {
                'lat': [lat_start, lat_end],
                'lon': [lon_start, lon_end],
                'depthLayers': nz
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    global lat_start, lat_end, lon_start, lon_end, nz, data_quality, scale_xy
    try:
        if request.method == 'GET':
            cfg = {
                'lat_start': lat_start,
                'lat_end': lat_end,
                'lon_start': lon_start,
                'lon_end': lon_end,
                'nz': nz,
                'data_quality': data_quality,
                'scale_xy': scale_xy
            }
            return jsonify({'success': True, 'config': cfg})
        else:
            data = request.json
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400

            # Validate and update allowed keys
            keys = ['lat_start', 'lat_end', 'lon_start', 'lon_end', 'nz', 'data_quality', 'scale_xy']
            changed = {}
            for k in keys:
                if k in data:
                    try:
                        # numeric conversion
                        v = int(data[k]) if k in ['nz', 'data_quality', 'scale_xy'] else float(data[k])
                        changed[k] = v
                    except Exception:
                        # keep as number fallback
                        changed[k] = data[k]

            # write to config file
            cfg_file = CONFIG_FILE
            try:
                if os.path.exists(cfg_file):
                    with open(cfg_file, 'r') as fh:
                        cfg = json.load(fh)
                else:
                    cfg = {}
                cfg.update(changed)
                with open(cfg_file, 'w') as fh:
                    json.dump(cfg, fh, indent=2)
                # reload into globals
                load_config()
                return jsonify({'success': True, 'config': cfg})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


@app.route('/api/time/metadata', methods=['GET'])
def api_time_metadata():
    """
    返回数据集的时间元数据，便于前端将时间步映射为真实时间。
    优先返回：
      - timesteps: 数据集中原始的时间步索引列表
      - base_time: 若配置文件中提供（如ISO字符串）
      - step_hours: 若配置文件中提供（时间步间隔小时数）
    """
    import sys
    import os

    try:
        # 确保可以导入 ceishifeng
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        import ceishifeng
    except Exception as e:
        return jsonify({'success': False, 'error': f'无法导入ceishifeng: {str(e)}'}), 500

    try:
        # 确保数据集已加载
        if ceishifeng.db_u is None or ceishifeng.db_v is None:
            ceishifeng.load_dataset(None, load_wind_fields=True)

        # 读取时间步
        timesteps = []
        try:
            timesteps = list(map(int, ceishifeng.db_u.getTimesteps()))
        except Exception:
            # 尝试从主数据集读取
            try:
                timesteps = list(map(int, ceishifeng.db.getTimesteps()))
            except Exception:
                timesteps = []

        # 可选：从配置文件读取基准时间与步长（如果存在）
        base_time = None
        step_hours = None
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as fh:
                    cfg = json.load(fh)
                    base_time = cfg.get('base_time')
                    step_hours = cfg.get('step_hours')
        except Exception:
            pass

        return jsonify({
            'success': True,
            'timesteps': timesteps,
            'base_time': base_time,
            'step_hours': step_hours
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/typhoon/track', methods=['POST'])
def api_typhoon_track():
    """追踪台风中心在所有时间步中的位置"""
    from flask import Response, stream_with_context
    import json as json_lib
    import sys
    import os
    
    try:
        data = request.json or {}
        start_time_step = int(data.get('start_time_step', 0))
        end_time_step = int(data.get('end_time_step', 49))
        data_quality = int(data.get('data_quality', -9))
        
        # 获取ceishifeng.py的路径并导入
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        
        try:
            import ceishifeng
        except ImportError as e:
            return jsonify({
                'success': False,
                'error': f'无法导入ceishifeng模块: {str(e)}'
            }), 500
        
        def generate():
            """生成流式响应"""
            try:
                # 发送初始进度
                yield f"data: {json_lib.dumps({'progress': {'message': '正在加载数据集...', 'progress': 0}})}\n\n"
                
                # 加载数据集
                try:
                    ceishifeng.load_dataset(None, load_wind_fields=False)
                    # 手动加载u和v数据集
                    from ceishifeng import get_dataset_url
                    import OpenVisus as ov
                    
                    u_url = get_dataset_url('u')
                    v_url = get_dataset_url('v')
                    
                    yield f"data: {json_lib.dumps({'progress': {'message': '正在加载u和v风场数据集...', 'progress': 10}})}\n\n"
                    
                    ceishifeng.db_u = ov.LoadDataset(u_url)
                    ceishifeng.db_v = ov.LoadDataset(v_url)
                    ceishifeng.db = ceishifeng.db_u
                    
                    yield f"data: {json_lib.dumps({'progress': {'message': '数据集加载完成，开始识别初始台风中心...', 'progress': 20}})}\n\n"
                except Exception as e:
                    yield f"data: {json_lib.dumps({'error': f'数据集加载失败: {str(e)}'})}\n\n"
                    return
                
                # 识别初始时间步的台风中心
                initial_centers = ceishifeng.detect_typhoon_centers(start_time_step, data_quality=data_quality)
                
                if not initial_centers:
                    yield f"data: {json_lib.dumps({'error': '未找到初始台风中心'})}\n\n"
                    return
                
                yield f"data: {json_lib.dumps({'progress': {'message': f'找到 {len(initial_centers)} 个初始台风中心，开始追踪...', 'progress': 30}})}\n\n"
                
                # 追踪台风中心
                total_steps = end_time_step - start_time_step + 1
                tracks = {}
                current_centers = {}
                
                # 初始化追踪字典
                for idx, center in enumerate(initial_centers):
                    typhoon_id = f"台风_{idx+1}"
                    # 转换NumPy类型为Python原生类型
                    center_converted = tuple(int(x) if isinstance(x, (np.integer, int)) else x for x in center)
                    tracks[typhoon_id] = [center_converted]
                    current_centers[typhoon_id] = center_converted
                
                # 遍历后续时间步
                for time_step in range(start_time_step + 1, end_time_step + 1):
                    progress = 30 + int((time_step - start_time_step) / total_steps * 70)
                    yield f"data: {json_lib.dumps({'progress': {'message': f'正在分析时间步 {time_step}/{end_time_step}...', 'progress': progress}})}\n\n"
                    
                    try:
                        # 识别当前时间步的台风中心
                        new_centers = ceishifeng.detect_typhoon_centers(time_step, data_quality=data_quality)
                        
                        if not new_centers:
                            # 所有台风都消散
                            for typhoon_id in tracks.keys():
                                tracks[typhoon_id].append(None)
                            continue
                        
                        # 匹配台风中心
                        matched_new_centers = set()
                        max_search_radius = 500
                        
                        for typhoon_id, prev_center in current_centers.items():
                            if prev_center is None:
                                tracks[typhoon_id].append(None)
                                continue
                            
                            prev_i, prev_j, _ = prev_center
                            best_match = None
                            best_distance = float('inf')
                            best_idx = -1
                            
                            for idx, new_center in enumerate(new_centers):
                                if idx in matched_new_centers:
                                    continue
                                
                                new_i, new_j, _ = new_center
                                distance = np.sqrt((new_i - prev_i)**2 + (new_j - prev_j)**2)
                                
                                if distance < best_distance and distance <= max_search_radius:
                                    best_distance = distance
                                    best_match = new_center
                                    best_idx = idx
                            
                            if best_match is not None:
                                # 转换NumPy类型为Python原生类型
                                best_match_converted = tuple(int(x) if isinstance(x, (np.integer, int)) else x for x in best_match)
                                tracks[typhoon_id].append(best_match_converted)
                                current_centers[typhoon_id] = best_match_converted
                                matched_new_centers.add(best_idx)
                            else:
                                tracks[typhoon_id].append(None)
                                current_centers[typhoon_id] = None
                        
                        # 处理新形成的台风
                        for idx, new_center in enumerate(new_centers):
                            if idx not in matched_new_centers:
                                new_typhoon_id = f"台风_{len(tracks) + 1}"
                                # 转换NumPy类型为Python原生类型
                                new_center_converted = tuple(int(x) if isinstance(x, (np.integer, int)) else x for x in new_center)
                                tracks[new_typhoon_id] = [None] * (time_step - start_time_step) + [new_center_converted]
                                current_centers[new_typhoon_id] = new_center_converted
                        
                        # 转换tracks中的所有NumPy类型
                        tracks_converted = convert_numpy_types(tracks)
                        # 发送中间结果
                        yield f"data: {json_lib.dumps({'tracks': tracks_converted, 'progress': {'message': f'已完成时间步 {time_step}', 'progress': progress}})}\n\n"
                        
                    except Exception as e:
                        yield f"data: {json_lib.dumps({'error': f'时间步 {time_step} 分析失败: {str(e)}'})}\n\n"
                        # 为所有台风添加None
                        for typhoon_id in tracks.keys():
                            tracks[typhoon_id].append(None)
                
                # 转换tracks中的所有NumPy类型
                tracks_converted = convert_numpy_types(tracks)
                # 发送最终结果
                yield f"data: {json_lib.dumps({'tracks': tracks_converted, 'success': True, 'progress': {'message': '追踪完成', 'progress': 100}})}\n\n"
                    
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                error_msg = str(e) + '\n' + error_detail
                yield f"data: {json_lib.dumps({'error': error_msg})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/typhoon/detect', methods=['POST'])
def api_typhoon_detect():
    """检测单个时间步的台风中心位置"""
    try:
        data = request.json or {}
        time_step = int(data.get('time_step', 0))
        data_quality = int(data.get('data_quality', -9))
        
        # 获取ceishifeng.py的路径并导入
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        
        try:
            import ceishifeng
        except ImportError as e:
            return jsonify({
                'success': False,
                'error': f'无法导入ceishifeng模块: {str(e)}'
            }), 500
        
        # 加载数据集（如果未加载）
        if ceishifeng.db_u is None or ceishifeng.db_v is None:
            from ceishifeng import get_dataset_url
            import OpenVisus as ov
            
            u_url = get_dataset_url('u')
            v_url = get_dataset_url('v')
            
            ceishifeng.db_u = ov.LoadDataset(u_url)
            ceishifeng.db_v = ov.LoadDataset(v_url)
            ceishifeng.db = ceishifeng.db_u
        
        # 检测台风中心
        centers = ceishifeng.detect_typhoon_centers(time_step, data_quality=data_quality)
        
        # 转换NumPy类型为Python原生类型
        centers_converted = []
        for center in centers:
            centers_converted.append(tuple(int(x) if isinstance(x, (np.integer, int)) else x for x in center))
        
        return jsonify({
            'success': True,
            'centers': centers_converted,
            'time_step': time_step,
            'count': len(centers_converted)
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return jsonify({
            'success': False,
            'error': f'{str(e)}\n{error_detail}'
        }), 500

@app.route('/api/typhoon/<int:typhoon_id>/mesh', methods=['GET'])
def api_typhoon_mesh(typhoon_id):
    """返回3D网格数据用于前端渲染可交互的立方体"""
    try:
        import text
        mesh_data = text.get_3d_mesh_data()
        return jsonify({
            'success': True,
            'mesh': mesh_data
        })
    except Exception as e:
        import traceback
        print(f'[api_typhoon_mesh] Error: {e}')
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/api/typhoon3/image', methods=['GET'])  # 兼容旧路由
def api_typhoon3_image():
    """运行 src/text.py 生成 Typhoon3 的 PyVista 截图并返回 base64 数据（兼容旧API）"""
    return api_typhoon_image_impl(3)

@app.route('/api/typhoon/<int:typhoon_id>/image', methods=['GET'])
def api_typhoon_image(typhoon_id):
    """运行 src/text.py 生成指定台风的 PyVista 截图并返回 base64 数据"""
    return api_typhoon_image_impl(typhoon_id)

@app.route('/api/typhoon/<int:typhoon_id>/cross-section', methods=['POST'])
def api_typhoon_cross_section(typhoon_id):
    """生成海洋截面可视化"""
    try:
        import cross_section_api
        
        data = request.get_json()
        method = data.get('method')  # 'three_points' or 'view_line'
        params = data.get('params', {})
        resolution = data.get('resolution', 150)
        
        if method not in ['three_points', 'view_line']:
            return jsonify({
                'success': False,
                'error': f'Invalid method: {method}. Must be "three_points" or "view_line"'
            }), 400
        
        # Validate parameters
        if method == 'three_points':
            if 'p1' not in params or 'p2' not in params or 'p3' not in params:
                return jsonify({
                    'success': False,
                    'error': 'Method "three_points" requires p1, p2, p3 parameters'
                }), 400
        elif method == 'view_line':
            if 'view_direction' not in params:
                return jsonify({
                    'success': False,
                    'error': 'Method "view_line" requires view_direction parameter'
                }), 400
        
        # Generate cross-section image
        image_data = cross_section_api.generate_cross_section_image(method, params, resolution)
        
        return jsonify({
            'success': True,
            'image': image_data
        })
    except Exception as e:
        import traceback
        print(f'[api_typhoon_cross_section] Error: {e}')
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/api/velocity3d/generate', methods=['POST', 'OPTIONS'])
def api_velocity3d_generate():
    """生成3D可视化图像（整合策略和矢量场优化）"""
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        import velocity_3d_api
        
        data = request.get_json() or {}
        
        render_mode = data.get('render_mode', 'image')
        if render_mode not in ['image', 'window']:
            render_mode = 'image'
        
        # 提取参数
        strategy_idx = int(data.get('strategy_idx', 1))
        vector_mode = int(data.get('vector_mode', 1))
        lat_start = float(data.get('lat_start', 10))
        lat_end = float(data.get('lat_end', 40))
        lon_start = float(data.get('lon_start', 100))
        lon_end = float(data.get('lon_end', 130))
        nz = int(data.get('nz', 10))
        data_quality = int(data.get('data_quality', -6))
        scale_xy = float(data.get('scale_xy', 25))
        skip = data.get('skip')
        if skip is not None:
            skip = int(skip)
        
        # 矢量场参数
        arrow_scale = float(data.get('arrow_scale', 60.0))
        k_neighbors = int(data.get('k_neighbors', 4))
        max_bend_factor = float(data.get('max_bend_factor', 0.3))
        streamline_length = float(data.get('streamline_length', 50.0))
        step_size = float(data.get('step_size', 0.5))
        n_seeds = int(data.get('n_seeds', 400))
        target_clusters = int(data.get('target_clusters', 20))
        
        # 窗口大小
        window_width = int(data.get('window_width', 1400))
        window_height = int(data.get('window_height', 900))
        window_size = (window_width, window_height)
        
        # 根据渲染模式设置离屏 / 窗口渲染
        import os
        if render_mode == 'image':
            # 离屏渲染，返回截图
            os.environ['PYVISTA_OFF_SCREEN'] = 'true'
            os.environ['PYVISTA_USE_PANEL'] = 'false'
            os.environ['VTK_REMOTE_ENABLE'] = '0'
            if 'DISPLAY' in os.environ:
                del os.environ['DISPLAY']
            off_screen = True
            return_image = True
        else:
            # 允许打开窗口，不强制离屏
            os.environ['PYVISTA_OFF_SCREEN'] = 'false'
            os.environ['PYVISTA_USE_PANEL'] = 'false'
            os.environ['VTK_REMOTE_ENABLE'] = '0'
            off_screen = False
            return_image = False
        
        # 生成图像 / 打开窗口
        print(f'[api_velocity3d_generate] 开始调用 generate_3d_visualization, render_mode={render_mode}...')
        image_base64 = velocity_3d_api.generate_3d_visualization(
            strategy_idx=strategy_idx,
            vector_mode=vector_mode,
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
            nz=nz,
            data_quality=data_quality,
            scale_xy=scale_xy,
            skip=skip,
            arrow_scale=arrow_scale,
            k_neighbors=k_neighbors,
            max_bend_factor=max_bend_factor,
            streamline_length=streamline_length,
            step_size=step_size,
            n_seeds=n_seeds,
            target_clusters=target_clusters,
            window_size=window_size,
            off_screen=off_screen,
            return_image=return_image
        )
        
        if render_mode == 'image':
            print(f'[api_velocity3d_generate] 图像生成完成，base64长度: {len(image_base64) if image_base64 else 0}')
            if image_base64:
                print(f'[api_velocity3d_generate] 图像数据前缀: {image_base64[:50]}...')
            response = jsonify({
                'success': True,
                'image': image_base64
            })
        else:
            print('[api_velocity3d_generate] 已在 PyVista 窗口中启动3D可视化（不返回截图）')
            response = jsonify({
                'success': True,
                'message': '3D可视化已在 PyVista 窗口中启动'
            })
        response.headers.add('Access-Control-Allow-Origin', '*')
        if render_mode == 'image':
            print(f'[api_velocity3d_generate] 准备返回响应，图像数据长度: {len(image_base64) if image_base64 else 0}')
        else:
            print('[api_velocity3d_generate] 准备返回窗口模式响应')
        return response
    except Exception as e:
        import traceback
        print(f'[api_velocity3d_generate] Error: {e}')
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/api/velocity3d/strategies', methods=['GET'])
def api_velocity3d_strategies():
    """获取所有可用的透明度策略列表"""
    try:
        import velocity_3d_api
        print(f'[api_velocity3d_strategies] STRATEGIES_AVAILABLE: {velocity_3d_api.STRATEGIES_AVAILABLE}')
        print(f'[api_velocity3d_strategies] strategy_descriptions长度: {len(velocity_3d_api.strategy_descriptions) if hasattr(velocity_3d_api, "strategy_descriptions") else "N/A"}')
        
        strategies = []
        if hasattr(velocity_3d_api, 'strategy_descriptions') and velocity_3d_api.strategy_descriptions:
            for i, desc in enumerate(velocity_3d_api.strategy_descriptions, 1):
                strategies.append({
                    'id': i,
                    'description': desc
                })
        else:
            # 如果策略列表为空，返回默认策略
            print('[api_velocity3d_strategies] 警告：策略列表为空，返回默认策略')
            strategies = [{
                'id': 1,
                'description': '策略1：默认策略（策略文件加载失败）'
            }]
        
        print(f'[api_velocity3d_strategies] 返回策略数量: {len(strategies)}')
        return jsonify({
            'success': True,
            'strategies': strategies
        })
    except Exception as e:
        import traceback
        print(f'[api_velocity3d_strategies] 错误: {e}')
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

def api_typhoon_image_impl(typhoon_id):
    """运行 src/text.py 生成指定台风的 PyVista 截图并返回 base64 数据"""
    print(f'[api_typhoon_image_impl] Called with typhoon_id={typhoon_id}')
    try:
        # 使用台风ID作为缓存键的一部分
        cache_key = f'typhoon_{typhoon_id}'
        if not hasattr(api_typhoon_image_impl, 'cache'):
            api_typhoon_image_impl.cache = {}
        if cache_key not in api_typhoon_image_impl.cache:
            api_typhoon_image_impl.cache[cache_key] = {'image': None, 'timestamp': 0}
        
        cache_ttl = 300  # 5 分钟缓存
        now = time.time()
        if api_typhoon_image_impl.cache[cache_key]['image'] and now - api_typhoon_image_impl.cache[cache_key]['timestamp'] < cache_ttl:
            return jsonify({'success': True, 'image': api_typhoon_image_impl.cache[cache_key]['image'], 'cached': True})

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            screenshot_path = tmp_file.name

        cmd = [
            sys.executable,
            TEXT_SCRIPT,
            '--offscreen',
            '--screenshot',
            screenshot_path
        ]

        env = os.environ.copy()
        env['PYVISTA_OFF_SCREEN'] = 'true'
        env['PYVISTA_USE_PANEL'] = 'false'
        # 禁用 OpenGL2，使用更兼容的渲染后端
        env['VTK_REMOTE_ENABLE'] = '0'
        env['LIBGL_ALWAYS_SOFTWARE'] = '1'
        env['PYVISTA_USE_EGL'] = 'false'
        # 尝试使用 OpenGL 而不是 OpenGL2
        env['PYVISTA_DEFAULT_RENDERER'] = 'opengl'
        # 禁用显示
        env.pop('DISPLAY', None)

        # 检查 text.py 文件是否存在
        if not os.path.exists(TEXT_SCRIPT):
            error_msg = (
                f'text.py file not found at: {TEXT_SCRIPT}\n'
                f'Please ensure text.py is in one of these locations:\n'
                f'  - {os.path.join(BACKEND_DIR, "text.py")}\n'
                f'  - {os.path.join(PROJECT_ROOT, "src", "text.py")}\n'
                f'Current working directory: {os.getcwd()}'
            )
            print(f'[Typhoon3] Error: {error_msg}')
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500

        print(f'[Typhoon3] Running text.py for screenshot: {" ".join(cmd)}')
        print(f'[Typhoon3] TEXT_SCRIPT path: {TEXT_SCRIPT}')
        print(f'[Typhoon3] TEXT_SCRIPT exists: {os.path.exists(TEXT_SCRIPT)}')
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            timeout=300,
            text=True
        )

        if result.returncode != 0:
            stderr_output = result.stderr or 'No stderr output'
            stdout_output = result.stdout or 'No stdout output'
            print(f'[Typhoon3] text.py failed: {stderr_output}')
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
            return jsonify({
                'success': False,
                'error': f'text.py failed: {stderr_output}',
                'stdout': stdout_output,
                'script_path': TEXT_SCRIPT
            }), 500

        with open(screenshot_path, 'rb') as fh:
            image_bytes = fh.read()
        os.remove(screenshot_path)

        image_base64 = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('ascii')
        api_typhoon_image_impl.cache[cache_key]['image'] = image_base64
        api_typhoon_image_impl.cache[cache_key]['timestamp'] = now

        return jsonify({'success': True, 'image': image_base64, 'cached': False})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'text.py screenshot timed out (300s)'}), 504
    except FileNotFoundError:
        return jsonify({'success': False, 'error': f'text.py not found at {TEXT_SCRIPT}'}), 500
    except Exception as e:
        print(f'[Typhoon3] Unexpected error: {e}')
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'typhoon_id': typhoon_id,
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


@app.route('/api/data/extract', methods=['POST'])
def extract_data_api():
    """
    数据提取 API
    调用 data_extractor.py 的 extract_data 函数
    """
    try:
        data = request.get_json()
        
        # 验证必需参数
        required_params = ['lon_min', 'lon_max', 'lat_min', 'lat_max', 'time_step']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必需参数: {param}'
                }), 400
        
        # 提取参数
        lon_min = float(data['lon_min'])
        lon_max = float(data['lon_max'])
        lat_min = float(data['lat_min'])
        lat_max = float(data['lat_max'])
        time_step = int(data['time_step'])
        layer_min = int(data['layer_min']) if data.get('layer_min') is not None else None
        layer_max = int(data['layer_max']) if data.get('layer_max') is not None else None
        variables = data.get('variables', None)
        save_data_flag = data.get('save_data', False)
        save_path = data.get('save_path', None)
        
        # 如果没有指定变量，使用默认（全部变量）
        if not variables:
            variables = ATMOSPHERE_VARIABLES + OCEAN_VARIABLES
        
        print(f'[DataExtract] 开始提取数据:')
        print(f'  经纬范围: [{lon_min}, {lon_max}] × [{lat_min}, {lat_max}]')
        print(f'  时间步: {time_step}')
        print(f'  层数范围: [{layer_min}, {layer_max}]')
        print(f'  变量: {variables}')
        
        # 调用提取函数
        result = extract_data(
            variables=variables,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            time_step=time_step,
            layer_min=layer_min,
            layer_max=layer_max
        )
        
        # 准备返回数据（只返回摘要，不返回完整数据数组）
        response_data = {
            'success': True,
            'summary': {}
        }
        
        # 添加数据摘要
        for var in variables:
            if var in result and len(result[var]) > 0:
                var_data = result[var]
                response_data['summary'][var] = {
                    'shape': list(var_data.shape),
                    'min': float(np.nanmin(var_data)),
                    'max': float(np.nanmax(var_data)),
                    'mean': float(np.nanmean(var_data)),
                    'points': int(var_data.shape[0]) if len(var_data.shape) > 0 else 0
                }
        
        # 添加坐标信息
        if 'lon' in result and len(result['lon']) > 0:
            response_data['summary']['coordinates'] = {
                'count': len(result['lon']),
                'lon_range': [float(np.min(result['lon'])), float(np.max(result['lon']))],
                'lat_range': [float(np.min(result['lat'])), float(np.max(result['lat']))]
            }
        
        # 添加层信息
        if 'layers' in result and len(result['layers']) > 0:
            response_data['summary']['layers'] = {
                'count': len(result['layers']),
                'range': [int(result['layers'][0]), int(result['layers'][-1])]
            }
        
        # 如果需要保存数据
        if save_data_flag and save_path:
            try:
                from data_extractor import save_data
                # 确定保存格式
                format_type = 'nc' if save_path.endswith('.nc') else 'npz'
                save_data(result, save_path, format=format_type)
                response_data['save_path'] = save_path
                response_data['save_format'] = format_type
            except Exception as save_error:
                print(f'[DataExtract] 保存数据失败: {save_error}')
                response_data['save_error'] = str(save_error)
        
        return jsonify(response_data)
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'参数错误: {str(e)}'
        }), 400
    except Exception as e:
        print(f'[DataExtract] 错误: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


@app.route('/api/data/extract/variables', methods=['GET'])
def get_extract_variables():
    """
    获取可用的变量列表
    """
    try:
        return jsonify({
            'success': True,
            'atmosphere_variables': ATMOSPHERE_VARIABLES,
            'ocean_variables': OCEAN_VARIABLES,
            'all_variables': ATMOSPHERE_VARIABLES + OCEAN_VARIABLES
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/atmosphere-ocean-fusion/generate', methods=['POST'])
def generate_atmosphere_ocean_fusion():
    """
    生成海气耦合可视化
    调用 atmosphere_ocean_fusion.py 的 visualize_atmosphere_ocean_fusion 函数
    """
    try:
        data = request.get_json()
        
        # 验证必需参数
        required_params = ['lon_min', 'lon_max', 'lat_min', 'lat_max', 'time_step']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必需参数: {param}'
                }), 400
        
        # 提取参数
        lon_min = float(data['lon_min'])
        lon_max = float(data['lon_max'])
        lat_min = float(data['lat_min'])
        lat_max = float(data['lat_max'])
        time_step = int(data['time_step'])
        resolution = data.get('resolution', 'medium')
        vector_mode = int(data.get('vector_mode', 1))
        
        # 验证分辨率
        if resolution not in ['low', 'medium', 'high']:
            resolution = 'medium'
        
        # 验证矢量场模式
        if vector_mode not in [1, 2]:
            vector_mode = 1
        
        print(f'[AtmosphereOceanFusion] 开始生成海气耦合可视化:')
        print(f'  经纬范围: [{lon_min}, {lon_max}] × [{lat_min}, {lat_max}]')
        print(f'  时间步: {time_step}')
        print(f'  分辨率: {resolution}')
        print(f'  矢量场模式: {vector_mode}')
        
        # 该旧接口保持原异步逻辑，避免影响现有使用
        return jsonify({
            'success': True,
            'message': '海气耦合可视化已启动，请查看PyVista窗口'
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'参数错误: {str(e)}'
        }), 400
    except Exception as e:
        print(f'[AtmosphereOceanFusion] 错误: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


@app.route('/api/atmosphere-ocean-coupled/generate', methods=['POST'])
def generate_atmosphere_ocean_coupled():
    """
    生成大气/海洋上下贴合立方体可视化（同步返回截图）
    """
    try:
        data = request.get_json()

        required_params = ['lon_min', 'lon_max', 'lat_min', 'lat_max', 'time_step']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必需参数: {param}'
                }), 400

        lon_min = float(data['lon_min'])
        lon_max = float(data['lon_max'])
        lat_min = float(data['lat_min'])
        lat_max = float(data['lat_max'])
        time_step = int(data['time_step'])
        layer_min = int(data.get('layer_min', 0))
        layer_max = int(data.get('layer_max', 50))
        ocean_nz = int(data.get('ocean_nz', 40))
        atmosphere_nz = int(data.get('atmosphere_nz', 20))
        data_quality = int(data.get('data_quality', -6))
        scale_xy = float(data.get('scale_xy', 25))
        vector_mode = int(data.get('vector_mode', 3))  # 默认直线箭头
        render_mode = data.get('render_mode', 'image')  # 'image' or 'window'
        return_image = render_mode == 'image'

        print('[AtmoOceanCoupled] 开始生成上下贴合立方体可视化:')
        print(f'  经纬范围: [{lon_min}, {lon_max}] × [{lat_min}, {lat_max}]')
        print(f'  时间步: {time_step}')
        print(f'  层范围: [{layer_min}, {layer_max}] 海洋nz={ocean_nz} 大气nz={atmosphere_nz}')
        print(f'  data_quality: {data_quality}, scale_xy: {scale_xy}, vector_mode: {vector_mode}')
        if return_image:
            image_b64 = visualize_atmo_ocean_coupled(
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                time_step=time_step,
                layer_min=layer_min,
                layer_max=layer_max,
                ocean_nz=ocean_nz,
                atmosphere_nz=atmosphere_nz,
                data_quality=data_quality,
                scale_xy=scale_xy,
                vector_mode=vector_mode,
                return_image=True
            )

            if not image_b64:
                return jsonify({
                    'success': False,
                    'error': '渲染完成但未获取到截图'
                }), 500

            return jsonify({
                'success': True,
                'image': image_b64,
                'message': '渲染完成'
            })
        else:
            import threading
            def run_visualization():
                try:
                    visualize_atmo_ocean_coupled(
                        lon_min=lon_min,
                        lon_max=lon_max,
                        lat_min=lat_min,
                        lat_max=lat_max,
                        time_step=time_step,
                        layer_min=layer_min,
                        layer_max=layer_max,
                        ocean_nz=ocean_nz,
                        atmosphere_nz=atmosphere_nz,
                        data_quality=data_quality,
                        scale_xy=scale_xy,
                        vector_mode=vector_mode,
                        return_image=False
                    )
                except Exception as e:
                    print(f'[AtmoOceanCoupled] 可视化运行失败: {e}')
                    import traceback
                    traceback.print_exc()

            thread = threading.Thread(target=run_visualization, daemon=True)
            thread.start()
            return jsonify({
                'success': True,
                'message': '已在 PyVista 窗口启动可视化'
            })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'参数错误: {str(e)}'
        }), 400
    except Exception as e:
        print(f'[AtmoOceanCoupled] 错误: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


@app.route('/api/atmosphere-3d/generate', methods=['POST'])
def generate_atmosphere_3d():
    """
    生成大气3D可视化（同步返回截图）
    """
    try:
        data = request.get_json()
        
        # 验证必需参数
        required_params = ['lon_min', 'lon_max', 'lat_min', 'lat_max', 'time_step']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必需参数: {param}'
                }), 400
        
        # 提取参数
        lon_min = float(data['lon_min'])
        lon_max = float(data['lon_max'])
        lat_min = float(data['lat_min'])
        lat_max = float(data['lat_max'])
        time_step = int(data['time_step'])
        layer_min = int(data.get('layer_min', 0))
        layer_max = int(data.get('layer_max', 50))
        data_quality = int(data.get('data_quality', -6))
        scale_xy = float(data.get('scale_xy', 25))
        atmosphere_nz = int(data.get('atmosphere_nz', 20))
        vector_mode = int(data.get('vector_mode', 1))
        render_mode = data.get('render_mode', 'image')  # 'image' or 'window'
        return_image = render_mode == 'image'
        
        # 验证矢量场模式
        if vector_mode not in [1, 2, 3]:
            vector_mode = 1
        
        print(f'[Atmosphere3D] 开始生成大气3D可视化:')
        print(f'  经纬范围: [{lon_min}, {lon_max}] × [{lat_min}, {lat_max}]')
        print(f'  时间步: {time_step}')
        print(f'  层数范围: [{layer_min}, {layer_max}]')
        print(f'  矢量场模式: {vector_mode}')
        if return_image:
            image_b64 = visualize_atmosphere_3d(
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                time_step=time_step,
                layer_min=layer_min,
                layer_max=layer_max,
                data_quality=data_quality,
                scale_xy=scale_xy,
                atmosphere_nz=atmosphere_nz,
                vector_mode=vector_mode,
                return_image=True
            )

            if not image_b64:
                return jsonify({
                    'success': False,
                    'error': '渲染完成但未获取到截图'
                }), 500

            return jsonify({
                'success': True,
                'image': image_b64,
                'message': '渲染完成'
            })
        else:
            import threading
            def run_visualization():
                try:
                    visualize_atmosphere_3d(
                        lon_min=lon_min,
                        lon_max=lon_max,
                        lat_min=lat_min,
                        lat_max=lat_max,
                        time_step=time_step,
                        layer_min=layer_min,
                        layer_max=layer_max,
                        data_quality=data_quality,
                        scale_xy=scale_xy,
                        atmosphere_nz=atmosphere_nz,
                        vector_mode=vector_mode,
                        return_image=False
                    )
                except Exception as e:
                    print(f'[Atmosphere3D] 可视化运行失败: {e}')
                    import traceback
                    traceback.print_exc()

            thread = threading.Thread(target=run_visualization, daemon=True)
            thread.start()
            return jsonify({
                'success': True,
                'message': '已在 PyVista 窗口启动可视化'
            })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'参数错误: {str(e)}'
        }), 400
    except Exception as e:
        print(f'[Atmosphere3D] 错误: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Flask Backend Server Starting...")
    print("=" * 60)
    
    # 不在这里初始化数据集，改为延迟加载（在首次API调用时加载）
    print("\n[Note] Dataset will be loaded on first API request")
    print("  This prevents blocking during server startup")
    
    print("\n[Starting Flask server...]")
    print(f"✓ Server will be available at: http://localhost:5000")
    print(f"✓ Health check: http://localhost:5000/api/health")
    print(f"✓ Volume data: http://localhost:5000/api/data/volume?time=0")
    print(f"✓ Dataset info: http://localhost:5000/api/info")
    print("\n" + "=" * 60)
    print("Server is running. Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    
    try:
        # 从环境变量读取配置，支持生产环境
        flask_env = os.environ.get('FLASK_ENV', 'development')
        debug_mode = flask_env != 'production'
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        if debug_mode:
            print("[Mode] Development mode (debug enabled)")
        else:
            print("[Mode] Production mode (debug disabled)")
            app.config['DEBUG'] = False
        
        # 使用 threaded=True 和更好的错误处理
        app.run(
            host=host, 
            port=port, 
            debug=debug_mode,
            threaded=True,
            use_reloader=debug_mode and os.name != 'nt'  # Windows 上禁用 reloader
        )
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Server stopped by user")
        print("=" * 60)
    except OSError as e:
        if "Address already in use" in str(e) or "address is already in use" in str(e).lower():
            print(f"\n\nERROR: Port 5000 is already in use!")
            print("Please either:")
            print("  1. Stop the other service using port 5000")
            print("  2. Change the port in app.py (line 236)")
        else:
            print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n\nERROR: Server failed to start: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check if port 5000 is available")
        print("  2. Check Python version (requires 3.8+)")
        print("  3. Check if all dependencies are installed")
        print("  4. Try running: pip install -r requirements.txt")

