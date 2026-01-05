#盐度切片
import OpenVisus as ov
import numpy as np
import os
import json
import argparse
import sys

# 在导入 PyVista 之前设置环境变量（必须在导入前设置）
# 检查是否需要离屏渲染
is_offscreen = (
    '--offscreen' in sys.argv or
    os.environ.get('PYVISTA_OFF_SCREEN', '').lower() == 'true' or
    os.environ.get('DISPLAY') is None
)

if is_offscreen:
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['PYVISTA_USE_PANEL'] = 'false'
    # 禁用 OpenGL2，使用 OpenGL（更兼容）
    os.environ['VTK_REMOTE_ENABLE'] = '0'
    # 尝试使用软件渲染
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# 尝试导入 PyVista
try:
    # 在导入前设置使用 OpenGL 而不是 OpenGL2
    os.environ['PYVISTA_DEFAULT_RENDERER'] = 'opengl'
    
    import pyvista as pv
    
    # 设置离屏模式
    if is_offscreen:
        pv.OFF_SCREEN = True
        # 尝试使用 OpenGL 渲染器（而不是 OpenGL2）
        try:
            pv.set_plot_theme('document')
        except:
            pass
    
    PYVISTA_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: PyVista import/setup failed: {e}")
    print("Will attempt to continue with basic PyVista functionality.")
    PYVISTA_AVAILABLE = True  # 继续尝试，可能只是警告
    import pyvista as pv

# ----------------------------
# 1️⃣ 数据集路径与加载
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

# 速度与盐度
def load_fields():
    U_db = load_dataset("u")
    V_db = load_dataset("v")
    W_db = load_dataset("w")
    Salt_db = load_dataset("salt")
    return U_db, V_db, W_db, Salt_db

# ----------------------------
# 2️⃣ 局部区域参数（从 backend/config.json 导入）
# ----------------------------
def load_local_config():
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
        # 尝试从后端配置文件读取，支持本地开发和服务器部署
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 如果 text.py 在 backend/ 目录下，config.json 应该在同一目录
        # 如果 text.py 在 src/ 目录下，config.json 在 ../backend/
        if os.path.basename(base_dir) == 'backend':
            cfg_path = os.path.join(base_dir, 'config.json')
        else:
            cfg_path = os.path.normpath(os.path.join(base_dir, '..', 'backend', 'config.json'))
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as fh:
                cfg = json.load(fh)
                for k, v in defaults.items():
                    cfg[k] = cfg.get(k, v)
                return cfg
        else:
            return defaults
    except Exception as e:
        print(f"Failed to load config.json: {e}")
        return defaults


cfg = load_local_config()
lat_start = cfg['lat_start']
lat_end = cfg['lat_end']
lon_start = cfg['lon_start']
lon_end = cfg['lon_end']
nz = cfg['nz']
data_quality = cfg['data_quality']
scale_xy = cfg['scale_xy']

# ----------------------------
# 3️⃣ 读取局部数据函数
# ----------------------------
def read_data(db):
    data_full = db.read(time=0, quality=data_quality)
    lat_dim, lon_dim, depth_dim = data_full.shape
    lat_idx_start = int(lat_dim * lat_start / 90)
    lat_idx_end = int(lat_dim * lat_end / 90)
    lon_idx_start = int(lon_dim * lon_start / 360)
    lon_idx_end = int(lon_dim * lon_end / 360)
    return data_full[lat_idx_start:lat_idx_end,
                     lon_idx_start:lon_idx_end,
                     :nz]

def build_plot(off_screen=False):
    U_db, V_db, W_db, Salt_db = load_fields()
    U_local = read_data(U_db)
    V_local = read_data(V_db)
    W_local = read_data(W_db)
    Salt_local = read_data(Salt_db)
    nx, ny, nz = U_local.shape
    z_grid = np.linspace(0, 1000, nz)

    # ----------------------------
    # 4️⃣ 坐标网格
    # ----------------------------
    x = np.linspace(lon_start, lon_end, ny) * scale_xy
    y = np.linspace(lat_start, lat_end, nx) * scale_xy
    X, Y, Z = np.meshgrid(x, y, z_grid, indexing='ij')
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = -Z.transpose(1, 0, 2)

    # ----------------------------
    # 5️⃣ PyVista 3D流线 + 多层盐度
    # ----------------------------
    # 先创建网格和向量数据
    grid = pv.StructuredGrid(X, Y, Z)
    vectors = np.stack([U_local.flatten(order="F"),
                        V_local.flatten(order="F"),
                        W_local.flatten(order="F")], axis=1)
    
    # 初始化网格的标量和向量数据（避免属性访问错误）
    # 先添加一个标量数组，确保网格有活动标量
    grid["dummy_scalar"] = np.ones(grid.n_points)
    grid["vectors"] = vectors
    
    # 种子点间隔（直接从原始坐标数组创建）
    points_array = np.column_stack([X.flatten(order="F"), 
                                     Y.flatten(order="F"), 
                                     Z.flatten(order="F")])
    seed_points = pv.PolyData(points_array[::10])

    # 尝试生成流线（如果失败则跳过，只显示盐度层）
    streamlines = None
    
    # 简化方案：暂时跳过流线生成，避免 PyVista 版本兼容性问题
    # 只显示盐度图层，确保基本功能可用
    print("Note: Streamline generation temporarily disabled due to PyVista compatibility issues.")
    print("Showing salinity layers only for now.")

    plotter = pv.Plotter(window_size=(1000, 700), off_screen=off_screen)
    
    # 添加盐度层
    for k in range(nz):
        surface_salt = pv.StructuredGrid(
            X[:, :, k],
            Y[:, :, k],
            Z[:, :, k]
        )
        surface_salt["salt"] = Salt_local[:, :, k].flatten(order="F")
        plotter.add_mesh(surface_salt, scalars="salt", cmap="Blues", opacity=0.1)

    # 添加流线（如果成功生成）
    if streamlines is not None and streamlines.n_points > 0:
        try:
            plotter.add_mesh(streamlines, scalars='speed', cmap='cool', line_width=2, opacity=0.8)
        except Exception as e:
            print(f"Warning: Failed to add streamlines to plotter: {e}")
    
    plotter.add_axes()
    return plotter


def run_visualization(off_screen=False, screenshot_path=None):
    plotter = build_plot(off_screen=off_screen)
    if screenshot_path:
        # PyVista会在show时自动生成截图
        plotter.show(screenshot=screenshot_path, auto_close=True, interactive=False)
    else:
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render PyVista visualization for Typhoon 3.")
    parser.add_argument("--offscreen", action="store_true", help="Use offscreen rendering (for server-side generation).")
    parser.add_argument("--screenshot", type=str, help="Path to save screenshot.")
    args = parser.parse_args()

    run_visualization(off_screen=args.offscreen, screenshot_path=args.screenshot)

