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
    # 尝试禁用 OpenGL2，使用更兼容的渲染器
    os.environ['VTK_REMOTE_ENABLE'] = '0'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    # 尝试使用软件渲染
    os.environ['PYVISTA_USE_EGL'] = 'false'

# 尝试导入 PyVista（可能需要处理导入错误）
try:
    # 尝试禁用 OpenGL2 渲染器
    os.environ['PYVISTA_DEFAULT_RENDERER'] = 'opengl'
    
    import pyvista as pv
    
    # 设置离屏模式
    if is_offscreen:
        try:
            pv.OFF_SCREEN = True
        except:
            pass
        # 尝试设置渲染主题为文档模式（更兼容）
        try:
            pv.set_plot_theme('document')
        except:
            pass
    
    PYVISTA_AVAILABLE = True
except Exception as e:
    print(f"Error: PyVista setup failed: {e}")
    print("This may be due to OpenGL/VTK compatibility issues on the server.")
    raise  # 重新抛出错误，让调用者知道问题

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

def get_3d_mesh_data():
    """提取3D网格数据用于前端渲染，返回JSON格式的顶点、面和颜色数据"""
    U_db, V_db, W_db, Salt_db = load_fields()
    Salt_local = read_data(Salt_db)
    nx, ny, nz = Salt_local.shape
    z_grid = np.linspace(0, 1000, nz)

    # 坐标网格
    x = np.linspace(lon_start, lon_end, ny) * scale_xy
    y = np.linspace(lat_start, lat_end, nx) * scale_xy
    X, Y, Z = np.meshgrid(x, y, z_grid, indexing='ij')
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = -Z.transpose(1, 0, 2)  # 负Z表示深度

    # 降采样以提高性能（每2个点采样一次）
    stride = 2
    X_sub = X[::stride, ::stride, ::stride]
    Y_sub = Y[::stride, ::stride, ::stride]
    Z_sub = Z[::stride, ::stride, ::stride]
    Salt_sub = Salt_local[::stride, ::stride, ::stride]
    
    nx_sub, ny_sub, nz_sub = X_sub.shape
    
    # 生成顶点数组 [x, y, z, salt_value]
    vertices = []
    vertex_indices = {}  # 用于去重和索引
    vertex_count = 0
    
    for i in range(nx_sub):
        for j in range(ny_sub):
            for k in range(nz_sub):
                x_val = float(X_sub[i, j, k])
                y_val = float(Y_sub[i, j, k])
                z_val = float(Z_sub[i, j, k])
                salt_val = float(Salt_sub[i, j, k])
                
                # 存储顶点（包含位置和盐度值）
                vertices.append([x_val, y_val, z_val, salt_val])
                vertex_indices[(i, j, k)] = vertex_count
                vertex_count += 1
    
    # 生成面的索引（多层盐度表面，类似PyVista的效果）
    faces = []
    layer_faces = []  # 按层组织的面，用于前端分层渲染
    
    # 为每一层生成表面（类似PyVista的add_mesh，每层透明度0.1）
    for k in range(nz_sub):
        layer_face_indices = []
        for i in range(nx_sub - 1):
            for j in range(ny_sub - 1):
                idx1 = vertex_indices[(i, j, k)]
                idx2 = vertex_indices[(i+1, j, k)]
                idx3 = vertex_indices[(i+1, j+1, k)]
                idx4 = vertex_indices[(i, j+1, k)]
                face = [idx1, idx2, idx3, idx4]
                faces.append(face)
                layer_face_indices.append(len(faces) - 1)
        layer_faces.append(layer_face_indices)
    
    # 计算盐度值的范围用于颜色映射
    salt_min = float(Salt_sub.min())
    salt_max = float(Salt_sub.max())
    
    # 返回数据
    return {
        'vertices': vertices,  # [[x, y, z, salt], ...]
        'faces': faces,  # [[idx1, idx2, idx3, idx4], ...]
        'layer_faces': layer_faces,  # 按层组织的面索引，用于分层渲染
        'bounds': {
            'x_min': float(X_sub.min()),
            'x_max': float(X_sub.max()),
            'y_min': float(Y_sub.min()),
            'y_max': float(Y_sub.max()),
            'z_min': float(Z_sub.min()),
            'z_max': float(Z_sub.max())
        },
        'salt_range': {
            'min': salt_min,
            'max': salt_max
        },
        'dimensions': {
            'nx': nx_sub,
            'ny': ny_sub,
            'nz': nz_sub
        }
    }

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
    parser.add_argument("--export-mesh", action="store_true", help="Export 3D mesh data as JSON.")
    args = parser.parse_args()

    if args.export_mesh:
        # 导出3D网格数据
        mesh_data = get_3d_mesh_data()
        print(json.dumps(mesh_data))
    else:
        run_visualization(off_screen=args.offscreen, screenshot_path=args.screenshot)
