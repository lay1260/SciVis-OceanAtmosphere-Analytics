import numpy as np
import pyvista as pv

# ===============================
# 假设你已经读取了局部截取的数据
# 大气: U,V,W,T,QI,QL (shape: [nx,ny,nz_atmos])
# 海洋: U,V,W,Theta,Salt (shape: [nx,ny,nz_ocean])
# ===============================

nx, ny, nz_atmos = 30, 30, 20
nz_ocean = 30

# --- 示例随机数据替代实际数据 ---
U_at = np.random.randn(nx, ny, nz_atmos).astype(np.float32)
V_at = np.random.randn(nx, ny, nz_atmos).astype(np.float32)
W_at = np.random.randn(nx, ny, nz_atmos).astype(np.float32)
T_at = np.linspace(280, 320, nx*ny*nz_atmos).reshape(nx, ny, nz_atmos).astype(np.float32)
QI = np.random.rand(nx, ny, nz_atmos).astype(np.float32)
QL = np.random.rand(nx, ny, nz_atmos).astype(np.float32)

U_oc = np.random.randn(nx, ny, nz_ocean).astype(np.float32)
V_oc = np.random.randn(nx, ny, nz_ocean).astype(np.float32)
W_oc = np.random.randn(nx, ny, nz_ocean).astype(np.float32)
Theta = np.linspace(2, 30, nx*ny*nz_ocean).reshape(nx, ny, nz_ocean).astype(np.float32)
Salt = np.random.rand(nx, ny, nz_ocean).astype(np.float32)

# ===============================
# 创建 PyVista 网格
# ===============================
def create_grid(U,V,W):
    nx, ny, nz = U.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X,Y,Z = np.meshgrid(x, y, z, indexing='ij')
    grid = pv.StructuredGrid(X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32))
    vectors = np.c_[U.ravel(), V.ravel(), W.ravel()]
    grid["vectors"] = vectors
    return grid

grid_atmos = create_grid(U_at, V_at, W_at)
grid_ocean = create_grid(U_oc, V_oc, W_oc)

# ===============================
# 可视化
# ===============================
pl = pv.Plotter()

# --- 大气温度 T 非线性颜色映射 (红-蓝) ---
T_norm = (T_at - T_at.min()) / (T_at.max() - T_at.min())
pl.add_volume(pv.wrap(T_norm), scalars=T_norm.ravel(),
              cmap='coolwarm', opacity='linear', blending='composite', opacity_unit_distance=1.0)

# --- 大气云水 QI/QL 透明度映射 ---
QI_QL = np.maximum(QI, QL)
pl.add_volume(pv.wrap(QI_QL),
              scalars=QI_QL.ravel(),
              cmap='gray',
              opacity='linear',  # 自动生成透明度映射
              blending='composite')

# --- 大气流线 (白色) ---
seed_at = pv.SeedSource(np.array([[nx/2, ny/2, nz_atmos/2]]), radius=5, number_of_seeds=100)
pl.add_streamlines(grid_atmos, source=seed_at, vectors="vectors", max_time=30,
                   integration_direction='forward', line_width=2, color='white')

# --- 海洋温度 Theta 非线性颜色映射 (黄-蓝) ---
Theta_norm = (Theta - Theta.min()) / (Theta.max()-Theta.min())
pl.add_volume(pv.wrap(Theta_norm), scalars=Theta_norm.ravel(),
              cmap='YlOrBr', opacity='linear', blending='composite', opacity_unit_distance=1.0)

# --- 海水盐度透明度映射 ---
pl.add_volume(pv.wrap(Salt),
              scalars=Salt.ravel(),
              cmap='gray',
              opacity='linear',  # 自动生成透明度映射
              blending='composite')

# --- 海洋流线 (白色) ---
seed_oc = pv.SeedSource(np.array([[nx/2, ny/2, nz_ocean/2]]), radius=5, number_of_seeds=100)
pl.add_streamlines(grid_ocean, source=seed_oc, vectors="vectors", max_time=30,
                   integration_direction='forward', line_width=2, color='white')

# --- 坐标轴 & 网格 ---
pl.add_axes()
pl.show_grid()
pl.show()
