"""
大气-海洋立方体上下贴合可视化
--------------------------------
功能：
- 读取同一经纬范围内的大气与海洋数据
- 构建两个上下贴合的立方体：大气 (Z为正) 与海洋 (Z为负)
- 体渲染：颜色映射温度/水含量（或盐度），透明度映射水含量/盐度
- 矢量场：简单直线箭头（可选，vector_mode=3）

依赖：
- numpy, pyvista, matplotlib
- data_extractor.extract_data (复用已有数据读取与插值能力)

快速使用：
    python atmo_ocean_coupled_cube.py
    # 或指定范围
    python atmo_ocean_coupled_cube.py 100 130 10 40 0
"""

import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import base64
import io
import imageio.v2 as iio

from data_extractor import extract_data


def _build_regular_grid(lon_min, lon_max, lat_min, lat_max, nz, scale_xy, z_min, z_max):
    """构建规则立方体网格"""
    nx = ny = max(2, int(np.sqrt( max(4, nz * 4) )))  # 简单估计XY分辨率
    x = np.linspace(lon_min, lon_max, nx) * scale_xy
    y = np.linspace(lat_min, lat_max, ny) * scale_xy
    z = np.linspace(z_min, z_max, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = Z.transpose(1, 0, 2)
    grid = pv.StructuredGrid(X, Y, Z)
    return grid, (nx, ny, nz)


def _interpolate_to_grid(src_lon, src_lat, values, grid, scale_xy, nz_target):
    """将点数据插值到规则网格，逐层最近邻插值"""
    from scipy.interpolate import griddata

    nx, ny, nz = grid.dimensions[0], grid.dimensions[1], nz_target

    # 仅使用单层的 XY 网格进行 2D 插值，避免把所有 z 层点混入导致尺寸不匹配
    # 获取唯一的 X/Y 坐标并构造目标 XY 平面
    unique_x = np.unique(grid.points[:, 0])
    unique_y = np.unique(grid.points[:, 1])
    target_X, target_Y = np.meshgrid(unique_x, unique_y, indexing="ij")
    target_xy = np.column_stack([
        target_X.flatten(order="F"),
        target_Y.flatten(order="F"),
    ])

    if len(src_lon) == 0 or len(src_lat) == 0:
        return np.zeros((nx, ny, nz))

    # 对齐源点和数值长度，避免“different number of values and points”
    n_points = min(len(src_lon), len(src_lat), values.shape[0])
    if n_points < 3:
        return np.zeros((nx, ny, nz))

    src_xy = np.column_stack([
        src_lon[:n_points] * scale_xy,
        src_lat[:n_points] * scale_xy,
    ])

    # 裁剪值到相同长度
    values = values[:n_points] if values.shape[0] != n_points else values

    def interp_layer(data_2d):
        return griddata(src_xy, data_2d, target_xy, method="nearest", fill_value=0).reshape(nx, ny, order="F")

    if values.ndim == 2:
        n_layers = values.shape[1]
        result = np.zeros((nx, ny, nz))
        for k in range(nz):
            src_k = min(int(k * n_layers / nz), n_layers - 1)
            result[:, :, k] = interp_layer(values[:, src_k])
    else:
        # 单层数据，直接插值到 2D 网格后扩展/截断到 nz_target
        single = interp_layer(values)
        result = np.repeat(single[:, :, None], nz, axis=2)

    return result


def visualize_atmo_ocean_coupled(
    lon_min=100,
    lon_max=130,
    lat_min=10,
    lat_max=40,
    time_step=0,
    layer_min=0,
    layer_max=50,
    ocean_nz=40,
    atmosphere_nz=20,
    data_quality=-6,
    scale_xy=25,
    vector_mode=3,  # 简化为直线箭头
    return_image=False
):
    """
    大气与海洋立方体上下贴合可视化
    - 大气位于Z>0，海洋位于Z<0
    - 体渲染：大气用温度/水含量；海洋用温度/盐度
    - 矢量场：默认直线箭头（vector_mode=3）
    """
    print("\n" + "=" * 60)
    print("开始提取大气数据")
    atm_vars = ["T", "QI", "QL", "RI", "RL", "U", "V", "W"]
    atm = extract_data(
        variables=atm_vars,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_step=time_step,
        layer_min=layer_min,
        layer_max=layer_max,
        quality=data_quality,
    )
    if len(atm.get("lon", [])) == 0:
        print("大气数据提取失败")
        return

    print("开始提取海洋数据")
    ocean_vars = ["Theta", "Salt", "U", "V", "W"]
    ocean = extract_data(
        variables=ocean_vars,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_step=time_step,
        layer_min=layer_min,
        layer_max=layer_max,
        quality=data_quality,
    )
    if len(ocean.get("lon", [])) == 0:
        print("海洋数据提取失败")
        return

    # 立方体尺度
    lon_range = (lon_max - lon_min) * scale_xy
    lat_range = (lat_max - lat_min) * scale_xy
    avg_xy = (lon_range + lat_range) / 2

    # 大气网格（Z>0）
    atm_grid, (atm_nx, atm_ny, atm_nz) = _build_regular_grid(
        lon_min, lon_max, lat_min, lat_max, atmosphere_nz, scale_xy, 0, avg_xy
    )
    # 海洋网格（Z<0）
    ocean_grid, (oc_nx, oc_ny, oc_nz) = _build_regular_grid(
        lon_min, lon_max, lat_min, lat_max, ocean_nz, scale_xy, -avg_xy, 0
    )

    # 计算大气可见水含量
    def compute_water(QI, QL, RI, RL):
        R_Lp = RL * 1e6
        R_Ip = RI * 1e6
        return 0.7 * QL * np.minimum(R_Lp / 10, 1.0) + 0.3 * QI * np.minimum(R_Ip / 20, 1.0)

    atm_T = atm.get("T", np.zeros(1))
    atm_QI = atm.get("QI", np.zeros_like(atm_T))
    atm_QL = atm.get("QL", np.zeros_like(atm_T))
    atm_RI = atm.get("RI", np.zeros_like(atm_T))
    atm_RL = atm.get("RL", np.zeros_like(atm_T))
    atm_water = compute_water(atm_QI, atm_QL, atm_RI, atm_RL)
    atm_lon = atm.get("lon", np.array([]))
    atm_lat = atm.get("lat", np.array([]))

    ocean_T = ocean.get("Theta", np.zeros(1))
    ocean_S = ocean.get("Salt", np.zeros_like(ocean_T))
    ocean_lon = ocean.get("lon", np.array([]))
    ocean_lat = ocean.get("lat", np.array([]))

    # 插值到网格
    atm_T_3d = _interpolate_to_grid(atm_lon, atm_lat, atm_T, atm_grid, scale_xy, atm_nz)
    atm_W_3d = _interpolate_to_grid(atm_lon, atm_lat, atm_water, atm_grid, scale_xy, atm_nz)

    oc_T_3d = _interpolate_to_grid(ocean_lon, ocean_lat, ocean_T, ocean_grid, scale_xy, oc_nz)
    oc_S_3d = _interpolate_to_grid(ocean_lon, ocean_lat, ocean_S, ocean_grid, scale_xy, oc_nz)

    # 简单速度场（如有）
    def velocity_to_grid(src_lon, src_lat, U, V, W, grid, nz_target):
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return np.zeros((grid.n_points, 3))
        U3 = _interpolate_to_grid(src_lon, src_lat, U, grid, scale_xy, nz_target)
        V3 = _interpolate_to_grid(src_lon, src_lat, V, grid, scale_xy, nz_target)
        W3 = _interpolate_to_grid(src_lon, src_lat, W, grid, scale_xy, nz_target)
        vel = np.stack([U3, V3, W3], axis=3)
        return vel.reshape(-1, 3, order="F")

    atm_vel = velocity_to_grid(
        atm_lon, atm_lat, atm.get("U", np.zeros_like(atm_T)), atm.get("V", np.zeros_like(atm_T)), atm.get("W", np.zeros_like(atm_T)),
        atm_grid, atm_nz
    )
    oc_vel = velocity_to_grid(
        ocean_lon, ocean_lat, ocean.get("U", np.zeros_like(ocean_T)), ocean.get("V", np.zeros_like(ocean_T)), ocean.get("W", np.zeros_like(ocean_T)),
        ocean_grid, oc_nz
    )

    # 写入网格属性
    atm_grid["Temperature"] = atm_T_3d.flatten(order="F")
    atm_grid["Water"] = atm_W_3d.flatten(order="F")
    atm_grid["velocity"] = atm_vel
    atm_grid["speed"] = np.linalg.norm(atm_vel, axis=1)

    ocean_grid["Temperature"] = oc_T_3d.flatten(order="F")
    ocean_grid["Salinity"] = oc_S_3d.flatten(order="F")
    ocean_grid["velocity"] = oc_vel
    ocean_grid["speed"] = np.linalg.norm(oc_vel, axis=1)

    # 体渲染设置
    plotter = pv.Plotter(window_size=(1500, 1000))
    plotter.background_color = (0.08, 0.12, 0.18)

    # 大气体渲染
    plotter.add_volume(
        atm_grid,
        scalars="Temperature",
        cmap="hot",
        opacity=0.2,
        opacity_unit_distance=6,
        show_scalar_bar=True,
        scalar_bar_args={"title": "大气温度"},
        blending="composite",
    )
    # 透明度基于Water
    # 简单方式：设置整体不透明度，突出体积即可

    # 海洋体渲染（反转Z，所以在下部）
    plotter.add_volume(
        ocean_grid,
        scalars="Temperature",
        cmap="coolwarm",
        opacity=0.2,
        opacity_unit_distance=6,
        show_scalar_bar=True,
        scalar_bar_args={"title": "海洋温度"},
        blending="composite",
    )

    # 矢量场（简化：直线箭头）
    if vector_mode == 3:
        # 大气箭头
        atm_arrows = pv.PolyData(atm_grid.points)
        atm_arrows["velocity"] = atm_grid["velocity"]
        atm_arrows["speed"] = atm_grid["speed"]
        atm_arrows = atm_arrows.glyph(
            orient="velocity",
            scale="speed",
            factor=0.5,
        )
        plotter.add_mesh(
            atm_arrows,
            scalars="speed",
            cmap="cool",
            opacity=1.0,
            show_scalar_bar=False,
            render_lines_as_tubes=True,
        )

        # 海洋箭头
        oc_arrows = pv.PolyData(ocean_grid.points)
        oc_arrows["velocity"] = ocean_grid["velocity"]
        oc_arrows["speed"] = ocean_grid["speed"]
        oc_arrows = oc_arrows.glyph(
            orient="velocity",
            scale="speed",
            factor=0.5,
        )
        plotter.add_mesh(
            oc_arrows,
            scalars="speed",
            cmap="viridis",
            opacity=1.0,
            show_scalar_bar=False,
            render_lines_as_tubes=True,
        )

    # 相机与坐标轴
    plotter.add_axes()
    center = (np.array(atm_grid.bounds).reshape(3, 2).mean(axis=1) + np.array(ocean_grid.bounds).reshape(3, 2).mean(axis=1)) / 2
    max_range = max(
        atm_grid.bounds[1] - atm_grid.bounds[0],
        atm_grid.bounds[3] - atm_grid.bounds[2],
        atm_grid.bounds[5] - ocean_grid.bounds[4],
    )
    cam_dist = max_range * 2.5
    plotter.camera_position = [
        (center[0] + cam_dist, center[1] + cam_dist, center[2] + cam_dist),
        center,
        (0, 0, 1),
    ]
    plotter.camera.azimuth = 45
    plotter.camera.elevation = 30

    print("✅ 可视化准备完成")
    if return_image:
        # 先执行一次渲染，再截图，避免“Nothing to screenshot”
        plotter.show(auto_close=False, interactive=False)
        img = plotter.screenshot(return_img=True)
        plotter.close()
        if img is not None:
            buf = io.BytesIO()
            iio.imwrite(buf, img, format='png')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        return None
    else:
        plotter.show()


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        lon_min = float(sys.argv[1])
        lon_max = float(sys.argv[2])
        lat_min = float(sys.argv[3])
        lat_max = float(sys.argv[4])
        time_step = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    else:
        lon_min, lon_max, lat_min, lat_max, time_step = 100, 130, 10, 40, 0

    visualize_atmo_ocean_coupled(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        time_step=time_step,
    )

