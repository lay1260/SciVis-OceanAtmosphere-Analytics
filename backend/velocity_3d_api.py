"""
3D可视化API封装：整合velocity_3D_strategies和velocity_3D_vector_optimized的功能
"""
import os
import sys
import tempfile
import base64

# 强制设置离屏渲染环境变量（必须在导入 pyvista 之前设置）
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_PANEL'] = 'false'
os.environ['VTK_REMOTE_ENABLE'] = '0'
os.environ['PYVISTA_DEFAULT_RENDERER'] = 'opengl'  # 使用 OpenGL 渲染器

# Windows 特定设置
if sys.platform == 'win32':
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['PYVISTA_USE_EGL'] = 'false'

# 清空DISPLAY变量（Linux系统）
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

# 现在才导入 pyvista（环境变量已设置）
import numpy as np
import pyvista as pv

# 导入策略脚本的功能
try:
    # 尝试导入策略函数（从velocity_3D_strategies(1)）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # 尝试导入带括号的文件名
    import importlib.util
    strategies_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_3D_strategies(1).py')
    if os.path.exists(strategies_file):
        spec = importlib.util.spec_from_file_location("velocity_3D_strategies", strategies_file)
        strategies_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategies_module)
        load_dataset = strategies_module.load_dataset
        read_data = strategies_module.read_data
        strategy_functions = strategies_module.strategy_functions
        strategy_descriptions = strategies_module.strategy_descriptions
        STRATEGIES_AVAILABLE = True
    else:
        raise ImportError(f"策略文件不存在: {strategies_file}")
except ImportError as e:
    print(f"警告：无法导入策略函数: {e}")
    STRATEGIES_AVAILABLE = False
    strategy_functions = []
    strategy_descriptions = []

try:
    # 尝试导入矢量场优化功能（从velocity_3D_vector_optimized）
    from velocity_3D_vector_optimized import (
        create_bent_arrows, create_3d_streamlines, cluster_flow_regions,
        create_cluster_bent_arrows, SCIPY_AVAILABLE, SCIPY_INTEGRATE_AVAILABLE,
        SKLEARN_AVAILABLE
    )
    VECTOR_OPTIMIZED_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入矢量场优化功能: {e}")
    VECTOR_OPTIMIZED_AVAILABLE = False

def generate_3d_visualization(
    strategy_idx=1,
    vector_mode=1,
    lat_start=10, lat_end=40,
    lon_start=100, lon_end=130,
    nz=10,
    data_quality=-6,
    scale_xy=25,
    skip=None,
    # 矢量场参数
    arrow_scale=60.0,
    k_neighbors=4,
    max_bend_factor=0.3,
    streamline_length=50.0,
    step_size=0.5,
    n_seeds=400,
    target_clusters=20,
    # 其他参数
    window_size=(1400, 900),
    off_screen=True,
    return_image=True
):
    """
    生成3D可视化图像
    
    Args:
        strategy_idx: 透明度策略索引 (1-20)
        vector_mode: 矢量场模式 (1=弯曲箭头, 2=三维流线, 3=聚类区域大箭头)
        lat_start, lat_end, lon_start, lon_end: 区域范围
        nz: 深度层数
        data_quality: 数据质量
        scale_xy: XY缩放因子
        skip: 采样间隔
        arrow_scale: 箭头缩放因子
        k_neighbors: 邻域点数
        max_bend_factor: 最大弯曲因子
        streamline_length: 流线长度
        step_size: 积分步长
        n_seeds: 种子点数量
        target_clusters: 目标聚类数
        window_size: 窗口大小
        off_screen: 是否离屏渲染
    
    Returns:
        base64编码的图像字符串
    """
    if not STRATEGIES_AVAILABLE:
        raise ValueError("策略函数不可用，请检查velocity_3D_strategies.py")
    
    # 1. 加载数据集
    U_db = load_dataset("u")
    V_db = load_dataset("v")
    W_db = load_dataset("w")
    Salt_db = load_dataset("salt")
    Theta_db = load_dataset("theta")
    
    # 2. 读取局部数据
    U_local = read_data(U_db, skip_value=skip)
    V_local = read_data(V_db, skip_value=skip)
    W_local = read_data(W_db, skip_value=skip)
    Salt_local = read_data(Salt_db, skip_value=skip)
    Theta_local = read_data(Theta_db, skip_value=skip)
    
    nx, ny, nz = U_local.shape
    z_grid = np.linspace(0, 1000, nz)
    
    # 3. 构建网格
    x = np.linspace(lon_start, lon_end, ny) * scale_xy
    y = np.linspace(lat_start, lat_end, nx) * scale_xy
    X, Y, Z = np.meshgrid(x, y, z_grid, indexing='ij')
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = -Z.transpose(1, 0, 2)
    
    # 4. 准备数据
    grid = pv.StructuredGrid(X, Y, Z)
    vectors = np.stack([
        U_local.flatten(order="F"),
        V_local.flatten(order="F"),
        W_local.flatten(order="F")
    ], axis=1)
    grid["velocity"] = vectors
    
    theta_data = Theta_local.flatten(order="F")
    salt_data = Salt_local.flatten(order="F")
    
    # 5. 计算盐度梯度
    salt_3d = salt_data.reshape(nx, ny, nz, order='F')
    grad_x, grad_y, grad_z = np.gradient(salt_3d)
    salt_gradient = np.stack([
        grad_x.flatten(order='F'),
        grad_y.flatten(order='F'),
        grad_z.flatten(order='F')
    ], axis=1)
    salt_gradient_mag = np.linalg.norm(salt_gradient, axis=1)
    if salt_gradient_mag.max() > salt_gradient_mag.min():
        salt_gradient_norm = (salt_gradient_mag - salt_gradient_mag.min()) / (salt_gradient_mag.max() - salt_gradient_mag.min())
    else:
        salt_gradient_norm = np.zeros_like(salt_gradient_mag)
    
    # 6. 应用透明度策略
    if strategy_idx < 1 or strategy_idx > 20:
        strategy_idx = 1
    strategy_func = strategy_functions[strategy_idx - 1]
    final_opacity = strategy_func(salt_data, salt_gradient_norm)
    
    # 7. 创建体积
    combined_volume = pv.StructuredGrid(X, Y, Z)
    combined_volume["Temperature"] = theta_data
    combined_volume["Salinity"] = salt_data
    
    # 8. 创建Plotter（根据 off_screen 决定是否离屏渲染）
    # 确保环境变量与参数一致（在文件开头已设置默认值，这里再次确认）
    os.environ['PYVISTA_OFF_SCREEN'] = 'true' if off_screen else 'false'
    os.environ['PYVISTA_USE_PANEL'] = 'false'
    os.environ['VTK_REMOTE_ENABLE'] = '0'
    
    # 在 Windows 上也需要设置
    if sys.platform == 'win32':
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['PYVISTA_USE_EGL'] = 'false'
    
    # 使用 off_screen 参数控制是否离屏渲染
    plotter = pv.Plotter(
        window_size=window_size, 
        off_screen=off_screen,
        title='' if off_screen else '3D可视化'
    )
    
    try:
        plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
    except Exception:
        pass
    
    # 9. 添加体积渲染（使用VTK底层API）
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        
        volume_actor = plotter.add_volume(
            combined_volume,
            scalars="Temperature",
            cmap="hot",
            opacity=0.1,
            opacity_unit_distance=5,
            show_scalar_bar=True,
            scalar_bar_args={'title': '温度 (Temperature) - 颜色'},
            shade=True,
            ambient=0.1,
            pickable=False,
            blending='composite'
        )
        
        mapper = volume_actor.GetMapper()
        vtk_volume = mapper.GetInput()
        volume_property = volume_actor.GetProperty()
        
        # 设置透明度
        salt_vtk_array = vtk_volume.GetPointData().GetArray("Salinity")
        if salt_vtk_array is None:
            salt_vtk_array = numpy_to_vtk(salt_data.astype(np.float32), array_type=vtk.VTK_FLOAT)
            salt_vtk_array.SetName("Salinity")
            vtk_volume.GetPointData().AddArray(salt_vtk_array)
        
        temp_min_val = np.min(theta_data)
        temp_max_val = np.max(theta_data)
        n_bins = 512
        opacity_func = vtk.vtkPiecewiseFunction()
        
        temp_vals = np.linspace(temp_min_val, temp_max_val, n_bins)
        temp_tolerance = (temp_max_val - temp_min_val) / n_bins * 2
        
        for t in temp_vals:
            temp_mask = np.abs(theta_data - t) <= temp_tolerance
            if np.any(temp_mask):
                corresponding_opacities = final_opacity[temp_mask]
                avg_opacity = np.mean(corresponding_opacities)
                opacity_func.AddPoint(t, np.clip(avg_opacity, final_opacity.min(), final_opacity.max()))
            else:
                temp_norm = (t - temp_min_val) / (temp_max_val - temp_min_val) if (temp_max_val - temp_min_val) > 0 else 0
                opacity = final_opacity.min() + (final_opacity.max() - final_opacity.min()) * temp_norm
                opacity_func.AddPoint(t, np.clip(opacity, final_opacity.min(), final_opacity.max()))
        
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.SetScalarOpacityUnitDistance(5.0)
        
        # 颜色映射
        temp_percentile_5 = np.percentile(theta_data, 5)
        temp_percentile_95 = np.percentile(theta_data, 95)
        
        try:
            import matplotlib.colormaps as cmaps
            hot_r_cmap = cmaps['hot_r']
        except (ImportError, KeyError):
            import matplotlib.pyplot as plt
            hot_r_cmap = plt.cm.get_cmap('hot_r')
        
        color_func = vtk.vtkColorTransferFunction()
        if (temp_max_val - temp_min_val) > 0:
            n_control_points = 10
            temp_vals = np.linspace(temp_percentile_5, temp_percentile_95, n_control_points)
            mid_end_idx = int(n_control_points * 0.7)
            mid_temp_vals = temp_vals[:mid_end_idx]
            mid_cmap_vals = np.linspace(0.1, 0.7, len(mid_temp_vals))
            extreme_temp_vals = temp_vals[mid_end_idx:]
            extreme_cmap_vals = np.linspace(0.7, 0.9, len(extreme_temp_vals))
            
            for temp_val, cmap_val in zip(mid_temp_vals, mid_cmap_vals):
                rgba = hot_r_cmap(cmap_val)
                color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
            for temp_val, cmap_val in zip(extreme_temp_vals, extreme_cmap_vals):
                rgba = hot_r_cmap(cmap_val)
                color_func.AddRGBPoint(temp_val, rgba[0], rgba[1], rgba[2])
            
            rgba_min = hot_r_cmap(0.1)
            rgba_max = hot_r_cmap(0.9)
            color_func.AddRGBPoint(temp_percentile_5, rgba_min[0], rgba_min[1], rgba_min[2])
            color_func.AddRGBPoint(temp_percentile_95, rgba_max[0], rgba_max[1], rgba_max[2])
        else:
            color_func.AddRGBPoint(temp_min_val, 0.5, 0.5, 0.5)
        
        volume_property.SetColor(color_func)
    except Exception as e:
        print(f"警告：VTK设置失败: {e}")
    
    # 10. 添加矢量场可视化
    # 采样点准备
    sampling_points_per_edge = 10
    n_samples_x = min(sampling_points_per_edge, nx)
    n_samples_y = min(sampling_points_per_edge, ny)
    n_samples_z = min(sampling_points_per_edge, nz)
    
    if nx > 1:
        x_indices = np.unique(np.concatenate([
            [0],
            np.linspace(1, nx-2, max(1, n_samples_x-2), dtype=int),
            [nx-1]
        ])) if nx > 2 and n_samples_x > 2 else np.linspace(0, nx-1, n_samples_x, dtype=int)
    else:
        x_indices = np.array([0], dtype=int)
    
    if ny > 1:
        y_indices = np.unique(np.concatenate([
            [0],
            np.linspace(1, ny-2, max(1, n_samples_y-2), dtype=int),
            [ny-1]
        ])) if ny > 2 and n_samples_y > 2 else np.linspace(0, ny-1, n_samples_y, dtype=int)
    else:
        y_indices = np.array([0], dtype=int)
    
    if nz > 1:
        z_indices = np.unique(np.concatenate([
            [0],
            np.linspace(1, nz-2, max(1, n_samples_z-2), dtype=int),
            [nz-1]
        ])) if nz > 2 and n_samples_z > 2 else np.linspace(0, nz-1, n_samples_z, dtype=int)
    else:
        z_indices = np.array([0], dtype=int)
    
    X_idx, Y_idx, Z_idx = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    X_idx = X_idx.flatten()
    Y_idx = Y_idx.flatten()
    Z_idx = Z_idx.flatten()
    
    sample_velocities = []
    sample_speeds = []
    sample_points_coords = []
    
    for i in range(len(X_idx)):
        x_idx, y_idx, z_idx = X_idx[i], Y_idx[i], Z_idx[i]
        x_idx = np.clip(x_idx, 0, nx-1)
        y_idx = np.clip(y_idx, 0, ny-1)
        z_idx = np.clip(z_idx, 0, nz-1)
        
        u_val = U_local[x_idx, y_idx, z_idx]
        v_val = V_local[x_idx, y_idx, z_idx]
        w_val = W_local[x_idx, y_idx, z_idx]
        
        vel = np.array([u_val, v_val, w_val])
        speed = np.linalg.norm(vel)
        sample_velocities.append(vel)
        sample_speeds.append(speed)
        
        point_idx = x_idx + y_idx * nx + z_idx * nx * ny
        coords = grid.points[point_idx]
        sample_points_coords.append(coords)
    
    sample_velocities = np.array(sample_velocities)
    sample_speeds = np.array(sample_speeds)
    sample_points_coords = np.array(sample_points_coords)
    
    sample_points = pv.PolyData(sample_points_coords)
    sample_points["velocity"] = sample_velocities
    sample_points["speed"] = sample_speeds
    
    # 根据vector_mode添加不同的矢量场可视化
    if vector_mode == 1 and VECTOR_OPTIMIZED_AVAILABLE and SCIPY_AVAILABLE:
        # 模式1：弯曲箭头
        try:
            arrows = create_bent_arrows(
                sample_points_coords,
                sample_velocities,
                sample_speeds,
                arrow_scale=arrow_scale,
                k_neighbors=k_neighbors,
                max_bend_factor=max_bend_factor
            )
            if arrows is not None and arrows.n_points > 0:
                plotter.add_mesh(
                    arrows,
                    scalars='speed',
                    cmap='cool',
                    opacity=1.0,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': '流速 (Speed)'}
                )
        except Exception as e:
            print(f"警告：弯曲箭头生成失败: {e}")
            # 回退到直线箭头
            speed_max = np.max(sample_speeds) if len(sample_speeds) > 0 else 1.0
            arrow_scale_factor = 50.0 / speed_max if speed_max > 0 else 1.0
            arrows = sample_points.glyph(orient='velocity', scale='speed', factor=arrow_scale_factor)
            plotter.add_mesh(arrows, scalars='speed', cmap='cool', opacity=1.0)
    elif vector_mode == 2 and VECTOR_OPTIMIZED_AVAILABLE and SCIPY_INTEGRATE_AVAILABLE:
        # 模式2：三维流线
        try:
            streamlines = create_3d_streamlines(
                grid,
                sample_points_coords,
                sample_velocities,
                streamline_length=streamline_length,
                step_size=step_size,
                n_seeds=n_seeds
            )
            if streamlines is not None and streamlines.n_points > 0:
                plotter.add_mesh(
                    streamlines,
                    scalars='speed',
                    cmap='cool',
                    line_width=2.0,
                    opacity=0.8,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': '流线速度 (Speed)'}
                )
        except Exception as e:
            print(f"警告：流线生成失败: {e}")
    elif vector_mode == 3 and VECTOR_OPTIMIZED_AVAILABLE and SKLEARN_AVAILABLE:
        # 模式3：聚类区域大箭头
        try:
            clusters, cluster_vels, labels = cluster_flow_regions(
                sample_points_coords,
                sample_velocities,
                spatial_eps=100.0,
                vel_eps=0.3,
                min_samples=3,
                target_clusters=target_clusters
            )
            if clusters and cluster_vels:
                cluster_arrows = create_cluster_bent_arrows(
                    clusters,
                    cluster_vels,
                    total_points=len(sample_points_coords),
                    arrow_scale=50.0,
                    spline_degree=3
                )
                if cluster_arrows is not None and cluster_arrows.n_points > 0:
                    plotter.add_mesh(
                        cluster_arrows,
                        scalars="Avg_Velocity" if "Avg_Velocity" in cluster_arrows.array_names else None,
                        cmap='viridis',
                        opacity=1.0,
                        show_scalar_bar=True,
                        scalar_bar_args={'title': '区域平均速度 (Avg Speed)'}
                    )
        except Exception as e:
            print(f"警告：聚类箭头生成失败: {e}")
    else:
        # 默认：直线箭头
        speed_max = np.max(sample_speeds) if len(sample_speeds) > 0 else 1.0
        arrow_scale_factor = 50.0 / speed_max if speed_max > 0 else 1.0
        arrows = sample_points.glyph(orient='velocity', scale='speed', factor=arrow_scale_factor)
        plotter.add_mesh(arrows, scalars='speed', cmap='cool', opacity=1.0)
    
    # 11. 添加坐标轴
    plotter.add_axes()
    
    if return_image:
        # 12. 保存截图（确保不打开窗口）
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            screenshot_path = tmp_file.name
        
        # 使用 screenshot 方法，确保不打开交互窗口
        # 重要：只使用 screenshot，绝不调用 show()
        try:
            # 直接使用 screenshot，不打开窗口
            plotter.screenshot(screenshot_path, window_size=window_size)
        except Exception as e:
            print(f"警告：screenshot 失败: {e}")
            # 如果 screenshot 失败，尝试渲染到内存
            try:
                # 渲染到 numpy 数组，然后保存
                img = plotter.screenshot(return_img=True)
                import matplotlib
                matplotlib.use('Agg')  # 使用非交互式后端
                import matplotlib.pyplot as plt
                plt.imsave(screenshot_path, img)
                plt.close('all')  # 关闭所有图形窗口
            except Exception as e2:
                print(f"错误：无法保存截图: {e2}")
                raise
        
        # 立即关闭 plotter，确保不保留窗口
        try:
            plotter.close()
        except Exception as e:
            print(f"警告：关闭plotter时出错: {e}")
        
        # 额外确保：如果 plotter 有窗口，强制关闭
        try:
            if hasattr(plotter, 'ren_win') and plotter.ren_win:
                plotter.ren_win.Finalize()
            if hasattr(plotter, 'render_window') and plotter.render_window:
                plotter.render_window.Finalize()
        except Exception as e:
            print(f"警告：Finalize窗口时出错: {e}")
        
        # 清理PyVista资源
        try:
            import gc
            gc.collect()  # 强制垃圾回收
        except:
            pass
        
        # 13. 读取并编码图像
        try:
          with open(screenshot_path, 'rb') as f:
              image_bytes = f.read()
          os.remove(screenshot_path)
        except Exception as e:
            print(f"警告：读取或删除截图文件时出错: {e}")
            # 尝试再次删除
            try:
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
            except:
                pass
            raise
        
        image_base64 = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('ascii')
        
        print(f"[generate_3d_visualization] 图像生成完成，准备返回")
        return image_base64
    else:
        # 交互窗口模式：直接显示，不返回图像
        try:
            plotter.show()
        finally:
            try:
                plotter.close()
            except Exception as e:
                print(f"警告：关闭plotter时出错: {e}")
        print("[generate_3d_visualization] 已以窗口方式显示，不返回图像")
        return None

