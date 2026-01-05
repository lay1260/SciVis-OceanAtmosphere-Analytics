# Cross-section API for typhoon detail page
# Wraps ocean_cross_section.py functionality for web API
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
import io
import tempfile
import os
from scipy.interpolate import RegularGridInterpolator, griddata

# Import functions from ocean_cross_section.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll need to load data and set up the grid
# This will be done when the module is imported or when API is called
import OpenVisus as ov

# Dataset loading
base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

def load_dataset(variable):
    if variable in ["theta", "w"]:
        base_dir = f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
    elif variable == "u":
        base_dir = "mit_output/llc2160_arco/visus.idx"
    else:
        base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
    dataset_url = base_url + base_dir
    db = ov.LoadDataset(dataset_url)
    return db

# Global data cache
_data_cache = {
    'loaded': False,
    'U_db': None,
    'V_db': None,
    'Salt_db': None,
    'Theta_db': None,
    'U_local': None,
    'V_local': None,
    'Salt_local': None,
    'Theta_local': None,
    'X': None,
    'Y': None,
    'Z': None,
    'bounds': None,
    'cube_center': None
}

def ensure_data_loaded():
    """Load data if not already loaded"""
    if _data_cache['loaded']:
        return
    
    print("[CrossSection] Loading datasets...")
    _data_cache['U_db'] = load_dataset("u")
    _data_cache['V_db'] = load_dataset("v")
    _data_cache['Salt_db'] = load_dataset("salt")
    _data_cache['Theta_db'] = load_dataset("theta")
    
    # Parameters (can be made configurable)
    lat_start, lat_end = 10, 40
    lon_start, lon_end = 100, 130
    nz = 10
    data_quality = -4
    scale_xy = 25
    
    def read_data(db):
        data_full = db.read(time=0, quality=data_quality)
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
        
        return data_full[lat_idx_start:lat_idx_end,
                       lon_idx_start:lon_idx_end,
                       :nz]
    
    print("[CrossSection] Reading data...")
    _data_cache['U_local'] = read_data(_data_cache['U_db'])
    _data_cache['V_local'] = read_data(_data_cache['V_db'])
    _data_cache['Salt_local'] = read_data(_data_cache['Salt_db'])
    _data_cache['Theta_local'] = read_data(_data_cache['Theta_db'])
    
    nx, ny, nz = _data_cache['U_local'].shape
    
    # Build 3D grid
    x = np.linspace(lon_start, lon_end, ny) * scale_xy
    y = np.linspace(lat_start, lat_end, nx) * scale_xy
    z_grid = np.linspace(0, 1000, nz)
    X, Y, Z = np.meshgrid(x, y, z_grid, indexing='ij')
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = -Z.transpose(1, 0, 2)
    
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    cube_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
    
    _data_cache['X'] = X
    _data_cache['Y'] = Y
    _data_cache['Z'] = Z
    _data_cache['bounds'] = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'z_min': z_min, 'z_max': z_max}
    _data_cache['cube_center'] = cube_center
    _data_cache['loaded'] = True
    
    print(f"[CrossSection] Data loaded. Grid: nx={nx}, ny={ny}, nz={nz}")
    print(f"[CrossSection] Bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")

# Import cross-section functions (simplified versions)
def get_cross_section_by_three_points(p1, p2, p3, resolution=100):
    """Method 1: Get cross-section defined by three points"""
    ensure_data_loaded()
    
    bounds = _data_cache['bounds']
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']
    z_min, z_max = bounds['z_min'], bounds['z_max']
    
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    
    if np.linalg.norm(normal) < 1e-6:
        raise ValueError("The three points are collinear.")
    
    normal = normal / np.linalg.norm(normal)
    
    # Find two orthogonal vectors in the plane
    if abs(normal[0]) < 0.9:
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 1, 0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Find intersection with cube (simplified)
    # Create grid on plane
    u_min, u_max = -500, 500  # Approximate range
    v_min, v_max = -500, 500
    
    u_grid = np.linspace(u_min, u_max, resolution)
    v_grid = np.linspace(v_min, v_max, resolution)
    U_grid, V_grid = np.meshgrid(u_grid, v_grid)
    
    cross_section_points = (p1 + 
                           U_grid[:, :, np.newaxis] * u[np.newaxis, np.newaxis, :] + 
                           V_grid[:, :, np.newaxis] * v[np.newaxis, np.newaxis, :])
    
    mask = ((cross_section_points[:, :, 0] >= x_min) & (cross_section_points[:, :, 0] <= x_max) &
            (cross_section_points[:, :, 1] >= y_min) & (cross_section_points[:, :, 1] <= y_max) &
            (cross_section_points[:, :, 2] >= z_min) & (cross_section_points[:, :, 2] <= z_max))
    
    valid_points = cross_section_points[mask]
    
    if len(valid_points) == 0:
        raise ValueError("No valid points found in cross-section.")
    
    return valid_points, {'u': u, 'v': v, 'p1': p1, 'normal': normal, 'mask': mask,
                          'U_grid': U_grid, 'V_grid': V_grid, 'cross_section_points': cross_section_points}

def get_cross_section_by_view_line(view_direction, depth_offset, resolution=100):
    """Method 2: Get cross-section perpendicular to view line"""
    ensure_data_loaded()
    
    bounds = _data_cache['bounds']
    cube_center = _data_cache['cube_center']
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']
    z_min, z_max = bounds['z_min'], bounds['z_max']
    
    view_direction = np.array(view_direction)
    view_norm = np.linalg.norm(view_direction)
    if view_norm < 1e-6:
        raise ValueError("View direction vector is zero.")
    view_direction = view_direction / view_norm
    
    view_point = cube_center + depth_offset * view_direction
    
    normal = view_direction
    
    if abs(normal[0]) < 0.9:
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 1, 0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Create grid
    u_min, u_max = -500, 500
    v_min, v_max = -500, 500
    
    u_grid = np.linspace(u_min, u_max, resolution)
    v_grid = np.linspace(v_min, v_max, resolution)
    U_grid, V_grid = np.meshgrid(u_grid, v_grid)
    
    cross_section_points = (view_point + 
                           U_grid[:, :, np.newaxis] * u[np.newaxis, np.newaxis, :] + 
                           V_grid[:, :, np.newaxis] * v[np.newaxis, np.newaxis, :])
    
    mask = ((cross_section_points[:, :, 0] >= x_min) & (cross_section_points[:, :, 0] <= x_max) &
            (cross_section_points[:, :, 1] >= y_min) & (cross_section_points[:, :, 1] <= y_max) &
            (cross_section_points[:, :, 2] >= z_min) & (cross_section_points[:, :, 2] <= z_max))
    
    valid_points = cross_section_points[mask]
    
    if len(valid_points) == 0:
        raise ValueError("No valid points found in cross-section.")
    
    return valid_points, {'u': u, 'v': v, 'view_point': view_point, 'normal': normal, 'mask': mask,
                          'U_grid': U_grid, 'V_grid': V_grid, 'cross_section_points': cross_section_points}

def interpolate_data_on_cross_section(cross_section_points, data_3d, grid_X, grid_Y, grid_Z):
    """Interpolate 3D data onto cross-section points"""
    x_unique = np.unique(grid_X[0, :, 0])
    y_unique = np.unique(grid_Y[:, 0, 0])
    z_unique = np.unique(grid_Z[0, 0, :])
    
    if data_3d.shape == (len(y_unique), len(x_unique), len(z_unique)):
        data_3d_interp = data_3d.transpose(1, 0, 2)
    elif data_3d.shape == (len(x_unique), len(y_unique), len(z_unique)):
        data_3d_interp = data_3d
    else:
        if data_3d.shape[0] == len(y_unique) and data_3d.shape[1] == len(x_unique):
            data_3d_interp = data_3d.transpose(1, 0, 2)
        else:
            raise ValueError(f"Cannot match data shape {data_3d.shape} to grid dimensions")
    
    interp = RegularGridInterpolator(
        (x_unique, y_unique, z_unique),
        data_3d_interp,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    interpolated = interp(cross_section_points)
    return interpolated

def opacity_mapping_linear(salt_data):
    """Linear opacity mapping"""
    salt_min = salt_data.min()
    salt_max = salt_data.max()
    if salt_max > salt_min:
        opacity = (salt_data - salt_min) / (salt_max - salt_min)
    else:
        opacity = np.ones_like(salt_data) * 0.5
    return np.clip(opacity, 0.0, 1.0)

def generate_cross_section_image(method, params, resolution=150):
    """
    Generate cross-section visualization image
    
    Args:
        method: 'three_points' or 'view_line'
        params: dict with parameters
            - For 'three_points': {'p1': [x,y,z], 'p2': [x,y,z], 'p3': [x,y,z]}
            - For 'view_line': {'view_direction': [x,y,z], 'depth_offset': float}
        resolution: resolution of cross-section grid
    
    Returns:
        base64 encoded image string
    """
    ensure_data_loaded()
    
    # Get cross-section
    if method == 'three_points':
        p1 = params['p1']
        p2 = params['p2']
        p3 = params['p3']
        cross_section_points, cross_section_data = get_cross_section_by_three_points(p1, p2, p3, resolution)
    elif method == 'view_line':
        view_direction = params['view_direction']
        depth_offset = params.get('depth_offset', 0.0)
        cross_section_points, cross_section_data = get_cross_section_by_view_line(view_direction, depth_offset, resolution)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Interpolate data
    X, Y, Z = _data_cache['X'], _data_cache['Y'], _data_cache['Z']
    theta_interp = interpolate_data_on_cross_section(cross_section_points, _data_cache['Theta_local'], X, Y, Z)
    salt_interp = interpolate_data_on_cross_section(cross_section_points, _data_cache['Salt_local'], X, Y, Z)
    u_interp = interpolate_data_on_cross_section(cross_section_points, _data_cache['U_local'], X, Y, Z)
    v_interp = interpolate_data_on_cross_section(cross_section_points, _data_cache['V_local'], X, Y, Z)
    
    # Project to plane coordinates
    u = cross_section_data['u']
    v = cross_section_data['v']
    if 'p1' in cross_section_data:
        origin = cross_section_data['p1']
    else:
        origin = cross_section_data['view_point']
    
    proj_points = cross_section_points - origin
    u_coords = np.dot(proj_points, u)
    v_coords = np.dot(proj_points, v)
    
    # Create regular grid
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    
    resolution_2d = max(100, min(200, int(np.sqrt(len(cross_section_points)))))
    u_grid = np.linspace(u_min, u_max, resolution_2d)
    v_grid = np.linspace(v_min, v_max, resolution_2d)
    U_grid, V_grid = np.meshgrid(u_grid, v_grid)
    
    # Interpolate onto grid
    valid_mask = ~(np.isnan(theta_interp) | np.isnan(salt_interp) | np.isnan(u_interp) | np.isnan(v_interp))
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid data points found.")
    
    u_coords_valid = u_coords[valid_mask]
    v_coords_valid = v_coords[valid_mask]
    theta_interp_valid = theta_interp[valid_mask]
    salt_interp_valid = salt_interp[valid_mask]
    u_interp_valid = u_interp[valid_mask]
    v_interp_valid = v_interp[valid_mask]
    
    theta_grid = griddata((u_coords_valid, v_coords_valid), theta_interp_valid, (U_grid, V_grid), method='linear')
    salt_grid = griddata((u_coords_valid, v_coords_valid), salt_interp_valid, (U_grid, V_grid), method='linear')
    u_grid_data = griddata((u_coords_valid, v_coords_valid), u_interp_valid, (U_grid, V_grid), method='linear')
    v_grid_data = griddata((u_coords_valid, v_coords_valid), v_interp_valid, (U_grid, V_grid), method='linear')
    
    speed_grid = np.sqrt(u_grid_data**2 + v_grid_data**2)
    speed_grid = np.nan_to_num(speed_grid, nan=0.0)
    
    salt_grid_clean = np.nan_to_num(salt_grid, nan=0.0)
    opacity_grid = opacity_mapping_linear(salt_grid_clean)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    theta_grid_clean = np.nan_to_num(theta_grid, nan=np.nanmean(theta_grid) if not np.isnan(np.nanmean(theta_grid)) else 0.0)
    temp_norm = mcolors.Normalize(vmin=np.nanmin(theta_grid_clean), vmax=np.nanmax(theta_grid_clean))
    temp_colors = plt.cm.hot_r(temp_norm(theta_grid_clean))
    temp_colors[:, :, 3] = opacity_grid
    
    nan_mask = np.isnan(theta_grid)
    temp_colors[nan_mask, 3] = 0.0
    
    ax.imshow(temp_colors, extent=[u_min, u_max, v_min, v_max],
              origin='lower', aspect='auto', interpolation='bicubic')
    
    # Add velocity arrows (simplified)
    step = max(1, resolution_2d // 15)
    for i in range(0, resolution_2d, step):
        for j in range(0, resolution_2d, step):
            if not np.isnan(u_grid_data[i, j]) and not np.isnan(v_grid_data[i, j]):
                u_val = u_grid_data[i, j]
                v_val = v_grid_data[i, j]
                speed = speed_grid[i, j]
                if speed > 0.01:
                    u_pos = u_grid[i]
                    v_pos = v_grid[j]
                    scale = 20.0
                    ax.arrow(u_pos, v_pos, u_val*scale, v_val*scale,
                            head_width=5, head_length=5,
                            fc='cyan', ec='cyan', alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Plane U Coordinate', fontsize=12)
    ax.set_ylabel('Plane V Coordinate', fontsize=12)
    ax.set_title('Ocean Cross-Section Visualization\n(Temperature: Color, Salinity: Transparency, Velocity: Arrows)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap='hot_r', norm=temp_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Temperature', fontsize=11)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    
    return 'data:image/png;base64,' + img_base64

