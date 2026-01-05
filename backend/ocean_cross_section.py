# Ocean Cube Cross-Section Visualization
# - Method 1: User inputs 3 points to define a plane, get cross-section
# - Method 2: User defines a line (view direction) through cube center, input depth, get perpendicular cross-section
# - Use linear interpolation to get data on cross-section
# - Visualize using the same method as ocean_2d_layers.py
import OpenVisus as ov
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator
import os

# Dependencies for bent arrows
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available, will use straight arrows")

# ----------------------------
# 1️⃣ Dataset Loading
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

print("Loading datasets...")
U_db = load_dataset("u")
V_db = load_dataset("v")
Salt_db = load_dataset("salt")
Theta_db = load_dataset("theta")

# ----------------------------
# 2️⃣ Local Region Parameters
# ----------------------------
lat_start, lat_end = 10, 40
lon_start, lon_end = 100, 130
nz = 10
data_quality = -4
scale_xy = 25

# ----------------------------
# 3️⃣ Read Local Data
# ----------------------------
def read_data(db):
    """Read local data (full, no sampling)"""
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
    
    result = data_full[lat_idx_start:lat_idx_end,
                       lon_idx_start:lon_idx_end,
                       :nz]
    
    return result

print("Reading data...")
U_local = read_data(U_db)
V_local = read_data(V_db)
Salt_local = read_data(Salt_db)
Theta_local = read_data(Theta_db)

nx, ny, nz = U_local.shape
print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")

# ----------------------------
# 4️⃣ Build 3D Grid Coordinates
# ----------------------------
x = np.linspace(lon_start, lon_end, ny) * scale_xy
y = np.linspace(lat_start, lat_end, nx) * scale_xy
z_grid = np.linspace(0, 1000, nz)
X, Y, Z = np.meshgrid(x, y, z_grid, indexing='ij')
X = X.transpose(1, 0, 2)
Y = Y.transpose(1, 0, 2)
Z = -Z.transpose(1, 0, 2)  # Negative Z (depth)

# Get grid bounds
x_min, x_max = X.min(), X.max()
y_min, y_max = Y.min(), Y.max()
z_min, z_max = Z.min(), Z.max()
cube_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])

print(f"Grid bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")
print(f"Cube center: ({cube_center[0]:.2f}, {cube_center[1]:.2f}, {cube_center[2]:.2f})")

# ----------------------------
# 5️⃣ Helper Functions
# ----------------------------
def opacity_mapping_linear(salt_data):
    """Linear opacity mapping: salinity range mapped to 0~1 opacity"""
    salt_min = salt_data.min()
    salt_max = salt_data.max()
    if salt_max > salt_min:
        opacity = (salt_data - salt_min) / (salt_max - salt_min)
    else:
        opacity = np.ones_like(salt_data) * 0.5
    return np.clip(opacity, 0.0, 1.0)

def create_2d_bent_arrows(x_coords, y_coords, u_vel, v_vel, speeds, arrow_scale=50.0, k_neighbors=4):
    """Generate 2D bent arrows (from ocean_2d_layers.py)"""
    if not SCIPY_AVAILABLE:
        return None
    
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    n_x_samples = 12
    n_y_samples = 12
    
    x_samples = np.linspace(x_min, x_max, n_x_samples)
    y_samples = np.linspace(y_min, y_max, n_y_samples)
    X_samples, Y_samples = np.meshgrid(x_samples, y_samples)
    sample_points_2d = np.column_stack([X_samples.flatten(), Y_samples.flatten()])
    
    sample_vels = []
    sample_speeds = []
    valid_sample_points = []
    
    nx_coords, ny_coords = x_coords.shape
    if u_vel.shape != (nx_coords, ny_coords):
        if u_vel.shape == (ny_coords, nx_coords):
            u_vel = u_vel.T
            v_vel = v_vel.T
            speeds = speeds.T
    
    for sp in sample_points_2d:
        x_dist = np.abs(x_coords - sp[0])
        y_dist = np.abs(y_coords - sp[1])
        total_dist = x_dist + y_dist
        min_idx = np.unravel_index(np.argmin(total_dist), total_dist.shape)
        y_idx, x_idx = min_idx
        
        y_idx = np.clip(y_idx, 0, nx_coords-1)
        x_idx = np.clip(x_idx, 0, ny_coords-1)
        
        u_val = u_vel[y_idx, x_idx]
        v_val = v_vel[y_idx, x_idx]
        speed_val = speeds[y_idx, x_idx]
        
        if speed_val > np.percentile(speeds.flatten(), 5):
            valid_sample_points.append(sp)
            sample_vels.append([u_val, v_val])
            sample_speeds.append(speed_val)
    
    if len(valid_sample_points) == 0:
        return None
    
    sample_points = np.array(valid_sample_points)
    sample_vels = np.array(sample_vels)
    sample_speeds = np.array(sample_speeds)
    
    arrows = []
    speed_max = np.max(sample_speeds) if len(sample_speeds) > 0 else 1.0
    arrow_scale_factor = arrow_scale * 1.5
    
    for i in range(len(sample_points)):
        try:
            current_point = sample_points[i]
            current_vel = sample_vels[i]
            speed = sample_speeds[i]
            
            if speed < 0.01 * speed_max:
                continue
            
            distances = np.linalg.norm(sample_points - current_point, axis=1)
            neighbor_indices = np.argsort(distances)[:k_neighbors]
            neighbor_points = sample_points[neighbor_indices]
            neighbor_vels = sample_vels[neighbor_indices]
            
            smoothed_vels = []
            for j in range(len(neighbor_vels)):
                weights = np.exp(-distances[neighbor_indices[j]]**2 / (2 * 1.0**2))
                smoothed_vels.append(neighbor_vels[j] * weights)
            smoothed_vels = np.array(smoothed_vels)
            avg_vel = np.mean(smoothed_vels, axis=0)
            
            num_points = 5
            total_length = speed * arrow_scale_factor / speed_max
            curve_points = []
            current_pos = current_point.copy()
            
            for j in range(num_points):
                t = j / (num_points - 1) if num_points > 1 else 0
                dir_vec = (1-t) * current_vel + t * avg_vel
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-6:
                    dir_vec = dir_vec / dir_norm
                else:
                    dir_vec = current_vel / (np.linalg.norm(current_vel) + 1e-6)
                
                step = dir_vec * (total_length / (num_points - 1)) if num_points > 1 else dir_vec * total_length
                current_pos = current_pos + step
                curve_points.append(current_pos.copy())
            
            if len(curve_points) >= 2:
                arrow_dir = curve_points[-1] - curve_points[0]
                arrow_norm = np.linalg.norm(arrow_dir)
                if arrow_norm > 1e-6:
                    arrows.append({
                        'pos': current_point,
                        'dir': arrow_dir / arrow_norm,
                        'length': arrow_norm,
                        'speed': speed
                    })
        except Exception:
            continue
    
    return arrows

# ----------------------------
# 6️⃣ Cross-Section Methods
# ----------------------------
def get_cross_section_by_three_points(p1, p2, p3, resolution=100):
    """
    Method 1: Get cross-section defined by three points
    
    Args:
        p1, p2, p3: Three 3D points (numpy arrays of shape (3,))
        resolution: Resolution of the cross-section grid
    
    Returns:
        cross_section_points: Points on the cross-section (N, 3)
        cross_section_data: Dictionary containing interpolated data
    """
    # Check if three points define a valid plane
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    
    if np.linalg.norm(normal) < 1e-6:
        raise ValueError("The three points are collinear and do not define a unique plane. Please re-enter.")
    
    # Normalize normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Define plane: normal * (point - p1) = 0
    # Find two orthogonal vectors in the plane
    if abs(normal[0]) < 0.9:
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 1, 0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Find intersection of plane with cube
    # Get all cube corners
    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_min, z_max],
        [x_min, y_max, z_max],
        [x_max, y_max, z_max]
    ])
    
    # Find intersection points with cube edges
    intersection_points = []
    edges = [
        ([x_min, y_min, z_min], [x_max, y_min, z_min]),
        ([x_min, y_min, z_min], [x_min, y_max, z_min]),
        ([x_min, y_min, z_min], [x_min, y_min, z_max]),
        ([x_max, y_max, z_min], [x_max, y_min, z_min]),
        ([x_max, y_max, z_min], [x_min, y_max, z_min]),
        ([x_max, y_max, z_min], [x_max, y_max, z_max]),
        ([x_min, y_max, z_max], [x_min, y_min, z_max]),
        ([x_min, y_max, z_max], [x_min, y_max, z_min]),
        ([x_min, y_max, z_max], [x_max, y_max, z_max]),
        ([x_max, y_min, z_max], [x_max, y_min, z_min]),
        ([x_max, y_min, z_max], [x_max, y_max, z_max]),
        ([x_max, y_min, z_max], [x_min, y_min, z_max])
    ]
    
    for edge_start, edge_end in edges:
        edge_start = np.array(edge_start)
        edge_end = np.array(edge_end)
        edge_dir = edge_end - edge_start
        
        # Line-plane intersection
        denom = np.dot(normal, edge_dir)
        if abs(denom) > 1e-6:
            t = np.dot(normal, p1 - edge_start) / denom
            if 0 <= t <= 1:
                intersection = edge_start + t * edge_dir
                # Check if intersection is within cube bounds
                if (x_min <= intersection[0] <= x_max and
                    y_min <= intersection[1] <= y_max and
                    z_min <= intersection[2] <= z_max):
                    intersection_points.append(intersection)
    
    if len(intersection_points) < 3:
        raise ValueError("Plane does not intersect with the cube sufficiently. Please check your points.")
    
    # Remove duplicate intersection points (within tolerance)
    intersection_points = np.array(intersection_points)
    if len(intersection_points) > 0:
        # Remove duplicates
        unique_points = []
        for point in intersection_points:
            is_duplicate = False
            for existing_point in unique_points:
                if np.linalg.norm(point - existing_point) < 1e-6:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        intersection_points = np.array(unique_points)
    
    if len(intersection_points) < 3:
        raise ValueError("Plane does not intersect with the cube sufficiently. Please check your points.")
    
    # Project intersection points to plane coordinate system
    proj_p1 = p1
    proj_points = intersection_points - proj_p1
    u_coords = np.dot(proj_points, u)
    v_coords = np.dot(proj_points, v)
    
    # Create grid on plane
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    
    u_grid = np.linspace(u_min, u_max, resolution)
    v_grid = np.linspace(v_min, v_max, resolution)
    U_grid, V_grid = np.meshgrid(u_grid, v_grid)
    
    # Convert back to 3D coordinates
    cross_section_points = (proj_p1 + 
                           U_grid[:, :, np.newaxis] * u[np.newaxis, np.newaxis, :] + 
                           V_grid[:, :, np.newaxis] * v[np.newaxis, np.newaxis, :])
    
    # Filter points within cube bounds
    mask = ((cross_section_points[:, :, 0] >= x_min) & (cross_section_points[:, :, 0] <= x_max) &
            (cross_section_points[:, :, 1] >= y_min) & (cross_section_points[:, :, 1] <= y_max) &
            (cross_section_points[:, :, 2] >= z_min) & (cross_section_points[:, :, 2] <= z_max))
    
    # Flatten for interpolation
    valid_points = cross_section_points[mask]
    
    if len(valid_points) == 0:
        raise ValueError("No valid points found in cross-section. Please check your input.")
    
    return valid_points, {'u': u, 'v': v, 'p1': p1, 'normal': normal, 'mask': mask, 
                          'U_grid': U_grid, 'V_grid': V_grid, 'cross_section_points': cross_section_points}

def get_cross_section_by_view_line(view_direction, depth_offset, resolution=100):
    """
    Method 2: Get cross-section perpendicular to view line through cube center
    
    Args:
        view_direction: Direction vector of the view line (numpy array of shape (3,))
        depth_offset: Depth along the view line from center (positive = forward, negative = backward)
        resolution: Resolution of the cross-section grid
    
    Returns:
        cross_section_points: Points on the cross-section (N, 3)
        cross_section_data: Dictionary containing plane information
    """
    # Normalize view direction
    view_direction = np.array(view_direction)
    view_norm = np.linalg.norm(view_direction)
    if view_norm < 1e-6:
        raise ValueError("View direction vector is zero. Please provide a valid direction.")
    view_direction = view_direction / view_norm
    
    # Calculate point on view line at specified depth
    view_point = cube_center + depth_offset * view_direction
    
    # Check if view point is within cube
    if not (x_min <= view_point[0] <= x_max and
            y_min <= view_point[1] <= y_max and
            z_min <= view_point[2] <= z_max):
        print(f"Warning: View point ({view_point[0]:.2f}, {view_point[1]:.2f}, {view_point[2]:.2f}) is outside cube bounds.")
        print("Will use the intersection point with cube.")
        # Find intersection with cube
        t_values = []
        for i in range(3):
            if abs(view_direction[i]) > 1e-6:
                bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)][i]
                for bound in bounds:
                    t = (bound - cube_center[i]) / view_direction[i]
                    point = cube_center + t * view_direction
                    if (x_min <= point[0] <= x_max and
                        y_min <= point[1] <= y_max and
                        z_min <= point[2] <= z_max):
                        t_values.append(t)
        if t_values:
            t_closest = min(t_values, key=lambda t: abs(t - depth_offset))
            view_point = cube_center + t_closest * view_direction
    
    # Plane normal is the view direction
    normal = view_direction
    
    # Find two orthogonal vectors in the plane (perpendicular to view direction)
    if abs(normal[0]) < 0.9:
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 1, 0])
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Determine plane extent (intersection with cube)
    # Project cube corners onto plane
    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_min, z_max],
        [x_min, y_max, z_max],
        [x_max, y_max, z_max]
    ])
    
    proj_corners = corners - view_point
    u_coords = np.dot(proj_corners, u)
    v_coords = np.dot(proj_corners, v)
    
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    
    # Create grid on plane
    u_grid = np.linspace(u_min, u_max, resolution)
    v_grid = np.linspace(v_min, v_max, resolution)
    U_grid, V_grid = np.meshgrid(u_grid, v_grid)
    
    # Convert to 3D coordinates
    cross_section_points = (view_point + 
                           U_grid[:, :, np.newaxis] * u[np.newaxis, np.newaxis, :] + 
                           V_grid[:, :, np.newaxis] * v[np.newaxis, np.newaxis, :])
    
    # Filter points within cube bounds
    mask = ((cross_section_points[:, :, 0] >= x_min) & (cross_section_points[:, :, 0] <= x_max) &
            (cross_section_points[:, :, 1] >= y_min) & (cross_section_points[:, :, 1] <= y_max) &
            (cross_section_points[:, :, 2] >= z_min) & (cross_section_points[:, :, 2] <= z_max))
    
    valid_points = cross_section_points[mask]
    
    if len(valid_points) == 0:
        raise ValueError("No valid points found in cross-section. Please check your input.")
    
    return valid_points, {'u': u, 'v': v, 'view_point': view_point, 'normal': normal, 'mask': mask,
                          'U_grid': U_grid, 'V_grid': V_grid, 'cross_section_points': cross_section_points}

# ----------------------------
# 7️⃣ Interpolation Functions
# ----------------------------
def interpolate_data_on_cross_section(cross_section_points, data_3d, grid_X, grid_Y, grid_Z):
    """
    Interpolate 3D data onto cross-section points using linear interpolation
    
    Args:
        cross_section_points: Points on cross-section (N, 3) - coordinates are (x, y, z)
        data_3d: 3D data array (nx, ny, nz) where nx=latitude, ny=longitude, nz=depth
        grid_X, grid_Y, grid_Z: 3D coordinate grids
    
    Returns:
        interpolated_data: Interpolated values (N,)
    """
    # Get unique coordinates for interpolation
    # Note: grid_X is (nx, ny, nz) after transpose, where nx corresponds to latitude (y), ny to longitude (x)
    # So we need to extract coordinates correctly
    x_unique = np.unique(grid_X[0, :, 0])  # Longitude (ny dimension)
    y_unique = np.unique(grid_Y[:, 0, 0])   # Latitude (nx dimension)
    z_unique = np.unique(grid_Z[0, 0, :])    # Depth (nz dimension)
    
    # Data shape is (nx, ny, nz) = (latitude, longitude, depth)
    # RegularGridInterpolator expects (x, y, z) = (longitude, latitude, depth)
    # So we need to transpose data from (nx, ny, nz) to (ny, nx, nz) = (longitude, latitude, depth)
    if data_3d.shape == (len(y_unique), len(x_unique), len(z_unique)):
        # Data is already in (latitude, longitude, depth) = (nx, ny, nz)
        # Need to transpose to (longitude, latitude, depth) = (ny, nx, nz)
        data_3d_interp = data_3d.transpose(1, 0, 2)
    elif data_3d.shape == (len(x_unique), len(y_unique), len(z_unique)):
        # Data is already in (longitude, latitude, depth)
        data_3d_interp = data_3d
    else:
        # Try to reshape
        print(f"Warning: Data shape {data_3d.shape} doesn't match expected shapes.")
        print(f"  Expected: (nx={len(y_unique)}, ny={len(x_unique)}, nz={len(z_unique)}) or (ny={len(x_unique)}, nx={len(y_unique)}, nz={len(z_unique)})")
        # Assume (nx, ny, nz) and transpose
        if data_3d.shape[0] == len(y_unique) and data_3d.shape[1] == len(x_unique):
            data_3d_interp = data_3d.transpose(1, 0, 2)
        else:
            raise ValueError(f"Cannot match data shape {data_3d.shape} to grid dimensions")
    
    # Create interpolator
    # RegularGridInterpolator expects coordinates in order (x, y, z) = (longitude, latitude, depth)
    interp = RegularGridInterpolator(
        (x_unique, y_unique, z_unique),  # (longitude, latitude, depth)
        data_3d_interp,  # Shape: (ny, nx, nz) = (longitude, latitude, depth)
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Interpolate
    # cross_section_points are in (x, y, z) = (longitude, latitude, depth) format
    interpolated = interp(cross_section_points)
    
    return interpolated

# ----------------------------
# 8️⃣ Visualization Function
# ----------------------------
def visualize_cross_section(cross_section_points, cross_section_data, 
                            u_interp, v_interp, salt_interp, theta_interp,
                            output_path="cross_section_output.png"):
    """
    Visualize cross-section using the same method as ocean_2d_layers.py
    """
    # Get plane coordinate system
    u = cross_section_data['u']
    v = cross_section_data['v']
    if 'p1' in cross_section_data:
        origin = cross_section_data['p1']
    else:
        origin = cross_section_data['view_point']
    
    # Project points to plane coordinates
    proj_points = cross_section_points - origin
    u_coords = np.dot(proj_points, u)
    v_coords = np.dot(proj_points, v)
    
    # Create 2D grid from points
    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    
    # Create regular grid (use a reasonable resolution)
    # Calculate resolution based on point density
    u_range = u_max - u_min
    v_range = v_max - v_min
    if u_range > 0 and v_range > 0:
        # Estimate resolution based on point density
        point_density = len(cross_section_points) / (u_range * v_range)
        resolution = max(100, min(200, int(np.sqrt(len(cross_section_points)))))
    else:
        resolution = 100
    
    u_grid = np.linspace(u_min, u_max, resolution)
    v_grid = np.linspace(v_min, v_max, resolution)
    U_grid, V_grid = np.meshgrid(u_grid, v_grid)
    
    # Interpolate data onto regular grid
    from scipy.interpolate import griddata
    
    # Filter out NaN values for interpolation
    valid_mask = ~(np.isnan(theta_interp) | np.isnan(salt_interp) | np.isnan(u_interp) | np.isnan(v_interp))
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid data points found in cross-section after interpolation.")
    
    u_coords_valid = u_coords[valid_mask]
    v_coords_valid = v_coords[valid_mask]
    theta_interp_valid = theta_interp[valid_mask]
    salt_interp_valid = salt_interp[valid_mask]
    u_interp_valid = u_interp[valid_mask]
    v_interp_valid = v_interp[valid_mask]
    
    # Interpolate each field
    theta_grid = griddata((u_coords_valid, v_coords_valid), theta_interp_valid, (U_grid, V_grid), method='linear')
    salt_grid = griddata((u_coords_valid, v_coords_valid), salt_interp_valid, (U_grid, V_grid), method='linear')
    u_grid_data = griddata((u_coords_valid, v_coords_valid), u_interp_valid, (U_grid, V_grid), method='linear')
    v_grid_data = griddata((u_coords_valid, v_coords_valid), v_interp_valid, (U_grid, V_grid), method='linear')
    
    # Calculate speed (handle NaN)
    speed_grid = np.sqrt(u_grid_data**2 + v_grid_data**2)
    speed_grid = np.nan_to_num(speed_grid, nan=0.0)
    
    # Calculate opacity (handle NaN)
    salt_grid_clean = np.nan_to_num(salt_grid, nan=0.0)
    opacity_grid = opacity_mapping_linear(salt_grid_clean)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create RGBA image
    # Handle NaN values in theta_grid
    theta_grid_clean = np.nan_to_num(theta_grid, nan=np.nanmean(theta_grid) if not np.isnan(np.nanmean(theta_grid)) else 0.0)
    temp_norm = mcolors.Normalize(vmin=np.nanmin(theta_grid_clean), vmax=np.nanmax(theta_grid_clean))
    temp_colors = plt.cm.hot_r(temp_norm(theta_grid_clean))
    temp_colors[:, :, 3] = opacity_grid
    
    # Set alpha to 0 for NaN regions
    nan_mask = np.isnan(theta_grid)
    temp_colors[nan_mask, 3] = 0.0
    
    # Display image
    ax.imshow(temp_colors, extent=[u_min, u_max, v_min, v_max],
              origin='lower', aspect='auto', interpolation='bicubic',
              filternorm=True, filterrad=4.0)
    
    # Add bent arrows
    arrows = create_2d_bent_arrows(U_grid, V_grid, u_grid_data, v_grid_data, speed_grid, arrow_scale=40.0)
    if arrows:
        for arrow in arrows:
            pos = arrow['pos']
            dir_vec = arrow['dir']
            length = arrow['length']
            
            ax.arrow(pos[0], pos[1], dir_vec[0]*length*0.9, dir_vec[1]*length*0.9,
                    head_width=length*0.25, head_length=length*0.3,
                    fc='cyan', ec='cyan', alpha=0.8, linewidth=2.0)
    
    # Set labels
    ax.set_xlabel('Plane U Coordinate', fontsize=12)
    ax.set_ylabel('Plane V Coordinate', fontsize=12)
    ax.set_title('Ocean Cross-Section Visualization\n(Temperature: Color, Salinity: Transparency, Velocity: Bent Arrows)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='hot_r', norm=temp_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Temperature', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved cross-section visualization: {output_path}")
    plt.close()

# ----------------------------
# 9️⃣ Main Program
# ----------------------------
print("\n" + "="*60)
print("Ocean Cube Cross-Section Visualization")
print("="*60)
print("\nSelect cross-section method:")
print("  1. Method 1: Define plane by three points")
print("  2. Method 2: Define plane perpendicular to view line through cube center")
print("="*60)

method_choice = input("Enter method number (1 or 2, default 1): ").strip()
if method_choice == "":
    method_choice = "1"

if method_choice == "1":
    # Method 1: Three points
    print("\nMethod 1: Define plane by three points")
    print(f"Cube bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")
    
    while True:
        try:
            print("\nEnter three points (x, y, z coordinates):")
            p1_input = input("Point 1 (x, y, z): ").strip().split(',')
            p2_input = input("Point 2 (x, y, z): ").strip().split(',')
            p3_input = input("Point 3 (x, y, z): ").strip().split(',')
            
            p1 = np.array([float(x.strip()) for x in p1_input])
            p2 = np.array([float(x.strip()) for x in p2_input])
            p3 = np.array([float(x.strip()) for x in p3_input])
            
            # Get cross-section
            cross_section_points, cross_section_data = get_cross_section_by_three_points(p1, p2, p3, resolution=150)
            print(f"✅ Cross-section obtained: {len(cross_section_points)} points")
            break
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            retry = input("Retry? (y/n, default y): ").strip().lower()
            if retry != 'y' and retry != '':
                raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            retry = input("Retry? (y/n, default y): ").strip().lower()
            if retry != 'y' and retry != '':
                raise

elif method_choice == "2":
    # Method 2: View line
    print("\nMethod 2: Define plane perpendicular to view line")
    print(f"Cube center: ({cube_center[0]:.2f}, {cube_center[1]:.2f}, {cube_center[2]:.2f})")
    
    while True:
        try:
            print("\nEnter view direction vector (x, y, z):")
            view_input = input("View direction (x, y, z): ").strip().split(',')
            view_direction = np.array([float(x.strip()) for x in view_input])
            
            depth_input = input("Depth offset from center (positive=forward, negative=backward, default 0): ").strip()
            depth_offset = float(depth_input) if depth_input else 0.0
            
            # Get cross-section
            cross_section_points, cross_section_data = get_cross_section_by_view_line(
                view_direction, depth_offset, resolution=150)
            print(f"✅ Cross-section obtained: {len(cross_section_points)} points")
            break
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            retry = input("Retry? (y/n, default y): ").strip().lower()
            if retry != 'y' and retry != '':
                raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            retry = input("Retry? (y/n, default y): ").strip().lower()
            if retry != 'y' and retry != '':
                raise
else:
    raise ValueError(f"Invalid method choice: {method_choice}")

# Interpolate data onto cross-section
print("\nInterpolating data onto cross-section...")
theta_interp = interpolate_data_on_cross_section(cross_section_points, Theta_local, X, Y, Z)
salt_interp = interpolate_data_on_cross_section(cross_section_points, Salt_local, X, Y, Z)
u_interp = interpolate_data_on_cross_section(cross_section_points, U_local, X, Y, Z)
v_interp = interpolate_data_on_cross_section(cross_section_points, V_local, X, Y, Z)

print(f"  Temperature range: [{np.nanmin(theta_interp):.4f}, {np.nanmax(theta_interp):.4f}]")
print(f"  Salinity range: [{np.nanmin(salt_interp):.4f}, {np.nanmax(salt_interp):.4f}]")

# Visualize
output_dir = "ocean_cross_section_output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cross_section.png")

visualize_cross_section(cross_section_points, cross_section_data,
                       u_interp, v_interp, salt_interp, theta_interp,
                       output_path=output_path)

print(f"\n{'='*60}")
print(f"✅ Cross-section visualization completed. Saved to: {output_path}")
print(f"{'='*60}")

