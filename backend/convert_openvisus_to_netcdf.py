"""
将OpenVisus数据转换为NetCDF格式
如果已安装OpenVisus，可以使用此脚本将数据保存为NetCDF文件
"""
import numpy as np
import os
import sys

def convert_openvisus_to_netcdf(dataset_url, output_file, time_steps=[0], 
                                 lat_range=None, lon_range=None):
    """
    从OpenVisus数据集读取数据并保存为NetCDF
    
    参数:
        dataset_url: OpenVisus数据集URL
        output_file: 输出NetCDF文件路径
        time_steps: 要转换的时间步列表
        lat_range: 纬度范围 (start, end)，None表示全部
        lon_range: 经度范围 (start, end)，None表示全部
    """
    try:
        import OpenVisus as ov
    except ImportError:
        print("错误: 未安装OpenVisus库")
        print("请安装: pip install OpenVisus")
        return False
    
    try:
        import xarray as xr
    except ImportError:
        try:
            import netCDF4
            USE_NETCDF4 = True
        except ImportError:
            print("错误: 需要安装 xarray 或 netCDF4")
            print("请安装: pip install xarray 或 pip install netcdf4")
            return False
    else:
        USE_NETCDF4 = False
    
    print(f"正在加载OpenVisus数据集: {dataset_url}")
    try:
        db = ov.LoadDataset(dataset_url)
        print("数据集加载成功")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return False
    
    # 获取数据集信息
    try:
        logic_box = db.getLogicBox()
        timesteps = db.getTimesteps()
        print(f"数据集维度: {logic_box[1]}")
        print(f"时间步数: {len(timesteps)}")
    except Exception as e:
        print(f"获取数据集信息失败: {e}")
        return False
    
    # 读取数据
    print(f"\n正在读取时间步 {time_steps}...")
    all_data = []
    
    for t in time_steps:
        try:
            print(f"  读取时间步 {t}...")
            data = db.read(time=t)
            print(f"  数据形状: {data.shape}")
            all_data.append(data)
        except Exception as e:
            print(f"  读取时间步 {t} 失败: {e}")
            continue
    
    if not all_data:
        print("错误: 未能读取任何数据")
        return False
    
    # 转换为xarray Dataset并保存
    print(f"\n正在保存为NetCDF: {output_file}")
    
    if USE_NETCDF4:
        # 使用netCDF4保存
        import netCDF4
        nc = netCDF4.Dataset(output_file, 'w', format='NETCDF4')
        
        # 创建维度
        if len(all_data[0].shape) == 3:
            ny, nx, nz = all_data[0].shape
            nc.createDimension('time', len(all_data))
            nc.createDimension('y', ny)
            nc.createDimension('x', nx)
            nc.createDimension('z', nz)
            
            # 创建变量
            var = nc.createVariable('data', 'f4', ('time', 'y', 'x', 'z'))
            for i, data in enumerate(all_data):
                var[i, :, :, :] = data
        else:
            ny, nx = all_data[0].shape
            nc.createDimension('time', len(all_data))
            nc.createDimension('y', ny)
            nc.createDimension('x', nx)
            
            var = nc.createVariable('data', 'f4', ('time', 'y', 'x'))
            for i, data in enumerate(all_data):
                var[i, :, :] = data
        
        nc.close()
    else:
        # 使用xarray保存
        data_array = np.array(all_data)
        
        if len(data_array.shape) == 4:  # (time, y, x, z)
            ds = xr.Dataset({
                'data': (['time', 'y', 'x', 'z'], data_array)
            })
        else:  # (time, y, x)
            ds = xr.Dataset({
                'data': (['time', 'y', 'x'], data_array)
            })
        
        ds.to_netcdf(output_file)
    
    print(f"✓ 转换完成: {output_file}")
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")
    return True

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python convert_openvisus_to_netcdf.py <OpenVisus_URL> <输出文件.nc> [时间步]")
        print("\n示例:")
        print("  python convert_openvisus_to_netcdf.py \\")
        print("    'https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx' \\")
        print("    output.nc 0")
        sys.exit(1)
    
    dataset_url = sys.argv[1]
    output_file = sys.argv[2]
    time_steps = [int(t) for t in sys.argv[3:]] if len(sys.argv) > 3 else [0]
    
    success = convert_openvisus_to_netcdf(dataset_url, output_file, time_steps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

