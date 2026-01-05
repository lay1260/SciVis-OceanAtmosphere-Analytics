"""
DYAMOND 数据文件查找和获取辅助工具
帮助用户找到或下载所需的气象数据文件
"""
import os
import sys
import requests
from pathlib import Path

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_local_files(directory="."):
    """检查本地目录中的NetCDF文件"""
    print_section("检查本地数据文件")
    
    nc_files = []
    search_dirs = [
        directory,
        os.path.join(directory, "data"),
        os.path.join(directory, "datasets"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for ext in ["*.nc", "*.nc4", "*.h5", "*.hdf5"]:
                files = list(Path(search_dir).rglob(ext))
                nc_files.extend(files)
    
    if nc_files:
        print(f"\n找到 {len(nc_files)} 个可能的数据文件:")
        for i, file in enumerate(nc_files[:20], 1):  # 只显示前20个
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {i}. {file}")
            print(f"     大小: {size_mb:.2f} MB")
        if len(nc_files) > 20:
            print(f"  ... 还有 {len(nc_files) - 20} 个文件")
        return str(nc_files[0]) if nc_files else None
    else:
        print("\n未找到本地NetCDF文件")
        print("搜索目录:")
        for d in search_dirs:
            print(f"  - {d}")
        return None

def show_data_sources():
    """显示数据源信息"""
    print_section("DYAMOND 数据源信息")
    
    sources = [
        {
            "name": "NASA NSDF (原始数据源)",
            "url": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/",
            "note": "通过OpenVisus访问，需要转换为NetCDF格式",
            "access": "需要OpenVisus库"
        },
        {
            "name": "ERA5 再分析数据 (推荐替代方案)",
            "url": "https://cds.climate.copernicus.eu/",
            "note": "免费注册后下载，包含MSLP、风场等变量",
            "access": "需要注册账户，支持NetCDF格式"
        },
        {
            "name": "NCEP/NCAR 再分析数据",
            "url": "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html",
            "note": "公开数据，包含海平面气压和风场",
            "access": "直接下载，NetCDF格式"
        },
        {
            "name": "JRA-55 再分析数据",
            "url": "https://jra.kishou.go.jp/JRA-55/index_en.html",
            "note": "日本气象厅数据，适合亚洲区域分析",
            "access": "需要注册，支持NetCDF格式"
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   URL: {source['url']}")
        print(f"   说明: {source['note']}")
        print(f"   访问方式: {source['access']}")

def show_download_instructions():
    """显示下载说明"""
    print_section("数据下载指南")
    
    print("""
方法1: 使用ERA5数据（推荐，最简单）
----------------------------------------
1. 访问: https://cds.climate.copernicus.eu/
2. 注册免费账户
3. 在"Datasets"中搜索"ERA5"
4. 选择"ERA5 hourly data on single levels"
5. 选择变量:
   - Mean sea level pressure (msl)
   - U-component of wind (u10, u850)
   - V-component of wind (v10, v850)
6. 选择时间和区域
7. 下载NetCDF格式文件

方法2: 使用NCEP/NCAR数据（无需注册）
----------------------------------------
1. 访问: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
2. 下载"Surface"和"Pressure Level"数据
3. 选择包含MSLP和风场的文件
4. 文件通常是NetCDF格式

方法3: 从OpenVisus转换（如果已安装OpenVisus）
----------------------------------------
可以使用转换脚本将OpenVisus数据保存为NetCDF格式
    """)

def create_sample_download_script():
    """创建示例下载脚本"""
    script_content = '''"""
示例：从ERA5下载数据的Python脚本
需要先安装: pip install cdsapi
需要配置CDS API密钥: https://cds.climate.copernicus.eu/how-to-use
"""
import cdsapi

c = cdsapi.Client()

# 下载海平面气压
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'mean_sea_level_pressure',
        'year': '2020',
        'month': '08',
        'day': '01',
        'time': '00:00',
        'format': 'netcdf',
    },
    'mslp_20200801.nc')

# 下载850hPa风场
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['u_component_of_wind', 'v_component_of_wind'],
        'pressure_level': '850',
        'year': '2020',
        'month': '08',
        'day': '01',
        'time': '00:00',
        'format': 'netcdf',
    },
    'wind_850_20200801.nc')
'''
    
    script_path = os.path.join(os.path.dirname(__file__), "download_era5_example.py")
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"\n已创建示例下载脚本: {script_path}")
    except Exception as e:
        print(f"\n无法创建脚本: {e}")

def suggest_test_data():
    """建议测试数据"""
    print_section("测试数据建议")
    
    print("""
如果暂时无法获取真实数据，可以：

1. 使用程序自带的模拟数据模式:
   python ceishi——feng.py

2. 使用小样本测试数据:
   - 访问: https://www.unidata.ucar.edu/software/netcdf/examples/
   - 下载示例NetCDF文件用于测试

3. 使用公开的小数据集:
   - NOAA示例数据: https://www.ncei.noaa.gov/data/
   - 选择较小的区域和时间范围进行测试
    """)

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  DYAMOND 数据文件查找工具")
    print("=" * 70)
    
    # 检查本地文件
    local_file = check_local_files()
    
    # 显示数据源
    show_data_sources()
    
    # 显示下载说明
    show_download_instructions()
    
    # 建议测试数据
    suggest_test_data()
    
    # 创建示例脚本
    print_section("创建下载示例脚本")
    create_sample_download_script()
    
    # 总结
    print_section("总结")
    if local_file:
        print(f"\n✓ 找到本地数据文件: {local_file}")
        print(f"\n可以直接使用:")
        print(f"  python ceishi——feng.py \"{local_file}\"")
    else:
        print("\n建议:")
        print("1. 如果只是想测试算法，直接运行: python ceishi——feng.py")
        print("2. 如果需要真实数据，访问ERA5或NCEP/NCAR网站下载")
        print("3. 下载后保存为 .nc 文件，然后运行: python ceishi——feng.py <文件路径>")

if __name__ == "__main__":
    main()

