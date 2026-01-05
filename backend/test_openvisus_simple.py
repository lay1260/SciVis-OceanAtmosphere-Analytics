#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试OpenVisus是否能正常工作
"""
import sys

print("=" * 60)
print("OpenVisus Simple Test")
print("=" * 60)

# 测试1: 导入
print("\n[1] Testing OpenVisus import...")
try:
    import OpenVisus as ov
    print("✓ OpenVisus imported successfully")
except Exception as e:
    print(f"✗ Failed to import OpenVisus: {str(e)}")
    sys.exit(1)

# 测试2: 基本功能
print("\n[2] Testing basic OpenVisus functionality...")
try:
    # 尝试创建一个简单的测试
    print("  OpenVisus version info...")
    if hasattr(ov, '__version__'):
        print(f"  Version: {ov.__version__}")
    else:
        print("  Version: unknown")
    print("✓ Basic functionality OK")
except Exception as e:
    print(f"✗ Basic functionality test failed: {str(e)}")
    sys.exit(1)

# 测试3: 尝试加载数据集（这可能会崩溃）
print("\n[3] Testing dataset loading...")
print("  WARNING: This may cause the process to crash!")
print("  If it crashes, you'll see an Access Violation error")

dataset_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"

try:
    print(f"  Attempting to load: {dataset_url}")
    print("  This may take a while or crash...")
    
    db = ov.LoadDataset(dataset_url)
    print("✓ Dataset loaded successfully!")
    
    try:
        logic_box = db.getLogicBox()
        print(f"  Logic box: {logic_box}")
    except:
        print("  ⚠ Could not get logic box")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    sys.exit(0)
    
except KeyboardInterrupt:
    print("\n✗ Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

