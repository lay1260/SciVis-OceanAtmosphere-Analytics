#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据集加载，诊断问题
"""

import sys
import traceback

def test_openvisus_import():
    """测试OpenVisus导入"""
    print("=" * 60)
    print("Test 1: OpenVisus Import")
    print("=" * 60)
    try:
        import OpenVisus as ov
        print("✓ OpenVisus imported successfully")
        print(f"  OpenVisus version: {ov.__version__ if hasattr(ov, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import OpenVisus: {str(e)}")
        print("  Please install: pip install OpenVisus")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing OpenVisus: {str(e)}")
        traceback.print_exc()
        return False

def test_dataset_load():
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("Test 2: Dataset Loading")
    print("=" * 60)
    
    try:
        import OpenVisus as ov
    except ImportError:
        print("✗ Skipping - OpenVisus not available")
        return False
    
    variable = "salt"
    base_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"
    base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"
    dataset_url = base_url + base_dir
    
    print(f"Dataset URL: {dataset_url}")
    print("Attempting to load dataset...")
    print("(This may take a while or cause the process to exit)")
    print()
    
    try:
        print("Calling ov.LoadDataset()...")
        db = ov.LoadDataset(dataset_url)
        print("✓ Dataset loaded successfully!")
        
        try:
            logic_box = db.getLogicBox()
            print(f"  Logic box: {logic_box}")
        except Exception as e:
            print(f"  ⚠ Could not get logic box: {str(e)}")
        
        try:
            timesteps = db.getTimesteps()
            print(f"  Timesteps: {len(timesteps)}")
        except Exception as e:
            print(f"  ⚠ Could not get timesteps: {str(e)}")
        
        try:
            field = db.getField()
            print(f"  Field: {field.name if field else 'N/A'}")
        except Exception as e:
            print(f"  ⚠ Could not get field: {str(e)}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        return False
    except SystemExit:
        print("\n✗ System exit during loading")
        return False
    except Exception as e:
        print(f"\n✗ Error loading dataset: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Dataset Loading Diagnostic Tool")
    print("=" * 60)
    print()
    
    # 测试1: 导入
    if not test_openvisus_import():
        print("\n" + "=" * 60)
        print("Diagnosis: OpenVisus is not properly installed")
        print("Solution: pip install OpenVisus")
        print("=" * 60)
        sys.exit(1)
    
    # 测试2: 加载数据集
    if not test_dataset_load():
        print("\n" + "=" * 60)
        print("Diagnosis: Dataset loading failed")
        print("Possible causes:")
        print("  1. Network connectivity issue")
        print("  2. Dataset server unavailable")
        print("  3. OpenVisus configuration problem")
        print("  4. Memory/resource limitations")
        print("=" * 60)
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

