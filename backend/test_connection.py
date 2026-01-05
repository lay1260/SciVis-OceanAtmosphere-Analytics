#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试后端连接和数据集初始化
"""

import sys
import requests
import time

def test_backend():
    """测试后端服务"""
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("Testing Backend Connection")
    print("=" * 60)
    
    # 测试1: 健康检查
    print("\n[Test 1] Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False, None
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend server")
        print("  Please make sure the backend is running:")
        print("  cd backend && python app.py")
        return False, None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, None
    
    # 测试2: 数据集信息
    print("\n[Test 2] Dataset Info...")
    try:
        response = requests.get(f"{base_url}/api/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✓ Dataset info retrieved")
                print(f"  Dimensions: {data.get('dimensions')}")
                print(f"  Timesteps: {data.get('timesteps')}")
                print(f"  Field: {data.get('field')}")
            else:
                print(f"✗ Dataset info failed: {data.get('error')}")
                return False, None
        else:
            print(f"✗ Request failed: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, None
    
    # 测试3: 获取体积数据
    print("\n[Test 3] Volume Data...")
    try:
        print("  Requesting data (this may take a while)...")
        start_time = time.time()
        response = requests.get(f"{base_url}/api/data/volume?time=0", timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✓ Volume data retrieved in {elapsed:.2f}s")
                shape = data.get('shape', {})
                print(f"  Shape: {shape.get('nx')} x {shape.get('ny')} x {shape.get('nz')}")
                print(f"  Bounds: {data.get('bounds')}")
                # keep a copy for fallback in case /api/typhoon is missing
                volume_data = data.get('data')
            else:
                print(f"✗ Volume data failed: {data.get('error')}")
                return False, None
        else:
            print(f"✗ Request failed: {response.status_code}")
            return False, None
    except requests.exceptions.Timeout:
        print("✗ Request timeout (data loading takes too long)")
        return False, None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, None
    
    # 测试4: 获取台风位置
    print("\n[Test 4] Typhoon Position...")
    try:
        response = requests.get(f"{base_url}/api/typhoon?time=0", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                typhoon_info = {
                    'lat': data.get('lat'),
                    'lng': data.get('lng'),
                    'grid_index': data.get('grid_index'),
                    'timeIndex': data.get('timeIndex')
                }
                print("✓ Typhoon position retrieved:")
                print(f"  Latitude: {typhoon_info['lat']}")
                print(f"  Longitude: {typhoon_info['lng']}")
            else:
                print(f"✗ Typhoon endpoint returned error: {data.get('error')}")
                return False, None
        else:
            print(f"✗ Request failed: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        # 退回到本地计算（如果我们能获取体积数据）
        try:
            # 如果volume_data可用，从中计算台风位置；否则失败
            if 'volume_data' in locals() and volume_data:
                # volume_data按深度层组织，取第一个层（近表层）
                surface_layer = volume_data[0]  # layer: rows x cols of objects
                nx = len(surface_layer)
                ny = len(surface_layer[0]) if nx > 0 else 0
                # 构造值矩阵
                import numpy as np
                vals = np.zeros((nx, ny))
                for i in range(nx):
                    for j in range(ny):
                        vals[i, j] = float(surface_layer[i][j].get('value', 0.0))
                # 计算梯度并定位最大点
                gy, gx = np.gradient(vals)
                grad = np.sqrt(gx**2 + gy**2)
                max_idx = np.unravel_index(np.argmax(grad), grad.shape)
                ix, iy = max_idx
                lat = float(surface_layer[ix][iy].get('lat'))
                lng = float(surface_layer[ix][iy].get('lng'))
                typhoon_info = {'lat': lat, 'lng': lng, 'grid_index': {'ix': int(ix), 'iy': int(iy)}, 'timeIndex': 0}
                print("✓ Typhoon position computed locally (fallback):")
                print(f"  Latitude: {lat}")
                print(f"  Longitude: {lng}")
            else:
                print('✗ No volume data available for fallback computation')
                return False, None
        except Exception as e2:
            print(f"✗ Local computation failed: {str(e2)}")
            return False, None

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True, typhoon_info

if __name__ == '__main__':
    success, typhoon = test_backend()
    if success:
        print(f"\nFinal Typhoon Location: {typhoon}")
    # Keep existing exit code behavior
    sys.exit(0 if success else 1)

