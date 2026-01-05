#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的数据加载工作进程
在单独的进程中加载OpenVisus数据，避免崩溃影响主服务器
"""
import sys
import pickle
import numpy as np

def load_dataset_worker(dataset_url, time_index, lat_start, lat_end, lon_start, lon_end, nz, data_quality):
    """在独立进程中加载数据集"""
    try:
        # 先测试网络连接
        print(f'[Worker] Testing network connection...', file=sys.stderr)
        try:
            import urllib.request
            import ssl
            # 创建不验证SSL的上下文（用于测试）
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            # 尝试连接服务器
            test_url = dataset_url.replace('/mit_output/', '/')
            req = urllib.request.Request(test_url)
            try:
                urllib.request.urlopen(req, timeout=10, context=ctx)
                print(f'[Worker] Network connection OK', file=sys.stderr)
            except:
                print(f'[Worker] Network test inconclusive, continuing...', file=sys.stderr)
        except Exception as e:
            print(f'[Worker] Network test failed: {str(e)}', file=sys.stderr)
        
        # 导入OpenVisus
        print(f'[Worker] Importing OpenVisus...', file=sys.stderr)
        import OpenVisus as ov
        print(f'[Worker] OpenVisus imported successfully', file=sys.stderr)
        
        # 加载数据集
        print(f'[Worker] Loading dataset from: {dataset_url}', file=sys.stderr)
        print(f'[Worker] This may cause the process to crash (Access Violation)...', file=sys.stderr)
        
        # 尝试加载，如果崩溃，进程会退出
        db = ov.LoadDataset(dataset_url)
        print(f'[Worker] Dataset loaded successfully!', file=sys.stderr)
        
        # 读取数据
        print(f'[Worker] Reading data for time_index={time_index}...', file=sys.stderr)
        data_full = db.read(time=time_index, quality=data_quality)
        lat_dim, lon_dim, depth_dim = data_full.shape
        print(f'[Worker] Full data shape: {lat_dim}x{lon_dim}x{depth_dim}', file=sys.stderr)
        
        # 提取区域
        lat_idx_start = int(lat_dim * lat_start / 90)
        lat_idx_end = int(lat_dim * lat_end / 90)
        lon_idx_start = int(lon_dim * lon_start / 360)
        lon_idx_end = int(lon_dim * lon_end / 360)
        
        data_local = data_full[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end, :nz]
        print(f'[Worker] Extracted region shape: {data_local.shape}', file=sys.stderr)
        
        # 转换为numpy数组（确保可以序列化）
        result = {
            'success': True,
            'data': np.array(data_local),
            'shape': data_local.shape
        }
        
        # 输出到stdout（pickle格式）
        pickle.dump(result, sys.stdout.buffer)
        sys.stdout.flush()
        return 0
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        try:
            import traceback
            error_result['traceback'] = traceback.format_exc()
        except:
            pass
        
        try:
            pickle.dump(error_result, sys.stdout.buffer)
            sys.stdout.flush()
        except:
            # 如果pickle也失败，至少输出错误信息
            print(f'ERROR: {str(e)}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    # 从命令行参数读取
    if len(sys.argv) < 9:
        print('Usage: load_data_worker.py <dataset_url> <time_index> <lat_start> <lat_end> <lon_start> <lon_end> <nz> <data_quality>', file=sys.stderr)
        sys.exit(1)
    
    dataset_url = sys.argv[1]
    time_index = int(sys.argv[2])
    lat_start = float(sys.argv[3])
    lat_end = float(sys.argv[4])
    lon_start = float(sys.argv[5])
    lon_end = float(sys.argv[6])
    nz = int(sys.argv[7])
    data_quality = int(sys.argv[8])
    
    exit_code = load_dataset_worker(dataset_url, time_index, lat_start, lat_end, lon_start, lon_end, nz, data_quality)
    sys.exit(exit_code)

