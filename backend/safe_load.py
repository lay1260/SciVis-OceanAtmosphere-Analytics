"""
安全加载数据集的辅助模块
使用超时和异常处理来防止进程崩溃
"""
import signal
import threading
import time

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("Operation timed out")

def safe_load_dataset(dataset_url, timeout=120):
    """
    安全加载数据集，带超时保护
    
    Args:
        dataset_url: 数据集URL
        timeout: 超时时间（秒），默认120秒
    
    Returns:
        加载的数据集对象
    
    Raises:
        TimeoutError: 如果加载超时
        Exception: 其他加载错误
    """
    import OpenVisus as ov
    
    result = [None]  # 使用列表以便在内部函数中修改
    exception = [None]
    
    def load_in_thread():
        """在线程中加载数据集"""
        try:
            print(f'  [Thread] Starting dataset load...')
            result[0] = ov.LoadDataset(dataset_url)
            print(f'  [Thread] Dataset loaded successfully')
        except Exception as e:
            exception[0] = e
            print(f'  [Thread] Error loading dataset: {str(e)}')
    
    # 创建并启动线程
    thread = threading.Thread(target=load_in_thread)
    thread.daemon = True  # 设置为守护线程
    thread.start()
    
    # 等待线程完成或超时
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # 线程仍在运行，说明超时了
        print(f'  [Timeout] Dataset loading exceeded {timeout} seconds')
        raise TimeoutError(f"Dataset loading timed out after {timeout} seconds")
    
    if exception[0] is not None:
        # 线程中发生了异常
        raise exception[0]
    
    if result[0] is None:
        raise Exception("Dataset loading failed for unknown reason")
    
    return result[0]

