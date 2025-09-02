"""
错误处理装饰器模块
提供通用的错误处理、性能监控和类型检查功能
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union
import torch
import psutil
import gc

logger = logging.getLogger(__name__)


def error_handler(
    default_return: Any = None,
    raise_on_error: bool = False,
    log_traceback: bool = True
) -> Callable:
    """
    错误处理装饰器
    
    Args:
        default_return: 发生错误时的默认返回值
        raise_on_error: 是否重新抛出异常
        log_traceback: 是否记录完整的错误堆栈
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                
                if log_traceback:
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                else:
                    logger.error(error_msg)
                
                if raise_on_error:
                    raise
                
                return default_return
        return wrapper
    return decorator


def performance_monitor(
    log_memory: bool = True,
    log_time: bool = True,
    memory_threshold_mb: float = 100.0
) -> Callable:
    """
    性能监控装饰器
    
    Args:
        log_memory: 是否监控内存使用
        log_time: 是否监控执行时间
        memory_threshold_mb: 内存使用警告阈值(MB)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # 记录初始内存使用
            if log_memory:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                if torch.cuda.is_available():
                    initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                else:
                    initial_gpu_memory = 0
            
            try:
                result = func(*args, **kwargs)
                
                # 记录性能指标
                if log_time:
                    execution_time = time.time() - start_time
                    logger.info(f"{func.__name__} executed in {execution_time:.4f}s")
                
                if log_memory:
                    final_memory = process.memory_info().rss / 1024 / 1024
                    memory_diff = final_memory - initial_memory
                    
                    if torch.cuda.is_available():
                        final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        gpu_memory_diff = final_gpu_memory - initial_gpu_memory
                        
                        logger.info(
                            f"{func.__name__} memory usage - "
                            f"CPU: {memory_diff:+.2f}MB, GPU: {gpu_memory_diff:+.2f}MB"
                        )
                    else:
                        logger.info(f"{func.__name__} memory usage - CPU: {memory_diff:+.2f}MB")
                    
                    # 内存使用警告
                    if abs(memory_diff) > memory_threshold_mb:
                        logger.warning(
                            f"{func.__name__} used significant memory: {memory_diff:+.2f}MB"
                        )
                
                return result
                
            except Exception as e:
                if log_time:
                    execution_time = time.time() - start_time
                    logger.error(f"{func.__name__} failed after {execution_time:.4f}s")
                raise
                
        return wrapper
    return decorator


def type_check(strict: bool = True) -> Callable:
    """
    类型检查装饰器
    
    Args:
        strict: 是否严格检查类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            import inspect
            sig = inspect.signature(func)
            
            # 检查参数类型
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, param_value in bound_args.arguments.items():
                param = sig.parameters[param_name]
                
                if param.annotation != inspect.Parameter.empty:
                    expected_type = param.annotation
                    
                    # 处理Union类型和Optional类型
                    if hasattr(expected_type, '__origin__'):
                        if expected_type.__origin__ is Union:
                            # Union类型检查
                            if not any(isinstance(param_value, t) for t in expected_type.__args__):
                                error_msg = (
                                    f"Parameter '{param_name}' in {func.__name__} "
                                    f"expected {expected_type}, got {type(param_value)}"
                                )
                                if strict:
                                    raise TypeError(error_msg)
                                else:
                                    logger.warning(error_msg)
                        continue
                    
                    # 基本类型检查
                    if not isinstance(param_value, expected_type):
                        error_msg = (
                            f"Parameter '{param_name}' in {func.__name__} "
                            f"expected {expected_type}, got {type(param_value)}"
                        )
                        if strict:
                            raise TypeError(error_msg)
                        else:
                            logger.warning(error_msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def memory_cleanup(force_gc: bool = True, clear_cuda_cache: bool = True) -> Callable:
    """
    内存清理装饰器
    
    Args:
        force_gc: 是否强制垃圾回收
        clear_cuda_cache: 是否清理CUDA缓存
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if force_gc:
                    gc.collect()
                
                if clear_cuda_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间(秒)
        backoff_factor: 延迟递增因子
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    
        return wrapper
    return decorator


def deprecated(reason: str = "", version: str = "") -> Callable:
    """
    弃用警告装饰器
    
    Args:
        reason: 弃用原因
        version: 弃用版本
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"{func.__name__} is deprecated"
            if version:
                warning_msg += f" since version {version}"
            if reason:
                warning_msg += f": {reason}"
            
            logger.warning(warning_msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 组合装饰器
def robust_method(
    log_errors: bool = True,
    monitor_performance: bool = True,
    cleanup_memory: bool = True,
    max_retries: int = 1
) -> Callable:
    """
    鲁棒性方法装饰器，组合多个功能
    
    Args:
        log_errors: 是否记录错误
        monitor_performance: 是否监控性能
        cleanup_memory: 是否清理内存
        max_retries: 最大重试次数
    """
    def decorator(func: Callable) -> Callable:
        # 应用装饰器链
        decorated_func = func
        
        if cleanup_memory:
            decorated_func = memory_cleanup()(decorated_func)
        
        if monitor_performance:
            decorated_func = performance_monitor()(decorated_func)
        
        if log_errors:
            decorated_func = error_handler(log_traceback=True)(decorated_func)
        
        if max_retries > 1:
            decorated_func = retry(max_attempts=max_retries)(decorated_func)
        
        return decorated_func
    return decorator 