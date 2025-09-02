"""
日志工具模块
"""
import logging
import os
from datetime import datetime
from typing import Optional

def setup_logging(save_dir: str, experiment_name: str) -> None:
    """
    设置日志配置
    
    Args:
        save_dir: 保存目录
        experiment_name: 实验名称
    """
    # 创建日志目录
    log_dir = os.path.join(save_dir, experiment_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 生成日志文件名（包含实验名称和时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    
    # 清除现有的处理器（避免重复日志）
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建文件处理器，添加错误处理
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 配置日志系统
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                file_handler,
                logging.StreamHandler()
            ]
        )
        
        # 记录日志配置信息
        logger = logging.getLogger(__name__)
        logger.info(f"日志系统已配置完成")
        logger.info(f"实验名称: {experiment_name}")
        logger.info(f"日志文件: {log_file}")
        
        # 在模块级别保存文件处理器引用，以便后续清理
        global _file_handler
        _file_handler = file_handler
        
    except Exception as e:
        # 如果文件日志失败，只使用控制台日志
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
        
        logger = logging.getLogger(__name__)
        logger.warning(f"无法创建日志文件 {log_file}: {e}")
        logger.info(f"日志系统已配置完成（仅控制台输出）")
        logger.info(f"实验名称: {experiment_name}")

def cleanup_logging():
    """清理日志处理器"""
    global _file_handler
    if '_file_handler' in globals() and _file_handler:
        try:
            _file_handler.close()
            logging.getLogger().removeHandler(_file_handler)
        except:
            pass

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称，默认为调用模块名
        
    Returns:
        logger: 配置好的日志记录器
    """
    return logging.getLogger(name)

def set_log_level(level: str) -> None:
    """
    设置日志级别
    
    Args:
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.getLogger().setLevel(numeric_level)
    
def log_model_info(model, logger: Optional[logging.Logger] = None):
    """
    记录模型信息
    
    Args:
        model: PyTorch模型
        logger: 日志记录器，如果为None则使用默认记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型信息:")
    logger.info(f"  总参数数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    logger.info(f"  模型类型: {type(model).__name__}")

def log_config_info(config, logger: Optional[logging.Logger] = None):
    """
    记录配置信息
    
    Args:
        config: 配置对象
        logger: 日志记录器，如果为None则使用默认记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("配置信息:")
    for key, value in vars(config).items():
        if not key.startswith('_'):
            logger.info(f"  {key}: {value}")

def log_training_progress(epoch: int, total_epochs: int, 
                         train_loss: float, val_loss: float,
                         metrics: dict, logger: Optional[logging.Logger] = None):
    """
    记录训练进度
    
    Args:
        epoch: 当前轮次
        total_epochs: 总轮次
        train_loss: 训练损失
        val_loss: 验证损失
        metrics: 评估指标字典
        logger: 日志记录器，如果为None则使用默认记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    progress = (epoch + 1) / total_epochs * 100
    
    logger.info(f"Epoch [{epoch+1}/{total_epochs}] ({progress:.1f}%)")
    logger.info(f"  训练损失: {train_loss:.4f}")
    logger.info(f"  验证损失: {val_loss:.4f}")
    
    if metrics:
        logger.info("  评估指标:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"    {metric_name}: {metric_value:.4f}") 