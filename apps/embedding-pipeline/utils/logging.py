import logging
from typing import Dict, Any


def setup_logger(name: str, config: Dict[str, Any]) -> logging.Logger:
    """
    Set up a logger with the given configuration.
    
    Args:
        name: Name of the logger
        config: Logger configuration
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    level_name = config.get('level', 'INFO')
    level = getattr(logging, level_name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger