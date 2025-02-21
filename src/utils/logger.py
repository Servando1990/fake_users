from typing import Dict, Any
from pathlib import Path
from loguru import logger
import sys

def setup_logging(log_path: str = "logs") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_path (str): Directory where log files will be stored
    """
    # Create logs directory if it doesn't exist
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler for all logs
    logger.add(
        f"{log_path}/all.log",
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    # Add file handler for errors only
    logger.add(
        f"{log_path}/error.log",
        rotation="100 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR"
    ) 