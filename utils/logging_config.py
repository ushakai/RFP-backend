"""
Centralized logging configuration for RFP Backend
"""
import logging
import sys
from datetime import datetime
import os

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Define log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create formatters
formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler - Application logs
app_log_file = os.path.join(LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(app_log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# File handler - Error logs
error_log_file = os.path.join(LOGS_DIR, f"error_{datetime.now().strftime('%Y%m%d')}.log")
error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# File handler - Worker logs
worker_log_file = os.path.join(LOGS_DIR, f"worker_{datetime.now().strftime('%Y%m%d')}.log")
worker_handler = logging.FileHandler(worker_log_file, encoding='utf-8')
worker_handler.setLevel(logging.DEBUG)
worker_handler.setFormatter(formatter)

# File handler - Tender ingestion logs
tender_log_file = os.path.join(LOGS_DIR, f"tender_{datetime.now().strftime('%Y%m%d')}.log")
tender_handler = logging.FileHandler(tender_log_file, encoding='utf-8')
tender_handler.setLevel(logging.DEBUG)
tender_handler.setFormatter(formatter)


def setup_logger(name: str, log_type: str = "app") -> logging.Logger:
    """
    Setup and return a logger with appropriate handlers
    
    Args:
        name: Logger name (usually __name__ from calling module)
        log_type: Type of log - "app", "worker", "tender"
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if not already added (prevents duplicate logs)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # All loggers get console and error handlers
        logger.addHandler(console_handler)
        logger.addHandler(error_handler)
        
        # Add specific file handler based on type
        if log_type == "worker":
            logger.addHandler(worker_handler)
        elif log_type == "tender":
            logger.addHandler(tender_handler)
        else:
            logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, log_type: str = "app") -> logging.Logger:
    """Get or create a logger - convenience wrapper"""
    return setup_logger(name, log_type)


# Default app logger
app_logger = setup_logger("rfp_backend", "app")

