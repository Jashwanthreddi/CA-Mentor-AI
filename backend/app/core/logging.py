from pathlib import Path
import sys

from loguru import logger 
# creat logs directory

LOG_DIR=Path("logs")
LOG_DIR.mkdir(parents=True,exist_ok=True)
LOG_FILE=LOG_DIR/ "ca_mentor_ai.log"

# Remove default logger

logger.remove()

#console logger
logger.add(
    sys.stdout,
    level='INFO',
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> |"
        "<level>{level:<8}</level> |"
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> |"
        "<level>{message}</level>"
    ),

)

#file logger

logger.add(
    LOG_FILE,
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    enqueue=True,
    backtrack=True,
    diagnose=True,
    encoding="ütf-8",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} |"
        "{level:<8} |"
        "{name}:{function}:{line} |"
        "{message}"
    ),

)