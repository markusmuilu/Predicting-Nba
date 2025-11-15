import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create a new log file for each run
LOG_FILE = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Log message format
LOG_FORMAT = "[ %(asctime)s ] %(name)s:%(lineno)d - %(levelname)s - %(message)s"

# File handler with rotation (5 MB per file, keep 5 backups)
file_handler = RotatingFileHandler(
    LOG_FILE_PATH,
    maxBytes=5_000_000,
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Console handler (output also printed to terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Export shared logger instance for entire project
logger = logging.getLogger("nba_predictor_logger")
