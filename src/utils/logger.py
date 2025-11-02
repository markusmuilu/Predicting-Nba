import logging
import os
from datetime import datetime

LOG_DIR=os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH=os.path.join(LOG_DIR,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger= logging.getLogger(__name__)

if __name__=="__main__":
    logging.info("Logging has started")