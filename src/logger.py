import logging
from datetime import datetime
import os

LOGS_DIR='logs'
os.makedirs(LOGS_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR,f'log_{datetime.now().strftime("%Y-%m-%d")}.log')

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s ',
    level=logging.INFO    
) 

def get_logger(name):
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
if __name__=="__main__":
    logger=get_logger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info("Logger is set up and running.")