"""
Centralized logging configuration.
"""

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
    level=logging.INFO,
)


#Also Logging into Console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s")
)
logging.getLogger().addHandler(console_handler)

