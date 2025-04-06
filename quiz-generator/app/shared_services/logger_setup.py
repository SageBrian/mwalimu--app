import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone


def setup_logger(log_level=logging.INFO):
    main_project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    log_folder = os.path.join(main_project_directory, "logs")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file_path = os.path.join(log_folder, f"query_state_log_{datetime.now().strftime('%Y-%m-%d')}.log")
    logger = logging.getLogger("QueryStateLogger")
    if not logger.handlers:
        logger.setLevel(log_level)
        file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=30)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger