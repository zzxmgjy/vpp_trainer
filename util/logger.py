import logging
from logging.handlers import RotatingFileHandler
import os
from config.app_config import config

# 读取配置文件
logging_config = config.logging

if not os.path.exists(logging_config.dir):
    os.makedirs(logging_config.dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging_config.level)

handler = RotatingFileHandler(
    os.path.join(logging_config.dir, logging_config.file),
    maxBytes=logging_config.max_size,
    backupCount=logging_config.backup_count
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)