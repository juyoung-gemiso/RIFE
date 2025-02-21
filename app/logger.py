import os
import logging
from pathlib import Path
from loguru import logger
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler


def setup_logger(config):
    # 기본 로그 디렉토리 설정
    log_dir = config.rife_config.log_save_path
    config = config.logging_config
    info_dir = os.path.join(log_dir, "info")
    error_dir = os.path.join(log_dir, "error")

    # 디렉토리 생성
    Path(info_dir).mkdir(parents=True, exist_ok=True)
    Path(error_dir).mkdir(parents=True, exist_ok=True)

    error_format = "{time:YYYY-MM-DD HH:mm:ss}[{level}|{file},{line}] {message}\n"
    info_format = "{time:YYYY-MM-DD HH:mm:ss}[{level}|{file},{line}] {message}"

    def get_size_handler(level):
        return RotatingFileHandler(
            os.path.join(f"{info_dir if level == logging.INFO else error_dir}", f'logger_{logging._levelToName[level]}_size.log'),
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )

    def get_time_handler(level):
        return TimedRotatingFileHandler(
            os.path.join(f"{info_dir if level == logging.INFO else error_dir}", f'logger_{logging._levelToName[level]}_time.log'),
            when='midnight',
            interval=1,
            backupCount=config.backup_count,
            encoding='utf-8'
        )

    # 로거 설정 추가 (크기 기반)
    logger.add(get_size_handler(logging.INFO), 
              format=info_format, 
              backtrace=False, 
              level="INFO")
    
    logger.add(get_size_handler(logging.ERROR), 
              format=error_format, 
              backtrace=False, 
              level="ERROR")

    # 로거 설정 추가 (시간 기반)
    logger.add(get_time_handler(logging.INFO), 
              format=info_format, 
              backtrace=False, 
              level="INFO")
    
    logger.add(get_time_handler(logging.ERROR), 
              format=error_format, 
              backtrace=False, 
              level="ERROR")

    return logger
