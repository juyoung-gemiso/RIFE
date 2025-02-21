import os
import yaml
import os.path as osp
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    base_path: str
    max_bytes: int
    backup_count: int

@dataclass
class RIFEConfig:
    checkpoint: str
    temp_dir: str
    log_save_path: str

class AppConfig:
    def __init__(self, base_path="."):
        # 환경변수에서 config 파일 경로를 가져옴
        self.config_path = osp.join(base_path, os.getenv('CONFIG_PATH', 'config.yaml'))
        self.base_path = base_path
        
        # 경로가 존재하는지 확인
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        self.load_config()
    
    def _load_yml(self):
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error occurred while loading config file. \n error_message = {e}")
        
    def load_config(self):
        self._load_yml()
        self.logging_config = LoggingConfig(**self.config['logging'])
        self.rife_config = RIFEConfig(**self.config['RIFE'])
