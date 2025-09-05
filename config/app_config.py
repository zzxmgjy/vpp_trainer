import yaml
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class ServerConfig:
    host: str
    port: int

@dataclass
class CronConfig:
    hour: int = 0
    minute: int = 0
    day: Optional[int] = None #添加可选的 day 字段
@dataclass
class LoggingConfig:
    dir: str
    file: str
    max_size: int
    backup_count: int
    level: str

@dataclass
class DirConfig:
    data: str
    model: str

@dataclass
class SftpConfig:
    host: str
    port: int
    username: str
    password: str
    upload_model_dir: str = ""
    upload_data_dir: str  = ""
    download_model_dir: str = ""
    download_data_dir: str = ""

class AppConfig:
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 AppConfig 类，加载并解析 YAML 文件。
        :param config_path: YAML 配置文件路径，默认为 None（使用默认路径）。
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        
        # 映射配置项到对应的类
        self.server = ServerConfig(**config_data.get("server", {}))
        self.cron = {
            "data": CronConfig(**config_data.get("cron", {}).get("data", {})),
            "train": CronConfig(**config_data.get("cron", {}).get("train", {})),
            "trainLstm": CronConfig(**config_data.get("cron", {}).get("trainLstm", {})),
            "forecast": CronConfig(**config_data.get("cron", {}).get("forecast", {}))
        }
        self.logging = LoggingConfig(**config_data.get("logging", {}))
        self.dir = DirConfig(**config_data.get("dir", {}))
        self.sftp = SftpConfig(**config_data.get("sftp", {}))

    def __repr__(self):
        return f"<AppConfig server={self.server} logging={self.logging}>"
    
    def get_data_dir(self):
        data_dir = self.dir.data
        p = Path(data_dir)
        if p.is_absolute():
            return data_dir
        else:
            return os.path.join(os.getcwd(), data_dir)
    
    def get_model_dir(self):
        model_dir = self.dir.model
        p = Path(model_dir)
        if p.is_absolute():
            return model_dir
        else:
            return os.path.join(os.getcwd(), model_dir)

    def getFtpUploadModelDir(self):
        sftp = self.sftp
        if sftp and sftp.host and sftp.upload_model_dir:
            return sftp.upload_model_dir
        else:
            return None
            
    def getFtpUploadDataDir(self):
        sftp = self.sftp
        if sftp and sftp.host and sftp.upload_data_dir:
            return sftp.upload_data_dir
        else:
            return None
        

    def getFtpDownloadModelDir(self):
        sftp = self.sftp
        if sftp and sftp.host and sftp.download_model_dir:
            return sftp.download_model_dir
        else:
            return None

    def getFtpDownloadDataDir(self):
        sftp = self.sftp
        if sftp and sftp.host and sftp.download_data_dir:
            return sftp.download_data_dir
        else:
            return None

# 全局配置对象
config = AppConfig()

