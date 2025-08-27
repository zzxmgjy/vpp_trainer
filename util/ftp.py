from util.logger import logger
import os
import paramiko
import stat
from config.app_config import config

def downloadFromFtp(remote_dir: str, local_dir: str, max_depth: int = 10):
    if not remote_dir or not local_dir:
        logger.error(f"Invalid remote or local directory: {remote_dir}, {local_dir}")
        return
    
    """
    下载远程FTP目录内容（非递归实现）
    :param remote_dir: 远程目录路径
    :param local_dir: 本地目录路径
    :param max_depth: 最大目录深度
    """
    sftp_config = config.sftp
    
    local_dir = os.path.abspath(local_dir)
    logger.info(f"Starting download from {remote_dir} to {local_dir}")

    try:
        with paramiko.Transport((sftp_config.host, sftp_config.port)) as transport:
            transport.connect(username=sftp_config.username, password=sftp_config.password)
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                # 使用队列处理目录
                from collections import deque
                queue = deque([(remote_dir, local_dir, 0)])

                while queue:
                    current_remote, current_local, depth = queue.popleft()
                    os.makedirs(current_local, exist_ok=True)

                    if depth >= max_depth:
                        logger.warning(f"Reached max depth {max_depth} at {current_remote}")
                        continue

                    for item in sftp.listdir(current_remote):
                        remote_path = os.path.join(current_remote, item)
                        local_path = os.path.join(current_local, item)
                        
                        if _process_remote_item(sftp, remote_path, local_path):
                            # 如果是目录则加入队列
                            queue.append((remote_path, local_path, depth + 1))
    except Exception as e:
        logger.error(f"FTP download failed: {e}")
        raise
    finally:
        if 'transport' in locals():
            transport.close()
            logger.info("FTP connection closed")

def _process_remote_item(sftp, remote_path: str, local_path: str):
    """处理远程目录项"""
    try:
        file_stat = sftp.stat(remote_path)
        if stat.S_ISDIR(file_stat.st_mode):
            logger.debug(f"Found directory: {remote_path}")
            return True
        else:
            _download_file(sftp, remote_path, local_path)
            return False
    except Exception as e:
        logger.error(f"Failed to process {remote_path}: {e}")
        return False
    
def _download_file(sftp, remote_path: str, local_path: str):
        """下载单个文件"""
        sftp.get(remote_path, local_path)
        logger.info(f"Downloaded: {remote_path} -> {local_path}")

def uploadToFtp(files : dict):
    if len(files) == 0:
        logger.info("No files to upload")
        return
    
    # Upload to SFTP server
    sftp_config = config.sftp
    try:
        transport = paramiko.Transport(sftp_config.host, sftp_config.port)
        transport.connect(username=sftp_config.username, password=sftp_config.password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        for local_path, remote_path in files.items():
            # Check if remote directory exists, create if not
            remote_dir = os.path.dirname(remote_path)
            try:
                sftp.listdir(remote_dir)
            except IOError:
                # Create directory recursively
                parts = remote_dir.split('/')
                current_path = ''
                for part in parts:
                    if not part:
                        continue
                    current_path += f'{part}/'
                    try:
                        sftp.listdir(current_path)
                    except IOError:
                        sftp.mkdir(current_path)
            # Upload file
            sftp.put(local_path, remote_path)
            logger.info(f"[SFTP] File {local_path} uploaded to {remote_path}")
        sftp.close()
        transport.close()
    except Exception as e:
        logger.error(f"[SFTP] Failed to upload file: {str(e)}")
