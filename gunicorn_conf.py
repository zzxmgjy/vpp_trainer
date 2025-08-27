import multiprocessing
import os
import logging

# 获取项目根目录
base_dir = "/opt/deploy/vpp-trainer"
log_dir = os.path.join(base_dir, "logs")

# 确保日志目录存在
try:
    os.makedirs(log_dir, exist_ok=True)
    # 设置目录权限（如果需要）
    os.chmod(log_dir, 0o755)
except Exception as e:
    # 如果目录创建失败，回退到临时目录
    print(f"Warning: Cannot create log directory {log_dir}: {e}")
    log_dir = "/tmp"

# 绑定地址和端口
bind = "127.0.0.1:8001"

# Worker 数量 (CPU核心数 * 2 + 1)
#workers = multiprocessing.cpu_count() * 2 + 1
workers = 1

# Worker 类型
worker_class = "uvicorn.workers.UvicornWorker"

# 每个worker处理的最大请求数后重启 (防止内存泄漏)
max_requests = 1000
max_requests_jitter = 100

# 超时设置
timeout = 600
graceful_timeout = 30
keepalive = 5

# 日志设置 - 输出到文件而不是标准输出
accesslog = os.path.join(log_dir, "gunicorn_access.log")
errorlog = os.path.join(log_dir, "gunicorn_error.log")
loglevel = "info"

# 进程名称
proc_name = "trainer"

# 禁用捕获输出，让应用程序自己处理日志
capture_output = False

# 在master进程启动时再次确保目录存在
def when_ready(server):
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory ensured: {log_dir}")
    except Exception as e:
        print(f"Error ensuring log directory: {e}")
