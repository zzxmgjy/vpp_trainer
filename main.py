from fastapi import FastAPI
from router.router import router
from router.lightgbm_router import lightgbm_router
from router.lstm_router import lstm_router
from router.prophet_router import prophet_router
from util.logger import logger
import uvicorn
from util.ftp import downloadFromFtp
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from util.scheduler_helper import SchedulerHelper
from service.lightgbm_service import LightgbmService
from service.prophet_service import ProphetService
from service.lstm_service import LstmService
from datetime import datetime
from config.app_config import config


# 加载配置
app = FastAPI()
app.include_router(router, prefix="/api/v1")
app.include_router(lightgbm_router, prefix="/api/v1")
app.include_router(lstm_router, prefix="/api/v1/lstm")
app.include_router(prophet_router, prefix="/api/v1/prophet")

def dowload_data():
    downloadFromFtp(config.getFtpDownloadDataDir(), config.get_data_dir())

def train_model():
    today = datetime.now().strftime("%Y-%m-%d")
    """lightgbm"""
    logger.info(f"train by lightgbm for {today}")
    LightgbmService.train_all_lightgbm()
    logger.info(f"train by lightgbm for {today} complete")
    
    """prophet"""
    logger.info(f"train by prophet for {today}")
    ProphetService.train_all_prophet()
    logger.info(f"train by prophet for {today} complete")

def train_lstm_model():
    today = datetime.now().strftime("%Y-%m-%d")
    """lstm"""
    logger.info(f"train by lstm for {today}")
    LstmService.train_all_lstm()
    logger.info(f"train by lstm for {today} complete")

def forecast_model():
    today = datetime.now().strftime("%Y-%m-%d")
    """lstm"""
    logger.info(f"forecast_model by lstm for {today}")
    LstmService.predict_all_lstm(today)
    logger.info(f"forecast_model by lstm for {today} complete")

# 配置调度任务
scheduler = BackgroundScheduler()
scheduler.add_job(
    dowload_data,
    trigger=CronTrigger(minute=config.cron["data"].minute),
    name="download_data"
)
scheduler.add_job(
    train_model,
    trigger=CronTrigger(hour=config.cron["train"].hour, minute=config.cron["train"].minute),
    name="daily_train"
)
# 使用调度器辅助类创建灵活的调度触发器
lstm_trigger = SchedulerHelper.create_trigger(config.cron["trainLstm"])
lstm_description = SchedulerHelper.get_schedule_description(config.cron["trainLstm"])
logger.info(f"LSTM 模型训练调度: {lstm_description}")

scheduler.add_job(
    train_lstm_model,
    trigger=lstm_trigger,
    name="lstm_train"
)
scheduler.add_job(
    forecast_model,
    trigger=CronTrigger(minute=config.cron["forecast"].minute),
    name="hourly_forecast"
)
scheduler.start()

logger.info("Application started")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host=config.server.host, port=config.server.port)
    except KeyboardInterrupt:
        scheduler.shutdown()
    except Exception as e:
        scheduler.shutdown()
        raise e