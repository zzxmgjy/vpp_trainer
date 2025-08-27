from util.logger import logger
from util.date_utils import iter_months
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os
from alg.lightgbm import train_meter,train_load,save_model
from util.ftp import uploadToFtp
from config.app_config import config


class LightgbmService:

    @staticmethod
    def train_by_lightgbm(customer_number: str = None,start_date: str = None, end_date: str = None, update_model: bool = None):
        model_dir = config.get_model_dir()
        data_dir = config.get_data_dir()
        logger.info(f"data_dir dir is {data_dir}")
        csv_dir = Path(f"{data_dir}/{customer_number}/data")
        meter_model_path = Path(f"{model_dir}/{customer_number}/lightgbm/meter.pkl")
        load_model_path = Path(f"{model_dir}/{customer_number}/lightgbm/load.pkl")
        time_str = datetime.now().strftime("%Y-%m-%d-%H%M")
        meter_model_his_path = Path(f"{model_dir}/{customer_number}/lightgbm/his/meter-{time_str}.pkl")
        load_model_his_path = Path(f"{model_dir}/{customer_number}/lightgbm/his/load-{time_str}.pkl")
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        actual_months = list(iter_months(start_datetime, end_datetime))
        train_data = pd.DataFrame()
        for month_datetime in actual_months:
            month_data_path = csv_dir / f"data-{customer_number}-{month_datetime.strftime('%Y-%m')}.csv"
            if not month_data_path.exists():
                logger.error(f"[train_by_lightgbm][{customer_number}] CSV file {month_data_path} not exists")
                continue
            month_data = pd.read_csv(month_data_path)
            train_data = pd.concat([
                train_data,
                month_data
            ])
        train_data = train_data[(train_data['time'] <= end_date) & (train_data['time'] >= start_date)]
        meter_model = train_meter(train_data)
        load_model = train_load(train_data)
        logger.info(f"[train_by_lightgbm][{customer_number}] train success")
        if update_model:
            save_model(meter_model_path, meter_model)
            save_model(meter_model_his_path, meter_model)
            save_model(load_model_path, load_model)
            save_model(load_model_his_path, meter_model)
            files = {}
            upload_dir = config.getFtpUploadModelDir()
            files[f"{meter_model_path}"] = f"{upload_dir}/{customer_number}/lightgbm/meter.pkl"
            files[f"{load_model_path}"] = f"{upload_dir}/{customer_number}/lightgbm/load.pkl"
            uploadToFtp(files)
            logger.info(f"[train_by_lightgbm][{customer_number}] uploadToFtp success")
        return {"status": "completed"}

    @staticmethod
    def train_all_lightgbm():
        try:
            model_dir = config.get_model_dir()
            data_dir = config.get_data_dir()
            upload_dir = config.getFtpUploadModelDir()
            logger.info(f"data_dir dir is {data_dir}")
            path = Path(data_dir)
            files = {}
            for item in path.iterdir():
                customer_number = item.name
                if not item.is_dir():
                    continue
                csv_dir = Path(f"{data_dir}/{customer_number}/data")
                meter_model_path = Path(f"{model_dir}/{customer_number}/lightgbm/meter.pkl")
                load_model_path = Path(f"{model_dir}/{customer_number}/lightgbm/load.pkl")
                time_str = datetime.now().strftime("%Y-%m-%d-%H%M")
                meter_model_his_path = Path(f"{model_dir}/{customer_number}/lightgbm/his/meter-{time_str}.pkl")
                load_model_his_path = Path(f"{model_dir}/{customer_number}/lightgbm/his/load-{time_str}.pkl")
                train_data = pd.DataFrame()
                path = Path(csv_dir)
                for month_data_path in path.glob("*.csv"):
                    month_data = pd.read_csv(month_data_path)
                    train_data = pd.concat([
                        train_data,
                        month_data
                    ])
                meter_model = train_meter(train_data)
                save_model(meter_model_path, meter_model)
                save_model(meter_model_his_path, meter_model)
                load_model = train_load(train_data)
                save_model(load_model_path, load_model)
                save_model(load_model_his_path, meter_model)
                files[f"{meter_model_path}"] = f"{upload_dir}/{customer_number}/lightgbm/meter.pkl"
                files[f"{load_model_path}"] = f"{upload_dir}/{customer_number}/lightgbm/load.pkl"
                logger.info(f"[train_all_lightgbm][{customer_number}] train success")
            uploadToFtp(files)
            logger.info(f"[train_all_lightgbm] uploadToFtp success")
            return {"status": "completed"}
        except Exception as e:
            logger.error(f"[train_all_lightgbm] {e}")
            return