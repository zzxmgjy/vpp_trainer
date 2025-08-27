from util.logger import logger
from util.ftp import uploadToFtp
from alg.lstm.forcast import predict_power
from alg.lstm.dataUtil import merge_station_data
from alg.lstm.train import main
from config.app_config import config
import os
import pandas as pd

class LstmService:
    @staticmethod
    def train_all_lstm():
        #获取到模型文件目录
        model_dir = config.get_model_dir()
        #获取到数据文件目录
        data_dir = config.get_data_dir()
        #获取所有场站id
        station_ids = []
        for item in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, item)) and not item.startswith('.'):
                station_ids.append(item)

        if not station_ids:
            logger.error("未找到任何场站ID")
            return {"status": "未找到任何场站ID"}
        #遍历所有场站id
        for station_id in station_ids:
            # 先合并数据文件
            merge_success = merge_station_data(station_id, data_dir)
            if not merge_success:
                logger.warning(f"场站 {station_id} 数据合并失败或无数据可合并，将尝试继续训练")
                continue

            # 构建训练数据文件路径
            data_file = f"{data_dir}/{station_id}/data/data-{station_id}-all.csv"
            
            logger.info(f"开始训练场站 {station_id}，数据文件: {data_file}")
            
            try:
                # 直接调用train.py中的main方法，传递路径参数
                main(station_id=station_id, data_file=data_file, enable_hyperopt=True,
                     output_path=data_dir, model_base_path=model_dir)
                logger.info(f"场站 {station_id} 训练任务完成")
            except Exception as e:
                logger.error(f"场站 {station_id} 训练任务失败: {e}")
                continue
        return {"status": "completed"}

    @staticmethod
    def predict_by_lstm(customer_number: str = None,date: str = None):
        #获取到模型文件目录
        model_dir = config.get_model_dir()
        #获取到数据文件目录
        data_dir = config.get_data_dir()
        logger.info(f"model_dir dir is {model_dir}")
        
        try:
            # 构建预测开始时间
            start_datetime = f"{date} 00:00:00"
            
            # 调用forcast.py中的predict_power方法进行预测
            # customer_number对应forcast.py中的station_id
            predictions = predict_power(customer_number, start_datetime, model_base_path=model_dir, data_path=data_dir)
            
            if not predictions:
                logger.error(f"[predict_by_lstm][{customer_number}][{date}] 预测失败，未返回结果")
                return

            #预测结果保存到相应的csv表中,{model_dir}/{station_id}/lstm/forcast/{date}.csv
            #csv中字段为time,load,meter
            try:
                # 构建保存目录
                forecast_dir = os.path.join(model_dir, customer_number, "lstm", "forcast")
                os.makedirs(forecast_dir, exist_ok=True)
                
                # 构建CSV文件路径
                csv_path = os.path.join(forecast_dir, f"{date}.csv")
                
                # 将预测结果转换为DataFrame并保存为CSV
                df = pd.DataFrame(predictions)
                df.to_csv(csv_path, index=False)
                
                logger.info(f" {customer_number}预测结果已保存到: {csv_path}")
                #上传到ftp
                uploadToFtp(csv_path)
                logger.info(f" {customer_number}预测结果已上传到ftp")
                return csv_path
                
            except Exception as e:
                logger.error(f"{customer_number}保存预测结果失败: {e}")
                return
        except Exception as e:
            logger.error(f"[predict_by_lstm][{customer_number}][{date}] {e}")
            return

    @staticmethod
    def predict_all_lstm(date: str = None):
        try:
            #获取到模型文件目录
            model_dir = config.get_model_dir()
            #获取到数据文件目录
            data_dir = config.get_data_dir()
            # 构建预测开始时间
            start_datetime = f"{date} 00:00:00"
            # 获取数据目录中所有场站ID
            station_ids = []
            for item in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, item)) and not item.startswith('.'):
                    station_ids.append(item)

            if not station_ids:
                logger.error("未找到任何场站ID")
                return {"status": "未找到任何场站ID"}
            # 预测每个场站
            for station_id in station_ids:
                logger.info(f"正在预测场站 {station_id} 的功率...")
                station_result = predict_power(station_id, start_datetime, model_base_path=model_dir, data_path=data_dir)
                #预测结果保存到相应的csv表中,{model_dir}/{station_id}/lstm/forcast/{date}.csv
                #csv中字段为time,load,meter
                try:
                    # 构建保存目录
                    forecast_dir = os.path.join(model_dir, station_id, "lstm", "forcast")
                    os.makedirs(forecast_dir, exist_ok=True)
                    
                    # 构建CSV文件路径
                    csv_path = os.path.join(forecast_dir, f"{date}.csv")
                    
                    # 将预测结果转换为DataFrame并保存为CSV
                    if station_result:
                        df = pd.DataFrame(station_result)
                        df.to_csv(csv_path, index=False)
                        logger.info(f"场站 {station_id} 的预测结果已保存到: {csv_path}")
                        #上传到ftp
                        files_to_upload = {
                            csv_path: csv_path
                        }
                        # 3. 传递字典给 uploadToFtp 函数
                        uploadToFtp(files_to_upload)
                        logger.info(f"场站 {station_id} 的预测结果已上传到ftp")
                    else:
                        logger.warning(f"场站 {station_id} 没有预测结果，跳过保存")
                        
                except Exception as e:
                    logger.error(f"保存场站 {station_id} 的预测结果失败: {e}")
                    continue

            return {"status": "completed"}

        except Exception as e:
            logger.error(f"[predict_by_lstm][{date}] {e}")
            return {"status": "failed"}
