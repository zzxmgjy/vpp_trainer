from util.logger import logger
from util.ftp import uploadToFtp
from config.app_config import config

class Service:
    @staticmethod
    def training():
        model_dir = config.get_model_dir()
        files = {}

        data_dir = config.get_data_dir()
        logger.info(f"data dir is {data_dir}")    

        # TODO: train here
        # TODO: generate model files
        # TODO: upload model to ftp

        # local_path = f"{model_dir}/your_model_file"
        # files[local_path] = f"{model_dir}/{company.customer_number}/xgboost/{local_path.name}"
                
        uploadToFtp(files)
        return {"status": "completed"}