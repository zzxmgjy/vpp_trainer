from fastapi import APIRouter
from service.service import Service
from util.ftp import downloadFromFtp
from config.app_config import config
router = APIRouter()

@router.get("/training")
def training():
    result = Service.training()
    return {"status": 200, "data": result}

@router.get("/download_data")
def download_data():
    downloadFromFtp(config.getFtpDownloadDataDir(), config.get_data_dir())
    return {"status": 200, "data": "download data complete"}