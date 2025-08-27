from fastapi import APIRouter,Body
from service.lstm_service import LstmService
from typing import Optional
import threading

lstm_router = APIRouter()


def _run_predict_async(date: str):
    """异步执行预测任务"""
    try:
        LstmService.predict_all_lstm(date)
    except Exception as e:
        # 这里可以添加日志记录
        print(f"异步预测任务执行失败: {e}")


def _run_train_async():
    """异步执行训练任务"""
    try:
        LstmService.train_all_lstm()
    except Exception as e:
        # 这里可以添加日志记录
        print(f"异步训练任务执行失败: {e}")


@lstm_router.post("/predict_all_lstm")
def predict_all_lstm(date: str = Body(..., embed=True)):
    """启动异步预测任务，不等待结果"""
    thread = threading.Thread(target=_run_predict_async, args=(date,))
    thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
    thread.start()
    return {"status": 200, "message": "预测任务已启动"}

@lstm_router.post("/train_all_lstm")
def train_all_lstm():
    """启动异步训练任务，不等待结果"""
    thread = threading.Thread(target=_run_train_async)
    thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
    thread.start()
    return {"status": 200, "message": "训练任务已启动"}