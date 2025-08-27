from fastapi import APIRouter,Body
from service.lstm_service import LstmService
from typing import Optional
lstm_router = APIRouter()


@lstm_router.post("/predict_all_lstm")
def predict_all_lstm(date: str = Body(..., embed=True)):
    result = LstmService.predict_all_lstm(date)
    return {"status": 200, "data": result}

@lstm_router.post("/train_all_lstm")
def train_all_lstm():
    result = LstmService.train_all_lstm()
    return {"status": 200, "data": result}