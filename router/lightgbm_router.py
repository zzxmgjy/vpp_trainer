from fastapi import APIRouter,Body
from service.lightgbm_service import LightgbmService
from typing import Optional
lightgbm_router = APIRouter()


@lightgbm_router.post("/train_by_lightgbm")
def train_by_lightgbm(
    customer_number: Optional[str] = Body(None, embed=True),
    start_date: Optional[str] = Body(None, embed=True),
    end_date: Optional[str] = Body(None, embed=True),
    update_model: Optional[bool] = Body(False, embed=True)
):
    result = LightgbmService.train_by_lightgbm(customer_number,start_date,end_date,update_model)
    return {"status": 200, "data": result}

@lightgbm_router.post("/train_all_lightgbm")
def train_all_lightgbm():
    result = LightgbmService.train_all_lightgbm()
    return {"status": 200, "data": result}