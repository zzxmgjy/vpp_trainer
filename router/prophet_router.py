from fastapi import APIRouter,Body
from service.prophet_service import ProphetService
from typing import Optional
prophet_router = APIRouter()


@prophet_router.post("/train_by_prophet")
def train_by_prophet(
    customer_number: Optional[str] = Body(None, embed=True),
    update_model: Optional[bool] = Body(False, embed=True)
):
    result = ProphetService.train_by_prophet(customer_number,update_model)
    return {"status": 200, "data": result}

@prophet_router.post("/train_all_prophet")
def train_all_prophet():
    result = ProphetService.train_all_prophet()
    return {"status": 200, "data": result}