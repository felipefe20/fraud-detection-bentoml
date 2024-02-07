# Logging
import logging
import pprint
from typing import List

import hydra
from fastapi import FastAPI
from helpers.request import ForecastRequest, create_forecast_index
from omegaconf import DictConfig
from registry.handler import MLFlowHandler

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

handlers = {}
models = {}
# MODEL_BASE_NAME = f"prophet-retail-forecaster-store-"

app = FastAPI()


@hydra.main(config_path="config", config_name="main")
def config_Setup(cfg: DictConfig) -> None:
    tracking_uri = cfg.tracking_uri
    MODEL_BASE_NAME = cfg.model_base_name
    return tracking_uri, MODEL_BASE_NAME


tracking_uri, MODEL_BASE_NAME = config_Setup()


# Start the in memory cache
@app.on_event("startup")
async def startup():
    logging.info("InMemory cache initiated")
    await get_service_handlers()
    logging.info("Updated global service handlers")


async def get_service_handlers():
    mlflow_handler = MLFlowHandler(tracking_uri=tracking_uri)
    global handlers
    handlers["mlflow"] = mlflow_handler
    logging.info("Retreving mlflow handler {}".format(mlflow_handler))
    return handlers


@app.get("/health/", status_code=200)
async def healthcheck():
    global handlers
    logging.info("Got handlers in healthcheck.")
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": handlers["mlflow"].check_mlflow_health(),
    }


# @cache(expire=30)
async def get_models(store_ids: list):
    global handlers
    global models
    models = []
    for store_id in store_ids:
        model = handlers["mlflow"].get_production_model(store_id=store_id)
        models.append(model)
    return models


async def get_model(store_id: str):
    global handlers
    global models
    model_name = MODEL_BASE_NAME + f"{store_id}"
    if model_name not in models:
        models[model_name] = handlers["mlflow"].get_production_model(store_id=store_id)
    return models[model_name]


@app.get("/testcache", status_code=200)
async def test_cache():
    global handlers
    logging.info(pprint.pprint(str(handlers["mlflow"])))
    return {"type": str(handlers["mlflow"].get_production_model(store_id="4"))}


@app.post("/forecast/", status_code=200)
async def return_forecast(forecast_request: List[ForecastRequest]):
    """
    Main route in the app for returning the forecast, steps are:

    1. iterate over forecast elements
    2. get model for each store, forecast request
    3. prepare forecast input time index
    4. perform forecast
    5. append to return object
    6. return
    """
    forecasts = []
    for item in forecast_request:
        model = await get_model(item.store_id)
        forecast_input = create_forecast_index(
            begin_date=item.begin_date, end_date=item.end_date
        )
        forecast_result = {}
        forecast_result["request"] = item.dict()
        model_prediction = model.predict(forecast_input)[["ds", "yhat"]].rename(
            columns={"ds": "timestamp", "yhat": "value"}
        )
        model_prediction["value"] = model_prediction["value"].astype(int)
        forecast_result["forecast"] = model_prediction.to_dict("records")
        forecasts.append(forecast_result)
    return forecasts
