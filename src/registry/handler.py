from pprint import pprint

import mlflow
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel


class MLFlowHandler:
    def __init__(self, tracking_uri: str) -> None:
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def check_mlflow_health(self) -> None:
        try:
            experiments = self.client.search_experiments()
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return "Service returning experiments"
        except:
            return "Error calling MLFlow"

    def get_production_model(self, store_id: str) -> PyFuncModel:
        model_name = f"prophet-retail-forecaster-store-{store_id}"
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
        return model


# Handle this properly later ...
def check_mlflow_health(tracking_uri: str):
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        experiments = client.search_experiments()
        for rm in experiments:
            pprint(dict(rm), indent=4)
        return "Service returning experiments"
    except:
        return "Error calling MLFlow"
