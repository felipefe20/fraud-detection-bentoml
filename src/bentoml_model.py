import json

import bentoml
import dagshub
from bentoml.io import JSON

dagshub.init(
    repo_owner="ofelipefernandez", repo_name="model-deployment-bentoml", mlflow=True
)

# preprocessor_runner = bentoml.Runner(PreProcessor)
runner = bentoml.mlflow.get("best_model_registered_bentoml").to_runner()
svc = bentoml.Service("best_model_registered_bentoml", runners=[runner])


@svc.api(input=JSON(), output=JSON())
def predict(self, data) -> json:
    return svc.predict(data)
