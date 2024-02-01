import bentoml
import dagshub
import mlflow

dagshub.init(
    repo_owner="ofelipefernandez", repo_name="model-deployment-bentoml", mlflow=True
)

# get all models from mlflow

# Get the best model
best_model = mlflow.search_runs(
    order_by=["metrics.training_roc_auc DESC"], max_results=2
)
print(best_model)

# register and save the best model into models registry

best_run_id = best_model["run_id"][0]
print(best_run_id)
artifact_name = "model"
model_name = "best_model_registered"
mlflow.register_model(f"runs:/{best_run_id}/{artifact_name}", model_name)

# mlflow.sklearn.save_model(best_model, "models/best_model_registered")

bento_model = bentoml.mlflow.import_model(
    "best_model_registered_bentoml",
    model_uri="models:/best_model_registered/latest",
)
