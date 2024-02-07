import logging

import dagshub
import hydra
import mlflow
import pandas as pd

# plt.rcParams.update({'font.size': 22})
import prophet
from mlflow.client import MlflowClient
from omegaconf import DictConfig
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
)




def prep_store_data(
    df: pd.DataFrame, store_id: int = 4, store_open: int = 1
) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Date": "ds", "Sales": "y"})
    df_store = df[(df["Store"] == store_id) & (df["Open"] == store_open)].reset_index(
        drop=True
    )
    return df_store.sort_values("ds", ascending=True)


def train_test_split_forecaster(
    df: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # grab split data
    train_index = int(train_fraction * df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]
    return df_train, df_test


def train_forecaster(
    df_train: pd.DataFrame, seasonality: dict
) -> prophet.forecaster.Prophet:
    # create Prophet model
    forecaster = Prophet(
        yearly_seasonality=seasonality["yearly"],
        weekly_seasonality=seasonality["weekly"],
        daily_seasonality=seasonality["daily"],
        interval_width=0.95,
    )
    forecaster.fit(df_train)
    return forecaster


def test_forecaster(df_test: pd.DataFrame) -> None:
    return None


def forecast(
    forecaster: prophet.forecaster.Prophet, forecast_index: pd.DataFrame
) -> pd.DataFrame:
    return forecaster.predict(forecast_index)


def train_predict(
    df: pd.DataFrame, train_fraction: float, seasonality: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    # grab split data
    train_index = int(train_fraction * df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]

    # create Prophet model
    model = Prophet(
        yearly_seasonality=seasonality["yearly"],
        weekly_seasonality=seasonality["weekly"],
        daily_seasonality=seasonality["daily"],
        interval_width=0.95,
    )

    # train and predict
    model.fit(df_train)
    predicted = model.predict(df_test)
    return predicted, df_train, df_test, train_index


# def plot_forecast(df_train: pd.DataFrame, df_test: pd.DataFrame, predicted: pd.DataFrame) -> None:
#     fig, ax = plt.subplots(figsize=(20,10))
#     df_test.plot(
#         x='ds',
#         y='y',
#         ax=ax,
#         label='Truth',
#         linewidth=1,
#         markersize=5,
#         color='tab:blue',
#         alpha=0.9,
#         marker='o'
#     )
#     predicted.plot(
#         x='ds',
#         y='yhat',
#         ax=ax,
#         label='Prediction + 95% CI',
#         linewidth=2,
#         markersize=5,
#         color='red'
#     )
#     ax.fill_between(
#         x=predicted['ds'],
#         y1=predicted['yhat_upper'],
#         y2=predicted['yhat_lower'],
#         alpha=0.15,
#         color='red',
#     )
#     df_train.iloc[train_index-100:].plot(
#         x='ds',
#         y='y',
#         ax=ax,
#         color='tab:blue',
#         label='_nolegend_',
#         alpha=0.5,
#         marker='o'
#     )
#     current_ytick_values = plt.gca().get_yticks()
#     plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_ytick_values])
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Sales')
#     plt.tight_layout()
#     plt.savefig('store_data_forecast.png')
import logging
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow

def setup_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)

def setup_mlflow(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    logging.info("Defined MLFlowClient and set tracking URI.")
    return client

def load_data(file_path):
    df = pd.read_csv(file_path)
    logging.info("Training data retrieved.")
    return df

def train_and_log_model(df, store_id, MODEL_BASE_NAME, seasonality):
    with mlflow.start_run():
        logging.info("Started MLFlow run")
        df_transformed = prep_store_data(df, store_id=int(store_id))
        logging.info("Transformed data")

        model_name = f"{MODEL_BASE_NAME}-{store_id}"
        mlflow.autolog()

        logging.info("Splitting data")
        df_train, df_test = train_test_split_forecaster(df=df_transformed, train_fraction=0.75)
        logging.info("Data split")

        logging.info("Training model")
        forecaster = train_forecaster(df_train=df_train, seasonality=seasonality)
        run_id = mlflow.active_run().info.run_id
        logging.info("Model trained")

        mlflow.prophet.log_model(forecaster, artifact_path="model")
        logging.info("Logged model")

        mlflow.log_params(seasonality)
        mlflow.log_metrics(
            {
                "rmse": mean_squared_error(y_true=df_test["y"], y_pred=forecaster.predict(df_test)["yhat"], squared=False),
                "mean_abs_perc_error": mean_absolute_percentage_error(y_true=df_test["y"], y_pred=forecaster.predict(df_test)["yhat"]),
                "mean_abs_error": mean_absolute_error(y_true=df_test["y"], y_pred=forecaster.predict(df_test)["yhat"]),
                "median_abs_error": median_absolute_error(y_true=df_test["y"], y_pred=forecaster.predict(df_test)["yhat"]),
            }
        )

        artifact_path = "model"
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info("Model registered")

        return model_details, run_id

def transition_model_to_prod(client, model_details):
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="production",
    )
    logging.info("Model transitioned to prod stage")




@hydra.main(config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    tracking_uri = cfg.tracking_uri
    MODEL_BASE_NAME = cfg.MODEL_BASE_NAME
    repo_owner = cfg.repo_owner
    repo_name = cfg.repo_name
    mlflow_true = cfg.mlflow
    stores = cfg.stores

    dagshub.init(repo_owner, repo_name, mlflow=mlflow_true)

    setup_logging()

    client = setup_mlflow(tracking_uri)
    file_path = "data/raw/train.csv"
    df = load_data(file_path)

    seasonality = {"yearly": True, "weekly": True, "daily": False}

    for store_id in stores:
        model_details, run_id = train_and_log_model(df, store_id, MODEL_BASE_NAME, seasonality)
        transition_model_to_prod(client, model_details)

if __name__ == "__main__":
    main()
