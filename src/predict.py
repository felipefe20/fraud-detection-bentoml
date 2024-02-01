import dagshub
import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder, StandardScaler

dagshub.init(
    repo_owner="ofelipefernandez", repo_name="model-deployment-bentoml", mlflow=True
)


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to train the model"""

    print(f"Train modeling using {config.data.processed}")
    print(f"Model used: {config.model.name}")
    # print(f"Save the output to {config.data.final}")

    # Load data and don't read first column

    df = pd.read_csv(config.data.processed, index_col=0)

    df.dropna(inplace=True)
    df["dob"] = pd.to_datetime(df["dob"])
    df["year_dob"] = df["dob"].dt.year
    df["year_dob"] = df["year_dob"].astype(str)
    # Define target and features
    df[config.process.target_column]
    # X = df.drop(config.process.use_columns , axis=1)
    X = df.drop(columns=config.process.drop_columns)
    print(X.columns)
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=["int", "float"]).columns
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns

    # Scale numerical columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        X[col] = encoder.fit_transform(X[col])

    model = mlflow.sklearn.load_model("models:/best_model_registered/latest")
    y_pred = model.predict(X)

    print(y_pred)
