"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""


import dagshub
import hydra
import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

dagshub.init(
    repo_owner="ofelipefernandez", repo_name="model-deployment-bentoml", mlflow=True
)

mlflow.autolog()


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
    y = df[config.process.target_column]
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

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=config.model.test_size, random_state=42
    )

    model = RandomForestClassifier(
        random_state=42,
        max_depth=config.model.hyperparameters.max_depth,
        min_samples_leaf=config.model.hyperparameters.min_samples_leaf,
        min_samples_split=config.model.hyperparameters.min_samples_split,
        criterion=config.model.hyperparameters.criterion,
        n_estimators=config.model.hyperparameters.n_estimators,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print f1-score
    print(f"F1-score: {classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    train_model()
