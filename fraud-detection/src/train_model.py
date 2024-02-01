"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

from sklearn.preprocessing import StandardScaler
from typing import Union, Optional
import hydra
from pathlib import Path

from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import mlflow
import joblib
import dagshub
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


dagshub.init(repo_owner='ofelipefernandez', repo_name='model-deployment-bentoml', mlflow=True)

mlflow.autolog()

def save_model(model, model_path: Union[str, Path]):
    """
    Template for saving a model.

    Args:
        model: Trained model.
        model_path: Path to save the model.
    """

    joblib.dump(model, model_path)




@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to train the model"""

    print(f"Train modeling using {config.data.processed}")
    print(f"Model used: {config.model.name}")
    #print(f"Save the output to {config.data.final}")

    # Load data
    df = pd.read_csv(config.data.processed)
    df=df[:5000]
    # Define target and features
    y = df[config.process.use_columns]
    X = df.drop(config.process.use_columns , axis=1)
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

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
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=config.model.test_size, random_state=42)
    
    # Create a Decision Tree Classifier model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #print f1-score
    print(f"F1-score: {classification_report(y_test, y_pred)}")
    

    save_model(model, "../models/model.pkl")


if __name__ == "__main__":
    train_model()
