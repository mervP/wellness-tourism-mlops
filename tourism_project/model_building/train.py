"""Train an XGBoost classifier with hyperparameter tuning + MLflow tracking."""

import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HF_USERNAME = os.getenv("HF_USERNAME", "prashanth-merwyn")
DATASET_REPO = f"{HF_USERNAME}/wellness-tourism-dataset"
MODEL_REPO = f"{HF_USERNAME}/wellness-tourism-model"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("wellness-tourism-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

class_counts = ytrain.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb_clf)])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
}

with mlflow.start_run(run_name="xgb_grid_search"):
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring="f1",
                        n_jobs=-1, verbose=1)
    grid.fit(Xtrain, ytrain)

    for params, mean_score in zip(
        grid.cv_results_["params"], grid.cv_results_["mean_test_score"]
    ):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", float(mean_score))

    mlflow.log_params(grid.best_params_)
    best_model = grid.best_estimator_

    pred_train = best_model.predict(Xtrain)
    pred_test = best_model.predict(Xtest)
    proba_test = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(ytrain, pred_train),
        "test_accuracy": accuracy_score(ytest, pred_test),
        "test_precision": precision_score(ytest, pred_test),
        "test_recall": recall_score(ytest, pred_test),
        "test_f1": f1_score(ytest, pred_test),
        "test_roc_auc": roc_auc_score(ytest, proba_test),
    }
    mlflow.log_metrics(metrics)
    print("Test metrics:", metrics)
    print(classification_report(ytest, pred_test, digits=4))

    model_path = "best_wellness_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Saved best model to {model_path}")

    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo '{MODEL_REPO}' already exists. Reusing it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{MODEL_REPO}' not found. Creating new repo...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False,
                    token=os.getenv("HF_TOKEN"))

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Best model pushed to {MODEL_REPO}")
