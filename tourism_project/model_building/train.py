"""
Model building + experiment tracking step.

- Loads the train/test splits produced by the data-prep job.
- Builds a preprocessing + XGBoost pipeline.
- Tunes hyperparameters with GridSearchCV.
- Logs every grid candidate (nested runs) and the best parameters/metrics to MLflow.
- Evaluates the best model and saves it to tourism_project/deployment/ so the
  workflow can commit it back into the repository.
"""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb

DEPLOY_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(DEPLOY_DIR, "best_tourism_model.joblib")


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    print("Loaded splits:", Xtrain.shape, Xtest.shape)
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(Xtrain):
    numeric_features = Xtrain.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = Xtrain.select_dtypes(exclude=["number"]).columns.tolist()
    print("Numeric features     :", numeric_features)
    print("Categorical features :", categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipe


def main():
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    Xtrain, Xtest, ytrain, ytest = load_splits()
    pipe = build_pipeline(Xtrain)

    # Handle class imbalance by weighting the positive class.
    neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
    scale_pos_weight = float(neg) / float(pos) if pos else 1.0

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__scale_pos_weight": [scale_pos_weight],
    }

    mlflow.set_experiment("tourism-package-prediction")

    with mlflow.start_run(run_name="xgboost_gridsearch"):
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(Xtrain, ytrain)

        best_model = grid.best_estimator_
        print("Best parameters:", grid.best_params_)

        # Log every grid candidate as a nested run (its parameters plus the
        # mean/std cross-validated F1), so all tuned combinations are tracked.
        results = grid.cv_results_
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_f1", float(results["mean_test_score"][i]))
                mlflow.log_metric("std_cv_f1", float(results["std_test_score"][i]))
                mlflow.log_metric("rank_cv_f1", int(results["rank_test_score"][i]))
        print(f"Logged {len(results['params'])} grid candidates as nested MLflow runs.")
        # Log the best parameters and CV score in the parent run.
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_f1", float(grid.best_score_))
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")

        # Evaluate on the held-out test set.
        ypred = best_model.predict(Xtest)
        yproba = best_model.predict_proba(Xtest)[:, 1]

        metrics = {
            "accuracy": accuracy_score(ytest, ypred),
            "precision": precision_score(ytest, ypred, zero_division=0),
            "recall": recall_score(ytest, ypred, zero_division=0),
            "f1": f1_score(ytest, ypred, zero_division=0),
            "roc_auc": roc_auc_score(ytest, yproba),
        }
        mlflow.log_metrics(metrics)

        print("\nTest-set performance:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification report:\n", classification_report(ytest, ypred))

        # Save the best model for deployment and log it to MLflow.
        joblib.dump(best_model, MODEL_PATH)
        print("Saved best model to", MODEL_PATH)
        # Input example (integer columns cast to float64 so the inferred
        # signature tolerates missing values) also records a model signature.
        int_cols = Xtrain.select_dtypes(include=["integer"]).columns
        input_example = Xtrain.head(5).astype({c: "float64" for c in int_cols})
        mlflow.sklearn.log_model(
            best_model, name="model", input_example=input_example
        )


if __name__ == "__main__":
    main()
