"""
Model Training Module
Healthcare AI - Hospital Readmission Prediction
Multi-model training, hyperparameter tuning, MLflow experiment tracking
"""

import logging
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from xgboost import XGBClassifier
import optuna
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        self.best_model = None
        self.best_score = 0
        self.best_model_name = None

    def get_models(self) -> dict:
        """Return base model definitions."""
        return {
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=100, random_state=42,
                eval_metric="logloss", verbosity=0
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=42
            )
        }

    def compute_metrics(self, y_true, y_pred, y_prob) -> dict:
        """Compute all evaluation metrics."""
        return {
            "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    def train_with_mlflow(self, model, model_name: str, X_train, X_test, y_train, y_test) -> dict:
        """Train a single model with full MLflow tracking."""
        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log model params
            mlflow.log_params(model.get_params())
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            # Cross-validation
            cv = StratifiedKFold(n_splits=self.config["training"]["cv_folds"], shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            mlflow.log_metric("cv_roc_auc_mean", round(cv_scores.mean(), 4))
            mlflow.log_metric("cv_roc_auc_std", round(cv_scores.std(), 4))

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = self.compute_metrics(y_test, y_pred, y_prob)

            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=self.config["mlflow"]["model_registry_name"]
            )

            run_id = mlflow.active_run().info.run_id
            logger.info(f"  {model_name} | AUC: {metrics['roc_auc']} | F1: {metrics['f1_score']} | Run: {run_id}")

        return {"model": model, "metrics": metrics, "run_id": run_id, "cv_mean": cv_scores.mean()}

    def tune_xgboost(self, X_train, y_train) -> dict:
        """Optuna hyperparameter tuning for XGBoost."""
        logger.info("Tuning XGBoost hyperparameters with Optuna...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": 42,
                "eval_metric": "logloss",
                "verbosity": 0
            }
            model = XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config["training"]["n_trials"])

        logger.info(f"Best XGBoost AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        return study.best_params

    def train_all_models(self, X_train, X_test, y_train, y_test) -> dict:
        """Train all models and track with MLflow."""
        logger.info("=" * 50)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 50)

        models = self.get_models()
        results = {}

        # Optionally tune XGBoost
        if self.config["training"]["hyperparameter_tuning"]:
            best_xgb_params = self.tune_xgboost(X_train, y_train)
            models["xgboost_tuned"] = XGBClassifier(
                **best_xgb_params, eval_metric="logloss", verbosity=0
            )

        for name, model in models.items():
            result = self.train_with_mlflow(model, name, X_train, X_test, y_train, y_test)
            results[name] = result

            # Track best model
            if result["metrics"]["roc_auc"] > self.best_score:
                self.best_score = result["metrics"]["roc_auc"]
                self.best_model = result["model"]
                self.best_model_name = name

        return results

    def save_best_model(self, path: str = "data/processed/best_model.pkl"):
        """Save the best performing model."""
        if self.best_model is None:
            raise ValueError("No model trained yet.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, path)
        logger.info(f"Best model ({self.best_model_name}, AUC={self.best_score:.4f}) saved to {path}")

    def print_leaderboard(self, results: dict):
        """Print model comparison leaderboard."""
        logger.info("\n" + "=" * 60)
        logger.info("MODEL LEADERBOARD")
        logger.info("=" * 60)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["metrics"]["roc_auc"], reverse=True)
        for rank, (name, result) in enumerate(sorted_results, 1):
            m = result["metrics"]
            logger.info(
                f"#{rank} {name:<25} AUC={m['roc_auc']:.4f} | "
                f"F1={m['f1_score']:.4f} | "
                f"Precision={m['precision']:.4f} | "
                f"Recall={m['recall']:.4f}"
            )
        logger.info(f"\n🏆 CHAMPION: {self.best_model_name} (AUC={self.best_score:.4f})")
        logger.info("=" * 60)


def run_training(config_path: str = "config/config.yaml"):
    """Main training pipeline."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load processed data
    df = pd.read_csv(config["data"]["processed_path"])
    target = config["data"]["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Target balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")

    trainer = ModelTrainer(config_path)
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    trainer.print_leaderboard(results)
    trainer.save_best_model()

    return trainer, results


if __name__ == "__main__":
    trainer, results = run_training()
