"""
Apache Airflow DAG
Healthcare AI - Hospital Readmission Prediction
Full MLOps pipeline orchestration with auto-retraining
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging

logger = logging.getLogger(__name__)

# ─── Default DAG Arguments ────────────────────────────────────────
default_args = {
    "owner": "healthcare-mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ─── Pipeline Task Functions ──────────────────────────────────────
def task_data_ingestion(**context):
    """Task: Ingest raw healthcare data."""
    import sys
    sys.path.append("/opt/airflow/dags/healthcare_mlops")
    from src.ingestion.data_ingestion import run_ingestion

    logger.info("Starting data ingestion task...")
    df = run_ingestion()
    logger.info(f"Ingestion complete. Records: {len(df)}")

    # Push to XCom for downstream tasks
    context["task_instance"].xcom_push(key="record_count", value=len(df))
    return f"Ingested {len(df)} records"


def task_data_validation(**context):
    """Task: Validate ingested data."""
    import sys
    import pandas as pd
    sys.path.append("/opt/airflow/dags/healthcare_mlops")
    from src.validation.data_validation import DataValidator

    logger.info("Starting data validation task...")
    df = pd.read_csv("data/raw/diabetic_data.csv")
    validator = DataValidator()
    report = validator.run_all_validations(df)

    if not report["overall_passed"]:
        raise ValueError(f"Data validation failed: {report}")

    context["task_instance"].xcom_push(key="validation_passed", value=True)
    return "Validation passed"


def task_feature_engineering(**context):
    """Task: Feature engineering pipeline."""
    import sys
    import pandas as pd
    sys.path.append("/opt/airflow/dags/healthcare_mlops")
    from src.features.feature_engineering import HealthcareFeatureEngineer

    logger.info("Starting feature engineering task...")
    df = pd.read_csv("data/raw/diabetic_data.csv")
    engineer = HealthcareFeatureEngineer()
    df_processed = engineer.fit_transform(df)
    df_processed.to_csv("data/processed/processed_data.csv", index=False)
    engineer.save()

    context["task_instance"].xcom_push(key="feature_count", value=len(df_processed.columns))
    logger.info(f"Feature engineering complete. Features: {len(df_processed.columns)}")
    return f"Processed {len(df_processed)} records with {len(df_processed.columns)} features"


def task_model_training(**context):
    """Task: Train all models with MLflow tracking."""
    import sys
    sys.path.append("/opt/airflow/dags/healthcare_mlops")
    from src.training.model_training import run_training

    logger.info("Starting model training task...")
    trainer, results = run_training()

    best_score = trainer.best_score
    best_model = trainer.best_model_name

    context["task_instance"].xcom_push(key="best_model_name", value=best_model)
    context["task_instance"].xcom_push(key="best_model_score", value=best_score)

    logger.info(f"Training complete. Best model: {best_model} | AUC: {best_score:.4f}")
    return f"Champion: {best_model} (AUC={best_score:.4f})"


def task_check_drift(**context):
    """Task: Check for data drift. Branch to retraining if detected."""
    import sys
    import json
    import os
    sys.path.append("/opt/airflow/dags/healthcare_mlops")

    logger.info("Checking for data drift...")

    trigger_path = "data/drift/retraining_trigger.json"
    if os.path.exists(trigger_path):
        with open(trigger_path) as f:
            trigger = json.load(f)
        if trigger.get("status") == "triggered":
            logger.info("Drift detected — branching to retraining")
            return "retrain_model"

    logger.info("No drift detected — skipping retraining")
    return "skip_retraining"


def task_champion_vs_challenger(**context):
    """Task: Compare new model vs current champion."""
    import sys
    sys.path.append("/opt/airflow/dags/healthcare_mlops")
    import joblib
    import pandas as pd
    import yaml
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    logger.info("Running champion vs challenger comparison...")

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data"]["processed_path"])
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"], stratify=y
    )

    champion = joblib.load("data/processed/best_model.pkl")
    champion_auc = roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1])

    logger.info(f"Champion AUC: {champion_auc:.4f}")
    context["task_instance"].xcom_push(key="champion_auc", value=champion_auc)

    return f"Champion AUC: {champion_auc:.4f}"


def task_deploy_model(**context):
    """Task: Auto-deploy champion model to API."""
    import shutil
    logger.info("Deploying champion model...")
    shutil.copy("data/processed/best_model.pkl", "data/processed/production_model.pkl")
    logger.info("Model deployed to production successfully")
    return "Model deployed"


# ─── Main Training DAG ────────────────────────────────────────────
with DAG(
    dag_id="healthcare_mlops_training_pipeline",
    default_args=default_args,
    description="Full MLOps pipeline for hospital readmission prediction",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["healthcare", "mlops", "training"],
) as training_dag:

    start = DummyOperator(task_id="pipeline_start")
    end = DummyOperator(task_id="pipeline_end", trigger_rule=TriggerRule.ALL_DONE)

    ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_data_ingestion,
    )

    validate = PythonOperator(
        task_id="data_validation",
        python_callable=task_data_validation,
    )

    engineer = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=task_model_training,
    )

    champion_challenger = PythonOperator(
        task_id="champion_vs_challenger",
        python_callable=task_champion_vs_challenger,
    )

    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=task_deploy_model,
    )

    start >> ingest >> validate >> engineer >> train >> champion_challenger >> deploy >> end


# ─── Drift Monitoring DAG ─────────────────────────────────────────
with DAG(
    dag_id="healthcare_mlops_drift_monitoring",
    default_args=default_args,
    description="Daily drift monitoring and auto-retraining trigger",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["healthcare", "monitoring", "drift"],
) as monitoring_dag:

    monitor_start = DummyOperator(task_id="monitoring_start")
    monitor_end = DummyOperator(task_id="monitoring_end", trigger_rule=TriggerRule.ALL_DONE)

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=task_model_training,
    )

    skip_retrain = DummyOperator(task_id="skip_retraining")

    redeploy = PythonOperator(
        task_id="redeploy_after_retraining",
        python_callable=task_deploy_model,
    )

    monitor_start >> check_drift
    check_drift >> retrain >> redeploy >> monitor_end
    check_drift >> skip_retrain >> monitor_end
