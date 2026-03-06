"""
Data Ingestion Module
Healthcare AI - Hospital Readmission Prediction
Handles loading, validating, and storing raw data
"""

import os
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    return config


def download_dataset(save_path: str) -> str:
    """
    Download the Diabetes 130-US Hospitals dataset.
    Dataset: UCI ML Repository - Diabetes 130-US hospitals for years 1999-2008
    """
    try:
        logger.info("Downloading Diabetes 130-US Hospitals dataset...")

        # Use UCI dataset URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # For demo/offline — generate realistic synthetic data
        logger.info("Generating synthetic healthcare dataset for demonstration...")
        df = generate_synthetic_data(n_samples=10000)
        df.to_csv(save_path, index=False)
        logger.info(f"Dataset saved to {save_path} with {len(df)} records")
        return save_path

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate realistic synthetic healthcare data mimicking Diabetes 130 dataset."""
    np.random.seed(42)

    data = {
        "encounter_id": range(1, n_samples + 1),
        "patient_nbr": np.random.randint(10000, 99999, n_samples),
        "race": np.random.choice(
            ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
            n_samples, p=[0.75, 0.19, 0.02, 0.02, 0.02]
        ),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "age": np.random.choice(
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            n_samples, p=[0.01, 0.01, 0.02, 0.04, 0.10, 0.18, 0.26, 0.24, 0.12, 0.02]
        ),
        "admission_type_id": np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
        "discharge_disposition_id": np.random.choice(range(1, 26), n_samples),
        "admission_source_id": np.random.choice(range(1, 18), n_samples),
        "time_in_hospital": np.random.randint(1, 14, n_samples),
        "num_lab_procedures": np.random.randint(1, 132, n_samples),
        "num_procedures": np.random.randint(0, 6, n_samples),
        "num_medications": np.random.randint(1, 82, n_samples),
        "number_outpatient": np.random.randint(0, 42, n_samples),
        "number_emergency": np.random.randint(0, 77, n_samples),
        "number_inpatient": np.random.randint(0, 21, n_samples),
        "diag_1": np.random.choice(["250", "276", "428", "414", "786"], n_samples),
        "diag_2": np.random.choice(["250", "276", "428", "414", "786"], n_samples),
        "diag_3": np.random.choice(["250", "276", "428", "414", "786"], n_samples),
        "number_diagnoses": np.random.randint(1, 16, n_samples),
        "max_glu_serum": np.random.choice(["None", ">200", ">300", "Norm"], n_samples),
        "A1Cresult": np.random.choice(["None", ">7", ">8", "Norm"], n_samples),
        "insulin": np.random.choice(["No", "Down", "Steady", "Up"], n_samples),
        "change": np.random.choice(["No", "Ch"], n_samples),
        "diabetesMed": np.random.choice(["Yes", "No"], n_samples, p=[0.77, 0.23]),
    }

    df = pd.DataFrame(data)

    # Create realistic target variable (readmitted within 30 days)
    readmit_prob = (
        0.1
        + 0.05 * (df["time_in_hospital"] > 7).astype(int)
        + 0.05 * (df["number_inpatient"] > 2).astype(int)
        + 0.03 * (df["number_emergency"] > 1).astype(int)
        + 0.04 * (df["num_medications"] > 20).astype(int)
        + 0.03 * (df["A1Cresult"] == "None").astype(int)
    )
    df["readmitted"] = (np.random.random(n_samples) < readmit_prob).astype(int)

    return df


def validate_raw_data(df: pd.DataFrame) -> bool:
    """Basic validation of raw data before processing."""
    logger.info("Validating raw data...")

    checks = {
        "not_empty": len(df) > 0,
        "has_target": "readmitted" in df.columns,
        "min_rows": len(df) >= 100,
        "no_all_null_columns": not df.isnull().all().any(),
    }

    for check, result in checks.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"  Validation [{check}]: {status}")

    all_passed = all(checks.values())
    if all_passed:
        logger.info("All raw data validations passed")
    else:
        logger.error("Some validations failed")

    return all_passed


def log_data_summary(df: pd.DataFrame):
    """Log summary statistics of ingested data."""
    logger.info("=" * 50)
    logger.info("DATA INGESTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total Records: {len(df):,}")
    logger.info(f"Total Features: {len(df.columns)}")
    logger.info(f"Target Distribution:\n{df['readmitted'].value_counts(normalize=True).round(3)}")
    logger.info(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    logger.info(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info("=" * 50)


def run_ingestion(config_path: str = "config/config.yaml"):
    """Main ingestion pipeline runner."""
    logger.info("Starting data ingestion pipeline...")
    start_time = datetime.now()

    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]

    # Download or load dataset
    if not os.path.exists(raw_path):
        download_dataset(raw_path)

    # Load data
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} records from {raw_path}")

    # Validate
    is_valid = validate_raw_data(df)
    if not is_valid:
        raise ValueError("Raw data validation failed. Stopping pipeline.")

    # Log summary
    log_data_summary(df)

    duration = (datetime.now() - start_time).seconds
    logger.info(f"Data ingestion completed in {duration}s")

    return df


if __name__ == "__main__":
    df = run_ingestion()
    print(f"\nIngestion complete. Shape: {df.shape}")
