"""
Feature Engineering Module
Healthcare AI - Hospital Readmission Prediction
Preprocessing, encoding, scaling, feature creation
"""

import logging
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HealthcareFeatureEngineer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.imputers = {}
        self.is_fitted = False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data - handle special values."""
        logger.info("Cleaning data...")
        df = df.copy()

        # Replace '?' with NaN (common in UCI datasets)
        df.replace("?", np.nan, inplace=True)

        # Drop high-cardinality or irrelevant ID columns
        drop_cols = ["encounter_id", "patient_nbr"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Remove expired/hospice patients (discharge 11, 13, 14, 19, 20, 21)
        if "discharge_disposition_id" in df.columns:
            expired_codes = [11, 13, 14, 19, 20, 21]
            df = df[~df["discharge_disposition_id"].isin(expired_codes)]

        logger.info(f"After cleaning: {len(df):,} records")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific healthcare features."""
        logger.info("Engineering features...")
        df = df.copy()

        # Age bucket to numeric
        age_map = {
            "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
            "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
            "[70-80)": 75, "[80-90)": 85, "[90-100)": 95
        }
        if "age" in df.columns:
            df["age_numeric"] = df["age"].map(age_map).fillna(50)

        # Service utilization score
        if all(c in df.columns for c in ["number_outpatient", "number_emergency", "number_inpatient"]):
            df["total_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
            df["high_utilizer"] = (df["total_visits"] > 5).astype(int)

        # Medication complexity
        if "num_medications" in df.columns:
            df["polypharmacy"] = (df["num_medications"] > 10).astype(int)

        # Long stay indicator
        if "time_in_hospital" in df.columns:
            df["long_stay"] = (df["time_in_hospital"] > 7).astype(int)

        # Diagnosis complexity
        if "number_diagnoses" in df.columns:
            df["complex_case"] = (df["number_diagnoses"] > 7).astype(int)

        # A1C missing (poor diabetes management indicator)
        if "A1Cresult" in df.columns:
            df["a1c_missing"] = (df["A1Cresult"] == "None").astype(int)
            df["a1c_high"] = (df["A1Cresult"].isin([">7", ">8"])).astype(int)

        # Emergency admission flag
        if "admission_type_id" in df.columns:
            df["emergency_admission"] = (df["admission_type_id"] == 1).astype(int)

        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical columns."""
        logger.info("Encoding categorical features...")
        df = df.copy()
        cat_cols = self.config["features"]["categorical_columns"]
        cat_cols = [c for c in cat_cols if c in df.columns]

        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                if col in self.encoders:
                    le = self.encoders[col]
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df

    def impute_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Impute missing values."""
        logger.info("Imputing missing values...")
        df = df.copy()

        num_cols = [c for c in self.config["features"]["numeric_columns"] if c in df.columns]
        cat_cols = [c for c in self.config["features"]["categorical_columns"] if c in df.columns]

        if num_cols:
            if fit:
                self.imputers["numeric"] = SimpleImputer(strategy="median")
                df[num_cols] = self.imputers["numeric"].fit_transform(df[num_cols])
            else:
                df[num_cols] = self.imputers["numeric"].transform(df[num_cols])

        if cat_cols:
            if fit:
                self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")
                df[cat_cols] = self.imputers["categorical"].fit_transform(df[cat_cols].astype(str))
            else:
                df[cat_cols] = self.imputers["categorical"].transform(df[cat_cols].astype(str))

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features."""
        logger.info("Scaling numeric features...")
        df = df.copy()

        target = self.config["data"]["target_column"]
        scale_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])

        return df

    def get_feature_target_split(self, df: pd.DataFrame):
        """Split into features and target."""
        target = self.config["data"]["target_column"]
        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline - fit and transform."""
        logger.info("Running full feature engineering pipeline (fit)...")
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.impute_missing(df, fit=True)
        df = self.encode_categoricals(df, fit=True)
        df = self.scale_features(df, fit=True)
        self.is_fitted = True
        logger.info("Feature engineering pipeline complete")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform only - no fitting."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Run fit_transform first.")
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.impute_missing(df, fit=False)
        df = self.encode_categoricals(df, fit=False)
        df = self.scale_features(df, fit=False)
        return df

    def save(self, path: str = "data/processed/feature_pipeline.pkl"):
        """Save fitted pipeline."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Feature pipeline saved to {path}")

    @staticmethod
    def load(path: str = "data/processed/feature_pipeline.pkl"):
        """Load saved pipeline."""
        pipeline = joblib.load(path)
        logger.info(f"Feature pipeline loaded from {path}")
        return pipeline


if __name__ == "__main__":
    from src.ingestion.data_ingestion import run_ingestion
    df = run_ingestion()

    engineer = HealthcareFeatureEngineer()
    df_processed = engineer.fit_transform(df)
    df_processed.to_csv("data/processed/processed_data.csv", index=False)
    engineer.save()

    print(f"\nProcessed shape: {df_processed.shape}")
    print(f"Features: {list(df_processed.columns)}")
