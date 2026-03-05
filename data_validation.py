"""
Data Validation Module
Healthcare AI - Hospital Readmission Prediction
Schema enforcement, distribution checks, data quality gates
"""

import logging
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Expected schema
SCHEMA = {
    "encounter_id": {"dtype": "int64", "nullable": False, "unique": True},
    "race": {"dtype": "object", "nullable": True, "allowed_values": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]},
    "gender": {"dtype": "object", "nullable": False, "allowed_values": ["Male", "Female"]},
    "age": {"dtype": "object", "nullable": False},
    "time_in_hospital": {"dtype": "int64", "nullable": False, "min": 1, "max": 14},
    "num_lab_procedures": {"dtype": "int64", "nullable": False, "min": 0, "max": 200},
    "num_procedures": {"dtype": "int64", "nullable": False, "min": 0, "max": 10},
    "num_medications": {"dtype": "int64", "nullable": False, "min": 0, "max": 100},
    "number_diagnoses": {"dtype": "int64", "nullable": False, "min": 1, "max": 20},
    "readmitted": {"dtype": "int64", "nullable": False, "allowed_values": [0, 1]},
}


class DataValidator:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.validation_report = {}
        self.passed = True

    def check_schema(self, df: pd.DataFrame) -> Dict:
        """Enforce schema on dataframe columns."""
        logger.info("Running schema validation...")
        results = {}

        for col, rules in SCHEMA.items():
            col_results = {"exists": col in df.columns}

            if col in df.columns:
                # Nullable check
                if not rules.get("nullable", True):
                    null_count = df[col].isnull().sum()
                    col_results["null_check"] = {"passed": null_count == 0, "null_count": int(null_count)}

                # Unique check
                if rules.get("unique", False):
                    dup_count = df[col].duplicated().sum()
                    col_results["unique_check"] = {"passed": dup_count == 0, "duplicate_count": int(dup_count)}

                # Range check
                if "min" in rules and pd.api.types.is_numeric_dtype(df[col]):
                    out_of_range = ((df[col] < rules["min"]) | (df[col] > rules["max"])).sum()
                    col_results["range_check"] = {"passed": out_of_range == 0, "out_of_range": int(out_of_range)}

                # Allowed values check
                if "allowed_values" in rules:
                    invalid = (~df[col].isin(rules["allowed_values"])).sum()
                    col_results["value_check"] = {"passed": invalid == 0, "invalid_count": int(invalid)}

            results[col] = col_results

        passed = all(
            all(v.get("passed", True) for k, v in col.items() if isinstance(v, dict))
            for col in results.values()
        )

        logger.info(f"Schema validation: {'PASSED' if passed else 'FAILED'}")
        return {"passed": passed, "details": results}

    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.3) -> Dict:
        """Check for excessive missing values."""
        logger.info("Checking missing values...")
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > threshold]

        result = {
            "passed": len(high_missing) == 0,
            "total_missing_pct": float(missing_pct.mean()),
            "high_missing_columns": high_missing.to_dict()
        }

        logger.info(f"Missing value check: {'PASSED' if result['passed'] else 'FAILED'}")
        return result

    def check_class_balance(self, df: pd.DataFrame) -> Dict:
        """Check target class distribution for severe imbalance."""
        logger.info("Checking class balance...")
        target = self.config["data"]["target_column"]
        dist = df[target].value_counts(normalize=True)

        minority_ratio = float(dist.min())
        severe_imbalance = minority_ratio < 0.05

        result = {
            "passed": not severe_imbalance,
            "distribution": dist.to_dict(),
            "minority_ratio": minority_ratio,
            "warning": "Severe class imbalance detected — consider SMOTE" if severe_imbalance else None
        }

        logger.info(f"Class balance check: {'PASSED' if result['passed'] else 'WARNING'}")
        return result

    def check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate records."""
        logger.info("Checking for duplicates...")
        dup_count = df.duplicated().sum()
        result = {
            "passed": dup_count == 0,
            "duplicate_count": int(dup_count),
            "duplicate_pct": float(dup_count / len(df))
        }
        logger.info(f"Duplicate check: {'PASSED' if result['passed'] else 'FAILED'}")
        return result

    def check_data_drift(self, df_new: pd.DataFrame, df_reference: pd.DataFrame) -> Dict:
        """PSI-based drift detection between reference and new data."""
        logger.info("Checking data drift...")
        numeric_cols = self.config["features"]["numeric_columns"]
        drift_results = {}

        for col in numeric_cols:
            if col in df_new.columns and col in df_reference.columns:
                psi = self._calculate_psi(df_reference[col], df_new[col])
                drift_results[col] = {
                    "psi": round(float(psi), 4),
                    "drift_detected": psi > self.config["monitoring"]["psi_threshold"]
                }

        drifted = [c for c, v in drift_results.items() if v["drift_detected"]]
        result = {
            "passed": len(drifted) == 0,
            "drifted_columns": drifted,
            "details": drift_results
        }

        logger.info(f"Drift check: {'PASSED' if result['passed'] else f'DRIFT in {drifted}'}")
        return result

    def _calculate_psi(self, reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index."""
        def get_bucket_pcts(data, breakpoints):
            counts = np.histogram(data.dropna(), bins=breakpoints)[0]
            pcts = counts / len(data)
            pcts = np.where(pcts == 0, 0.0001, pcts)
            return pcts

        breakpoints = np.percentile(reference.dropna(), np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 3:
            return 0.0

        ref_pcts = get_bucket_pcts(reference, breakpoints)
        cur_pcts = get_bucket_pcts(current, breakpoints)

        min_len = min(len(ref_pcts), len(cur_pcts))
        psi = np.sum((cur_pcts[:min_len] - ref_pcts[:min_len]) * np.log(cur_pcts[:min_len] / ref_pcts[:min_len]))
        return float(psi)

    def run_all_validations(self, df: pd.DataFrame, df_reference: pd.DataFrame = None) -> Dict:
        """Run complete validation suite."""
        logger.info("=" * 50)
        logger.info("RUNNING FULL DATA VALIDATION SUITE")
        logger.info("=" * 50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "schema": self.check_schema(df),
            "missing_values": self.check_missing_values(df),
            "class_balance": self.check_class_balance(df),
            "duplicates": self.check_duplicates(df),
        }

        if df_reference is not None:
            report["drift"] = self.check_data_drift(df, df_reference)

        # Overall pass/fail
        critical_checks = ["schema", "missing_values", "duplicates"]
        report["overall_passed"] = all(report[c]["passed"] for c in critical_checks)

        self._log_report_summary(report)
        self._save_report(report)

        return report

    def _log_report_summary(self, report: Dict):
        logger.info("=" * 50)
        logger.info("VALIDATION REPORT SUMMARY")
        logger.info("=" * 50)
        for key, value in report.items():
            if isinstance(value, dict) and "passed" in value:
                status = "✅ PASSED" if value["passed"] else "❌ FAILED"
                logger.info(f"  {key.upper()}: {status}")
        overall = "✅ ALL PASSED" if report["overall_passed"] else "❌ VALIDATION FAILED"
        logger.info(f"\nOVERALL: {overall}")
        logger.info("=" * 50)

    def _save_report(self, report: Dict):
        """Save validation report to file."""
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        report_path = "data/processed/validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to {report_path}")


if __name__ == "__main__":
    from src.ingestion.data_ingestion import run_ingestion
    df = run_ingestion()
    validator = DataValidator()
    report = validator.run_all_validations(df)
    print(f"\nValidation passed: {report['overall_passed']}")
