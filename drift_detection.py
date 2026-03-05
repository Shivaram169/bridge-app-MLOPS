"""
Drift Detection & Auto-Retraining Module
Healthcare AI - Hospital Readmission Prediction
KS Test + PSI drift detection with automated retraining trigger
"""

import logging
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.drift_threshold = self.config["monitoring"]["drift_threshold"]
        self.psi_threshold = self.config["monitoring"]["psi_threshold"]

    def ks_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float, bool]:
        """Kolmogorov-Smirnov test for distribution shift."""
        stat, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
        drift_detected = p_value < self.drift_threshold
        return float(stat), float(p_value), drift_detected

    def psi_score(self, reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
        """Population Stability Index — measures feature distribution shift."""
        breakpoints = np.percentile(reference.dropna(), np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 3:
            return 0.0

        def bucket_pcts(data):
            counts = np.histogram(data.dropna(), bins=breakpoints)[0]
            pcts = counts / max(len(data), 1)
            return np.where(pcts == 0, 0.0001, pcts)

        ref_pcts = bucket_pcts(reference)
        cur_pcts = bucket_pcts(current)
        min_len = min(len(ref_pcts), len(cur_pcts))
        psi = np.sum((cur_pcts[:min_len] - ref_pcts[:min_len]) * np.log(cur_pcts[:min_len] / ref_pcts[:min_len]))
        return float(psi)

    def psi_interpretation(self, psi: float) -> str:
        if psi < 0.1:
            return "No significant change"
        elif psi < 0.2:
            return "Moderate change — monitor closely"
        else:
            return "Significant change — retraining recommended"

    def detect_feature_drift(self, df_reference: pd.DataFrame, df_current: pd.DataFrame) -> Dict:
        """Run drift detection on all numeric features."""
        logger.info("Running feature drift detection...")
        numeric_cols = self.config["features"]["numeric_columns"]
        cols_to_check = [c for c in numeric_cols if c in df_reference.columns and c in df_current.columns]

        drift_report = {}
        drifted_features = []

        for col in cols_to_check:
            ks_stat, ks_pval, ks_drift = self.ks_test(df_reference[col], df_current[col])
            psi = self.psi_score(df_reference[col], df_current[col])
            psi_drift = psi > self.psi_threshold

            drift_detected = ks_drift or psi_drift
            if drift_detected:
                drifted_features.append(col)

            drift_report[col] = {
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(ks_pval, 4),
                "ks_drift": ks_drift,
                "psi_score": round(psi, 4),
                "psi_drift": psi_drift,
                "psi_interpretation": self.psi_interpretation(psi),
                "drift_detected": drift_detected
            }

        overall_drift = len(drifted_features) > len(cols_to_check) * 0.3

        result = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(df_reference),
            "current_size": len(df_current),
            "total_features_checked": len(cols_to_check),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "overall_drift_detected": overall_drift,
            "retraining_required": overall_drift,
            "feature_drift": drift_report
        }

        self._log_drift_summary(result)
        self._save_drift_report(result)

        return result

    def detect_prediction_drift(self, reference_preds: list, current_preds: list) -> Dict:
        """Detect drift in model output distribution."""
        ref = pd.Series(reference_preds)
        cur = pd.Series(current_preds)

        ks_stat, ks_pval, ks_drift = self.ks_test(ref, cur)
        psi = self.psi_score(ref, cur)

        result = {
            "timestamp": datetime.now().isoformat(),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(ks_pval, 4),
            "prediction_drift": ks_drift,
            "psi_score": round(psi, 4),
            "psi_interpretation": self.psi_interpretation(psi),
            "reference_mean_prob": round(float(ref.mean()), 4),
            "current_mean_prob": round(float(cur.mean()), 4),
            "retraining_required": ks_drift or psi > self.psi_threshold
        }

        logger.info(f"Prediction drift: {'DETECTED' if result['prediction_drift'] else 'NONE'} | "
                    f"PSI: {psi:.4f} | KS p-value: {ks_pval:.4f}")
        return result

    def _log_drift_summary(self, result: Dict):
        logger.info("=" * 50)
        logger.info("DRIFT DETECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Features checked: {result['total_features_checked']}")
        logger.info(f"Drifted features: {result['drifted_features_count']}")
        if result['drifted_features']:
            logger.info(f"Drifted columns: {result['drifted_features']}")
        status = "🚨 DRIFT DETECTED — RETRAINING REQUIRED" if result['overall_drift_detected'] else "✅ NO SIGNIFICANT DRIFT"
        logger.info(f"Status: {status}")
        logger.info("=" * 50)

    def _save_drift_report(self, result: Dict):
        Path("data/drift").mkdir(parents=True, exist_ok=True)
        filename = f"data/drift/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Drift report saved to {filename}")
        return filename


class AutoRetrainingTrigger:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.detector = DriftDetector(config_path)

    def check_and_trigger(self, df_reference: pd.DataFrame, df_current: pd.DataFrame) -> bool:
        """Check drift and trigger retraining if needed."""
        logger.info("Checking drift and evaluating retraining need...")

        drift_report = self.detector.detect_feature_drift(df_reference, df_current)

        if drift_report["retraining_required"] and self.config["monitoring"]["retraining_trigger"]:
            logger.info("🔄 AUTO-RETRAINING TRIGGERED")
            self._trigger_retraining(drift_report)
            return True
        else:
            logger.info("✅ No retraining needed")
            return False

    def _trigger_retraining(self, drift_report: Dict):
        """Trigger the retraining pipeline via Airflow or direct call."""
        trigger_log = {
            "timestamp": datetime.now().isoformat(),
            "trigger_reason": "drift_detected",
            "drifted_features": drift_report["drifted_features"],
            "status": "triggered"
        }

        Path("data/drift").mkdir(parents=True, exist_ok=True)
        with open("data/drift/retraining_trigger.json", "w") as f:
            json.dump(trigger_log, f, indent=2)

        logger.info("Retraining trigger logged. Airflow DAG will pick this up on next schedule.")


def simulate_production_drift(df_reference: pd.DataFrame, drift_magnitude: float = 0.3) -> pd.DataFrame:
    """Simulate production data drift for testing."""
    logger.info(f"Simulating production drift with magnitude {drift_magnitude}...")
    df_drifted = df_reference.copy()

    numeric_cols = ["time_in_hospital", "num_medications", "number_inpatient", "num_lab_procedures"]
    for col in numeric_cols:
        if col in df_drifted.columns:
            noise = np.random.normal(drift_magnitude * df_drifted[col].std(), 0.1, len(df_drifted))
            df_drifted[col] = (df_drifted[col] + noise).clip(lower=0)

    logger.info("Production drift simulation complete")
    return df_drifted


if __name__ == "__main__":
    import pandas as pd
    from src.ingestion.data_ingestion import generate_synthetic_data

    df_reference = generate_synthetic_data(5000)
    df_current = simulate_production_drift(df_reference, drift_magnitude=0.5)

    detector = DriftDetector()
    report = detector.detect_feature_drift(df_reference, df_current)

    print(f"\nDrift detected: {report['overall_drift_detected']}")
    print(f"Drifted features: {report['drifted_features']}")
    print(f"Retraining required: {report['retraining_required']}")
