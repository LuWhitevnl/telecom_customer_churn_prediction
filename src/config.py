from pathlib import Path
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT / "data" / "raw" / "Telecom Data.csv"
PRED_DATA_PATH = ROOT / "data" / "raw" / "Telecom - Prediction Data.csv"
SPLIT_IDS_PATH = ROOT / "data" / "processed" / "train_test_split_ids.csv"

METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = "CustomerID"
TARGET_COL = "Churn"
DROP_COLS = [
    "CustomerID",
    "Churn",
    "RetentionCalls",
    "RetentionOffersAccepted",
    "MadeCallToRetentionTeam",
]

SCORING = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "ap": "average_precision",
}


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)