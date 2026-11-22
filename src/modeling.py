import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import METRICS_DIR, SCORING


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    min_frequency: int = 50,
) -> ColumnTransformer:
    """Trả về ColumnTransformer CHƯA fit — luôn clone() để fit riêng trong từng fold CV."""
    num_pipe = FeatureUnion([
        ("scaled_num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])),
        ("missing_ind", MissingIndicator(features="missing-only", error_on_new=False)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(model)),
    ])


def evaluate_cv(pipe, X_train, y_train, cv):
    res = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        error_score="raise",
    )
    summary = pd.DataFrame({
        "metric": list(SCORING),
        "mean": [res[f"test_{k}"].mean() for k in SCORING],
        "std": [res[f"test_{k}"].std() for k in SCORING],
    }).round(4)
    return res, summary


def ensure_proba_model(model):
    m = clone(model)
    if hasattr(m, "predict_proba"):
        return m
    return CalibratedClassifierCV(estimator=m, method="sigmoid", cv=3)


def scan_oof_thresholds(
    preprocessor: ColumnTransformer,
    model,
    X_train,
    y_train,
    cv,
    start: float = 0.15,
    stop: float = 0.76,
    step: float = 0.05,
):
    proba_pipe = make_pipeline(preprocessor, ensure_proba_model(model))
    oof_prob = cross_val_predict(
        proba_pipe,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    rows = []
    for th in np.arange(start, stop, step):
        pred = (oof_prob >= th).astype(int)
        rows.append({
            "threshold": round(float(th), 2),
            "precision": precision_score(y_train, pred, zero_division=0),
            "recall": recall_score(y_train, pred, zero_division=0),
            "f1": f1_score(y_train, pred, zero_division=0),
        })

    df = pd.DataFrame(rows).round(4)
    best_row = df.loc[df["f1"].idxmax()]
    return df, float(best_row["threshold"]), float(best_row["f1"])


def compute_test_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
    }


def save_cv_report(filename: str, payload: dict):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / filename
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path