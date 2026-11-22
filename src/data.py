import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    SPLIT_IDS_PATH,
    ID_COL,
    TARGET_COL,
    DROP_COLS,
    TEST_SIZE,
    RANDOM_STATE,
)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch tất định (không fit thống kê) để đảm bảo 0% data leakage."""
    X = df.drop(columns=DROP_COLS, errors="ignore").copy()
    if "HandsetPrice" in X.columns:
        X["HandsetPrice"] = pd.to_numeric(X["HandsetPrice"], errors="coerce")
    return X


def load_train_test():
    """Trả về X_train, X_test, y_train, y_test, num_cols, cat_cols đồng nhất toàn nhóm."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc: {RAW_DATA_PATH}")

    raw = pd.read_csv(RAW_DATA_PATH)
    raw[ID_COL] = raw[ID_COL].astype(str)

    y_all = raw[TARGET_COL].map({"No": 0, "Yes": 1})
    assert y_all.notna().all(), "Cột Churn chứa giá trị ngoài 'No'/'Yes'!"
    y_all = y_all.astype(int)

    if SPLIT_IDS_PATH.exists():
        split_df = pd.read_csv(SPLIT_IDS_PATH)
        split_df[ID_COL] = split_df[ID_COL].astype(str)
        train_ids = split_df.loc[split_df["split"] == "train", ID_COL].values
        test_ids = split_df.loc[split_df["split"] == "test", ID_COL].values

        indexed = raw.set_index(ID_COL)
        train_df = indexed.loc[train_ids]
        test_df = indexed.loc[test_ids]
    else:
        train_idx, test_idx = train_test_split(
            raw.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_all,
        )
        SPLIT_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        split_out = pd.DataFrame({
            ID_COL: raw[ID_COL],
            "split": np.where(raw.index.isin(train_idx), "train", "test"),
        })
        split_out.to_csv(SPLIT_IDS_PATH, index=False)

        train_df = raw.loc[train_idx].set_index(ID_COL)
        test_df = raw.loc[test_idx].set_index(ID_COL)

    X_train = clean_features(train_df)
    y_train = train_df[TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)
    X_test = clean_features(test_df)
    y_test = test_df[TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)

    assert X_train.index.intersection(X_test.index).empty, "Train và Test bị trùng CustomerID!"

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
    return X_train, X_test, y_train, y_test, num_cols, cat_cols


def prepare_raw_input(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Chuẩn hóa bảng dữ liệu thô (CSV / API) trước khi gọi pipeline.predict_proba()."""
    X_in = df.reindex(columns=num_cols + cat_cols).copy()
    for col in num_cols:
        X_in[col] = pd.to_numeric(X_in[col], errors="coerce")
    return X_in