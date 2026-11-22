import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ID_COL
from src.data import prepare_raw_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
META_PATH = ROOT / "models" / "model_meta.json"

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.error("Không tìm thấy `models/churn_pipeline.joblib` hoặc `models/model_meta.json`. Hãy chạy `07_comparison.ipynb` trước.")
        st.stop()
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return pipe, meta


pipe, meta = load_artifacts()
default_th = float(meta["threshold"])
test_m = meta.get("test_metrics", {})

# --- SIDEBAR: Thông tin mô hình ---
with st.sidebar:
    st.header("Thông tin mô hình")
    st.markdown(f"**Mô hình:** `{meta['model_name']}`")
    st.markdown(f"**Ngưỡng OOF mặc định:** `{default_th:.2f}`")

    threshold = st.slider(
        "Điều chỉnh ngưỡng phân loại (Threshold)",
        min_value=0.10,
        max_value=0.90,
        value=default_th,
        step=0.01,
        help="Mặc định dùng ngưỡng OOF tối ưu đã chốt trên tập Train.",
    )

    if test_m:
        st.divider()
        st.subheader("Kết quả Hold-out Test")
        c1, c2 = st.columns(2)
        c1.metric("PR-AUC", f"{test_m.get('pr_auc', 0):.4f}")
        c2.metric("ROC-AUC", f"{test_m.get('roc_auc', 0):.4f}")
        c1.metric("F1-Score", f"{test_m.get('f1', 0):.4f}")
        c2.metric("Recall", f"{test_m.get('recall', 0):.4f}")

# --- HEADER ---
st.title("📡 Dự đoán Khách hàng Rời mạng (Telecom Churn)")
st.caption("Pipeline tự động tiền xử lý (`ColumnTransformer`) và suy luận từ mô hình tốt nhất đã huấn luyện.")

tab_csv, tab_single = st.tabs(["📂 Dự đoán hàng loạt (CSV)", "👤 Dự đoán 1 khách hàng"])

# ==========================================
# TAB 1: UPLOAD CSV HÀNG LOẠT
# ==========================================
with tab_csv:
    uploaded = st.file_uploader(
        "Tải lên file CSV dữ liệu khách hàng (định dạng giống Telecom - Prediction Data.csv)",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Không đọc được file CSV: {e}")
            st.stop()

        with st.spinner("Đang tiền xử lý và dự đoán..."):
            X_in = prepare_raw_input(raw_df, meta["num_cols"], meta["cat_cols"])
            probs = pipe.predict_proba(X_in)[:, 1]
            preds = np.where(probs >= threshold, "Yes", "No")

        out_df = pd.DataFrame({
            ID_COL: raw_df[ID_COL] if ID_COL in raw_df.columns else range(1, len(raw_df) + 1),
            "churn_probability": np.round(probs, 4),
            "prediction": preds,
        })

        n_total = len(out_df)
        n_churn = int((preds == "Yes").sum())
        churn_rate = n_churn / n_total * 100

        k1, k2, k3 = st.columns(3)
        k1.metric("Tổng khách hàng", f"{n_total:,}")
        k2.metric("Dự đoán rời mạng (Yes)", f"{n_churn:,} ({churn_rate:.1f}%)")
        k3.metric("Xác suất Churn trung bình", f"{probs.mean():.2%}")

        col_chart, col_table = st.columns([1.1, 1.3])
        with col_chart:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.hist(probs, bins=35, color="#3182ce", edgecolor="white", alpha=0.85)
            ax.axvline(threshold, color="#e53e3e", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
            ax.set_title("Phân bố xác suất rời mạng")
            ax.set_xlabel("Churn Probability")
            ax.set_ylabel("Số lượng khách hàng")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col_table:
            st.markdown("**Top khách hàng có nguy cơ rời mạng cao nhất**")
            st.dataframe(
                out_df.sort_values("churn_probability", ascending=False).head(50),
                use_container_width=True,
                height=340,
            )

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Tải về kết quả (churn_predictions.csv)",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
            type="primary",
        )

# ==========================================
# TAB 2: DỰ ĐOÁN 1 KHÁCH HÀNG
# ==========================================
with tab_single:
    st.info("Nhập các chỉ số chính của khách hàng. Các cột còn lại sẽ tự động được `SimpleImputer` trong pipeline xử lý theo đúng phân phối tập Train.")

    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)
        monthly_rev = c1.number_input("MonthlyRevenue ($)", min_value=0.0, value=55.0, step=1.0)
        monthly_min = c2.number_input("MonthlyMinutes", min_value=0.0, value=450.0, step=10.0)
        recurring = c3.number_input("TotalRecurringCharge ($)", min_value=0.0, value=45.0, step=1.0)

        months_in_service = c1.number_input("MonthsInService (tháng)", min_value=0, value=18, step=1)
        equip_days = c2.number_input("CurrentEquipmentDays (ngày)", min_value=0, value=365, step=10)
        care_calls = c3.number_input("CustomerCareCalls", min_value=0.0, value=1.5, step=0.5)

        overage_min = c1.number_input("OverageMinutes", min_value=0.0, value=25.0, step=5.0)
        handset_price = c2.number_input("HandsetPrice ($)", min_value=0.0, value=80.0, step=10.0)
        credit_rating = c3.selectbox(
            "CreditRating",
            ["1-Highest", "2-High", "3-Good", "4-Medium", "5-Low", "6-VeryLow", "7-Lowest"],
            index=1,
        )

        submitted = st.form_submit_button("🔍 Dự đoán ngay", type="primary")

    if submitted:
        row = {
            "MonthlyRevenue": monthly_rev,
            "MonthlyMinutes": monthly_min,
            "TotalRecurringCharge": recurring,
            "MonthsInService": months_in_service,
            "CurrentEquipmentDays": equip_days,
            "CustomerCareCalls": care_calls,
            "OverageMinutes": overage_min,
            "HandsetPrice": handset_price,
            "CreditRating": credit_rating,
        }
        X_one = prepare_raw_input(pd.DataFrame([row]), meta["num_cols"], meta["cat_cols"])
        prob = float(pipe.predict_proba(X_one)[0, 1])
        pred = "Yes" if prob >= threshold else "No"

        st.divider()
        r1, r2 = st.columns(2)
        r1.metric("Xác suất rời mạng (Churn Probability)", f"{prob:.2%}")
        if pred == "Yes":
            r2.error(f"**Kết luận: CÓ NGUY CƠ RỜI MẠNG (Yes)** — vượt ngưỡng {threshold:.2f}")
        else:
            r2.success(f"**Kết luận: AN TOÀN (No)** — dưới ngưỡng {threshold:.2f}")