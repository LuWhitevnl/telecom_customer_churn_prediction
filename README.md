# Telecom Customer Churn Prediction (Machine Learning)

Dự án xây dựng quy trình Học máy (Machine Learning) dự đoán khả năng rời bỏ dịch vụ (Churn) của khách hàng viễn thông, đảm bảo **tính tái lập (reproducibility)** và **không rò rỉ dữ liệu (zero data leakage)** từ khâu tiền xử lý, huấn luyện Cross-Validation đến đánh giá tập Test và đóng gói mô hình triển khai.

---

## 1. Dữ liệu (Dataset)

- **Link tải dữ liệu:** [Tải bộ dữ liệu tại đây](https://www.kaggle.com/datasets/pranavchandaliya/telcom-churn-dataset?resource=download) *(Kaggle)*
- **Các file cần thiết:**
  1. `Telecom Data.csv` — Dữ liệu có nhãn (51.047 dòng, tỷ lệ Churn ~28.82%), được chia cố định 80% Train (`40.837` dòng) và 20% Test (`10.210` dòng) theo `stratify=y`, `random_state=42`.
  2. `Telecom - Prediction Data.csv` — Dữ liệu chưa gán nhãn (`20.000` khách hàng) dùng để kiểm thử suy luận thực tế sau khi đóng gói mô hình.

> **Lưu ý:** Sau khi tải về, đặt cả 2 file CSV vào đúng thư mục `data/raw/`.

---

## 2. Cấu trúc thư mục

```text
telecom-churn-ml/
├── data/
│   ├── raw/                            # Đặt 2 file CSV tải từ Kaggle tại đây
│   │   ├── Telecom Data.csv
│   │   └── Telecom - Prediction Data.csv
│   └── processed/
│       └── train_test_split_ids.csv    # Chỉ số ID phân chia tập Train (80%) / Test (20%) cố định
├── src/                                # Module dùng chung (chống lặp code & data leakage)
│   ├── __init__.py
│   ├── config.py                       # Hằng số, đường dẫn, 5-fold StratifiedKFold
│   ├── data.py                         # Làm sạch tất định, chia Train/Test, prepare_raw_input()
│   └── modeling.py                     # ColumnTransformer, pipeline CV, quét ngưỡng OOF
├── notebooks/
│   ├── 00_data_audit.ipynb             # Kiểm tra chất lượng dữ liệu gốc
│   ├── 01_eda.ipynb                    # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb          # Trực quan hóa & kiểm chứng tiền xử lý từ src/
│   ├── 03_logistic_regression.ipynb    # Thí nghiệm E1–E6: Logistic Regression
│   ├── 04_decision_tree.ipynb          # Thí nghiệm E1–E6: Decision Tree
│   ├── 05_random_forest.ipynb          # Thí nghiệm E1–E6: Random Forest
│   ├── 06_svm.ipynb                    # Thí nghiệm E1–E6: Support Vector Machine (LinearSVC)
│   └── 07_comparison.ipynb             # So sánh CV -> Chấm Test 1 lần -> Đóng gói models/
├── models/                             # Đóng gói mô hình và metadata
│   ├── churn_pipeline.joblib           # Pipeline hoàn chỉnh (preprocessor + model thắng cuộc)
│   └── model_meta.json                 # Danh sách cột, ngưỡng tối ưu, kết quả CV/Test
├── results/
│   ├── metrics/                        # Các file JSON kết quả CV (03–06) và bảng Test (07)
│   ├── figures/                        # Biểu đồ EDA & kết quả thực nghiệm
│   └── predictions/                    # File dự đoán 20.000 khách hàng (telecom_predictions.csv - gitignore)
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 3. Cài đặt môi trường

Yêu cầu **Python 3.10+**. Mở terminal tại thư mục gốc `telecom-churn-ml/` và chạy:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Hướng dẫn chạy dự án

Mở dự án bằng **VS Code** hoặc **Jupyter**, chọn đúng kernel **`.venv`** của dự án, sau đó bấm **Restart & Run All** theo đúng thứ tự:

1. **Khám phá & kiểm chứng tiền xử lý:**
* `notebooks/00_data_audit.ipynb`
* `notebooks/01_eda.ipynb`
* `notebooks/02_preprocessing.ipynb` *(kiểm chứng làm sạch 53 đặc trưng, ma trận sau `ColumnTransformer`, 0 giá trị `NaN`/`Inf`)*


2. **Huấn luyện & thí nghiệm 4 mô hình (chạy độc lập trên tập Train, có thể chia 4 thành viên chạy song song):**
* `notebooks/03_logistic_regression.ipynb` $\rightarrow$ xuất `results/metrics/03_logistic_regression.json`
* `notebooks/04_decision_tree.ipynb` $\rightarrow$ xuất `results/metrics/04_decision_tree.json`
* `notebooks/05_random_forest.ipynb` $\rightarrow$ xuất `results/metrics/05_random_forest.json`
* `notebooks/06_svm.ipynb` $\rightarrow$ xuất `results/metrics/06_svm.json`


3. **So sánh, đánh giá Hold-out Test & đóng gói thành phẩm:**
* Khi `results/metrics/` đã có đủ 4 file JSON (`03`–`06`), chạy `notebooks/07_comparison.ipynb`.
* Notebook sẽ tự động chọn mô hình thắng cuộc dựa trên CV, đánh giá **1 lần duy nhất** trên tập Test (`10.210` dòng), xuất `models/churn_pipeline.joblib` + `models/model_meta.json`, và chạy dự đoán ra `results/predictions/telecom_predictions.csv`.



---

## 5. Quy chuẩn kỹ thuật áp dụng

* **Chống Data Leakage:** Loại bỏ 5 cột (`CustomerID`, `Churn`, `RetentionCalls`, `RetentionOffersAccepted`, `MadeCallToRetentionTeam`). Mọi phép biến đổi thống kê (`SimpleImputer`, `StandardScaler`, `MissingIndicator`, `OneHotEncoder`, `SelectPercentile`, `SMOTE`) đều nằm trong `Pipeline` và chỉ `fit` trên training fold.
* **Hệ thống thí nghiệm thống nhất (E1–E6):**
* **E1:** Baseline 5-fold Stratified CV (`F1`, `Recall`, `Precision`, `ROC-AUC`, `PR-AUC`).
* **E2:** Hyperparameter Tuning (`GridSearchCV` / `RandomizedSearchCV`).
* **E3:** Xử lý mất cân bằng lớp (`class_weight="balanced"` vs `SMOTE`).
* **E4:** Chọn lọc đặc trưng (`SelectPercentile` top 60%).
* **E5:** Phân tích độ ổn định qua 5 fold.
* **E6:** Khảo sát ngưỡng phân loại tối ưu trên xác suất out-of-fold (OOF) của tập Train.

---

## 6. Kết quả thực nghiệm trên tập Hold-out Test (10.210 dòng)

Sau khi kiểm chứng chéo 5-fold (5-fold Stratified CV) trên tập Huấn luyện và quét ngưỡng phân loại tối ưu (OOF Threshold Tuning), các mô hình được đánh giá **1 lần duy nhất** trên tập Hold-out Test độc lập:

| Mô hình | Ngưỡng (Threshold) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Winner)** | **0.45** | **47.97%** | **33.93%** | **85.04%** | **0.4851** | **0.6480** | **0.4146** |
| Decision Tree | 0.40 | 48.22% | 33.66% | 82.09% | 0.4774 | 0.6164 | 0.3666 |
| Logistic Regression | 0.45 | 48.88% | 33.34% | 77.43% | 0.4661 | 0.6132 | 0.3725 |
| Support Vector Machine (LinearSVC) | 0.25 | 48.53% | 33.24% | 77.97% | 0.4661 | 0.6118 | 0.3701 |

> **Nhận xét:**
> - **Random Forest** đạt hiệu quả tổng thể cao nhất với **ROC-AUC = 0.6480**, **PR-AUC = 0.4146** và **Recall = 85.04%** tại ngưỡng $0.45$.
> - Trong bài toán giữ chân khách hàng (Retention), chỉ số **Recall** đóng vai trò tối quan trọng để bắt trúng tối đa khách hàng có nguy cơ rời mạng viễn thông. Do đó, pipeline Random Forest được chọn làm giải pháp chính thức và đóng gói tại `models/churn_pipeline.joblib`.


