# Data-Science-Forecasting-System

# End-to-End Time Series Forecasting System 📈

A production-grade forecasting engine designed to predict 8-week sales trends across multiple regions. This system automates the entire lifecycle: from raw data ingestion and feature engineering to automated model selection and REST API deployment.

## 🚀 Key Features
* **Automated Model Selection:** Evaluates **SARIMA**, **Facebook Prophet**, **XGBoost**, and **LSTM** for every state and selects the champion model based on Mean Absolute Error (MAE).
* **Advanced Feature Engineering:** Implements time-dependent features including:
    * **Lags:** $t-1$, $t-7$, $t-30$ to capture autocorrelation.
    * **Window Statistics:** Rolling mean and standard deviation for volatility tracking.
    * **Temporal Features:** Day of week, month, and holiday flags to handle seasonality.
* **Production API:** A high-performance **FastAPI** backend that serves 8-week-ahead forecasts with sub-millisecond latency.
* **Modern Frontend:** Interactive visualization dashboard using Chart.js to present forecasts to stakeholders.

---

## 🛠️ Technical Stack
* **Languages:** Python 3.9+
* **Forecasting:** Statsmodels (SARIMA), Prophet, XGBoost, PyTorch (LSTM)
* **Backend:** FastAPI, Uvicorn
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Artifact Management:** Joblib (Model Registry)

---

## 📁 Project Structure
```bash
├── artifacts/               # Saved model registry and prediction CSVs
├── src/
│   ├── data.py              # Preprocessing & cleaning pipeline
│   ├── features.py          # Engineering lag & rolling features
│   ├── models.py            # Model architectures (Classic to Deep Learning)
│   ├── train.py             # Orchestration & automated model selection
│   └── api.py               # REST API implementation
├── index.html               # Frontend dashboard
├── data.xlsx                # Historical sales data
└── requirements.txt         # Dependency manifest
```

---

## ⚡ Execution Guide

### 1. Training & Selection
The [train.py](https://github.com/veena-madhuri123/Data-Science-end-to-end-Time-Series-Forecasting-System-with-API-/blob/main/train.py) script handles the heavy lifting. It splits data using time-series logic (preventing data leakage), trains the four mandatory models, and benchmarks them.
```bash
python train.py
```

### 2. API Deployment
Once training is complete, the best-performing models are serialized. Serve the predictions via the REST API:
```bash
uvicorn api:app --reload
```
*Access documentation at:* `http://localhost:8000/docs`

---

## 📊 Evaluation Logic
The system uses a **Time-Series Holdout Validation** strategy. For each state, the last 8 weeks of data are reserved for validation. The model achieving the lowest MAE is promoted to production.

| Model | Strengths |
| :--- | :--- |
| **SARIMA** | Captures linear trends and seasonal cycles. |
| **Prophet** | Robust to missing data and shifts in trend. |
| **XGBoost** | Captures non-linear relationships via lag features. |
| **LSTM** | Learns complex long-term dependencies in the data. |

---

## 👤 Author
**[Your Name]**
* **Role:** GDG On Campus Organizer | Microsoft Gold Ambassador
* **Focus:** AI/ML, Cloud Architecture, & Open Source

---

### Pro-Tip for the Apprenticeship:
When you record your video, emphasize that you used **XGBoost with manual lag features** and **LSTM** to show you can handle both traditional and modern deep learning approaches. This "Hybrid" knowledge is what recruiters look for!

How does this look? If you want, I can also draft a short script for your video to go along with this.
