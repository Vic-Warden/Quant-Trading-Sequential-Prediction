# Quant-Trading-Sequential-Prediction

REST API for sequential price prediction and trading signal generation using a Stacked LSTM model on OHLCV time-series data.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Model | TensorFlow / Keras (Stacked LSTM) |
| Feature Engineering | Pandas, NumPy, TA-Lib |
| Data Source | yfinance (BTC-USD) |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker / Docker Compose |

---

## Project Structure

```
├── api/
│   ├── main.py          # FastAPI app & /signal endpoint
│   ├── predict.py       # Inference logic (LSTM + MinMaxScaler)
│   ├── schemas.py       # Pydantic input schemas
│   └── models/          # Saved model (.h5) and scaler (.pkl)
├── data/
│   ├── get_data.py      # yfinance ingestion script
│   ├── raw/             # Raw OHLCV data
│   └── processed/       # Feature-engineered & backtest data
├── notebooks/
│   ├── Data_Exploration.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Model_Training_LSTM.ipynb
│   └── Backtesting.ipynb
├── dashboard/
│   └── app.py           # Streamlit dashboard
├── Dockerfile
└── docker-compose.yml
```

---

## API

### `POST /signal`

**Input:** A sequence of exactly 30 OHLCV + indicator records.

```json
{
  "sequence": [
    {
      "Open": 60000, "High": 61000, "Low": 59500, "Close": 60500,
      "Volume": 1500000000,
      "SMA_20": 59000, "SMA_100": 55000,
      "RSI_14": 58.3,
      "BB_UPPER": 62000, "BB_MIDDLE": 59000, "BB_LOWER": 56000
    }
  ]
}
```

**Output:**

```json
{
  "prediction_probability": 0.72,
  "trading_signal": 1,
  "interpretation": "Buy",
  "threshold": 0.40
}
```

Signal `1` = Buy, `0` = Hold/Sell. Threshold: `0.40`.

---

## Quickstart

### Docker (recommended)

```bash
docker-compose up --build
```

API available at `http://localhost:8000` — docs at `http://localhost:8000/docs`.

### Local

```bash
# Install TA-Lib (macOS)
brew install ta-lib

pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Model & Features

- **Architecture:** Stacked LSTM (window size = 30 days)
- **Task:** Binary classification — price direction (Up / Down)
- **Features:** `Open`, `High`, `Low`, `Close`, `Volume`, `SMA_20`, `SMA_100`, `RSI_14`, `BB_UPPER`, `BB_MIDDLE`, `BB_LOWER`
- **Preprocessing:** `MinMaxScaler` (fitted on training set, saved as `scaler.pkl`)

---

## Backtesting Results (BTC-USD)

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.88 |
| Max Drawdown | 18.66% |

**Known limitation:** The model is biased toward the underlying bullish trend, making performance close to Buy-and-Hold. Planned improvements: Temporal Convolutional Networks (TCN), macro-economic features, sentiment analysis (FinBERT).