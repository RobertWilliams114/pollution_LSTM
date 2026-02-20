# Time-Series Air Quality Forecasting System

A multi-output LSTM neural network that simultaneously forecasts 6 air pollutants (SO2, NO2, O3, CO, PM10, PM2.5) from Seoul metropolitan monitoring stations, trained on 647K+ hourly temporal sequences.

Built with **TensorFlow/Keras**, **Pandas**, **scikit-learn**, and **Matplotlib**.

---

## What It Does

Air quality monitoring stations collect hourly readings across multiple pollutants, but predicting future levels requires capturing complex temporal patterns — seasonal cycles, rush-hour spikes, and weather-driven fluctuations. This project builds a single LSTM model that takes 24 hours of historical pollution data and predicts the next hour's levels for all 6 pollutants simultaneously.

---

## Architecture

```
Input: 24-hour pollution sequence
┌─────────────────────────────────┐
│  X = (batch, 24 timesteps, 6)  │   6 pollutants x 24 hours
└──────────────┬──────────────────┘
               │
       ┌───────▼───────┐
       │   LSTM Layer   │   50 hidden units
       │  (return last) │   Captures temporal dependencies
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │  Dense Layer   │   6 output neurons (one per pollutant)
       │   (linear)     │   Simultaneous multi-output regression
       └───────┬───────┘
               │
┌──────────────▼──────────────────┐
│  y = (batch, 6)                 │   Next-hour prediction for all 6
└─────────────────────────────────┘
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Epochs | 20 |
| Batch Size | 32 |
| Validation Split | 10% of training data |
| Sequence Length | 24 timesteps (hours) |

---

## Dataset

**Source:** Seoul metropolitan area air quality monitoring stations

| Attribute | Details |
|---|---|
| Total sequences | 647,463 (517,984 train / 129,479 test) |
| Time granularity | Hourly measurements |
| Monitoring stations | Multiple stations across Seoul |
| Features per timestep | 6 pollutant concentrations |
| Date range | 2017 onwards |

**Pollutants tracked:**

| Pollutant | Type | Description |
|---|---|---|
| SO2 | Gas | Sulfur dioxide |
| NO2 | Gas | Nitrogen dioxide |
| O3 | Gas | Ground-level ozone |
| CO | Gas | Carbon monoxide |
| PM10 | Particulate | Coarse particulate matter (≤10μm) |
| PM2.5 | Particulate | Fine particulate matter (≤2.5μm) |

---

## Model Performance

### R² Scores by Pollutant

| Pollutant | R² Score | Rating |
|---|---|---|
| **SO2** | **0.9310** | Excellent |
| **NO2** | **0.8990** | Excellent |
| **O3** | **0.8606** | Very Good |
| CO | 0.3726 | Poor |
| PM10 | 0.3797 | Poor |
| PM2.5 | 0.2443 | Poor |
| **Overall** | **0.6145** | — |

### Test Set Metrics

| Metric | Value |
|---|---|
| Test Loss (MSE) | 0.0001 |
| Test MAE | 0.0010 |

**Key insight:** The model excels at forecasting gaseous pollutants (SO2, NO2, O3) which follow smoother temporal patterns driven by traffic and industrial cycles. Particulate matter (PM10, PM2.5) is harder to predict because it depends heavily on external factors like wind, weather fronts, and construction activity that aren't captured in the input features.

---

## Data Pipeline

The preprocessing pipeline was designed to prevent data leakage — a common pitfall in time-series ML projects:

```
Raw CSV (Measurement_summary.csv)
    │
    ├── 1. Feature engineering
    │       Extract: Hour, Day of Week, Month
    │       Preserve: Latitude, Longitude, Station Code
    │
    ├── 2. Temporal train/test split (80/20)
    │       NO shuffling — preserves time ordering
    │       Split happens BEFORE any scaling
    │
    ├── 3. MinMaxScaler fit on TRAINING data only
    │       Test data transformed with training scaler
    │       Prevents future information from leaking into training
    │
    └── 4. Sliding window sequence generation
            Window size: 24 timesteps
            Stride: 1 (overlapping sequences)
            X: hours [t-23, t-22, ..., t]
            y: hour [t+1] (all 6 pollutants)
```

**Why this matters:** A naive approach (scaling all data before splitting, or random train/test splits) artificially inflates R² scores by letting the model "see" future statistical distributions during training. The temporal split ensures the model is evaluated on genuinely unseen future data.

---

## Getting Started

### Prerequisites
- Python 3.10+
- ~2GB RAM for training

### Installation

```bash
# Clone the repo
git clone https://github.com/RobertWilliams114/pollution_LSTM.git
cd pollution_LSTM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install pandas tensorflow scikit-learn numpy matplotlib
```

### Running the Notebook

```bash
jupyter notebook pollution_LSTM.ipynb
```

Run all cells sequentially. The notebook will:
1. Load and preprocess `Measurement_summary.csv`
2. Generate temporal sequences (24-hour windows)
3. Train the LSTM model (~13 minutes on CPU)
4. Evaluate and display R² scores per pollutant
5. Generate comparison plots (true vs. predicted)
6. Save the trained model to `multi_output_lstm_model.h5`

---

## Visualizations

The notebook generates per-pollutant comparison plots showing true vs. predicted values across the test set. Gaseous pollutants (SO2, NO2, O3) track the actual values closely, while particulate matter predictions are visibly smoother and less responsive to spikes.

---

## Project Structure

```
pollution_LSTM/
├── pollution_LSTM.ipynb              # Main notebook (preprocessing, training, evaluation)
├── Measurement_summary.csv           # Raw dataset (Seoul air quality stations)
├── organized_air_pollution_data.csv  # Preprocessed dataset with temporal features
├── multi_output_lstm_model.h5        # Trained model weights (HDF5 format)
└── README.md
```

---

## Roadmap

### Potential Improvements
- [ ] Add weather features (temperature, humidity, wind speed) to improve PM2.5/PM10 predictions
- [ ] Experiment with deeper architectures (stacked LSTMs, bidirectional LSTM)
- [ ] Implement attention mechanism for variable-length sequence weighting
- [ ] Add station-level embeddings to capture geographic differences
- [ ] Switch to native Keras `.keras` format for model persistence
- [ ] Add hyperparameter tuning (learning rate, hidden units, sequence length)
- [ ] Deploy as a REST API for real-time pollution forecasting

---

## Technologies

- **Python & Jupyter Notebook** — Interactive data exploration and iterative development
- **TensorFlow 2.x & Keras** — High-level APIs for building and training LSTM networks
- **Pandas & NumPy** — Data manipulation and numerical operations
- **Matplotlib & Seaborn** — Time-series visualization and model performance plots
- **scikit-learn** — MinMaxScaler preprocessing and R² evaluation metrics

---

## Acknowledgments

- Seoul metropolitan government for air quality monitoring data
- TensorFlow/Keras team for the LSTM implementation
