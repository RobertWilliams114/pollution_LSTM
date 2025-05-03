## Pollution LSTM

![LSTM Prediction](./plots/prediction_viz.png)

# Air Pollution Forecasting with LSTM

A time series forecasting project that leverages Long Short-Term Memory (LSTM) neural networks to predict future air pollution levels, aiding environmental monitoring and decision-making.

---

## 📂 Project Structure

```bash
pollution_LSTM/
├── data/                     # Raw & processed CSV datasets
├── notebooks/                # Jupyter notebooks for exploration & modeling
├── models/                   # Saved model weights and artifacts
├── requirements.txt          # Python dependencies
├── preprocess.py             # Data cleaning & normalization scripts
├── train.py                  # Model training script
├── evaluate.py               # Evaluation & visualization scripts
└── README.md                 # This overview
```

---

## 🔨 Technologies & Why I Chose Them

- **Python & Jupyter Notebook**: Interactive data exploration and iterative development.
- **TensorFlow 2.x & Keras**: High-level APIs for building and training LSTM networks efficiently.
- **Pandas & NumPy**: Robust data manipulation and numerical operations.
- **Matplotlib & Seaborn**: Visualize time series and model performance metrics.
- **scikit-learn**: Preprocessing utilities and performance evaluation metrics.

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/RobertWilliams114/pollution_LSTM.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess data
python preprocess.py --input data/raw.csv --output data/processed.csv

# 4. Train the model
python train.py --data data/processed.csv --epochs 50

# 5. Evaluate & visualize
python evaluate.py --model models/lstm.h5 --data data/processed.csv
```

---

## 📊 Model Performance
- R-Squared value of PM2.5 = .88
- R-Squared value of PM10 = .81

---
