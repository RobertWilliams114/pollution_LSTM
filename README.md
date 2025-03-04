# Pollution LSTM

This repository contains an LSTM (Long Short-Term Memory) model for forecasting pollution levels. The project utilizes time series data to predict future pollution metrics and is designed to aid environmental monitoring and analysis.

## Features
- **Time Series Forecasting:** Uses LSTM neural networks to predict pollution levels.
- **Data Preprocessing:** Includes scripts for data cleaning and normalization.
- **Model Evaluation:** Compares predicted values with actual data.
- Visualization:** Generates plots to visualize predictions versus real measurements.

## Technologies Used
- Python 3.7+
- Deep Learning Framework (TensorFlow/Keras or PyTorch)
- Data Manipulation Libraries (Pandas, NumPy)
- Visualization (Matplotlib)

## Dataset
This project uses a pollution dataset (e.g., from [here](https://www.kaggle.com/datasets/bappekim/air-pollution-in-seoul)). Ensure that the dataset is placed in the correct directory or update the data path in the code accordingly.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Required libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RobertWilliams114/pollution_LSTM.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pollution_LSTM
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Data Preprocessing: Run the preprocessing script if needed:
   ```bash
   python preprocess.py
   ```
2. Training the Model: Train the LSTM model by executing:
   ```bash
   python train.py
   ```
3. Evaluation: Generate predictions and evaluate the model:
   ```bash
   python evaluate.py
   ```
4. Visualization: The evaluation script will produce plots comparing actual vs. predicted pollution levels.

## Thank You
Thank you for reading. I hope that you learned something new or gained something of value.
