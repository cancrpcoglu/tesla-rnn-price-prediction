# Tesla Stock Price Prediction using RNN

## 📌 Project Overview

This project uses a Recurrent Neural Network (RNN) architecture to predict the future stock prices of Tesla Inc. by analyzing historical time series data. The goal is to explore sequential modeling and understand how RNNs perform in financial data forecasting.

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- Tesla stock CSV data (sourced from Kaggle)

## 🗂️ Dataset

- Source: [Tesla historical stock price CSV](https://www.kaggle.com/code/serkanp/tesla-stock-price-prediction)
- Used Features: `Open`, `High`, `Low`, `Close`, `Volume`
- Data Split: 70% Training / 30% Testing

## 🧠 Model Details

- Implemented a standard RNN model using `SimpleRNN` layers
- Applied time series preprocessing (normalization, sequence windowing)
- Tuned epochs, learning rate, batch size for better performance
- Used `EarlyStopping` to prevent overfitting

## 📊 Results

- Visualized training and validation loss over epochs
- Compared predicted and actual prices using line graphs

## 🚀 How to Run
pip install -r requirements.txt

python rnn_model.py

## ✍️ Author
Can Çorapçıoğlu
[GitHub](https://github.com/cancrpcoglu) | [LinkedIn](https://www.linkedin.com/in/can-%C3%A7orap%C3%A7%C4%B1o%C4%9Flu-15a340247/)
