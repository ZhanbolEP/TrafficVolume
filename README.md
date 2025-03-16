# 🚗 Traffic Volume Prediction with LSTM

This project uses **deep learning (LSTM)** to predict **hourly traffic volume** on I-94 in Minnesota, USA. The model leverages **weather**, **holiday**, and **time-based features** to understand and forecast vehicle flow, helping improve traffic management, planning, and safety.

---

## 📂 Dataset Overview

The dataset is provided by the **UCI Machine Learning Repository** and includes:

| Column                        | Type         | Description                                         |
|------------------------------|--------------|-----------------------------------------------------|
| `temp`                       | Numeric      | Temperature in Kelvin                              |
| `rain_1h`, `snow_1h`         | Numeric      | Precipitation in mm                                |
| `clouds_all`                 | Numeric      | Cloud coverage percentage                          |
| `date_time`                  | DateTime     | Hour of data collection (CST)                      |
| `holiday_` (11 columns)      | Categorical  | US holidays and Minnesota State Fair               |
| `weather_main_` (11 columns) | Categorical  | Weather summary (one-hot encoded)                  |
| `weather_description_` (35 columns) | Categorical | Detailed weather (one-hot encoded)          |
| `hour_of_day`, `day_of_week`, `day_of_month`, `month` | Numeric | Time indicators                        |
| `traffic_volume`             | Numeric      | **Target** – hourly traffic volume                 |

Data has been preprocessed and normalized into:
- `train_scaled.csv`
- `test_scaled.csv`

---

## 🧠 Model Architecture

- **Model**: 2-layer **LSTM** with `66` input features, `64` hidden units  
- **Activation**: LeakyReLU  
- **Loss Function**: MSE (Mean Squared Error)  
- **Optimizer**: Adam (learning rate = 0.0001)

---

## 🔧 Steps to Run

1. **Load and preprocess data**  
2. **Create sequences** of 12-hour time windows  
3. **Prepare DataLoaders** for efficient training  
4. **Define and train LSTM model** (2 epochs)  
5. **Evaluate** using Test MSE  
6. **Plot predictions vs actual values**

---

## 📈 Sample Output

- **Train Loss**: Decreases each epoch (e.g., 0.07 → 0.06)  
- **Test MSE**: e.g., `0.072`  
- **Prediction Plot**: Visual comparison of predicted vs real traffic volume

---

## 🚀 Future Improvements

- Increase training epochs for better accuracy  
- Add **validation set** and use **early stopping**  
- Experiment with **other architectures** (GRU, CNN-LSTM)  
- Deploy model using a web app (e.g., **Streamlit** or **Flask**)

---

## 📎 Dependencies

- Python 3.x  
- PyTorch  
- Pandas, NumPy, Matplotlib
