import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

print("\n\n")
# -------------------------------
# 1. โหลดและเตรียมข้อมูล
# -------------------------------
data_path = "data/XAUUSD/XAUUSD1440_formatted.csv"
data = pd.read_csv(data_path)

data.rename(
    columns={
        "timestamp": "Date",
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
    },
    inplace=True,
)
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# ✅ ใช้เฉพาะคอลัมน์ที่เกี่ยวข้องกับราคา
data_prices = data[["Open", "Close", "High", "Low"]]

# ✅ สร้าง Scaler แยกเฉพาะสำหรับราคาทองคำ
scaler_price = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler_price.fit_transform(data_prices)


def create_dataset(data, look_back=90):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i])
        y.append(data[i])
    return np.array(X), np.array(y)


look_back = 60
X, y = create_dataset(scaled_prices, look_back)

ts_split = TimeSeriesSplit(n_splits=5)
for train_index, test_index in ts_split.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# ✅ ตรวจสอบขนาดของข้อมูล
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


def build_lstm_model(input_shape, learning_rate=0.0003):
    model = Sequential(
        [
            Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.2),
            Dense(50, activation="relu"),
            Dense(4),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error"
    )
    return model


model_path = "saved_model/XAUUSD_LSTM_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")
else:
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Model Input Shape: {input_shape}")  # ✅ ตรวจสอบขนาด input
    model = build_lstm_model(input_shape)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
    model_checkpoint = ModelCheckpoint(
        filepath=model_path, save_best_only=True, monitor="val_loss", verbose=1
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=2,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler, model_checkpoint],
        verbose=1,
    )
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss during Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# ✅ ป้องกันปัญหาขนาดข้อมูลไม่ตรง
if X_test.shape[2] != X_train.shape[2]:
    X_test = X_test[:, :, : X_train.shape[2]]

predictions = model.predict(X_test)
predictions = scaler_price.inverse_transform(predictions)
y_test_actual = scaler_price.inverse_transform(y_test)


def generate_trade_signal(predicted_prices, actual_prices):
    return [
        "BUY" if predicted_prices[i, 1] > actual_prices[i, 1] else "SELL"
        for i in range(len(predicted_prices))
    ]


def compute_tp_sl(predicted_prices, actual_prices):
    atr = np.mean(
        np.abs(actual_prices[:, 2] - actual_prices[:, 3])
    )  # ATR คำนวณจาก High - Low
    tp_values = predicted_prices[:, 1] + (atr * 1.5)
    sl_values = predicted_prices[:, 1] - (atr * 1.5)
    return tp_values, sl_values


signals = generate_trade_signal(predictions, y_test_actual)
tp_values, sl_values = compute_tp_sl(predictions, y_test_actual)

result_df = pd.DataFrame(
    {
        "Date": data.index[-len(y_test_actual) :],
        "Open": y_test_actual[:, 0],
        "Close": y_test_actual[:, 1],
        "High": y_test_actual[:, 2],
        "Low": y_test_actual[:, 3],
        "Predicted Close": predictions[:, 1],
        "Trade Signal": signals,
        "Take Profit": tp_values,
        "Stop Loss": sl_values,
    }
)

result_df.to_csv("trade_signals.csv", index=False)
print("Trade signals saved to 'trade_signals.csv'")
