import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
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


# -------------------------------
# 2. การเตรียมชุดข้อมูลสำหรับ LSTM
# -------------------------------
def create_dataset(data, look_back=90):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ✅ สร้าง Scaler แยกเฉพาะสำหรับราคาทองคำ
scaler_price = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler_price.fit_transform(data_prices)

look_back = 60
X, y = create_dataset(scaled_prices, look_back)

# -------------------------------
# 3. Cross-Validation และแบ่งชุดข้อมูล
# -------------------------------
ts_split = TimeSeriesSplit(n_splits=5)
for train_index, test_index in ts_split.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# ✅ ตรวจสอบขนาดของข้อมูล
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# -------------------------------
# ฟังก์ชันสร้างโมเดล LSTM
# -------------------------------
def build_lstm_model(input_shape, learning_rate=0.0005):
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(4),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error"
    )
    return model


# -------------------------------
# 4. การฝึกโมเดล
# -------------------------------
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

# -------------------------------
# 5. การพยากรณ์และวิเคราะห์ผล
# -------------------------------
# ✅ ป้องกันปัญหาขนาดข้อมูลไม่ตรง
if X_test.shape[2] != X_train.shape[2]:
    X_test = X_test[:, :, : X_train.shape[2]]

predictions = model.predict(X_test)
predictions = scaler_price.inverse_transform(predictions)
y_test_actual = scaler_price.inverse_transform(y_test)

print(f"MAE: {mean_absolute_error(y_test_actual, predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_actual, predictions)):.4f}")

# -------------------------------
# การแสดงผล
# -------------------------------
variables = ["Close", "High", "Low"]
colors = ["blue", "green", "red"]

for i, var in enumerate(variables):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual[:, i], label=f"Actual {var} Price", color=colors[i])
    plt.plot(predictions[:, i], label=f"Predicted {var} Price", color="orange")
    plt.fill_between(
        range(len(predictions[:, i])),
        predictions[:, i] - 1.96 * np.std(predictions[:, i]),
        predictions[:, i] + 1.96 * np.std(predictions[:, i]),
        color="lightblue",
        alpha=0.2,
        label=f"Confidence Interval for {var}",
    )
    plt.title(f"{var} Price Prediction with Confidence Interval")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

# -------------------------------
# 6. การบันทึกผลลัพธ์
# -------------------------------
result_summary = []
for i, col in enumerate(variables):
    mae = mean_absolute_error(y_test_actual[:, i], predictions[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_actual[:, i], predictions[:, i]))
    result_summary.append({"Variable": col, "MAE": mae, "RMSE": rmse})

summary_df = pd.DataFrame(result_summary)
summary_df.to_csv("result_summary.csv", index=False)
summary_df.plot(kind="bar", x="Variable", y=["MAE", "RMSE"], figsize=(10, 6))

plt.title("Model Performance Metrics")
plt.ylabel("Error")
plt.show()

print("Results saved to 'result_summary.csv'")


# -------------------------------
# 7. การวิเคราะห์ความสำคัญของตัวแปร (แก้ไข)
# -------------------------------
def variable_importance(data, look_back=60):
    feature_importance = []
    for col in data.columns:
        temp_data = data.copy()
        temp_data[col] = 0
        scaled_temp_data = scaler_price .transform(temp_data)
        X_temp, y_temp = create_dataset(scaled_temp_data, look_back)  # เพิ่ม y_temp
        temp_predictions = model.predict(X_temp)
        temp_predictions = scaler_price .inverse_transform(temp_predictions)

        # ตัดขนาด y_temp ให้ตรงกับ temp_predictions
        y_temp_actual = scaler_price .inverse_transform(y_temp[: len(temp_predictions)])

        # คำนวณ MAE โดยใช้ข้อมูลที่มีขนาดตรงกัน
        mae = mean_absolute_error(y_temp_actual, temp_predictions)
        feature_importance.append({"Variable": col, "Importance": mae})
    return pd.DataFrame(feature_importance)


# เรียกใช้ฟังก์ชันและบันทึกผล
importance_df = variable_importance(data, look_back)
importance_df.to_csv("variable_importance.csv", index=False)
print("Variable importance saved to 'variable_importance.csv'")
