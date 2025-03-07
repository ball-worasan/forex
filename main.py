import os
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def create_dataset(dataset, look_back=60):
    """
    สร้างชุดข้อมูล X, y สำหรับ LSTM
    โดยในที่นี้ y มี 3 ค่า [Close, High, Low] ของแถว i
    """
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back : i, :])
        y.append(dataset[i, :])  # [Close, High, Low] ของแถว i
    return np.array(X), np.array(y)


# -----------------------------------------------------
# 1) โหลดข้อมูล CSV
# -----------------------------------------------------
data_path = "data/XAUUSD/XAUUSD1440_formatted.csv"
df = pd.read_csv(
    data_path,
    usecols=["timestamp", "open", "high", "low", "close"],
    parse_dates=["timestamp"],
    index_col="timestamp",
)

df.rename(
    columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
    inplace=True,
)
df.ffill(inplace=True)  # เติม NaN ถ้ามี

# -----------------------------------------------------
# 2) ทำ Scaling เฉพาะ [Close, High, Low] (ตามที่โมเดลเคยเทรน)
# -----------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Close", "High", "Low"]])

# -----------------------------------------------------
# 3) สร้างชุดข้อมูล X_all, y_all
# -----------------------------------------------------
look_back = 60
X_all, y_all = create_dataset(scaled_data, look_back)
print("X_all.shape =", X_all.shape)
print("y_all.shape =", y_all.shape)

if X_all.shape[0] == 0:
    raise ValueError("ข้อมูลไม่พอสำหรับ look_back=60")

# -----------------------------------------------------
# 4) โหลดโมเดล LSTM
# -----------------------------------------------------
model_path = "saved_model/XAUUSD_LSTM_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"ไม่พบโมเดลที่ {model_path}")

model = load_model(model_path)
print("✅ โหลดโมเดล LSTM สำเร็จ!")

# -----------------------------------------------------
# 5) ทำการทำนาย (Predict)
# -----------------------------------------------------
predictions = model.predict(X_all)  # shape=(n_samples, 3)
predictions_inv = scaler.inverse_transform(predictions)

# -----------------------------------------------------
# 6) ใส่ค่าทำนายลงใน DataFrame
#    (แถวที่เริ่มต้นได้ค่าทำนายคือ index = look_back)
# -----------------------------------------------------
df["PredClose"] = np.nan
df["PredHigh"] = np.nan
df["PredLow"] = np.nan

pred_index_start = look_back
prediction_dates = df.index[pred_index_start:]  # ช่วงวันที่มีการทำนาย

for i, date_idx in enumerate(prediction_dates):
    df.at[date_idx, "PredClose"] = predictions_inv[i, 0]
    df.at[date_idx, "PredHigh"] = predictions_inv[i, 1]
    df.at[date_idx, "PredLow"] = predictions_inv[i, 2]

# -----------------------------------------------------
# 7) ลบเส้นทำนายบนกราฟ -> แสดงแค่แท่งเทียนจริง (ตามต้องการ)
#    (จะแสดงกราฟเฉพาะ 120 วันล่าสุด - 1)
# -----------------------------------------------------
if len(df) >= 121:
    df_last_120 = df.iloc[-121:-1].copy()
    mpf.plot(
        df_last_120,
        type="candle",
        style="charles",
        title="XAU/USD (Last 120 Days, Skipped Final Day)",
        ylabel="Price (USD)",
        volume=False,
        figscale=1.2,
        figratio=(16, 9),
        tight_layout=True,
    )
    # plt.show()
else:
    print("ข้อมูลน้อยกว่า 121 แถว จึงขอข้ามการแสดงกราฟ 120 วันล่าสุด")

# -----------------------------------------------------
# 8) คำนวณ "Agent" โดยปรับจากโมเดล (ตัวอย่าง: ลด/เพิ่ม 5 หน่วย)
# -----------------------------------------------------
# คำนวณเบื้องต้นจากโมเดล
df["AgentClose"] = df["PredClose"]
df["AgentHigh"] = df["PredHigh"] - 5  # ปรับให้ลดลง 5 หน่วย
df["AgentLow"] = df["PredLow"] + 5  # ปรับให้เพิ่มขึ้น 5 หน่วย

# ปรับค่าให้อยู่ในเกณฑ์ถูกต้อง (ให้แน่ใจว่า AgentLow <= AgentClose <= AgentHigh)
# โดยใช้ np.clip เพื่อบังคับค่า AgentClose ให้อยู่ในช่วง [AgentLow, AgentHigh]
df["AgentClose"] = np.clip(df["AgentClose"], df["AgentLow"], df["AgentHigh"])

# นอกจากนี้ ถ้าในบางกรณี AgentHigh น้อยกว่า AgentClose เราก็ปรับให้ AgentHigh เท่ากับ AgentClose
df["AgentHigh"] = np.where(
    df["AgentHigh"] < df["AgentClose"], df["AgentClose"], df["AgentHigh"]
)
# และถ้า AgentLow มากกว่า AgentClose ให้ปรับให้ AgentLow เท่ากับ AgentClose
df["AgentLow"] = np.where(
    df["AgentLow"] > df["AgentClose"], df["AgentClose"], df["AgentLow"]
)

# -----------------------------------------------------
# 9) วนลูปคำนวณคะแนน (Score) ตามเงื่อนไข
#    - ให้คะแนนแยกออกเป็น 3 หมวด (High, Low, Direction)
#    - เปรียบเทียบทั้ง "Model" และ "Agent"
# -----------------------------------------------------
model_score_high = 0
model_score_low = 0
model_score_dir = 0

agent_score_high = 0
agent_score_low = 0
agent_score_dir = 0

results = []  # เก็บผลลัพธ์ทุกรายการ

# วนตั้งแต่ look_back ถึงสิ้นสุด (เฉพาะวันที่มีค่าทำนาย)
for date_idx in prediction_dates:
    # ถ้าไม่มีข้อมูลจริงก็ข้าม
    if pd.isna(df.at[date_idx, "Open"]) or pd.isna(df.at[date_idx, "Close"]):
        continue

    # ราคาจริง
    a_open = df.at[date_idx, "Open"]
    a_close = df.at[date_idx, "Close"]
    a_high = df.at[date_idx, "High"]
    a_low = df.at[date_idx, "Low"]

    # ราคาโมเดลทำนาย
    m_close = df.at[date_idx, "PredClose"]
    m_high = df.at[date_idx, "PredHigh"]
    m_low = df.at[date_idx, "PredLow"]

    # ราคา "Agent"
    ag_close = df.at[date_idx, "AgentClose"]
    ag_high = df.at[date_idx, "AgentHigh"]
    ag_low = df.at[date_idx, "AgentLow"]

    # ข้ามถ้าไม่มี pred
    if pd.isna(m_close) or pd.isna(m_high) or pd.isna(m_low):
        continue

    # -----------------------------
    # 1) Model High Score
    #    if PredHigh < ActualHigh => +1
    # -----------------------------
    m_h_score = 1 if (m_high < a_high) else 0

    # -----------------------------
    # 2) Model Low Score
    #    if PredLow > ActualLow => +1
    # -----------------------------
    m_l_score = 1 if (m_low > a_low) else 0

    # -----------------------------
    # 3) Model Direction Score
    #    ถ้า predClose > open => Buy,
    #       ถ้าราคาจริง close > open => ทายถูก (+1), ไม่งั้น (-1)
    #    ถ้า predClose < open => Sell,
    #       ถ้าราคาจริง close < open => ทายถูก (+1), ไม่งั้น (-1)
    # -----------------------------
    if m_close > a_open:
        # โมเดลบอก Buy
        m_dir_score = 1 if (a_close > a_open) else -1
    else:
        # โมเดลบอก Sell
        m_dir_score = 1 if (a_close < a_open) else -1

    # -----------------------------
    # Agent Scores (high, low, dir) แบบเดียวกัน
    # -----------------------------
    ag_h_score = 1 if (ag_high < a_high) else 0
    ag_l_score = 1 if (ag_low > a_low) else 0

    if ag_close > a_open:
        # agent บอก Buy
        ag_dir_score = 1 if (a_close > a_open) else -1
    else:
        # agent บอก Sell
        ag_dir_score = 1 if (a_close < a_open) else -1

    # สะสมคะแนนรวม
    model_score_high += m_h_score
    model_score_low += m_l_score
    model_score_dir += m_dir_score

    agent_score_high += ag_h_score
    agent_score_low += ag_l_score
    agent_score_dir += ag_dir_score

    # เก็บข้อมูลใน results เพื่อตรวจสอบรายวัน
    day_result = {
        "Date": date_idx.strftime("%Y-%m-%d"),
        "Actual(OHLC)": (a_open, a_high, a_low, a_close),
        "Model(High,Low,Close)": (m_high, m_low, m_close),
        "Agent(High,Low,Close)": (ag_high, ag_low, ag_close),
        "ModelScores": {"High": m_h_score, "Low": m_l_score, "Dir": m_dir_score},
        "AgentScores": {"High": ag_h_score, "Low": ag_l_score, "Dir": ag_dir_score},
    }
    results.append(day_result)

# -----------------------------------------------------
# 10) แสดงรายการทุกรายวัน (ตรวจสอบตั้งแต่ต้น)
# -----------------------------------------------------
for r in results:
    print("Date:", r["Date"])
    print("  Actual (O,H,L,C):", r["Actual(OHLC)"])
    print("  Model (H,L,C)   :", r["Model(High,Low,Close)"])
    print("  Agent (H,L,C)   :", r["Agent(High,Low,Close)"])
    print("  Model Scores    :", r["ModelScores"])
    print("  Agent Scores    :", r["AgentScores"])
    print("-" * 60)

# -----------------------------------------------------
# 11) สรุปคะแนนรวม
# -----------------------------------------------------
model_total_score = model_score_high + model_score_low + model_score_dir
agent_total_score = agent_score_high + agent_score_low + agent_score_dir

print("\n=== Summary Scores ===")
print(
    f"Model => High={model_score_high}, Low={model_score_low}, Dir={model_score_dir}, Total={model_total_score}"
)
print(
    f"Agent => High={agent_score_high}, Low={agent_score_low}, Dir={agent_score_dir}, Total={agent_total_score}"
)
print("✅ เสร็จสิ้นการทำนายและให้คะแนนตามเงื่อนไข!")
