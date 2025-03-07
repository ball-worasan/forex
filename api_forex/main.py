# File: main.py
import os
import time
from datetime import datetime, timedelta
from api_forex.api import APIClient
from api_forex.utils import save_to_csv, load_status, save_status

API_KEY = "NjaVQPuHBD5hOd_s7ckTZkhPkyODsE3j"
# รายชื่อคู่เหรียญที่ต้องการเพิ่ม ซึ่งมีรูปแบบการเคลื่อนที่คล้าย XAUUSD
TICKERS = [
    "C:EURUSD",
    "C:USDJPY",
    "C:GBPUSD",
    "C:AUDUSD",
    "C:USDCAD",
    "C:USDCHF",
    "C:NZDUSD",
    "C:EURGBP",
    "C:EURJPY",
    "C:GBPJPY",
    "C:AUDJPY",
    "C:NZDJPY",
    "C:EURCHF",
    "C:GBPCHF",
    "C:EURCAD",
    "C:GBP-AUD",
    "C:GBP-NZD",
    "C:CHFJPY",
    "C:AUDNZD",
    "C:AUDCAD",
    "C:AUDCHF",
    "C:CADJPY",
    "C:CADCHF",
    "C:NZDCHF",
    "C:NZDCAD",
    "C:GBPCAD",
    "C:GBPZAR",
    "C:EURTRY",
    "C:USDTRY",
    "C:EURZAR",
    "C:USDCNH",
    "C:USDHKD",
    "C:USDINR",
    "C:USDSGD",
    "C:USDMXN",
    "C:USDPLN",
    "C:USDSEK",
    "C:USDNOK",
    "C:USDKRW",
    "C:USDTWD",
    "C:USDTHB",
    "C:USDIDR",
    "C:USDPHP",
    "C:USDHUF",
    "C:USDILS",
    "C:USDCLP",
    "C:USDCZK",
    "C:USDTRY",
    "C:USDRON",
    "C:USDKZT",
    "C:USDBRL",
    # สามารถเพิ่มหรือปรับปรุงคู่เหรียญเพิ่มเติมตามต้องการ
]
OUTPUT_FOLDER = "data"
DELAY = 15  # วินาที


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # โหลดสถานะการโหลดข้อมูลที่ผ่านมา (สำหรับทุกคู่เหรียญ)
    status = load_status()

    start_date_default = datetime(2023, 1, 1)  # วันที่เริ่มต้นสำหรับคู่เหรียญใหม่
    end_date = datetime(2025, 1, 15)  # วันที่สิ้นสุด

    api_client = APIClient(API_KEY)

    # วนลูปสำหรับแต่ละคู่เหรียญ
    for ticker in TICKERS:
        print(f"\n=== เริ่มการประมวลผลคู่เหรียญ: {ticker} ===")
        # กำหนดวันเริ่มต้นจากสถานะที่บันทึกไว้ หรือค่าเริ่มต้น
        start_date = start_date_default
        if ticker in status:
            # อ่านวันที่สิ้นสุดจากการโหลดครั้งล่าสุด แล้วเริ่มจากวันถัดไป
            start_date = datetime.strptime(status[ticker], "%Y-%m-%d") + timedelta(
                days=1
            )

        # โหลดข้อมูลจนกว่าจะครบช่วงเวลาที่กำหนด
        while start_date < end_date:
            next_end_date = min(
                start_date + timedelta(days=730), end_date
            )  # ช่วงสูงสุด 2 ปี
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = next_end_date.strftime("%Y-%m-%d")

            print(f"Fetching data for {ticker} from {start_str} to {end_str}...")
            data = api_client.fetch_data(ticker, start_str, end_str)

            if data and "results" in data:
                save_to_csv(ticker, data["results"], OUTPUT_FOLDER)
                # บันทึกสถานะการโหลดสำหรับคู่เหรียญนี้
                status[ticker] = end_str
                save_status(status)  # บันทึกสถานะลงไฟล์
                print(f"Saved data for {ticker} from {start_str} to {end_str}.")
            else:
                print(f"No data for {ticker} from {start_str} to {end_str}.")

            # เลื่อนช่วงการค้นหาข้อมูลไปในอนาคต
            start_date = next_end_date + timedelta(days=1)
            time.sleep(DELAY)  # ดีเลย์เพื่อลดภาระ API


if __name__ == "__main__":
    main()
