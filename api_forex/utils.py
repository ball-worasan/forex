# File: utils.py
import os
import json
import csv

STATUS_FILE = "status.json"

def save_to_csv(ticker, data, output_folder):
    filename = os.path.join(output_folder, f"{ticker.replace(':', '_')}.csv")
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "volume", "vw", "open", "close", "high", "low", "transactions"])
        for record in data:
            writer.writerow([
                record["t"], record["v"], record["vw"], record["o"], record["c"], record["h"], record["l"], record["n"]
            ])

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as file:
            return json.load(file)
    return {}

def save_status(status):
    with open(STATUS_FILE, "w") as file:
        json.dump(status, file)
