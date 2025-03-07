# File: plot.py
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_data(ticker, output_folder):
    filename = os.path.join(output_folder, f"{ticker.replace(':', '_')}.csv")
    if not os.path.exists(filename):
        print(f"No data found for {ticker} in {output_folder}")
        return

    # Load data
    data = pd.read_csv(filename)
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["close"], label="Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Closing Prices")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    TICKER = "C:XAUUSD"
    OUTPUT_FOLDER = "data"
    plot_data(TICKER, OUTPUT_FOLDER)
