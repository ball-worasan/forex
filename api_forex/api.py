# File: api.py
import requests

class APIClient:
    BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"

    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()

    def fetch_data(self, ticker, start_date, end_date):
        url = self.BASE_URL.format(ticker=ticker, start=start_date, end=end_date)
        params = {"adjusted": "true", "sort": "asc", "apiKey": self.api_key}
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None
