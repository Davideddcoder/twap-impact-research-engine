"""Input loading and basic validation for generated market data."""
from pathlib import Path
import json
import pandas as pd


class DataLoader:
    REQUIRED = {"timestamp", "open", "high", "low", "close", "volume"}

    def load_market_data(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        missing = self.REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError(f"Market data is missing columns: {sorted(missing)}")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        if frame["timestamp"].duplicated().any():
            raise ValueError("Market data contains duplicate timestamps")
        return frame

    def load_events(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open() as handle:
            return json.load(handle)
