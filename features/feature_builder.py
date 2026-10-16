"""Causal rolling features for exploratory OHLCV analysis."""
import numpy as np


class FeatureBuilder:
    FEATURE_NAMES = ["return_1m", "return_5m", "volume_ratio", "volume_zscore",
                     "range_ratio", "realized_volatility"]

    def __init__(self, window_size=30):
        self.window_size = max(2, int(window_size))

    @staticmethod
    def get_feature_names():
        return list(FeatureBuilder.FEATURE_NAMES)

    def build_features(self, data):
        df = data.copy()
        returns = df.close.pct_change()
        mean = df.volume.rolling(self.window_size, min_periods=2).mean()
        std = df.volume.rolling(self.window_size, min_periods=2).std().replace(0, np.nan)
        df["return_1m"] = returns
        df["return_5m"] = df.close.pct_change(5)
        df["volume_ratio"] = df.volume / mean
        df["volume_zscore"] = (df.volume - mean) / std
        df["range_ratio"] = (df.high - df.low) / df.close.replace(0, np.nan)
        df["realized_volatility"] = returns.rolling(self.window_size, min_periods=2).std()
        return df.replace([np.inf, -np.inf], np.nan)
