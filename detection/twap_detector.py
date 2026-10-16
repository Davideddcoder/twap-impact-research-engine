"""Heuristic detector for sustained volume anomalies in OHLCV data.

This transparent baseline is for synthetic demonstrations, not validated
execution identification.
"""
import numpy as np

class TWAPDetector:
    def __init__(self, volume_threshold=1.8, lookback_window=60,
                 min_event_duration=5, smoothing_window=5):
        self.volume_threshold = float(volume_threshold)
        self.lookback_window = int(lookback_window)
        self.min_event_duration = int(min_event_duration)
        self.smoothing_window = int(smoothing_window)

    def detect(self, data):
        if data.empty:
            return []
        frame = data.copy().reset_index(drop=True)
        # Compare a short smoothed volume average with a trailing baseline.
        smooth = frame.volume.rolling(self.smoothing_window,
                                      min_periods=self.smoothing_window).mean()
        baseline = smooth.shift(self.lookback_window).rolling(
            self.lookback_window, min_periods=max(5, self.lookback_window // 2)
        ).median()
        ratio = smooth / baseline.replace(0, np.nan)
        active = ratio.ge(self.volume_threshold).fillna(False).to_numpy()
        indices = np.flatnonzero(active)
        if not len(indices):
            return []
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        returns = frame.close.pct_change().fillna(0)
        events = []
        for group in groups:
            start, end = int(group[0]), int(group[-1]) + 1
            if end - start < self.min_event_duration:
                continue
            window = frame.iloc[start:end]
            move = float(returns.iloc[start:end].sum())
            base = float(baseline.iloc[start:end].median())
            events.append({
                "timestamp_start": window.timestamp.iloc[0],
                "timestamp_end": window.timestamp.iloc[-1],
                "start_idx": start,
                "end_idx": end,
                "duration_minutes": end - start,
                "side": "buy" if move >= 0 else "sell",
                "confidence": float(min(1.0, ratio.iloc[start:end].mean() /
                                        (self.volume_threshold * 2))),
                "volume_ratio": float(window.volume.mean() / base) if base > 0 else float("nan"),
            })
        return events
