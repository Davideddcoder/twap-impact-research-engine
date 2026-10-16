"""Explainable baseline detector for sustained volume anomalies.

The detector is intentionally simple and calibrated only for this repository's
synthetic signal. Its score is not a probability or a production model.
"""
import numpy as np

class TWAPDetector:
    def __init__(self, volume_threshold=2.2, lookback_window=60,
                 min_event_duration=5, smoothing_window=5):
        self.volume_threshold = float(volume_threshold)
        self.lookback_window = int(lookback_window)
        self.min_event_duration = int(min_event_duration)
        self.smoothing_window = int(smoothing_window)

    def detect(self, data):
        if data.empty:
            return []
        frame = data.copy().reset_index(drop=True)
        smooth = frame.volume.rolling(self.smoothing_window,
                                      min_periods=self.smoothing_window).mean()
        baseline = smooth.shift(self.lookback_window).rolling(
            self.lookback_window,
            min_periods=max(5, self.lookback_window // 2)
        ).median()
        ratio = smooth / baseline.replace(0, np.nan)
        active = ratio.ge(self.volume_threshold).fillna(False).to_numpy()
        indices = np.flatnonzero(active)
        if not len(indices):
            return []
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        returns = np.log(frame.close).diff().fillna(0)
        found = []
        for group in groups:
            start, end = int(group[0]), int(group[-1]) + 1
            if end - start < self.min_event_duration:
                continue
            window = frame.iloc[start:end]
            base = float(baseline.iloc[start:end].median())
            volume_ratio = float(window.volume.mean() / base) if base > 0 else float("nan")
            signed_move = float(returns.iloc[start:end].sum())
            prior = returns.iloc[max(0, start - self.lookback_window):start]
            sigma = float(prior.std()) if len(prior) > 1 else 0.0
            directional_z = (abs(signed_move) / (sigma * np.sqrt(end - start))
                              if sigma > 0 else 0.0)
            volume_evidence = min(1.0, volume_ratio / (self.volume_threshold * 1.5))
            direction_evidence = min(1.0, directional_z / 2.0)
            signal_score = 0.6 * volume_evidence + 0.4 * direction_evidence
            found.append({
                "timestamp_start": window.timestamp.iloc[0],
                "timestamp_end": window.timestamp.iloc[-1],
                "start_idx": start,
                "end_idx": end,
                "duration_minutes": end - start,
                "side": "buy" if signed_move >= 0 else "sell",
                "volume_ratio": volume_ratio,
                "directional_z": float(directional_z),
                "signal_score": float(signal_score),
                "evidence": {
                    "volume_anomaly": volume_ratio >= self.volume_threshold,
                    "directional_move_bps": float(signed_move * 10000),
                },
            })
        return found
