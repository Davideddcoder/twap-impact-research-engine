"""Descriptive event-window statistics; no causal attribution is implied."""
import numpy as np
import pandas as pd


class EventStudyAnalyzer:
    def __init__(self, pre_window=60, post_window=120):
        self.pre_window = int(pre_window)
        self.post_window = int(post_window)

    def analyze_events(self, data, events):
        rows = []
        close = data.close.reset_index(drop=True)
        volume = data.volume.reset_index(drop=True)
        for event in events:
            start = int(event.get("start_idx", 0))
            end = int(event.get("end_idx", start + event.get("duration_minutes", 1)))
            if start >= len(data) or end <= start:
                continue
            pre = max(0, start - self.pre_window)
            post = min(len(data) - 1, end + self.post_window)
            start_price, end_price = float(close.iloc[start]), float(close.iloc[end - 1])
            baseline = float(close.iloc[max(0, start - 1)])
            event_volume = float(volume.iloc[start:end].mean())
            baseline_volume = float(volume.iloc[pre:start].mean()) if start > pre else np.nan
            sign = 1 if event.get("side") == "buy" else -1
            impact = sign * (end_price / baseline - 1) * 10000 if baseline else np.nan
            row = {
                "timestamp_start": event.get("timestamp_start", data.timestamp.iloc[start]),
                "side": event.get("side", "unknown"),
                "duration_minutes": end - start,
                "temporary_impact_bps": impact,
                "volume_ratio": event_volume / baseline_volume if baseline_volume else np.nan,
            }
            for horizon, col in ((60, "return_1h_bps"), (240, "return_4h_bps"),
                                 (1440, "return_24h_bps")):
                idx = min(end - 1 + horizon, len(data) - 1)
                row[col] = (float(close.iloc[idx]) / end_price - 1) * 10000 if end_price else np.nan
            row["permanent_impact_bps"] = (
                (float(close.iloc[post]) / end_price - 1) * 10000 if end_price else np.nan
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=["timestamp_start", "side", "duration_minutes",
            "temporary_impact_bps", "permanent_impact_bps", "volume_ratio",
            "return_1h_bps", "return_4h_bps", "return_24h_bps"])

    def generate_summary(self, results):
        def mean(col):
            return float(results[col].mean()) if col in results and results[col].notna().any() else 0.0
        return {
            "event_count": int(len(results)),
            "mean_duration_minutes": mean("duration_minutes"),
            "mean_temporary_impact_bps": mean("temporary_impact_bps"),
            "mean_permanent_impact_bps": mean("permanent_impact_bps"),
            "mean_return_1h_bps": mean("return_1h_bps"),
            "mean_return_4h_bps": mean("return_4h_bps"),
            "mean_return_24h_bps": mean("return_24h_bps"),
        }
