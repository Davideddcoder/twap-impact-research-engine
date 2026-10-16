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
            post = min(len(data) - 1, end - 1 + self.post_window)
            reference = float(close.iloc[max(0, start - 1)])
            event_end = float(close.iloc[end - 1])
            baseline_volume = float(volume.iloc[pre:start].mean()) if start > pre else np.nan
            event_volume = float(volume.iloc[start:end].mean())
            side_sign = 1 if event.get("side") == "buy" else -1
            rows.append({
                "timestamp_start": event.get("timestamp_start", data.timestamp.iloc[start]),
                "side": event.get("side", "unknown"),
                "duration_minutes": end - start,
                "event_direction_move_bps": side_sign * (event_end / reference - 1) * 10000 if reference else np.nan,
                "post_window_return_bps": (float(close.iloc[post]) / event_end - 1) * 10000 if event_end else np.nan,
                "volume_ratio": event_volume / baseline_volume if baseline_volume else np.nan,
                "return_1h_bps": (float(close.iloc[min(end - 1 + 60, len(data)-1)]) / event_end - 1) * 10000 if event_end else np.nan,
                "return_4h_bps": (float(close.iloc[min(end - 1 + 240, len(data)-1)]) / event_end - 1) * 10000 if event_end else np.nan,
                "return_24h_bps": (float(close.iloc[min(end - 1 + 1440, len(data)-1)]) / event_end - 1) * 10000 if event_end else np.nan,
            })
        return pd.DataFrame(rows, columns=[
            "timestamp_start", "side", "duration_minutes",
            "event_direction_move_bps", "post_window_return_bps", "volume_ratio",
            "return_1h_bps", "return_4h_bps", "return_24h_bps"
        ])

    def generate_summary(self, results):
        def mean(column):
            return (float(results[column].mean()) if column in results and
                    results[column].notna().any() else 0.0)
        return {
            "event_count": int(len(results)),
            "mean_duration_minutes": mean("duration_minutes"),
            "mean_event_direction_move_bps": mean("event_direction_move_bps"),
            "mean_post_window_return_bps": mean("post_window_return_bps"),
            "mean_return_1h_bps": mean("return_1h_bps"),
            "mean_return_4h_bps": mean("return_4h_bps"),
            "mean_return_24h_bps": mean("return_24h_bps"),
        }
