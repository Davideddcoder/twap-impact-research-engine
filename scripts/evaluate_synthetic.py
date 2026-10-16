"""Evaluate candidate intervals against injected synthetic event labels."""
import argparse
from data.synthetic_dataset_generator import SyntheticDatasetGenerator
from detection.twap_detector import TWAPDetector

def evaluate(seeds=10, threshold=2.2, num_candles=10080, num_events=5):
    tp = fp = fn = 0
    for seed in range(seeds):
        market, truth = SyntheticDatasetGenerator(seed=seed).generate(
            num_candles=num_candles, num_events=num_events
        )
        predictions = TWAPDetector(volume_threshold=threshold).detect(market)
        remaining = set(range(len(truth)))
        for prediction in predictions:
            overlaps = [
                i for i in remaining
                if prediction["start_idx"] < truth[i]["start_idx"] + truth[i]["duration_minutes"]
                and prediction["end_idx"] > truth[i]["start_idx"]
            ]
            if not overlaps:
                fp += 1
                continue
            best = max(overlaps, key=lambda i:
                min(prediction["end_idx"], truth[i]["start_idx"] + truth[i]["duration_minutes"])
                - max(prediction["start_idx"], truth[i]["start_idx"]))
            tp += 1
            remaining.remove(best)
        fn += len(remaining)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"seeds": seeds, "events_per_seed": num_events, "true_positive": tp,
            "false_positive": fp, "false_negative": fn, "precision": precision,
            "recall": recall, "f1": f1, "volume_threshold": threshold}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=2.2)
    args = parser.parse_args()
    print(evaluate(args.seeds, args.threshold))
