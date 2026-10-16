"""Run the reproducible synthetic TWAP research pipeline.""" 
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from data.synthetic_dataset_generator import SyntheticDatasetGenerator
from ingestion.loader import DataLoader
from detection.twap_detector import TWAPDetector
from features.feature_builder import FeatureBuilder
from events.event_study import EventStudyAnalyzer

def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = Path("data/raw")
    results_dir = Path("results")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "synthetic_data.csv"
    events_path = data_dir / "synthetic_data_events.json"
    if args.regenerate or not (csv_path.exists() and events_path.exists()):
        generator = SyntheticDatasetGenerator(str(data_dir), seed=args.seed)
        df, events = generator.generate(num_candles=args.num_candles, num_events=args.num_events)
        generator.save_dataset(df, events)
    loader = DataLoader()
    market = loader.load_market_data(csv_path)
    detector = TWAPDetector(volume_threshold=1.8, lookback_window=60, min_event_duration=5)
    detected = detector.detect(market)
    featured = FeatureBuilder(window_size=30).build_features(market)
    results = EventStudyAnalyzer(pre_window=60, post_window=120).analyze_events(featured, detected)
    summary = EventStudyAnalyzer().generate_summary(results)
    results.to_csv(results_dir / "event_study_results.csv", index=False)
    with (results_dir / "summary_statistics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (results_dir / "detected_events.json").open("w") as f:
        json.dump([{**e, "timestamp_start": str(e["timestamp_start"]), "timestamp_end": str(e["timestamp_end"])} for e in detected], f, indent=2)
    try:
        from visualization.plots import ResultsPlotter
        plotter = ResultsPlotter(str(results_dir))
        plotter.plot_event_count(detected)
        plotter.plot_return_distribution(results)
        plotter.plot_cumulative_performance(results)
        if not results.empty:
            plotter.plot_impact_analysis(results)
    except ImportError as exc:
        logging.info("Optional plotting skipped: %s", exc)
    logging.info("Pipeline complete: %d candidates; outputs in %s", len(detected), results_dir)
    logging.info("These are synthetic, exploratory results; they are not validated trading signals.")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic input data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-candles", type=int, default=10080)
    parser.add_argument("--num-events", type=int, default=5)
    run(parser.parse_args())
