import unittest
import pandas as pd
from data.synthetic_dataset_generator import SyntheticDatasetGenerator
from detection.twap_detector import TWAPDetector
from features.feature_builder import FeatureBuilder
from events.event_study import EventStudyAnalyzer
from scripts.evaluate_synthetic import evaluate

class PipelineTests(unittest.TestCase):
    def test_injected_event_is_detectable_and_exposes_evidence(self):
        generator = SyntheticDatasetGenerator(output_dir="/tmp/twap-test-data", seed=11)
        market, truth = generator.generate(
            num_candles=1200, num_events=1,
            event_duration_range=(25, 25), participation_range=(0.15, 0.15)
        )
        candidates = TWAPDetector(volume_threshold=2.2).detect(market)
        target = truth[0]["start_idx"]
        self.assertTrue(any(abs(x["start_idx"] - target) < 10 for x in candidates))
        event = min(candidates, key=lambda x: abs(x["start_idx"] - target))
        self.assertIn("volume_ratio", event)
        self.assertIn("directional_z", event)
        self.assertIn("signal_score", event)
        self.assertIn("evidence", event)
        features = FeatureBuilder().build_features(market)
        self.assertTrue(set(FeatureBuilder.get_feature_names()).issubset(features.columns))
        results = EventStudyAnalyzer().analyze_events(features, [event])
        self.assertGreater(len(results), 0)
        self.assertIn("post_window_return_bps", results.columns)
        self.assertNotIn("permanent_impact_bps", results.columns)

    def test_empty_event_summary_is_safe(self):
        analyzer = EventStudyAnalyzer()
        results = analyzer.analyze_events(pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "close": [100.0], "volume": [1.0]
        }), [])
        self.assertEqual(analyzer.generate_summary(results)["event_count"], 0)

    def test_multi_seed_synthetic_evaluation_has_documented_floor(self):
        metrics = evaluate(seeds=3)
        self.assertGreaterEqual(metrics["precision"], 0.60)
        self.assertGreaterEqual(metrics["recall"], 0.80)

if __name__ == "__main__":
    unittest.main()
