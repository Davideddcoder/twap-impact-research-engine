import unittest
from data.synthetic_dataset_generator import SyntheticDatasetGenerator
from detection.twap_detector import TWAPDetector
from features.feature_builder import FeatureBuilder
from events.event_study import EventStudyAnalyzer

class PipelineTests(unittest.TestCase):
    def test_injected_event_is_detectable_and_analyzable(self):
        generator = SyntheticDatasetGenerator(output_dir="/tmp/twap-test-data", seed=11)
        market, truth = generator.generate(
            num_candles=1200, num_events=1,
            event_duration_range=(25, 25), participation_range=(0.15, 0.15)
        )
        candidates = TWAPDetector(volume_threshold=1.8, lookback_window=60,
                                  min_event_duration=5).detect(market)
        self.assertTrue(candidates, "synthetic signature should produce a candidate")
        target = truth[0]["start_idx"]
        self.assertTrue(any(abs(x["start_idx"] - target) < 10 for x in candidates))
        features = FeatureBuilder().build_features(market)
        self.assertTrue(set(FeatureBuilder.get_feature_names()).issubset(features.columns))
        results = EventStudyAnalyzer().analyze_events(features, candidates)
        self.assertGreater(len(results), 0)
        self.assertIn("return_24h_bps", results.columns)

    def test_empty_event_summary_is_safe(self):
        import pandas as pd
        analyzer = EventStudyAnalyzer()
        results = analyzer.analyze_events(pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "close": [100.0], "volume": [1.0]
        }), [])
        summary = analyzer.generate_summary(results)
        self.assertEqual(summary["event_count"], 0)

if __name__ == "__main__":
    unittest.main()
