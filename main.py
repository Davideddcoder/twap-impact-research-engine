"""
Main entry point for TWAP impact research engine.

Orchestrates the complete research workflow:
1. Load synthetic data
2. Detect TWAP events
3. Build features
4. Run event study analysis
5. Generate visualizations
6. Save results
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Local imports
from ingestion.loader import DataLoader
from detection.twap_detector import TWAPDetector
from features.feature_builder import FeatureBuilder
from events.event_study import EventStudyAnalyzer
from visualization.plots import ResultsPlotter


def setup_logging():
    """Configure structured logging."""
    log_dir = Path('results')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Execute complete research pipeline."""
    logger = setup_logging()
    
    try:
        logger.info("="*70)
        logger.info("TWAP Impact Research Engine - Pipeline Start")
        logger.info("="*70)
        
        # ===== STEP 1: LOAD DATA =====
        logger.info("\n[1/6] Loading synthetic data...")
        
        data_path = 'data/raw/synthetic_data.csv'
        events_path = 'data/raw/synthetic_data_events.json'
        
        loader = DataLoader()
        df = loader.load_market_data(data_path)
        ground_truth_events = loader.load_events(events_path)
        
        logger.info(f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Dataset shape: {df.shape}, Columns: {list(df.columns)}")
        
        # ===== STEP 2: DETECT EVENTS =====
        logger.info("\n[2/6] Detecting TWAP events...")
        
        detector = TWAPDetector(
            volume_threshold=1.5,
            lookback_window=60,
            min_event_duration=5
        )
        detected_events = detector.detect(df)
        
        logger.info(f"Detected {len(detected_events)} events")
        if detected_events:
            logger.info(f"  - First event: {detected_events[0]['timestamp_start']} "
                       f"({detected_events[0]['duration_minutes']} min, {detected_events[0]['side'].upper()})")
        
        # ===== STEP 3: BUILD FEATURES =====
        logger.info("\n[3/6] Building features...")
        
        feature_builder = FeatureBuilder(window_size=30)
        df_with_features = feature_builder.build_features(df)
        
        feature_cols = FeatureBuilder.get_feature_names()
        logger.info(f"Built {len(feature_cols)} features: {feature_cols}")
        logger.info(f"Feature summary:\n{df_with_features[feature_cols].describe().to_string()}")
        
        # ===== STEP 4: EVENT STUDY ANALYSIS =====
        logger.info("\n[4/6] Running event study analysis...")
        
        analyzer = EventStudyAnalyzer(pre_window=60, post_window=120)
        results_df = analyzer.analyze_events(df_with_features, detected_events)
        
        logger.info(f"Analyzed {len(results_df)} events")
        
        # Generate summary statistics
        summary = analyzer.generate_summary(results_df)
        logger.info("Event Study Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # ===== STEP 5: SAVE RESULTS =====
        logger.info("\n[5/6] Saving results...")
        
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save event study results
        results_path = results_dir / 'event_study_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved event study results to {results_path}")
        
        # Save summary statistics
        import json
        summary_path = results_dir / 'summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary statistics to {summary_path}")
        
        # Save detected events
        detected_events_path = results_dir / 'detected_events.json'
        # Convert timestamps to strings for JSON serialization
        events_to_save = []
        for event in detected_events:
            event_copy = event.copy()
            event_copy['timestamp_start'] = str(event_copy['timestamp_start'])
            event_copy['timestamp_end'] = str(event_copy['timestamp_end'])
            events_to_save.append(event_copy)
        
        with open(detected_events_path, 'w') as f:
            json.dump(events_to_save, f, indent=2)
        logger.info(f"Saved detected events to {detected_events_path}")
        
        # ===== STEP 6: GENERATE VISUALIZATIONS =====
        logger.info("\n[6/6] Generating visualizations...")
        
        plotter = ResultsPlotter(output_dir='results')
        
        plot_1 = plotter.plot_event_count(detected_events)
        logger.info(f"Saved event count chart: {plot_1}")
        
        plot_2 = plotter.plot_return_distribution(results_df)
        logger.info(f"Saved return distribution chart: {plot_2}")
        
        plot_3 = plotter.plot_cumulative_performance(results_df)
        logger.info(f"Saved cumulative performance chart: {plot_3}")
        
        if len(results_df) > 0:
            plot_4 = plotter.plot_impact_analysis(results_df)
            logger.info(f"Saved impact analysis chart: {plot_4}")
        
        # ===== GENERATE SUMMARY REPORT =====
        logger.info("\n" + "="*70)
        logger.info("RESEARCH PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\nOutput Files Generated:")
        logger.info(f"  ✓ {results_path}")
        logger.info(f"  ✓ {summary_path}")
        logger.info(f"  ✓ {detected_events_path}")
        logger.info(f"  ✓ {results_dir}/event_count.png")
        logger.info(f"  ✓ {results_dir}/return_distribution.png")
        logger.info(f"  ✓ {results_dir}/cumulative_performance.png")
        if len(results_df) > 0:
            logger.info(f"  ✓ {results_dir}/impact_analysis.png")
        
        logger.info("\nKey Results:")
        logger.info(f"  • Events detected: {len(detected_events)}")
        logger.info(f"  • Mean event duration: {summary['mean_duration_minutes']:.1f} minutes")
        logger.info(f"  • Mean temporary impact: {summary['mean_temporary_impact_bps']:.2f} bps")
        logger.info(f"  • Mean 1-hour return: {summary['mean_return_1h_bps']:.2f} bps")
        logger.info(f"  • Mean 4-hour return: {summary['mean_return_4h_bps']:.2f} bps")
        logger.info(f"  • Mean 24-hour return: {summary['mean_return_24h_bps']:.2f} bps")
        
        logger.info("\nAll results saved to: results/")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure synthetic data is generated first:")
        logger.error("  python data/synthetic_dataset_generator.py")
        return 1
    
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
