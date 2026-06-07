# TWAP Impact Research Engine

## Overview

This project implements a systematic framework for detecting, analyzing, and modeling market impact from Time-Weighted Average Price (TWAP) execution strategies. The research engine provides end-to-end capabilities from synthetic data generation through machine learning classification, enabling investigation of execution microstructure without external data dependencies.

TWAP algorithms execute orders incrementally over a specified time horizon at rates proportional to observed market volume. This project models the measurable price impact induced by such systematic execution patterns.

## Project Goals

1. **Generate realistic synthetic market microstructure data** with controlled parameters and reproducible scenarios
2. **Detect TWAP-like execution patterns** using multi-scale temporal and volumetric analysis
3. **Engineer domain-specific features** capturing market impact signatures and execution dynamics
4. **Quantify permanent and temporary price impact** through event study methodology
5. **Classify execution types** via supervised machine learning on observed patterns
6. **Support academic and applied research** into execution algorithms and market impact

## Methodology

### Data Generation

The synthetic dataset generator produces realistic 1-minute candlestick data with five key features:

- **OHLC prices**: Generated via geometric Brownian motion with calibrated volatility and drift
- **Volume**: Modeled with gamma distribution and price-level dependency
- **Open Interest**: Cumulative process with controlled mean dynamics
- **Funding Rate**: Cyclic component with mean reversion (applicable to derivatives markets)

All data generation is deterministic and seeded for reproducibility.

### Synthetic Event Creation

TWAP events are injected into the dataset following a model of execution impact:

**Event Parameters:**
- `start_idx`: Initiation time
- `duration_minutes`: Execution window (5-60 minutes)
- `volume_participation`: Fraction of market volume executed (5%-20%)
- `side`: Buy or sell direction
- `aggressive_factor`: Scaling multiplier on price impact (0.8-1.5x)

**Price Impact Model:**

The model decomposes price impact into two components:

1. **Temporary Impact** (intra-event):
   - Proportional to participation rate
   - Peaks at mid-execution (concave profile)
   - Decays after event completion (mean reversion)
   - Formula: `temp_move = baseline_price * (participation * 0.005) * (1 - (progress - 0.5)²)`

2. **Permanent Impact** (persistent):
   - Square-root relationship with participation (market impact law)
   - Accumulates linearly during execution
   - Remains post-event, creating new local price equilibrium
   - Formula: `perm_move = baseline_price * (0.001 * √participation) * progress`

**Impact Application:**
- Combined impact applied to all OHLC prices during event
- Volume increased by `participation * 0.5`
- Post-event mean reversion occurs over `duration / 2` minutes
- Partial reversion factor: 0.5 (50% of temporary impact reverts)

This model reflects empirical findings from market microstructure literature on the concave nature of temporary impact and square-root scaling of permanent impact.

## Event Detection Logic

### Multi-Scale Analysis

The detection framework operates across multiple timescales:

1. **Intra-event horizon (1-minute)**: Identify elevated volume and price volatility
2. **Event horizon (15-45 minutes)**: Detect sustained directional bias and volume clustering
3. **Post-event horizon (30-45 minutes)**: Observe mean reversion and impact decay

### Detection Features

The TWAP detector computes the following signals:

- **Volume Anomaly**: Deviation from rolling mean, normalized by volatility
- **Directional Bias**: Skewness of volume-weighted returns over event window
- **Price Impact**: Magnitude of price movement relative to volume
- **Persistence**: Autocorrelation of returns (mean reversion indicator)
- **Relative Spread**: Bid-ask spread estimation from high-low range

## Feature Engineering

### Derived Features

The feature builder constructs domain-specific indicators:

#### Volume Features
- `vol_ma_ratio`: Current volume / 20-period MA
- `vol_accel`: Rate of change of volume
- `vol_clustering`: Standard deviation of volume within window

#### Price Features
- `returns_momentum`: Cumulative returns over window
- `price_acceleration`: Rate of change of mid-price
- `ohlc_range`: (High - Low) / Close normalized range

#### Impact Features
- `realized_impact`: Absolute price change vs. baseline
- `impact_decay`: Magnitude of post-event price reversal
- `impact_persistence`: Fraction of impact persisting 30+ minutes

#### Temporal Features
- `time_of_day`: Hour of execution (0-23)
- `day_of_week`: Weekday indicator (0-4, business days)
- `market_phase`: Market hour phase (opening, mid, close)

## Event Study Framework

### Analysis Window Structure

```
[Pre-event]  [Event]  [Post-event]  [Recovery]
(-60 min)    (15-45min)  (30-45 min)  (60+ min)
```

### Measurement Metrics

**Execution Efficiency:**
- `execution_price`: Volume-weighted average execution price
- `vs_open`: Deviation from period open
- `vs_vwap`: Deviation from volume-weighted average price

**Price Impact:**
- `temporary_impact`: Intra-event price displacement
- `permanent_impact`: Price level change post-recovery
- `total_impact`: Sum of temporary and permanent components
- `impact_recovery_time`: Minutes until 80% price recovery

**Market Response:**
- `volume_spike`: Elevated volume ratio during event
- `volatility_increase`: Realized volatility during event vs. baseline
- `liquidity_response`: Change in effective spread

### Event Study Output

For each detected event, the framework generates:

```json
{
  "event_id": 1,
  "timestamp_start": "2024-01-01T10:15:00",
  "timestamp_end": "2024-01-01T10:45:00",
  "duration_minutes": 30,
  "side": "buy",
  "volume_participation": 0.12,
  "temporary_impact_bps": 8.5,
  "permanent_impact_bps": 3.2,
  "recovery_time_minutes": 42,
  "market_conditions": {
    "baseline_volatility": 0.015,
    "baseline_volume": 1250,
    "market_liquidity": "normal"
  }
}
```

## Machine Learning Extension

### Classification Task

The supervised model classifies execution windows as:

- **TWAP Event**: Systematic execution with measurable impact (positive class)
- **Background**: Normal market activity or noise (negative class)

### Model Architecture

A gradient boosting classifier (XGBoost/LightGBM) trained on engineered features:

**Input:** 25-dimensional feature vector
**Output:** Binary classification + confidence score

**Training:**
- Features from synthetic events with known ground truth
- Balanced dataset (50% events, 50% background)
- 5-fold cross-validation
- Feature importance ranking

### Validation Metrics

- Precision: True positives / predicted positives
- Recall: True positives / actual positives
- F1-score: Harmonic mean of precision and recall
- ROC-AUC: Discrimination ability across thresholds

## Project Structure

```
twap-impact-research-engine/
├── data/
│   ├── raw/                          # Generated synthetic datasets
│   │   ├── synthetic_data.csv
│   │   ├── synthetic_data_events.json
│   │   └── synthetic_data_summary.json
│   └── processed/                    # Feature-engineered data
│
├── ingestion/
│   └── loader.py                     # Data loading and validation
│
├── detection/
│   └── twap_detector.py             # Event detection algorithms
│
├── features/
│   └── feature_builder.py           # Feature engineering pipeline
│
├── events/
│   └── event_study.py               # Event study analysis framework
│
├── models/
│   └── classifier.py                # Supervised classification model
│
├── visualization/
│   └── plots.py                     # Analysis and result visualization
│
├── utils/
│   ├── config.py                    # Configuration management
│   └── logger.py                    # Structured logging
│
├── notebooks/                        # Research notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_event_detection.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_event_study.ipynb
│   └── 05_model_training.ipynb
│
├── data/
│   └── synthetic_dataset_generator.py  # Standalone generator script
│
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Requirements

- Python 3.8+
- NumPy, Pandas, SciPy for numerical computing
- scikit-learn, XGBoost for machine learning
- Matplotlib, Seaborn for visualization
- Jupyter for notebooks

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Davideddcoder/twap-impact-research-engine.git
   cd twap-impact-research-engine
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate synthetic dataset:**
   ```bash
   python data/synthetic_dataset_generator.py
   ```

   This creates:
   - `data/raw/synthetic_data.csv` (market data)
   - `data/raw/synthetic_data_events.json` (event ground truth)
   - `data/raw/synthetic_data_summary.json` (dataset statistics)

## Usage

### Generate Custom Synthetic Dataset

```python
from data.synthetic_dataset_generator import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator(output_dir='data/raw', seed=42)

# Generate 2 weeks of data with 8 injected events
df, events = generator.generate(
    num_candles=20160,           # 2 weeks @ 1-min
    num_events=8,
    event_duration_range=(20, 50),
    participation_range=(0.08, 0.25),
)

# Save to disk
output_files = generator.save_dataset(df, events, dataset_name='synthetic_data_custom')
```

### Load Dataset

```python
from ingestion.loader import DataLoader

loader = DataLoader(data_dir='data/raw')
market_data = loader.load_market_data('synthetic_data.csv')
events_metadata = loader.load_events('synthetic_data_events.json')
```

### Run Event Detection

```python
from detection.twap_detector import TWAPDetector

detector = TWAPDetector(lookback_window=60, sensitivity=0.7)
detected_events = detector.detect(market_data)

print(f"Detected {len(detected_events)} events")
for event in detected_events:
    print(f"  Event at {event['timestamp']}, confidence: {event['confidence']:.2f}")
```

### Build Features

```python
from features.feature_builder import FeatureBuilder

builder = FeatureBuilder(window_size=30)
features_df = builder.build_features(market_data, events_metadata)

print(features_df.head())
print(f"Feature matrix shape: {features_df.shape}")
```

### Conduct Event Study

```python
from events.event_study import EventStudyAnalyzer

analyzer = EventStudyAnalyzer(
    pre_window=60,
    event_window=30,
    post_window=60
)

results = analyzer.analyze_events(market_data, events_metadata)

print(f"Average permanent impact: {results['avg_permanent_impact']:.2f} bps")
print(f"Average temporary impact: {results['avg_temporary_impact']:.2f} bps")
```

### Train Classification Model

```python
from models.classifier import TWAPClassifier

model = TWAPClassifier(model_type='xgboost')
model.train(features_df, labels)

predictions = model.predict(test_features)
confidence = model.predict_proba(test_features)

print(f"Test F1-score: {model.score(test_features, test_labels):.3f}")
```

### Run Full Pipeline

```bash
python main.py --config config.yaml
```

## Example Outputs

### Dataset Statistics

```json
{
  "dataset_name": "synthetic_data",
  "num_candles": 10080,
  "candle_interval_minutes": 1,
  "date_range": {
    "start": "2024-01-01T09:30:00",
    "end": "2024-01-12T16:00:00"
  },
  "price_range": {
    "min": 96.82,
    "max": 103.45,
    "mean": 100.12
  },
  "volume_range": {
    "min": 45,
    "max": 3850,
    "mean": 1247
  },
  "num_events": 5,
  "generation_seed": 42
}
```

### Event Metadata (JSON)

```json
[
  {
    "event_id": 0,
    "start_idx": 342,
    "duration_minutes": 28,
    "volume_participation": 0.142,
    "side": "buy",
    "aggressive_factor": 1.23,
    "timestamp_start": "2024-01-02T11:42:00",
    "timestamp_end": "2024-01-02T12:10:00",
    "is_synthetic_event": true
  }
]
```

### Detected Events Output

```
Event 0:
  Timestamp: 2024-01-02 11:42:00
  Duration: 28 minutes
  Confidence: 0.87
  Estimated Participation: 14.2%
  Estimated Side: BUY
```

### Event Study Summary

```
=== Event Study Results ===

Sample Size: 5 events

Price Impact (basis points):
  Temporary Impact:
    Mean: 7.3 bps
    Std:  2.1 bps
  
  Permanent Impact:
    Mean: 2.8 bps
    Std:  1.4 bps
```

## Skills Demonstrated

This project demonstrates proficiency in:

### Quantitative Finance & Market Microstructure
- Geometric Brownian motion for price simulation
- Market impact modeling (temporary vs. permanent)
- TWAP algorithm mechanics and detection
- Event study methodology
- Execution microstructure analysis

### Data Science & Machine Learning
- Synthetic data generation with controlled parameters
- Feature engineering for financial time series
- Supervised classification (binary)
- Model validation and cross-validation

### Software Engineering
- Modular, production-ready code structure
- Object-oriented design patterns
- Comprehensive logging
- Reproducible research (seeded randomness)
- Full documentation

### Financial Data Processing
- Time series manipulation with Pandas
- OHLCV data handling
- Multi-timeframe analysis
- Temporal feature engineering

## License

This project is provided as-is for research and educational purposes.

---

**Last Updated**: January 2024
