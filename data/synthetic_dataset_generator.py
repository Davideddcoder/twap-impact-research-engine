"""
Synthetic dataset generator for TWAP impact research.

Generates realistic market microstructure data with controlled synthetic TWAP-like events.
This module produces self-contained datasets that enable the entire analysis pipeline
to run without external data dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class MarketDataGenerator:
    """Generates synthetic OHLCV market data with realistic microstructure."""
    
    def __init__(
        self,
        seed: int = 42,
        num_candles: int = 10080,  # 1 week of 1-min candles
        base_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
    ):
        """
        Initialize market data generator.
        
        Args:
            seed: Random seed for reproducibility
            num_candles: Number of 1-minute candles to generate
            base_price: Initial price level
            volatility: Daily volatility (annualized)
            drift: Price drift component
        """
        np.random.seed(seed)
        self.seed = seed
        self.num_candles = num_candles
        self.base_price = base_price
        self.volatility = volatility
        self.drift = drift
        
    def _generate_price_path(self) -> np.ndarray:
        """
        Generate price path using geometric Brownian motion.
        
        Returns:
            Array of price values
        """
        # Adjust volatility from daily to per-candle (1 min)
        dt = 1.0 / (252 * 24 * 60)  # 1 minute in years
        sigma_scaled = self.volatility * np.sqrt(dt)
        
        returns = np.random.normal(
            self.drift * dt,
            sigma_scaled,
            self.num_candles
        )
        
        price_path = self.base_price * np.exp(np.cumsum(returns))
        return price_path
    
    def _generate_volume(self, price_path: np.ndarray) -> np.ndarray:
        """
        Generate volume with mean reversion and price level dependence.
        
        Args:
            price_path: Array of prices
            
        Returns:
            Array of volumes
        """
        base_volume = 1000
        volume = np.random.gamma(
            shape=2.0,
            scale=base_volume / 2.0,
            size=self.num_candles
        )
        
        # Add price level effect (higher prices -> higher notional volume)
        price_ratio = price_path / self.base_price
        volume *= price_ratio
        
        return np.maximum(volume, 10)  # Minimum volume
    
    def _generate_ohlc(self, price_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate OHLC from price path.
        
        Args:
            price_path: Array of prices
            
        Returns:
            Tuple of (open, high, low, close) arrays
        """
        close = price_path
        
        # Generate intracandle volatility
        open_prices = close + np.random.normal(0, close * 0.001, self.num_candles)
        
        high = np.maximum.reduce([
            open_prices,
            close,
            open_prices + np.abs(np.random.normal(0, close * 0.005, self.num_candles))
        ])
        
        low = np.minimum.reduce([
            open_prices,
            close,
            open_prices - np.abs(np.random.normal(0, close * 0.005, self.num_candles))
        ])
        
        return open_prices, high, low, close
    
    def generate_base_data(self) -> pd.DataFrame:
        """
        Generate base market microstructure data.
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        price_path = self._generate_price_path()
        open_prices, high, low, close = self._generate_ohlc(price_path)
        volume = self._generate_volume(price_path)
        
        # Generate timestamps (1-minute candles, business hours)
        start_time = datetime(2024, 1, 1, 9, 30)
        timestamps = []
        current_time = start_time
        
        for i in range(self.num_candles):
            timestamps.append(current_time)
            current_time += timedelta(minutes=1)
            
            # Skip to next day at 9:30 if we hit 4 PM (16:00)
            if current_time.hour >= 16:
                current_time = current_time.replace(
                    hour=9, minute=30, second=0, microsecond=0
                ) + timedelta(days=1)
        
        df = pd.DataFrame({
            'timestamp': timestamps[:len(close)],
            'open': open_prices[:len(close)],
            'high': high[:len(close)],
            'low': low[:len(close)],
            'close': close[:len(close)],
            'volume': volume[:len(close)],
        })
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add open_interest and funding_rate features.
        
        Args:
            df: Base market data
            
        Returns:
            DataFrame with additional features
        """
        # Open interest (typically follows volume with lag and smoother dynamics)
        base_oi = 50000
        oi = base_oi + np.cumsum(
            np.random.normal(0, 100, len(df))
        )
        oi = np.maximum(oi, 10000)
        
        df['open_interest'] = oi
        
        # Funding rate (cyclic with noise, mean-reverting)
        t = np.arange(len(df))
        base_funding = 0.0001 * np.sin(t * 2 * np.pi / 1440)  # 24-hour cycle
        noise = np.random.normal(0, 0.00005, len(df))
        df['funding_rate'] = base_funding + noise
        
        return df


class TWAPEventInjector:
    """Injects synthetic TWAP-like execution events into market data."""
    
    def __init__(self, seed: int = 42):
        """Initialize event injector."""
        np.random.seed(seed)
        self.seed = seed
        self.events: List[Dict] = []
        
    def create_twap_event(
        self,
        start_idx: int,
        duration_minutes: int,
        volume_participation: float,
        side: str = 'buy',
        aggressive_factor: float = 1.0,
    ) -> Dict:
        """
        Create a synthetic TWAP-like event specification.
        
        A TWAP event models systematic execution with temporal spread,
        inducing measurable price impact through volume participation.
        
        Args:
            start_idx: Starting candle index
            duration_minutes: Event duration in minutes
            volume_participation: Participation rate (fraction of market volume)
            side: 'buy' or 'sell'
            aggressive_factor: Multiplier on price impact (1.0 = baseline)
            
        Returns:
            Event dictionary with parameters
        """
        event = {
            'start_idx': start_idx,
            'duration_minutes': duration_minutes,
            'volume_participation': volume_participation,
            'side': side,
            'aggressive_factor': aggressive_factor,
            'num_slices': duration_minutes,  # One slice per minute
            'is_synthetic_event': True,
        }
        return event
    
    def apply_event_to_data(
        self,
        df: pd.DataFrame,
        event: Dict,
    ) -> pd.DataFrame:
        """
        Apply a TWAP event to market data by modifying prices and volumes.
        
        Implementation:
        - Volume is increased proportionally to participation rate
        - Price impact is modeled as temporary (concave) and permanent (linear)
        - Temporary impact decays after event completion
        - Permanent impact persists and creates new local mean
        
        Args:
            df: Market data
            event: Event specification
            
        Returns:
            Modified DataFrame with injected event
        """
        df = df.copy()
        start_idx = event['start_idx']
        duration = event['duration_minutes']
        end_idx = min(start_idx + duration, len(df))
        
        if start_idx >= len(df):
            logger.warning(f"Event start index {start_idx} exceeds data length {len(df)}")
            return df
        
        participation = event['volume_participation']
        side = event['side']
        aggressive = event['aggressive_factor']
        
        # Extract baseline metrics
        event_slice = df.iloc[start_idx:end_idx]
        avg_volume = event_slice['volume'].mean()
        baseline_price = event_slice['close'].iloc[0] if len(event_slice) > 0 else df.iloc[start_idx]['close']
        
        # Calculate temporary and permanent impact
        # Temporary: proportional to participation, decays after event
        temp_impact_ratio = participation * 0.005 * aggressive
        
        # Permanent: square-root market impact
        perm_impact_ratio = 0.001 * np.sqrt(participation) * aggressive
        
        # Apply to event period
        for i in range(start_idx, end_idx):
            idx = i - start_idx
            progress = idx / duration
            
            # Volume increase
            df.loc[i, 'volume'] *= (1 + participation * 0.5)
            
            # Temporary impact (peaks mid-execution)
            temp_factor = 1.0 - ((progress - 0.5) ** 2)
            temp_move = baseline_price * temp_impact_ratio * temp_factor
            
            # Permanent impact (accumulates)
            perm_move = baseline_price * perm_impact_ratio * progress
            
            # Combined price impact
            total_move = (temp_move + perm_move) * (1 if side == 'buy' else -1)
            
            # Apply to OHLC
            df.loc[i, 'open'] += total_move
            df.loc[i, 'high'] += total_move if side == 'buy' else 0
            df.loc[i, 'low'] += total_move if side == 'sell' else 0
            df.loc[i, 'close'] += total_move
        
        # Post-event period: decay temporary impact (mean reversion)
        decay_period = duration // 2
        for i in range(end_idx, min(end_idx + decay_period, len(df))):
            decay_idx = i - end_idx
            decay_factor = np.exp(-2 * decay_idx / decay_period)
            
            temp_residual = baseline_price * temp_impact_ratio * decay_factor
            residual_move = temp_residual * (1 if side == 'buy' else -1)
            
            # Partial mean reversion
            df.loc[i, 'open'] += residual_move * 0.5
            df.loc[i, 'close'] += residual_move * 0.5
            df.loc[i, 'high'] += residual_move * 0.5 if side == 'buy' else 0
            df.loc[i, 'low'] += residual_move * 0.5 if side == 'sell' else 0
        
        return df
    
    def inject_events(
        self,
        df: pd.DataFrame,
        num_events: int = 5,
        avg_duration: int = 30,
        volume_participation_range: Tuple[float, float] = (0.05, 0.20),
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Inject multiple TWAP-like events into dataset.
        
        Args:
            df: Base market data
            num_events: Number of events to inject
            avg_duration: Average event duration in minutes
            volume_participation_range: Range of participation rates
            
        Returns:
            Tuple of (modified DataFrame, event metadata list)
        """
        df_with_events = df.copy()
        self.events = []
        
        # Ensure minimum spacing between events
        min_spacing = 120  # 2 hours
        available_indices = list(range(min_spacing, len(df) - min_spacing))
        
        if len(available_indices) < num_events:
            logger.warning(
                f"Cannot fit {num_events} events with {min_spacing} spacing. "
                f"Using {len(available_indices)} events instead."
            )
            num_events = len(available_indices)
        
        selected_indices = sorted(
            np.random.choice(available_indices, num_events, replace=False)
        )
        
        for event_num, start_idx in enumerate(selected_indices):
            # Randomize event parameters
            duration = max(
                5,
                int(np.random.normal(avg_duration, avg_duration * 0.3))
            )
            participation = np.random.uniform(*volume_participation_range)
            side = np.random.choice(['buy', 'sell'])
            aggressive = np.random.uniform(0.8, 1.5)
            
            event = self.create_twap_event(
                start_idx=start_idx,
                duration_minutes=duration,
                volume_participation=participation,
                side=side,
                aggressive_factor=aggressive,
            )
            
            df_with_events = self.apply_event_to_data(df_with_events, event)
            
            # Store metadata
            event['event_id'] = event_num
            event['timestamp_start'] = df_with_events.iloc[start_idx]['timestamp']
            event['timestamp_end'] = df_with_events.iloc[min(start_idx + duration - 1, len(df_with_events) - 1)]['timestamp']
            
            self.events.append(event)
            logger.info(f"Injected event {event_num}: {side.upper()} at idx {start_idx}, duration {duration}min, participation {participation:.2%}")
        
        return df_with_events, self.events
    
    def get_event_metadata(self) -> List[Dict]:
        """Return injected event metadata."""
        return self.events


class SyntheticDatasetGenerator:
    """Orchestrates synthetic dataset generation end-to-end."""
    
    def __init__(self, output_dir: str = 'data/raw', seed: int = 42):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Directory to save generated files
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
    
    def generate(
        self,
        num_candles: int = 10080,
        num_events: int = 5,
        event_duration_range: Tuple[int, int] = (15, 45),
        participation_range: Tuple[float, float] = (0.05, 0.20),
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate complete synthetic dataset with injected events.
        
        Args:
            num_candles: Number of 1-minute candles
            num_events: Number of synthetic TWAP events
            event_duration_range: Min, max event duration in minutes
            participation_range: Min, max volume participation rate
            
        Returns:
            Tuple of (market_data, event_metadata)
        """
        logger.info("Generating base market data...")
        market_gen = MarketDataGenerator(
            seed=self.seed,
            num_candles=num_candles,
        )
        
        df = market_gen.generate_base_data()
        df = market_gen.add_derived_features(df)
        
        logger.info(f"Injecting {num_events} synthetic TWAP events...")
        event_injector = TWAPEventInjector(seed=self.seed)
        
        avg_duration = (event_duration_range[0] + event_duration_range[1]) // 2
        df_with_events, events = event_injector.inject_events(
            df,
            num_events=num_events,
            avg_duration=avg_duration,
            volume_participation_range=participation_range,
        )
        
        return df_with_events, events
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        events: List[Dict],
        dataset_name: str = 'synthetic_data',
    ) -> Dict[str, Path]:
        """
        Save generated dataset and metadata.
        
        Args:
            df: Market data
            events: Event metadata
            dataset_name: Base name for output files
            
        Returns:
            Dictionary of output file paths
        """
        output_files = {}
        
        # Save market data
        csv_path = self.output_dir / f'{dataset_name}.csv'
        df.to_csv(csv_path, index=False)
        output_files['market_data'] = csv_path
        logger.info(f"Saved market data: {csv_path}")
        
        # Save event metadata
        events_path = self.output_dir / f'{dataset_name}_events.json'
        
        # Serialize events (convert datetime to string)
        events_serializable = []
        for event in events:
            event_copy = event.copy()
            event_copy['timestamp_start'] = event_copy['timestamp_start'].isoformat()
            event_copy['timestamp_end'] = event_copy['timestamp_end'].isoformat()
            events_serializable.append(event_copy)
        
        with open(events_path, 'w') as f:
            json.dump(events_serializable, f, indent=2)
        
        output_files['events'] = events_path
        logger.info(f"Saved event metadata: {events_path}")
        
        # Save summary
        summary = {
            'dataset_name': dataset_name,
            'num_candles': len(df),
            'candle_interval_minutes': 1,
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
            },
            'price_range': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'mean': float(df['close'].mean()),
            },
            'volume_range': {
                'min': float(df['volume'].min()),
                'max': float(df['volume'].max()),
                'mean': float(df['volume'].mean()),
            },
            'num_events': len(events),
            'generation_seed': self.seed,
        }
        
        summary_path = self.output_dir / f'{dataset_name}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        output_files['summary'] = summary_path
        logger.info(f"Saved summary: {summary_path}")
        
        return output_files


def main():
    """Generate synthetic dataset with default parameters."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = SyntheticDatasetGenerator(output_dir='data/raw')
    
    logger.info("Starting synthetic dataset generation...")
    df, events = generator.generate(
        num_candles=10080,  # ~1 week
        num_events=5,
        event_duration_range=(15, 45),
        participation_range=(0.05, 0.20),
    )
    
    output_files = generator.save_dataset(
        df, events, dataset_name='synthetic_data'
    )
    
    logger.info("Dataset generation complete")
    logger.info(f"Output files: {output_files}")
    
    return df, events, output_files


if __name__ == '__main__':
    main()
