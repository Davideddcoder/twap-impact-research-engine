"""
Visualization module for TWAP research engine.

Generates publication-quality plots and summary visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResultsPlotter:
    """Generate visualizations for research results."""
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory to save plot files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_event_count(self, events: List[Dict]) -> str:
        """
        Plot count of detected events by type.
        
        Args:
            events: List of detected events
            
        Returns:
            Path to saved plot
        """
        side_counts = {}
        for event in events:
            side = event.get('side', 'unknown')
            side_counts[side] = side_counts.get(side, 0) + 1
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sides = list(side_counts.keys())
        counts = list(side_counts.values())
        colors = ['#2ecc71' if s == 'buy' else '#e74c3c' for s in sides]
        
        ax.bar(sides, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
        ax.set_xlabel('Event Type', fontsize=12, fontweight='bold')
        ax.set_title('Detected TWAP Events by Type', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, (side, count) in enumerate(zip(sides, counts)):
            ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / 'event_count.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved event count chart: {filepath}")
        return str(filepath)
    
    def plot_return_distribution(self, results_df: pd.DataFrame) -> str:
        """
        Plot distribution of forward returns.
        
        Args:
            results_df: Event study results
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        returns_data = [
            ('1-Hour', results_df['return_1h_bps']),
            ('4-Hour', results_df['return_4h_bps']),
            ('24-Hour', results_df['return_24h_bps']),
        ]
        
        for idx, (label, returns) in enumerate(returns_data):
            ax = axes[idx]
            
            ax.hist(returns, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
            ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.1f} bps')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Return (basis points)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{label} Forward Return Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'return_distribution.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved return distribution chart: {filepath}")
        return str(filepath)
    
    def plot_cumulative_performance(self, results_df: pd.DataFrame) -> str:
        """
        Plot cumulative performance by event side.
        
        Args:
            results_df: Event study results
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Separate buy and sell events
        buy_events = results_df[results_df['side'] == 'buy'].sort_values('timestamp_start')
        sell_events = results_df[results_df['side'] == 'sell'].sort_values('timestamp_start')
        
        # Compute cumulative returns
        if len(buy_events) > 0:
            buy_cumulative = (1 + buy_events['return_1h_bps'] / 10000).cumprod()
            ax.plot(range(len(buy_cumulative)), (buy_cumulative - 1) * 100, 
                   label='BUY Events', color='#2ecc71', linewidth=2, marker='o', markersize=4)
        
        if len(sell_events) > 0:
            sell_cumulative = (1 + sell_events['return_1h_bps'] / 10000).cumprod()
            ax.plot(range(len(sell_events)), (sell_cumulative - 1) * 100,
                   label='SELL Events', color='#e74c3c', linewidth=2, marker='s', markersize=4)
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Event Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative 1-Hour Forward Returns by Event', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'cumulative_performance.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cumulative performance chart: {filepath}")
        return str(filepath)
    
    def plot_impact_analysis(self, results_df: pd.DataFrame) -> str:
        """
        Plot impact analysis by event characteristics.
        
        Args:
            results_df: Event study results
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Temporary impact by side
        ax = axes[0, 0]
        buy_impact = results_df[results_df['side'] == 'buy']['temporary_impact_bps']
        sell_impact = results_df[results_df['side'] == 'sell']['temporary_impact_bps']
        
        ax.boxplot([buy_impact, sell_impact], labels=['BUY', 'SELL'])
        ax.set_ylabel('Temporary Impact (bps)', fontsize=11, fontweight='bold')
        ax.set_title('Temporary Impact Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Duration vs immediate impact
        ax = axes[0, 1]
        scatter = ax.scatter(results_df['duration_minutes'], results_df['temporary_impact_bps'],
                            c=results_df['volume_ratio'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Event Duration (minutes)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Temporary Impact (bps)', fontsize=11, fontweight='bold')
        ax.set_title('Duration vs Impact', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Volume Ratio', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 3: Recovery profile
        ax = axes[1, 0]
        horizons = ['return_1h_bps', 'return_4h_bps', 'return_24h_bps']
        labels = ['1h', '4h', '24h']
        
        mean_returns = [results_df[h].mean() for h in horizons]
        std_returns = [results_df[h].std() for h in horizons]
        
        ax.errorbar(labels, mean_returns, yerr=std_returns, fmt='o-', linewidth=2,
                   markersize=8, capsize=5, capthick=2, color='#3498db', ecolor='#e74c3c')
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Mean Return (bps)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Forward Horizon', fontsize=11, fontweight='bold')
        ax.set_title('Forward Return Profile', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 4: Volume ratio distribution
        ax = axes[1, 1]
        ax.hist(results_df['volume_ratio'], bins=15, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax.axvline(results_df['volume_ratio'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f"Mean: {results_df['volume_ratio'].mean():.2f}x")
        ax.set_xlabel('Volume Ratio', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Event Volume Intensity Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'impact_analysis.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved impact analysis chart: {filepath}")
        return str(filepath)
