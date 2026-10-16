# TWAP Impact Research Engine — Public Demonstration

## Purpose and compliance boundary

This is a **public research demonstration**, not a copy of a production model or
trading system. To respect confidentiality and compliance obligations related
to my work at YIP Capital, this repository excludes proprietary source code,
parameters, market data, strategies, and performance results. The code here is
an independently runnable example built entirely around synthetic data.

The goal is to make the reasoning workflow inspectable: state a synthetic
signal, compute interpretable evidence, apply a documented rule, compare
detections with injected labels, and report errors and limits.

## What the pipeline does

1. Generates seeded one-minute OHLCV data and injects labeled buy/sell events.
   Event signatures are deliberately clear so the demonstration can be tested;
   their parameters are not calibrated to a real venue.
2. Smooths volume over five bars and compares it with a trailing median
   baseline. Consecutive threshold exceedances form candidate intervals.
3. Reports volume ratio, a directional-move z-score, side, and a bounded
   signal score. The score is an explainability aid, **not** a probability.
4. Builds causal rolling features and computes side-adjusted event-window moves
   plus forward returns. These are descriptive summaries, not causal impact
   estimates or trading returns.
5. Evaluates candidate intervals against synthetic labels over multiple seeds.

## Reproduce

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --regenerate
python -m unittest discover -s tests -v
python -m scripts.evaluate_synthetic --seeds 10
```

The pipeline writes generated data under data/raw/ and CSV/JSON results
under results/. Matplotlib and Seaborn generate optional charts; the core
pipeline still writes its CSV/JSON outputs if chart libraries are unavailable.

## Synthetic validation snapshot

With seeds 0–9, 10,080 candles and 5 injected events per seed, the documented
volume threshold of 2.2 produced:

| Measure | Count |
|---|---:|
| Injected events | 50 |
| Matched candidate intervals (TP) | 43 |
| Unmatched candidate intervals (FP) | 24 |
| Missed injected events (FN) | 7 |
| Precision | 64.2% |
| Recall | 86.0% |
| F1 | 73.5% |

A prediction counts as a match when its interval overlaps an unmatched injected
event; matching is one-to-one. These figures measure recovery of this generator's
labels only. They are not estimates of real-world TWAP detection accuracy. The
false positives are intentionally reported rather than hidden.

## Limitations

There is no exchange feed, order book, real execution data, calibrated impact
model, production strategy, profitability claim, or out-of-sample market
validation. Synthetic event parameters and the detection threshold are chosen
for an inspectable teaching example. The code demonstrates research reasoning
and engineering practices; it does not substantiate claims about a live or
proprietary system.
