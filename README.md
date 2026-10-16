# TWAP Impact Research Engine — Public Demonstration

## Purpose and scope

This repository is a **demonstration project** showing a reproducible workflow for
simulating TWAP-like execution events, identifying candidate intervals, and
summarizing their observed effects in a controlled environment.

It is not the production model or trading system used in any employment. To
respect confidentiality and compliance obligations related to my work at YIP
Capital, I have not included proprietary source code, parameters, market data,
strategies, or performance results. The public code is an independently
runnable educational demonstration built around synthetic data.

## What this repository demonstrates

- Seeded generation of one-minute synthetic OHLCV data and injected events.
- A transparent baseline heuristic that flags sustained volume anomalies.
- Rolling time-series features and descriptive event-window statistics.
- A command-line pipeline that saves event, summary, and tabular results.
- Basic automated checks for event injection, detection, and analysis.

The detector is a deliberately simple demo heuristic, not an attempt to reproduce
a proprietary model. It may produce false positives. The reported event returns
are descriptive and must not be interpreted as causal market impact or trading
performance.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --regenerate
python -m unittest discover -s tests -v
```

The pipeline writes generated input data under `data/raw/` and CSV/JSON
summaries under `results/`. Matplotlib and Seaborn are included for optional
charts; if they are unavailable, the core CSV/JSON pipeline still runs.

Use `--seed`, `--num-candles`, and `--num-events` to configure the
synthetic scenario. Results are reproducible for a fixed environment and seed.

## Limitations

The market is simulated: there is no exchange feed, order book, real execution
data, calibrated impact model, trading strategy, profitability claim, or
out-of-sample validation. The injected signal and detector threshold are tuned
to make the educational pipeline inspectable. This repository demonstrates
research and engineering workflow; it does not substantiate claims about a live
or production trading system.
