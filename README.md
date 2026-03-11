# 🧬 Genetic Algorithm for Trading Strategy Optimization

> **Automatically discover optimal crypto trading parameters using evolutionary computation.**
> Built for OKX Futures (BTC-USDT) | Python | Custom GA Engine | No Black-Box Libraries

---

## 📸 Results Preview

| Evolution Progress | Equity Curve | Parameter Analysis |
|:-:|:-:|:-:|
| ![evolution](results/evolution_history.png) | ![equity](results/equity_curve.png) | ![params](results/parameter_analysis.png) |

---

## 🏆 Actual Backtest Results (BTC-USD, 2 Years Live Data)

| Metric | Baseline | After GA Optimization |
|---|---|---|
| **Total Return** | +0.00% | **+7.27%** |
| **Sharpe Ratio** | 0.000 | **1.6225** |
| **Sortino Ratio** | — | **0.8179** |
| **Calmar Ratio** | — | **8.9480** |
| **Max Drawdown** | — | **0.81%** |
| **Win Rate** | — | **90.9%** |
| **Profit Factor** | — | **9.670** |
| **Total Trades** | 0 | **11** |
| **Avg Trade Return** | — | **+3.60%** |
| **Fitness Score** | -999 | **+2.1424** |

> Data: BTC-USD daily candles | Period: March 2024 – March 2026 | Capital: $10,000 | Commission: 0.1%

---

## 🎯 What This Project Does

Traditional trading strategies require manual parameter tuning — a time-consuming and biased process. This project uses a **Genetic Algorithm (GA)** to automatically search the parameter space and find configurations that maximize risk-adjusted returns.

### ✅ What Gets Optimized

| Category | Parameters |
|---|---|
| **Technical Indicators** | EMA fast/slow periods, RSI period, RSI thresholds, ATR period & multiplier, Bollinger Band period & std dev |
| **Risk Management** | Stop loss %, take profit %, position size % |
| **Signal Filters** | Volume filter multiplier |

---

## 🧠 How Genetic Algorithm Works for Trading

```
POPULATION (40 individuals, each = one set of strategy params)
      │
      ▼
EVALUATION (backtest each individual → compute fitness score)
      │
      ▼
SELECTION (tournament selection → pick the fittest parents)
      │
      ▼
CROSSOVER (uniform + arithmetic blend → create offspring)
      │
      ▼
MUTATION (adaptive gaussian noise → explore new parameter space)
      │
      ▼
ELITISM (preserve top 4 → prevent fitness regression)
      │
      ▼
NEXT GENERATION → repeat for 25 generations
      │
      ▼
BEST INDIVIDUAL = OPTIMAL STRATEGY PARAMETERS ✅
```

### 🏋️ Fitness Function

A **multi-objective composite score** that balances return and risk:

```python
fitness = (
    0.35 × sharpe_ratio     +  # Risk-adjusted return
    0.25 × calmar_ratio     +  # Return / max drawdown
    0.20 × win_rate         +  # Trade success rate
    0.15 × total_return     +  # Raw performance
    0.05 × profit_factor       # Gross profit / gross loss
)
```

### 📈 Optimized Parameters Found

```json
{
  "ema_fast": 18,
  "ema_slow": 27,
  "rsi_period": 7,
  "rsi_oversold": 30.31,
  "rsi_overbought": 71.03,
  "atr_period": 17,
  "atr_multiplier": 2.2637,
  "bb_period": 13,
  "bb_std": 1.3314,
  "stop_loss_pct": 0.0411,
  "take_profit_pct": 0.0440,
  "position_size_pct": 0.1886,
  "volume_filter": 1.0414
}
```

> Win Rate: **90.9%** | Max Drawdown: **0.81%** | Calmar Ratio: **8.95**

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/roybeey0/ga-trading-optimizer-.git
cd ga-trading-optimizer-
pip install -r requirements.txt
```

### 2. Run Optimization

```bash
python main.py
```

### 3. View Results

```
results/
├── evolution_history.png    # Fitness per generation chart
├── equity_curve.png         # Portfolio performance + drawdown
├── parameter_analysis.png   # Optimized parameter visualization
├── best_params.json         # Load directly into your trading bot
├── metrics.json             # Full performance metrics
├── trade_log.csv            # Every trade entry/exit/pnl
└── evolution_history.csv    # GA convergence data per generation
```

---

## ⚙️ Configuration

Edit `CONFIG` in `main.py`:

```python
CONFIG = {
    # Data
    'symbol':           'BTC-USD',   # Yahoo Finance ticker
    'period':           '2y',        # 1y, 2y, 5y
    'interval':         '1d',        # 1d, 1h, 4h

    # GA Parameters
    'population_size':  40,          # Individuals per generation
    'n_generations':    25,          # Increase for better results (try 50+)
    'crossover_rate':   0.80,
    'mutation_rate':    0.15,        # Adaptive — auto-increases if diversity drops
    'elite_size':       4,
    'tournament_size':  3,

    # Backtester
    'initial_capital':  10_000.0,
    'commission':       0.001,       # 0.1% per trade (OKX Futures taker fee)
}
```

---

## 📁 Project Structure

```
ga-trading-optimizer/
│
├── main.py                    # Entry point — run this
│
├── src/
│   ├── backtester.py          # Custom backtest engine + 8 performance metrics
│   ├── data_loader.py         # yfinance OHLCV fetcher + synthetic fallback
│   ├── genetic_algorithm.py   # Full GA: selection, crossover, mutation, elitism
│   ├── visualizer.py          # Dark-theme Matplotlib charts
│   └── __init__.py
│
├── results/                   # Auto-generated after run
│   ├── best_params.json
│   ├── metrics.json
│   ├── trade_log.csv
│   └── *.png
│
├── pyrightconfig.json         # Pylance/VS Code type checking config
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Performance Metrics Explained

| Metric | Formula | Target |
|---|---|---|
| **Total Return** | `(final - initial) / initial` | As high as possible |
| **Sharpe Ratio** | `mean(r) / std(r) × √252` | > 1.0 good, > 2.0 excellent |
| **Sortino Ratio** | `mean(r) / std(downside) × √252` | Like Sharpe, downside only |
| **Max Drawdown** | `max((peak - trough) / peak)` | As low as possible |
| **Calmar Ratio** | `annual_return / max_drawdown` | > 1.0 good, > 3.0 excellent |
| **Win Rate** | `winning_trades / total_trades` | > 50% |
| **Profit Factor** | `gross_profit / gross_loss` | > 1.2 good, > 2.0 excellent |

---

## 🔬 GA Technical Details

### Gene Encoding (Search Space)

| Gene | Min | Max | Type |
|---|---|---|---|
| `ema_fast` | 5 | 50 | Integer |
| `ema_slow` | 20 | 200 | Integer |
| `rsi_period` | 5 | 30 | Integer |
| `rsi_oversold` | 15 | 40 | Float |
| `rsi_overbought` | 60 | 85 | Float |
| `atr_multiplier` | 0.5 | 4.0 | Float |
| `stop_loss_pct` | 0.5% | 8% | Float |
| `take_profit_pct` | 1% | 25% | Float |
| `position_size_pct` | 5% | 50% | Float |

### Evolutionary Operators

- **Selection** — Tournament (k=3): competitive pressure without premature convergence
- **Crossover** — Uniform + BLX-α blend: inherits traits from both parents intelligently
- **Mutation** — Adaptive Gaussian: rate auto-increases when population diversity drops below threshold
- **Elitism** — Top 4 preserved unchanged each generation to prevent regression
- **Seeds** — Population initialized with 3 domain-knowledge starting points

---

## 🔗 Integration with OKX Trading Bot

After optimization, load `results/best_params.json` directly:

```python
import json

with open('results/best_params.json') as f:
    params = json.load(f)

# Plug into your OKX bot strategy
ema_fast      = params['ema_fast']           # 18
ema_slow      = params['ema_slow']           # 27
stop_loss     = params['stop_loss_pct']      # 0.0411 = 4.11%
take_profit   = params['take_profit_pct']    # 0.0440 = 4.40%
position_size = params['position_size_pct']  # 0.1886 = 18.86%
```

---

## 🛠️ Extending the Project

### Add New Parameters
```python
# In src/genetic_algorithm.py → GENE_DEFINITIONS
'macd_fast': {'min': 8, 'max': 20, 'type': 'int', 'step': 1},
'macd_slow': {'min': 20, 'max': 50, 'type': 'int', 'step': 1},
```

### Increase Optimization Power
```python
# In main.py → CONFIG
'population_size': 80,   # More diversity
'n_generations':   50,   # More evolution time
```

### Use Hourly Data
```python
'interval': '1h',
'period':   '6mo',
```

---

## 📄 License

MIT License — free to use, modify, and deploy.

---

## 👨‍💻 Author

Built as part of a machine learning portfolio project.
Context: Companion optimizer for an OKX Futures trading bot (Trend/Range/Breakout strategies).

> ⚠️ **Disclaimer**: Educational and research purposes only. Past backtest performance does not guarantee future results. Always manage risk appropriately.