"""
Main Runner — Genetic Algorithm Trading Strategy Optimizer
=========================================================
Orchestrates data loading, GA optimization, backtesting, and visualization.
"""

import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Path fix: works on Windows, Linux, Mac ──
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_data, add_technical_indicators
from src.backtester import Backtester, BacktestResult
from src.genetic_algorithm import GeneticAlgorithm, GENE_DEFINITIONS
from src.visualizer import plot_evolution_history, plot_equity_curve, plot_parameter_analysis

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
CONFIG = {
    # Data
    'symbol':         'BTC-USD',
    'period':         '2y',
    'interval':       '1d',
    'use_live_data':  True,

    # GA Parameters
    'population_size':  40,
    'n_generations':    25,
    'crossover_rate':   0.80,
    'mutation_rate':    0.15,
    'elite_size':       4,
    'tournament_size':  3,

    # Backtester
    'initial_capital':  10_000.0,
    'commission':       0.001,    # 0.1% per trade (OKX Futures)

    # Output
    'results_dir':      'results',
}


def fitness_function(params: dict, df: pd.DataFrame) -> BacktestResult:
    """Fitness function passed to the GA. Runs backtest and returns result."""
    backtester = Backtester(
        initial_capital=CONFIG['initial_capital'],
        commission=CONFIG['commission']
    )
    return backtester.run(df, params)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🧬 GA TRADING STRATEGY OPTIMIZER                          ║
║   Genetic Algorithm x Crypto Backtesting                    ║
║   Language: Python | Exchange: OKX Futures (BTC-USDT)       ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_best_params(params: dict, result: BacktestResult):
    print("\n" + "=" * 60)
    print("   BEST OPTIMIZED PARAMETERS")
    print("=" * 60)

    categories = {
        'Technical Indicators': ['ema_fast', 'ema_slow', 'rsi_period', 'rsi_oversold',
                                  'rsi_overbought', 'atr_period', 'atr_multiplier',
                                  'bb_period', 'bb_std'],
        'Risk Management':      ['stop_loss_pct', 'take_profit_pct', 'position_size_pct'],
        'Filters':              ['volume_filter'],
    }

    for cat, keys in categories.items():
        print(f"\n  >> {cat}:")
        for key in keys:
            val = params.get(key, 'N/A')
            if isinstance(val, float):
                if 'pct' in key:
                    print(f"     {key:<25} {val*100:.2f}%")
                else:
                    print(f"     {key:<25} {val:.4f}")
            else:
                print(f"     {key:<25} {val}")

    print("\n" + "=" * 60)
    print("   PERFORMANCE METRICS")
    print("=" * 60)
    print(f"   Total Return      : {result.total_return*100:+.2f}%")
    print(f"   Sharpe Ratio      : {result.sharpe_ratio:.4f}")
    print(f"   Sortino Ratio     : {result.sortino_ratio:.4f}")
    print(f"   Calmar Ratio      : {result.calmar_ratio:.4f}")
    print(f"   Max Drawdown      : {result.max_drawdown*100:.2f}%")
    print(f"   Win Rate          : {result.win_rate*100:.1f}%")
    print(f"   Profit Factor     : {result.profit_factor:.3f}")
    print(f"   Total Trades      : {result.total_trades}")
    print(f"   Avg Trade Return  : {result.avg_trade_return*100:+.2f}%")
    print(f"   Fitness Score     : {result.fitness:.4f}")
    print("=" * 60)


def save_results(best_params: dict, final_result: BacktestResult, history_df: pd.DataFrame):
    """Save all results to the results/ directory."""
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    params_path = os.path.join(CONFIG['results_dir'], 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\n  Saved: {params_path}")

    metrics = {
        'total_return':     round(final_result.total_return, 6),
        'sharpe_ratio':     round(final_result.sharpe_ratio, 6),
        'sortino_ratio':    round(final_result.sortino_ratio, 6),
        'calmar_ratio':     round(final_result.calmar_ratio, 6),
        'max_drawdown':     round(final_result.max_drawdown, 6),
        'win_rate':         round(final_result.win_rate, 6),
        'profit_factor':    round(final_result.profit_factor, 6),
        'total_trades':     final_result.total_trades,
        'avg_trade_return': round(final_result.avg_trade_return, 6),
        'fitness':          round(final_result.fitness, 6),
    }
    metrics_path = os.path.join(CONFIG['results_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    history_path = os.path.join(CONFIG['results_dir'], 'evolution_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"  Saved: {history_path}")

    if final_result.trades:
        trades_data = [{
            'entry_date': t.entry_date, 'exit_date': t.exit_date,
            'direction': t.direction, 'entry_price': t.entry_price,
            'exit_price': t.exit_price, 'pnl': round(t.pnl, 4),
            'pnl_pct': round(t.pnl_pct * 100, 4), 'exit_reason': t.exit_reason
        } for t in final_result.trades]
        trades_path = os.path.join(CONFIG['results_dir'], 'trade_log.csv')
        pd.DataFrame(trades_data).to_csv(trades_path, index=False)
        print(f"  Saved: {trades_path}")


def main():
    print_banner()
    total_start = time.time()

    # 1. Load Data
    print("Step 1/5 - Loading market data...")
    raw_df = load_data(
        use_live=CONFIG['use_live_data'],
        symbol=CONFIG['symbol'],
        period=CONFIG['period'],
        interval=CONFIG['interval']
    )

    default_params = {k: (v['min'] + v['max']) / 2 for k, v in GENE_DEFINITIONS.items()}
    df_with_indicators = add_technical_indicators(raw_df, default_params)

    print(f"\nStep 2/5 - Data ready: {len(df_with_indicators)} candles | {len(df_with_indicators.columns)} features")

    # 2. Baseline
    print("\nRunning baseline strategy (default params)...")
    baseline_backtester = Backtester(CONFIG['initial_capital'], CONFIG['commission'])
    baseline_result = baseline_backtester.run(df_with_indicators, default_params)
    print(f"   Baseline Return: {baseline_result.total_return*100:+.2f}% | "
          f"Sharpe: {baseline_result.sharpe_ratio:.3f} | Trades: {baseline_result.total_trades}")

    # 3. Genetic Algorithm
    print("\nStep 3/5 - Running Genetic Algorithm optimization...")

    def fitness_fn_with_indicators(params, df):
        try:
            df_ind = add_technical_indicators(raw_df, params)
            return fitness_function(params, df_ind)
        except Exception:
            return BacktestResult(fitness=-999.0)

    ga = GeneticAlgorithm(
        population_size=CONFIG['population_size'],
        n_generations=CONFIG['n_generations'],
        crossover_rate=CONFIG['crossover_rate'],
        mutation_rate=CONFIG['mutation_rate'],
        elite_size=CONFIG['elite_size'],
        tournament_size=CONFIG['tournament_size'],
        fitness_fn=fitness_fn_with_indicators,
        verbose=True,
    )

    best_individual = ga.evolve(df_with_indicators)
    history_df = ga.get_evolution_history()

    # 4. Final Backtest
    print("\nStep 4/5 - Running final backtest with optimized parameters...")
    best_params = best_individual.genes
    df_final = add_technical_indicators(raw_df, best_params)
    final_backtester = Backtester(CONFIG['initial_capital'], CONFIG['commission'])
    final_result = final_backtester.run(df_final, best_params)

    print_best_params(best_params, final_result)

    ret_improvement = final_result.total_return - baseline_result.total_return
    print(f"\n  GA Improvement vs Baseline:")
    print(f"     Return:  {baseline_result.total_return*100:+.2f}% -> {final_result.total_return*100:+.2f}% ({ret_improvement*100:+.2f}%)")
    print(f"     Sharpe:  {baseline_result.sharpe_ratio:.3f} -> {final_result.sharpe_ratio:.3f}")
    print(f"     Fitness: {baseline_result.fitness:.4f} -> {final_result.fitness:.4f}")

    # 5. Visualize & Save
    print("\nStep 5/5 - Generating visualizations...")
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    plot_evolution_history(history_df, save_path=os.path.join(CONFIG['results_dir'], 'evolution_history.png'))
    print("   evolution_history.png saved")

    plot_equity_curve(final_result, df_final, save_path=os.path.join(CONFIG['results_dir'], 'equity_curve.png'))
    print("   equity_curve.png saved")

    plot_parameter_analysis(best_params, save_path=os.path.join(CONFIG['results_dir'], 'parameter_analysis.png'))
    print("   parameter_analysis.png saved")

    save_results(best_params, final_result, history_df)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("\nDone! Check the results/ folder.")

    return best_params, final_result, history_df


if __name__ == "__main__":
    main()
