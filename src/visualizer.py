"""
Visualization Module
Creates comprehensive charts for GA evolution and trading performance.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

DARK_BG  = '#0d1117'
CARD_BG  = '#161b22'
GREEN    = '#3fb950'
RED      = '#f85149'
BLUE     = '#58a6ff'
YELLOW   = '#e3b341'
PURPLE   = '#bc8cff'
GRAY     = '#8b949e'
WHITE    = '#f0f6fc'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    CARD_BG,
    'axes.edgecolor':    '#30363d',
    'axes.labelcolor':   WHITE,
    'text.color':        WHITE,
    'xtick.color':       GRAY,
    'ytick.color':       GRAY,
    'grid.color':        '#21262d',
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'legend.facecolor':  CARD_BG,
    'legend.edgecolor':  '#30363d',
})


def plot_evolution_history(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> matplotlib.figure.Figure:

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle('Genetic Algorithm — Evolution Progress',
                 fontsize=16, color=WHITE, fontweight='bold', y=0.98)

    gens = history_df['generation'].values

    ax = axes[0, 0]
    ax.fill_between(gens, history_df['worst_fitness'], history_df['best_fitness'],
                    alpha=0.15, color=BLUE)
    ax.plot(gens, history_df['best_fitness'],  color=GREEN,  linewidth=2.5, label='Best')
    ax.plot(gens, history_df['avg_fitness'],   color=BLUE,   linewidth=1.5, linestyle='--', label='Average', alpha=0.8)
    ax.plot(gens, history_df['worst_fitness'], color=RED,    linewidth=1,   linestyle=':',  label='Worst',   alpha=0.6)
    ax.axhline(y=0, color=GRAY, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Score')
    ax.set_title('Fitness Evolution', color=WHITE, fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _style_axis(ax)

    ax = axes[0, 1]
    diversity = history_df['diversity'].values
    colors = [GREEN if d > 0.15 else YELLOW if d > 0.08 else RED for d in diversity]
    ax.bar(gens, diversity, color=colors, alpha=0.8, width=0.8)
    ax.axhline(y=0.15, color=GREEN,  linewidth=1, linestyle='--', alpha=0.6, label='High diversity')
    ax.axhline(y=0.08, color=YELLOW, linewidth=1, linestyle='--', alpha=0.6, label='Low diversity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity Score')
    ax.set_title('Population Diversity', color=WHITE, fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    _style_axis(ax)

    ax = axes[1, 0]
    best_fit = history_df['best_fitness'].values
    improvement = np.diff(np.array(best_fit, dtype=float), prepend=float(best_fit[0]))
    pos_imp = np.where(improvement > 0, improvement, 0)
    neg_imp = np.where(improvement < 0, improvement, 0)
    ax.bar(gens, pos_imp, color=GREEN, alpha=0.8, width=0.8, label='Improvement')
    ax.bar(gens, neg_imp, color=RED,   alpha=0.6, width=0.8, label='Regression')
    ax.axhline(y=0, color=GRAY, linewidth=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Delta Fitness')
    ax.set_title('Fitness Change per Generation', color=WHITE, fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    _style_axis(ax)

    ax = axes[1, 1]
    if 'elapsed_time' in history_df.columns:
        elapsed = history_df['elapsed_time'].values
        time_per_gen = np.diff(np.array(elapsed, dtype=float), prepend=0.0)
        time_per_gen[0] = float(elapsed[0])
        ax.plot(gens, time_per_gen, color=PURPLE, linewidth=2, label='Time/Gen')
        ax.fill_between(gens, 0, time_per_gen, alpha=0.2, color=PURPLE)
        ax.plot(gens, pd.Series(time_per_gen).rolling(5, min_periods=1).mean().values,
                color=YELLOW, linewidth=1.5, linestyle='--', label='5-gen avg')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time', color=WHITE, fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _style_axis(ax)

    plt.tight_layout(pad=2.0)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    return fig


def plot_equity_curve(
    result,
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> matplotlib.figure.Figure:

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    equity  = np.array(result.equity_curve)
    dates   = result.dates
    initial = equity[0]

    ax_equity = fig.add_subplot(gs[0, :])

    if len(df) == len(equity):
        bh_returns = df['close'] / df['close'].iloc[0] * initial
        ax_equity.plot(dates, bh_returns, color=GRAY, linewidth=1,
                       linestyle='--', alpha=0.7, label='Buy & Hold')

    for i in range(1, len(equity)):
        color = GREEN if equity[i] >= initial else RED
        ax_equity.plot(dates[i-1:i+1], equity[i-1:i+1], color=color, linewidth=1.5, alpha=0.8)

    ax_equity.axhline(y=initial, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.5)

    if result.trades:
        for trade in result.trades[:50]:
            try:
                entry_idx = next((i for i, d in enumerate(dates)
                                  if str(d.date()) == trade.entry_date), None)
                if entry_idx:
                    color  = GREEN if trade.pnl > 0 else RED
                    marker = '^' if trade.direction == 'long' else 'v'
                    ax_equity.scatter(dates[entry_idx], equity[entry_idx],
                                      color=color, marker=marker, s=50, zorder=5, alpha=0.8)
            except Exception:
                pass

    ax_equity.set_title('Equity Curve (Optimized Strategy vs Buy & Hold)',
                        color=WHITE, fontsize=13, fontweight='bold', pad=10)
    ax_equity.set_ylabel('Portfolio Value ($)')
    ax_equity.legend(fontsize=9)
    ax_equity.grid(True, alpha=0.3)
    _style_axis(ax_equity)

    ax_dd = fig.add_subplot(gs[1, :])
    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak > 0, peak, 1) * 100
    ax_dd.fill_between(range(len(drawdown)), drawdown, 0, color=RED, alpha=0.4)
    ax_dd.plot(drawdown, color=RED, linewidth=1)
    ax_dd.axhline(y=0, color=GRAY, linewidth=0.8)
    ax_dd.set_title('Drawdown (%)', color=WHITE, fontsize=11, pad=8)
    ax_dd.set_ylabel('Drawdown (%)')
    ax_dd.grid(True, alpha=0.3)
    _style_axis(ax_dd)

    ax_pnl = fig.add_subplot(gs[2, 0])
    if result.trades:
        pnls   = [t.pnl for t in result.trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        if wins:
            ax_pnl.hist(wins,   bins=12, color=GREEN, alpha=0.7, label=f'Wins ({len(wins)})')
        if losses:
            ax_pnl.hist(losses, bins=12, color=RED,   alpha=0.7, label=f'Losses ({len(losses)})')
        ax_pnl.axvline(x=0, color=WHITE, linewidth=1)
    ax_pnl.set_title('Trade PnL Distribution', color=WHITE, fontsize=10, pad=8)
    ax_pnl.set_xlabel('PnL ($)')
    ax_pnl.legend(fontsize=8)
    ax_pnl.grid(True, alpha=0.3)
    _style_axis(ax_pnl)

    ax_monthly = fig.add_subplot(gs[2, 1])
    _plot_monthly_returns(ax_monthly, equity, dates)

    ax_metrics = fig.add_subplot(gs[2, 2])
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis('off')

    metrics = [
        ('Total Return',  f"{result.total_return*100:+.2f}%",
         GREEN if result.total_return > 0 else RED),
        ('Sharpe Ratio',  f"{result.sharpe_ratio:.3f}",
         GREEN if result.sharpe_ratio > 1 else YELLOW),
        ('Max Drawdown',  f"{result.max_drawdown*100:.2f}%", RED),
        ('Win Rate',      f"{result.win_rate*100:.1f}%",
         GREEN if result.win_rate > 0.5 else YELLOW),
        ('Profit Factor', f"{result.profit_factor:.2f}",
         GREEN if result.profit_factor > 1.2 else YELLOW),
        ('Total Trades',  f"{result.total_trades}", BLUE),
        ('Calmar Ratio',  f"{result.calmar_ratio:.3f}",
         GREEN if result.calmar_ratio > 0.5 else YELLOW),
        ('Sortino Ratio', f"{result.sortino_ratio:.3f}",
         GREEN if result.sortino_ratio > 1 else YELLOW),
    ]

    ax_metrics.set_title('Performance Metrics', color=WHITE, fontsize=10, pad=8)
    for i, (label, value, color) in enumerate(metrics):
        y = 0.88 - i * 0.115
        ax_metrics.text(0.05, y, label, color=GRAY,  fontsize=9,  transform=ax_metrics.transAxes)
        ax_metrics.text(0.95, y, value, color=color, fontsize=10, fontweight='bold',
                        transform=ax_metrics.transAxes, ha='right')
        ax_metrics.plot([0.02, 0.98], [y - 0.015, y - 0.015], color='#21262d',
                        linewidth=0.5, transform=ax_metrics.transAxes)
    _style_axis(ax_metrics)

    fig.suptitle('Trading Strategy — Backtest Results',
                 fontsize=15, color=WHITE, fontweight='bold', y=0.99)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    return fig


def plot_parameter_analysis(
    best_params: dict,
    save_path: Optional[str] = None
) -> matplotlib.figure.Figure:

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle('Optimized Parameter Analysis',
                 fontsize=15, color=WHITE, fontweight='bold')

    from src.genetic_algorithm import GENE_DEFINITIONS

    axes_flat = axes.flatten()
    keys = list(GENE_DEFINITIONS.keys())

    for i, key in enumerate(keys):
        if i >= len(axes_flat):
            break
        ax  = axes_flat[i]
        g   = GENE_DEFINITIONS[key]
        value    = best_params.get(key, g['min'])
        norm_val = (value - g['min']) / (g['max'] - g['min']) if g['max'] != g['min'] else 0.5

        ax.barh(0, 1,        color='#21262d', height=0.3)
        ax.barh(0, norm_val, color=BLUE,      height=0.3, alpha=0.9)
        ax.axvline(x=norm_val, color=YELLOW, linewidth=2, alpha=0.9)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(
            [f"{g['min']}", f"{(g['min']+g['max'])/2:.1f}", f"{g['max']}"],
            fontsize=7, color=GRAY
        )
        label   = key.replace('_', '\n')
        val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
        ax.set_title(f"{label}\n{val_str}", color=WHITE, fontsize=8, pad=4)
        _style_axis(ax)

    for i in range(len(keys), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(pad=2.0)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    return fig


def _plot_monthly_returns(ax: Axes, equity: np.ndarray, dates: list) -> None:
    try:
        eq_series = pd.Series(equity, index=pd.DatetimeIndex(dates))
        monthly   = eq_series.resample('ME').last().pct_change().dropna() * 100
        colors    = [GREEN if r > 0 else RED for r in monthly.values]
        ax.bar(range(len(monthly)), np.array(monthly.values, dtype=float), color=colors, alpha=0.8)
        ax.axhline(y=0, color=GRAY, linewidth=0.8)
        ax.set_title('Monthly Returns (%)', color=WHITE, fontsize=10, pad=8)
        ax.set_ylabel('%')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([d.strftime('%b\n%y') for d in monthly.index],
                           fontsize=6, color=GRAY)
        ax.grid(True, alpha=0.3, axis='y')
        _style_axis(ax)
    except Exception:
        ax.text(0.5, 0.5, 'Insufficient\ndata', transform=ax.transAxes,
                ha='center', va='center', color=GRAY)


def _style_axis(ax: Axes) -> None:
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')