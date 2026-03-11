"""
Custom Backtesting Engine
Simulates trading based on strategy signals and parameters
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    entry_date: str
    entry_price: float
    direction: str
    position_size: float
    stop_loss: float
    take_profit: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[str] = None


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    dates: List = field(default_factory=list)

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0

    fitness: float = -999.0


class TradingStrategy:
    def __init__(self, params: dict):
        self.params = params
        self._validate_params()

    def _validate_params(self):
        p = self.params
        p['ema_fast'] = max(5, min(50, int(p.get('ema_fast', 10))))
        p['ema_slow'] = max(p['ema_fast'] + 5, min(200, int(p.get('ema_slow', 30))))
        p['rsi_period'] = max(5, min(50, int(p.get('rsi_period', 14))))
        p['rsi_oversold'] = max(10, min(40, float(p.get('rsi_oversold', 30))))
        p['rsi_overbought'] = max(60, min(90, float(p.get('rsi_overbought', 70))))
        p['atr_multiplier'] = max(0.5, min(5.0, float(p.get('atr_multiplier', 2.0))))
        p['stop_loss_pct'] = max(0.005, min(0.10, float(p.get('stop_loss_pct', 0.02))))
        p['take_profit_pct'] = max(0.01, min(0.30, float(p.get('take_profit_pct', 0.05))))
        p['position_size_pct'] = max(0.05, min(1.0, float(p.get('position_size_pct', 0.2))))
        p['bb_std'] = max(1.0, min(3.5, float(p.get('bb_std', 2.0))))
        p['volume_filter'] = max(0.5, min(3.0, float(p.get('volume_filter', 1.2))))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        p = self.params

        ema_cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        ema_cross_down = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))

        rsi_not_overbought = df['rsi'] < p['rsi_overbought']
        rsi_not_oversold = df['rsi'] > p['rsi_oversold']

        volume_ok = df['volume'] > (df['volume_ma'] * p['volume_filter'])

        macd_bull = df['macd'] > df['macd_signal']
        macd_bear = df['macd'] < df['macd_signal']

        bb_oversold = df['close'] <= df['bb_lower']
        bb_overbought = df['close'] >= df['bb_upper']

        long_signal = (
            (ema_cross_up & rsi_not_overbought & volume_ok & macd_bull) |
            (bb_oversold & (df['rsi'] < p['rsi_oversold'] + 5) & macd_bull)
        )

        short_signal = (
            (ema_cross_down & rsi_not_oversold & volume_ok & macd_bear) |
            (bb_overbought & (df['rsi'] > p['rsi_overbought'] - 5) & macd_bear)
        )

        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals


class Backtester:
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, df: pd.DataFrame, params: dict) -> BacktestResult:
        result = BacktestResult()

        try:
            strategy = TradingStrategy(params)
            signals = strategy.generate_signals(df)

            capital = self.initial_capital
            position: Optional[Trade] = None
            equity: List[float] = []

            for i in range(len(df)):
                row = df.iloc[i]
                date = df.index[i]
                signal = signals.iloc[i]

                if position is not None:
                    exit_price: Optional[float] = None
                    exit_reason: Optional[str] = None

                    if position.direction == 'long':
                        if row['low'] <= position.stop_loss:
                            exit_price = position.stop_loss
                            exit_reason = 'stop_loss'
                        elif row['high'] >= position.take_profit:
                            exit_price = position.take_profit
                            exit_reason = 'take_profit'
                        elif signal == -1:
                            exit_price = float(row['close'])
                            exit_reason = 'signal_reverse'

                    elif position.direction == 'short':
                        if row['high'] >= position.stop_loss:
                            exit_price = position.stop_loss
                            exit_reason = 'stop_loss'
                        elif row['low'] <= position.take_profit:
                            exit_price = position.take_profit
                            exit_reason = 'take_profit'
                        elif signal == 1:
                            exit_price = float(row['close'])
                            exit_reason = 'signal_reverse'

                    if exit_price is not None and exit_reason is not None:
                        if position.direction == 'long':
                            pnl_pct = (exit_price - position.entry_price) / position.entry_price
                        else:
                            pnl_pct = (position.entry_price - exit_price) / position.entry_price

                        pnl = position.position_size * pnl_pct
                        commission_cost = position.position_size * self.commission * 2
                        pnl -= commission_cost

                        capital += pnl

                        position.exit_date = str(date.date())
                        position.exit_price = exit_price
                        position.pnl = pnl
                        position.pnl_pct = float(pnl_pct)
                        position.exit_reason = exit_reason

                        result.trades.append(position)
                        position = None

                if position is None and signal != 0 and capital > 0:
                    pos_size = capital * float(params.get('position_size_pct', 0.2))
                    sl_pct = float(params.get('stop_loss_pct', 0.02))
                    tp_pct = float(params.get('take_profit_pct', 0.05))
                    atr_mult = float(params.get('atr_multiplier', 2.0))
                    atr_val = float(row.get('atr', row['close'] * 0.02))

                    if signal == 1:
                        sl = float(row['close']) - (atr_val * atr_mult)
                        tp = float(row['close']) + (atr_val * atr_mult * (tp_pct / sl_pct))
                        position = Trade(
                            entry_date=str(date.date()),
                            entry_price=float(row['close']),
                            direction='long',
                            position_size=pos_size,
                            stop_loss=max(sl, float(row['close']) * (1 - sl_pct)),
                            take_profit=min(tp, float(row['close']) * (1 + tp_pct))
                        )

                    elif signal == -1:
                        sl = float(row['close']) + (atr_val * atr_mult)
                        tp = float(row['close']) - (atr_val * atr_mult * (tp_pct / sl_pct))
                        position = Trade(
                            entry_date=str(date.date()),
                            entry_price=float(row['close']),
                            direction='short',
                            position_size=pos_size,
                            stop_loss=min(sl, float(row['close']) * (1 + sl_pct)),
                            take_profit=max(tp, float(row['close']) * (1 - tp_pct))
                        )

                current_equity = capital
                if position is not None:
                    unrealized = self._calc_unrealized(position, float(row['close']))
                    current_equity += unrealized

                equity.append(current_equity)

            result.equity_curve = equity
            result.dates = df.index.tolist()
            result.total_trades = len(result.trades)

            if len(equity) > 0 and self.initial_capital > 0:
                self._calculate_metrics(result)

        except Exception:
            result.fitness = -999.0

        return result

    def _calc_unrealized(self, position: Trade, current_price: float) -> float:
        if position.direction == 'long':
            return position.position_size * (current_price - position.entry_price) / position.entry_price
        else:
            return position.position_size * (position.entry_price - current_price) / position.entry_price

    def _calculate_metrics(self, result: BacktestResult):
        equity = np.array(result.equity_curve)

        if len(equity) < 2:
            return

        result.total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return

        if returns.std() > 0:
            result.sharpe_ratio = float((returns.mean() * 252) / (returns.std() * np.sqrt(252)))

        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            result.sortino_ratio = float((returns.mean() * 252) / (downside.std() * np.sqrt(252)))

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.where(peak > 0, peak, 1)
        result.max_drawdown = float(abs(drawdown.min()))

        if result.max_drawdown > 0:
            result.calmar_ratio = float(result.total_return / result.max_drawdown)

        if result.total_trades > 0:
            winning_trades = [t for t in result.trades if t.pnl > 0]
            losing_trades = [t for t in result.trades if t.pnl <= 0]

            result.win_rate = len(winning_trades) / result.total_trades

            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))

            if gross_loss > 0:
                result.profit_factor = gross_profit / gross_loss

            result.avg_trade_return = float(np.mean([t.pnl_pct for t in result.trades]))

        result.fitness = self._compute_fitness(result)

    def _compute_fitness(self, result: BacktestResult) -> float:
        if result.total_trades < 5:
            return -999.0
        if result.total_return <= -0.5:
            return -999.0

        sharpe_score = float(np.clip(result.sharpe_ratio, -3, 5))
        calmar_score = float(np.clip(result.calmar_ratio, -2, 5))
        winrate_score = result.win_rate * 2 - 1
        return_score = float(np.clip(result.total_return, -1, 5))
        pf_score = float(np.clip(result.profit_factor - 1, -1, 3))

        fitness = (
            0.35 * sharpe_score +
            0.25 * calmar_score +
            0.20 * winrate_score +
            0.15 * return_score +
            0.05 * pf_score
        )

        return float(fitness)