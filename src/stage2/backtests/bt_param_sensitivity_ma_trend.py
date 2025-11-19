"""阶段 2 第 5 块：双均线趋势策略参数敏感性小测试脚本。

主要职责：
- 在一组候选参数网格上（短期/长期均线窗口组合）跑双均线趋势策略回测；
- 输出各参数组合在全样本上的基础指标（总收益、年化收益、年化波动、Sharpe、最大回撤）；
- 帮助直观感受「策略对参数的敏感程度」，为后续回顾与重构提供依据。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.stage2.backtests.bt_ma_trend_single_symbol import (
    compute_performance_stats,
    run_ma_trend_backtest,
)

logger = logging.getLogger(__name__)


def run_ma_param_grid(
    csv_path: Path,
    short_windows: List[int],
    long_windows: List[int],
    min_gap: int = 5,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    trading_days: int = 365,
    date_col: str = "date",
    price_col: str = "close",
) -> pd.DataFrame:
    """在参数网格上运行双均线趋势策略回测。

    参数:
        csv_path: 历史数据 CSV 文件路径。
        short_windows: 候选短期均线窗口列表。
        long_windows: 候选长期均线窗口列表。
        min_gap: 长短均线之间的最小间隔，避免短期过于接近长期。
        initial_capital: 初始资金规模。
        fee_rate: 手续费比例。
        trading_days: 一年折算的交易日数量。
        date_col: CSV 中日期列名称。
        price_col: CSV 中收盘价列名称。

    返回:
        一个包含不同参数组合绩效指标的 DataFrame。
    """
    results: List[Dict[str, float]] = []

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue
            if long_window - short_window < min_gap:
                continue

            logger.info(
                "运行双均线参数组合: short=%d, long=%d", short_window, long_window
            )

            _, stats = run_ma_trend_backtest(
                csv_path=csv_path,
                short_window=short_window,
                long_window=long_window,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                trading_days=trading_days,
                date_col=date_col,
                price_col=price_col,
                plot=False,
            )

            s = stats.get("strategy", {})
            results.append(
                {
                    "short_window": float(short_window),
                    "long_window": float(long_window),
                    "total_return": float(s.get("total_return", np.nan)),
                    "annual_return": float(s.get("annual_return", np.nan)),
                    "annual_vol": float(s.get("annual_vol", np.nan)),
                    "sharpe": float(s.get("sharpe", np.nan)),
                    "max_drawdown": float(s.get("max_drawdown", np.nan)),
                }
            )

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="sharpe", ascending=False)

    return df_res


def _format_pct(x: float) -> str:
    if np.isnan(x):
        return "   NaN "
    return f"{x * 100:7.2f}%"


def print_param_grid_table(df_res: pd.DataFrame) -> None:
    """以文本表格形式打印参数敏感性结果。"""
    if df_res.empty:
        print("参数结果为空，请检查数据或参数范围。")
        return

    print("\n>>> 双均线参数敏感性结果（按 Sharpe 从高到低排序）<<<")
    header = (
        f"{'short':>5} {'long':>5} | {'TotRet':>8} {'AnnRet':>8} "
        f"{'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8}"
    )
    print(header)
    print("-" * len(header))

    for _, row in df_res.iterrows():
        print(
            f"{int(row['short_window']):5d} {int(row['long_window']):5d} | "
            f"{_format_pct(row['total_return'])} "
            f"{_format_pct(row['annual_return'])} "
            f"{_format_pct(row['annual_vol'])} "
            f"{row['sharpe']:8.2f} "
            f"{_format_pct(row['max_drawdown'])}"
        )


def main() -> None:
    """命令行入口：运行一个小规模的参数网格测试。"""
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "dist" / "btc_data.csv"

    short_windows = [10, 20, 30]
    long_windows = [50, 60, 90]

    df_res = run_ma_param_grid(
        csv_path=default_csv,
        short_windows=short_windows,
        long_windows=long_windows,
        min_gap=5,
        initial_capital=10_000.0,
        fee_rate=0.001,
        trading_days=365,
        date_col="date",  # 如列名不同，可在此调整
        price_col="close",
    )

    print_param_grid_table(df_res)


if __name__ == "__main__":
    main()
