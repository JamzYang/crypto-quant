"""双均线趋势策略 + 波动率调仓回测脚本。

主要职责：
- 在现有双均线趋势策略基础上，引入简单的波动率调仓模块；
- 对比「简单持有 / 原始趋势策略 / 叠加波动率调仓的趋势策略」三条资金曲线及其绩效指标；
- 结合阶段 2 第 3 块，演示如何在不引入未来函数的前提下使用滚动波动率调整仓位大小。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.stage2.data.loader import add_return_columns, load_ohlcv_csv
from src.stage2.strategies.ma_trend import MATrendConfig, MATrendStrategy
from src.stage2.strategies.volatility_adjustment import (
    VolatilityAdjustmentConfig,
    VolatilityAdjuster,
)
from src.stage2.backtests.bt_ma_trend_single_symbol import compute_performance_stats

logger = logging.getLogger(__name__)


def run_trend_with_vol_adjust_backtest(
    csv_path: Path,
    short_window: int = 20,
    long_window: int = 60,
    vol_window: int = 30,
    low_threshold: float = 0.01,
    high_threshold: float = 0.03,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    trading_days: int = 365,
    date_col: str = "date",
    price_col: str = "close",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """运行「双均线趋势策略 + 波动率调仓」回测。

    参数:
        csv_path: 历史数据 CSV 文件路径。
        short_window: 双均线短期窗口长度。
        long_window: 双均线长期窗口长度。
        vol_window: 波动率滚动窗口长度（基于日收益率）。
        low_threshold: 低波动阈值（滚动标准差）。
        high_threshold: 高波动阈值。
        initial_capital: 初始资金规模。
        fee_rate: 单次换手所支付的手续费比例（双边合计近似）。
        trading_days: 一年折算的交易日数量。
        date_col: CSV 中的日期列名称。
        price_col: CSV 中的收盘价列名称。
        plot: 是否绘制资金曲线图（需要 matplotlib）。

    返回:
        data: 含有价格、均线、波动率、仓位与资金曲线等信息的 DataFrame。
        stats: 一个字典，包含三组绩效指标：
            - "buy_and_hold": 简单持有；
            - "trend": 原始双均线趋势策略；
            - "trend_vol_adjusted": 叠加波动率调仓的趋势策略。
    """
    df = load_ohlcv_csv(csv_path, date_col=date_col, price_col=price_col)
    df = add_return_columns(df, price_col="close", log_return=False)

    # 1. 原始双均线趋势策略信号
    trend_config = MATrendConfig(short_window=short_window, long_window=long_window, price_col="close")
    trend_strategy = MATrendStrategy(trend_config)
    data = trend_strategy.generate_signals(df)

    # 对齐日收益率
    data["ret"] = df["ret"]

    # 2. 简单持有基准：始终满仓
    data["bh_ret"] = data["ret"]
    data["bh_equity"] = initial_capital * (1.0 + data["bh_ret"]).cumprod()

    # 3. 原始趋势策略收益（未计手续费）
    data["trend_position"] = data["position"].astype(float)
    data["trend_ret_gross"] = data["trend_position"] * data["ret"]

    # 4. 波动率调仓模块：在趋势仓位基础上缩放仓位
    vol_config = VolatilityAdjustmentConfig(
        window=vol_window,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        ret_col="ret",
        base_position_col="trend_position",
        position_low=1.0,
        position_medium=0.6,
        position_high=0.3,
    )
    vol_adjuster = VolatilityAdjuster(vol_config)
    data = vol_adjuster.adjust_positions(data)

    # 5. 叠加波动率调仓后的策略收益（未计手续费）
    data["trend_vol_position"] = data["position_vol"]
    data["trend_vol_ret_gross"] = data["trend_vol_position"] * data["ret"]

    # 6. 手续费估算与净收益
    # 简单持有不产生换手，这里只对两条策略曲线扣费
    trend_position_change = data["trend_position"].diff().abs().fillna(data["trend_position"].abs())
    data["trend_fee"] = trend_position_change * fee_rate
    data["trend_ret"] = data["trend_ret_gross"] - data["trend_fee"]

    trend_vol_position_change = data["trend_vol_position"].diff().abs().fillna(
        data["trend_vol_position"].abs()
    )
    data["trend_vol_fee"] = trend_vol_position_change * fee_rate
    data["trend_vol_ret"] = data["trend_vol_ret_gross"] - data["trend_vol_fee"]

    # 7. 资金曲线
    data["trend_equity"] = initial_capital * (1.0 + data["trend_ret"]).cumprod()
    data["trend_vol_equity"] = initial_capital * (1.0 + data["trend_vol_ret"]).cumprod()

    # 8. 绩效指标
    stats: Dict[str, Dict[str, float]] = {
        "buy_and_hold": compute_performance_stats(data["bh_ret"], trading_days=trading_days),
        "trend": compute_performance_stats(data["trend_ret"], trading_days=trading_days),
        "trend_vol_adjusted": compute_performance_stats(
            data["trend_vol_ret"], trading_days=trading_days
        ),
    }

    if plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data["bh_equity"], label="简单持有", alpha=0.6)
            plt.plot(data.index, data["trend_equity"], label="双均线趋势", alpha=0.9)
            plt.plot(
                data.index,
                data["trend_vol_equity"],
                label="趋势 + 波动率调仓",
                alpha=0.9,
            )
            plt.title("双均线趋势策略：叠加波动率调仓前后对比")
            plt.xlabel("日期")
            plt.ylabel("资金曲线")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError as exc:
            logger.warning("未安装 matplotlib，跳过资金曲线绘图: %s", exc, exc_info=True)

    return data, stats


def _format_pct(x: float) -> str:
    """将小数形式的收益/回撤转为百分比字符串。"""
    if pd.isna(x):
        return "   NaN "
    return f"{x * 100:7.2f}%"


def print_trend_with_vol_stats(stats: Dict[str, Dict[str, float]]) -> None:
    """以易读格式打印三条曲线的绩效指标。"""
    for name, s in stats.items():
        print(f"\n===== {name} =====")
        print(f"总收益:      {_format_pct(s.get('total_return', np.nan))}")
        print(f"年化收益:    {_format_pct(s.get('annual_return', np.nan))}")
        print(f"年化波动率:  {_format_pct(s.get('annual_vol', np.nan))}")
        sharpe = s.get("sharpe", np.nan)
        if pd.isna(sharpe):
            print("Sharpe 比率:    NaN")
        else:
            print(f"Sharpe 比率:{sharpe:8.2f}")
        print(f"最大回撤:    {_format_pct(s.get('max_drawdown', np.nan))}")


def main() -> None:
    """命令行入口函数。

    默认使用项目根目录下 dist/btc_data.csv 作为数据源，
    并对比简单持有 / 原始趋势 / 叠加波动率调仓三种资金曲线。
    """
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "dist" / "btc_data.csv"

    print("使用的数据文件:", default_csv)

    data, stats = run_trend_with_vol_adjust_backtest(
        csv_path=default_csv,
        short_window=20,
        long_window=60,
        vol_window=30,
        low_threshold=0.01,
        high_threshold=0.03,
        initial_capital=10_000.0,
        fee_rate=0.001,
        trading_days=365,
        date_col="date",  # 如果你的 CSV 使用其他列名，请在此处调整
        price_col="close",
        plot=False,
    )

    print_trend_with_vol_stats(stats)

    cols_to_show = [
        "close",
        "ma_short",
        "ma_long",
        "vol",
        "vol_regime",
        "trend_position",
        "trend_vol_position",
        "bh_equity",
        "trend_equity",
        "trend_vol_equity",
    ]
    existing_cols = [c for c in cols_to_show if c in data.columns]

    print("\n回测结果（尾部 5 行预览）:")
    print(data[existing_cols].tail().round(4))


if __name__ == "__main__":
    main()
