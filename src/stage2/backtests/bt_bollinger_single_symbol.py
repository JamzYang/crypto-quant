"""单标的布林带均值回归策略回测脚本。

主要职责：
- 读入 BTC/USDT 日线数据；
- 调用布林带均值回归策略生成持仓信号；
- 计算策略与简单持有的资金曲线与基础绩效指标；
- 为阶段 2 第 2 块「均值回归原型 + 布林带策略」提供一个可运行的最小回测示例。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.stage2.data.loader import add_return_columns, load_ohlcv_csv
from src.stage2.strategies.bollinger_mean_reversion import (
    BollingerConfig,
    BollingerMeanReversionStrategy,
)
from src.stage2.backtests.bt_ma_trend_single_symbol import (
    compute_performance_stats,
    print_stats,
)

logger = logging.getLogger(__name__)


def run_bollinger_backtest(
    csv_path: Path,
    window: int = 20,
    num_std: float = 2.0,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    trading_days: int = 365,
    date_col: str = "date",
    price_col: str = "close",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """运行单标的布林带均值回归策略回测。

    参数:
        csv_path: 历史数据 CSV 文件路径。
        window: 布林带滚动窗口长度（天数）。
        num_std: 上下轨与中轨之间的标准差倍数。
        initial_capital: 初始资金规模。
        fee_rate: 单次换手所支付的手续费比例（双边合计的近似值）。
        trading_days: 一年折算的交易日数量。
        date_col: CSV 中的日期列名称。
        price_col: CSV 中的收盘价列名称。
        plot: 是否绘制价格 + 布林带 + 资金曲线图（需要 matplotlib）。

    返回:
        data: 含有价格、布林带、信号、资金曲线等信息的 DataFrame。
        stats: 一个字典，包含「strategy」与「buy_and_hold」两套绩效指标。
    """
    df = load_ohlcv_csv(csv_path, date_col=date_col, price_col=price_col)
    df = add_return_columns(df, price_col="close", log_return=False)

    config = BollingerConfig(window=window, num_std=num_std, price_col="close")
    strategy = BollingerMeanReversionStrategy(config)
    data = strategy.generate_signals(df)

    # 简单持有策略：始终满仓
    data["ret"] = df["ret"]
    data["bh_ret"] = data["ret"]
    data["bh_equity"] = initial_capital * (1.0 + data["bh_ret"]).cumprod()

    # 布林带均值回归策略收益（未计手续费）
    data["strategy_ret_gross"] = data["position"] * data["ret"]

    # 根据持仓变动计算近似换手率，并扣除手续费
    position_change = data["position"].diff().abs().fillna(data["position"].abs())
    data["fee_rate"] = position_change * fee_rate
    data["strategy_ret"] = data["strategy_ret_gross"] - data["fee_rate"]

    data["strategy_equity"] = initial_capital * (1.0 + data["strategy_ret"]).cumprod()

    stats = {
        "strategy": compute_performance_stats(data["strategy_ret"], trading_days=trading_days),
        "buy_and_hold": compute_performance_stats(data["bh_ret"], trading_days=trading_days),
    }

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # 价格 + 布林带
            ax_price.plot(data.index, data["close"], label="收盘价", color="black", linewidth=1.0)
            ax_price.plot(data.index, data["bb_mid"], label="中轨", color="blue", linewidth=0.9)
            ax_price.plot(data.index, data["bb_upper"], label="上轨", color="green", linewidth=0.9)
            ax_price.plot(data.index, data["bb_lower"], label="下轨", color="red", linewidth=0.9)

            # 标记开仓、平仓点（只做简单散点标记）
            if "enter_long" in data.columns:
                ax_price.scatter(
                    data.index[data["enter_long"]],
                    data.loc[data["enter_long"], "close"],
                    marker="^",
                    color="green",
                    s=30,
                    label="开多",
                )
            if "exit_long" in data.columns:
                ax_price.scatter(
                    data.index[data["exit_long"]],
                    data.loc[data["exit_long"], "close"],
                    marker="v",
                    color="red",
                    s=30,
                    label="平多",
                )

            ax_price.set_title("BTC 价格与布林带")
            ax_price.legend(loc="best")
            ax_price.grid(True, alpha=0.3)

            # 资金曲线
            ax_equity.plot(data.index, data["bh_equity"], label="简单持有", alpha=0.7)
            ax_equity.plot(data.index, data["strategy_equity"], label="布林带策略", alpha=0.9)
            ax_equity.set_title("布林带均值回归策略 vs 简单持有")
            ax_equity.set_xlabel("日期")
            ax_equity.set_ylabel("资金曲线")
            ax_equity.legend(loc="best")
            ax_equity.grid(True, alpha=0.3)

            fig.tight_layout()
            plt.show()
        except ImportError as exc:  # 遵循异常规范，记录日志后降级处理
            logger.warning("未安装 matplotlib，跳过画图: %s", exc, exc_info=True)

    return data, stats


def _format_pct(x: float) -> str:
    """将小数形式的收益/回撤转为百分比字符串。"""
    if pd.isna(x):
        return "   NaN "
    return f"{x * 100:7.2f}%"


def print_bollinger_stats(stats: Dict[str, Dict[str, float]]) -> None:
    """以易读格式打印布林带策略与基准的绩效指标。"""
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
    你也可以修改 csv_path 指向自己的数据文件。
    """
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "dist" / "btc_data.csv"

    print("使用的数据文件:", default_csv)

    data, stats = run_bollinger_backtest(
        csv_path=default_csv,
        window=20,
        num_std=2.0,
        initial_capital=10_000.0,
        fee_rate=0.001,
        trading_days=365,
        date_col="date",  # 如果你的 CSV 使用其他列名，请在此处调整
        price_col="close",
        plot=False,
    )

    print_bollinger_stats(stats)

    cols_to_show = [
        "close",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "position",
        "bh_equity",
        "strategy_equity",
    ]
    existing_cols = [c for c in cols_to_show if c in data.columns]

    print("\n回测结果（尾部 5 行预览）:")
    print(data[existing_cols].tail().round(4))


if __name__ == "__main__":
    main()
