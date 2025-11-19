"""单标的双均线趋势策略回测脚本。

主要职责：
- 读入 BTC/USDT 日线数据；
- 调用双均线策略生成持仓信号；
- 计算策略与简单持有的资金曲线与基础绩效指标；
- 为阶段 2 第 1 块提供一个可运行的最小回测示例。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.stage2.data.loader import add_return_columns, load_ohlcv_csv
from src.stage2.strategies.ma_trend import MATrendConfig, MATrendStrategy


def compute_performance_stats(
    returns: pd.Series,
    trading_days: int = 365,
) -> Dict[str, float]:
    """根据日度收益率计算基础绩效指标。

    参数:
        returns: 日收益率序列（假定为等间隔，通常为日度）。
        trading_days: 一年折算的交易日数量，现货加密货币可近似为 365。

    返回:
        包含总收益、年化收益、年化波动率、Sharpe 比率、最大回撤的字典。
    """
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    # 资金曲线
    equity = (1.0 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0

    n = len(returns)
    annual_return = (1.0 + total_return) ** (trading_days / n) - 1.0

    # 年化波动率与 Sharpe
    daily_vol = returns.std(ddof=1)
    annual_vol = daily_vol * np.sqrt(trading_days)
    sharpe = np.nan
    if annual_vol > 0:
        sharpe = annual_return / annual_vol

    # 最大回撤
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }


def run_ma_trend_backtest(
    csv_path: Path,
    short_window: int = 20,
    long_window: int = 60,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    trading_days: int = 365,
    date_col: str = "date",
    price_col: str = "close",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """运行单标的双均线趋势策略回测。

    参数:
        csv_path: 历史数据 CSV 文件路径。
        short_window: 短期均线窗口。
        long_window: 长期均线窗口。
        initial_capital: 初始资金。
        fee_rate: 单次换手所支付的手续费比例（双边合计的近似值）。
        trading_days: 一年折算的交易日数量。
        date_col: CSV 中的日期列名称。
        price_col: CSV 中的收盘价列名称。
        plot: 是否绘制资金曲线图（需要 matplotlib）。

    返回:
        data: 含有价格、信号、资金曲线等信息的 DataFrame。
        stats: 一个字典，包含“strategy”与“buy_and_hold”两套绩效指标。
    """
    df = load_ohlcv_csv(csv_path, date_col=date_col, price_col=price_col)
    df = add_return_columns(df, price_col="close", log_return=False)

    config = MATrendConfig(short_window=short_window, long_window=long_window, price_col="close")
    strategy = MATrendStrategy(config)
    data = strategy.generate_signals(df)

    # 简单持有策略：始终满仓
    data["ret"] = df["ret"]
    data["bh_ret"] = data["ret"]
    data["bh_equity"] = initial_capital * (1.0 + data["bh_ret"]).cumprod()

    # 双均线策略收益（未计手续费）
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

            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data["bh_equity"], label="简单持有", alpha=0.7)
            plt.plot(data.index, data["strategy_equity"], label="双均线策略", alpha=0.9)
            plt.title("BTC 双均线趋势策略 vs 简单持有")
            plt.xlabel("日期")
            plt.ylabel("资金曲线")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("未安装 matplotlib，跳过画图。")

    return data, stats


def _format_pct(x: float) -> str:
    """将小数形式的收益/回撤转为百分比字符串。"""
    if pd.isna(x):
        return "   NaN "
    return f"{x * 100:7.2f}%"


def print_stats(stats: Dict[str, Dict[str, float]]) -> None:
    """以易读格式打印策略与基准的绩效指标。"""
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
    # 通过文件位置反推项目根目录: .../crypto-quant/src/stage2/backtests/*.py
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "dist" / "btc_data.csv"

    print("使用的数据文件:", default_csv)

    data, stats = run_ma_trend_backtest(
        csv_path=default_csv,
        short_window=20,
        long_window=60,
        initial_capital=10_000.0,
        fee_rate=0.001,
        trading_days=365,
        date_col="date",  # 如果你的 CSV 使用其他列名，请在此处调整
        price_col="close",
        plot=False,
    )

    print_stats(stats)

    # 打印尾部几行，方便快速查看结果是否合理
    cols_to_show = [
        "close",
        "ma_short",
        "ma_long",
        "position",
        "bh_equity",
        "strategy_equity",
    ]
    existing_cols = [c for c in cols_to_show if c in data.columns]

    print("\n回测结果（尾部 5 行预览）:")
    print(data[existing_cols].tail().round(4))


if __name__ == "__main__":
    main()
