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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
            plt.plot(data.index, data["bh_equity"], label="Buy and Hold", alpha=0.7)
            plt.plot(data.index, data["strategy_equity"], label="MA Trend Strategy", alpha=0.9)
            plt.title("BTC MA Trend Strategy vs Buy and Hold")
            plt.xlabel("Date")
            plt.ylabel("Equity Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib is not installed, skip plotting.")

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


def _plot_kline_with_signals(
    data: pd.DataFrame,
    output_dir: Path,
    filename: str = "ma_trend_kline.png",
) -> Path:
    """生成带双均线与买卖点标记的 K 线图并保存到指定目录。"""
    required_price_cols = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_price_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"绘制 K 线图需要列 {required_price_cols}，当前缺少: {missing_cols}")

    for col in ["ma_short", "ma_long", "position", "golden_cross", "death_cross"]:
        if col not in data.columns:
            raise KeyError(f"绘制信号图需要列: {col}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制 K 线：上下影线 + 实体
    up = data["close"] >= data["open"]
    down = ~up

    ax.vlines(data.index, data["low"], data["high"], color="black", linewidth=0.5)

    ax.bar(
        data.index[up],
        (data.loc[up, "close"] - data.loc[up, "open"]),
        bottom=data.loc[up, "open"],
        width=0.8,
        color="red",
        align="center",
        edgecolor="none",
    )
    ax.bar(
        data.index[down],
        (data.loc[down, "open"] - data.loc[down, "close"]),
        bottom=data.loc[down, "close"],
        width=0.8,
        color="green",
        align="center",
        edgecolor="none",
    )

    # 绘制短期 / 长期均线
    ax.plot(data.index, data["ma_short"], label="Short MA", color="orange", linewidth=1.0)
    ax.plot(data.index, data["ma_long"], label="Long MA", color="blue", linewidth=1.0)

    # 金叉 / 死叉标记（用收盘价位置）
    golden_idx = data.index[data["golden_cross"]]
    death_idx = data.index[data["death_cross"]]
    ax.scatter(
        golden_idx,
        data.loc[golden_idx, "close"],
        marker="^",
        color="gold",
        s=50,
        label="Golden Cross",
        zorder=5,
    )
    ax.scatter(
        death_idx,
        data.loc[death_idx, "close"],
        marker="v",
        color="black",
        s=50,
        label="Death Cross",
        zorder=5,
    )

    # 买入 / 卖出点：根据持仓变动
    position_change = data["position"].diff().fillna(data["position"])
    buy_mask = position_change == 1
    sell_mask = position_change == -1

    buy_idx = data.index[buy_mask]
    sell_idx = data.index[sell_mask]
    ax.scatter(
        buy_idx,
        data.loc[buy_idx, "close"] * 0.995,
        marker="^",
        color="green",
        s=60,
        label="Buy",
        zorder=6,
    )
    ax.scatter(
        sell_idx,
        data.loc[sell_idx, "close"] * 1.005,
        marker="v",
        color="red",
        s=60,
        label="Sell",
        zorder=6,
    )

    ax.set_title("BTC MA Trend Strategy - K Line and Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def _plot_interactive_kline_with_signals(
    data: pd.DataFrame,
    output_path: Path,
) -> Path:
    """使用 Plotly 生成交互式 K 线与信号图并保存为 HTML。

    图中包含：
    - K 线（OHLC）
    - 短期 / 长期均线
    - 金叉 / 死叉位置
    - 买入 / 卖出点
    """
    required_price_cols = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_price_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Interactive K-line requires columns {required_price_cols}, missing: {missing_cols}")

    for col in ["ma_short", "ma_long", "position", "golden_cross", "death_cross"]:
        if col not in data.columns:
            raise KeyError(f"Interactive K-line requires column: {col}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    # 蜡烛图
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
            opacity=0.8,
        )
    )

    # 均线
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["ma_short"],
            mode="lines",
            name="Short MA",
            line=dict(color="orange", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["ma_long"],
            mode="lines",
            name="Long MA",
            line=dict(color="blue", width=1),
        )
    )

    # 金叉 / 死叉
    golden_mask = data["golden_cross"]
    death_mask = data["death_cross"]
    fig.add_trace(
        go.Scatter(
            x=data.index[golden_mask],
            y=data.loc[golden_mask, "close"],
            mode="markers",
            name="Golden Cross",
            marker=dict(
                symbol="triangle-up",
                color="gold",
                size=14,
                line=dict(color="white", width=1),
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index[death_mask],
            y=data.loc[death_mask, "close"],
            mode="markers",
            name="Death Cross",
            marker=dict(
                symbol="triangle-down",
                color="black",
                size=14,
                line=dict(color="white", width=1),
            ),
        )
    )

    # 买入 / 卖出点：根据持仓变动
    position_change = data["position"].diff().fillna(data["position"])
    buy_mask = position_change == 1
    sell_mask = position_change == -1

    fig.add_trace(
        go.Scatter(
            x=data.index[buy_mask],
            y=data.loc[buy_mask, "close"],
            mode="markers",
            name="Buy",
            marker=dict(
                symbol="triangle-up",
                color="green",
                size=15,
                line=dict(color="white", width=1),
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index[sell_mask],
            y=data.loc[sell_mask, "close"],
            mode="markers",
            name="Sell",
            marker=dict(
                symbol="triangle-down",
                color="red",
                size=15,
                line=dict(color="white", width=1),
            ),
        )
    )

    fig.update_layout(
        title="BTC MA Trend Strategy - Interactive K Line and Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=800,
    )

    # 默认聚焦在最近一段数据，避免全历史压缩在一起导致线过陡
    if len(data.index) > 200:
        fig.update_xaxes(range=[data.index[-200], data.index[-1]])

    # 使用 inline 模式嵌入 Plotly JS，生成的 HTML 可单文件离线查看
    fig.write_html(str(output_path), include_plotlyjs="inline")
    return output_path


def main() -> None:
    """命令行入口函数。

    默认使用项目根目录下 dist/btc_data.csv 作为数据源，
    你也可以修改 csv_path 指向自己的数据文件。
    """
    # 通过文件位置反推项目根目录: .../crypto-quant/src/stage2/backtests/*.py
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "dist" / "btc_data.csv"
    kline_dir = project_root / "dist" / "stage2"
    interactive_path = kline_dir / "ma_trend_kline_interactive.html"

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

    # 生成并保存带信号的静态 K 线图
    try:
        kline_path = _plot_kline_with_signals(data, kline_dir)
        print(f"K 线图已保存到: {kline_path}")
    except KeyError as exc:
        print(f"生成 K 线图失败: {exc}")

    # 生成交互式 K 线 HTML
    try:
        interactive = _plot_interactive_kline_with_signals(data, interactive_path)
        print(f"交互式 K 线图已保存到: {interactive}")
    except KeyError as exc:
        print(f"生成交互式 K 线图失败: {exc}")

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
