"""阶段 2 第 4 块：综合样本内(IS) vs 样本外(OOS)回测脚本。

主要职责：
- 将历史数据按时间切分为「样本内 (Train)」和「样本外 (Test)」两段；
- 在这两段区间上分别运行阶段 2 已实现的三个核心策略：
  1. 双均线趋势 (MA Trend)
  2. 布林带均值回归 (Bollinger Mean Reversion)
  3. 趋势 + 波动率调仓 (Trend + Vol Adjust)
- 对比各策略在 IS 和 OOS 的表现，检查是否存在过拟合现象。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.stage2.data.loader import add_return_columns, load_ohlcv_csv
from src.stage2.strategies.ma_trend import MATrendConfig, MATrendStrategy
from src.stage2.strategies.bollinger_mean_reversion import (
    BollingerConfig,
    BollingerMeanReversionStrategy,
)
from src.stage2.strategies.volatility_adjustment import (
    VolatilityAdjustmentConfig,
    VolatilityAdjuster,
)
from src.stage2.backtests.bt_ma_trend_single_symbol import compute_performance_stats

logger = logging.getLogger(__name__)


def run_strategy_on_data(
    df: pd.DataFrame, strategy_name: str, initial_capital: float = 10000.0, fee_rate: float = 0.001
) -> pd.Series:
    """在给定数据段上运行指定策略，返回资金曲线序列。"""
    data = df.copy()
    
    # 策略路由
    if strategy_name == "ma_trend":
        # 1. 双均线趋势
        config = MATrendConfig(short_window=20, long_window=60)
        strat = MATrendStrategy(config)
        data = strat.generate_signals(data)
        # 趋势策略持仓
        data["final_pos"] = data["position"]

    elif strategy_name == "bollinger":
        # 2. 布林带均值回归
        config = BollingerConfig(window=20, num_std=2.0)
        strat = BollingerMeanReversionStrategy(config)
        data = strat.generate_signals(data)
        # 均值回归策略持仓
        data["final_pos"] = data["position"]

    elif strategy_name == "trend_vol":
        # 3. 趋势 + 波动率调仓
        # 先生成基础趋势
        trend_conf = MATrendConfig(short_window=20, long_window=60)
        trend_strat = MATrendStrategy(trend_conf)
        data = trend_strat.generate_signals(data)
        data = data.rename(columns={"position": "trend_pos"})
        
        # 再叠加波动率
        vol_conf = VolatilityAdjustmentConfig(
            window=30,
            low_threshold=0.01,
            high_threshold=0.03,
            base_position_col="trend_pos"
        )
        vol_adj = VolatilityAdjuster(vol_conf)
        data = vol_adj.adjust_positions(data)
        # 最终持仓
        data["final_pos"] = data["position_vol"]

    elif strategy_name == "buy_and_hold":
        # 4. 简单持有
        data["final_pos"] = 1.0

    else:
        raise ValueError(f"未知策略名: {strategy_name}")

    # 计算收益
    # 毛收益
    data["ret_gross"] = data["final_pos"] * data["ret"]
    # 换手费
    pos_change = data["final_pos"].diff().abs().fillna(data["final_pos"].abs())
    data["fee"] = pos_change * fee_rate
    # 净收益
    data["strategy_ret"] = data["ret_gross"] - data["fee"]
    # 资金曲线
    equity = initial_capital * (1.0 + data["strategy_ret"]).cumprod()
    
    return equity


def run_comprehensive_comparison(
    csv_path: Path,
    split_date: str = "2023-01-01",
    initial_capital: float = 10000.0,
) -> None:
    """运行综合对比：样本内 vs 样本外。"""
    
    # 1. 加载全量数据
    df_all = load_ohlcv_csv(csv_path)
    df_all = add_return_columns(df_all)
    
    # 2. 切分样本内 (IS) 和 样本外 (OOS)
    # split_date 之前为 IS，之后为 OOS
    mask_is = df_all.index < split_date
    mask_oos = df_all.index >= split_date
    
    df_is = df_all.loc[mask_is].copy()
    df_oos = df_all.loc[mask_oos].copy()
    
    print(f"数据切分日: {split_date}")
    print(f"样本内 (IS) 区间: {df_is.index.min().date()} ~ {df_is.index.max().date()} (共 {len(df_is)} 天)")
    print(f"样本外 (OOS) 区间: {df_oos.index.min().date()} ~ {df_oos.index.max().date()} (共 {len(df_oos)} 天)")
    print("-" * 60)

    strategies = ["buy_and_hold", "ma_trend", "bollinger", "trend_vol"]
    
    results = []

    for strat in strategies:
        # 跑样本内
        equity_is = run_strategy_on_data(df_is, strat, initial_capital)
        ret_series_is = equity_is.pct_change().fillna(0.0)
        stats_is = compute_performance_stats(ret_series_is)
        
        # 跑样本外
        equity_oos = run_strategy_on_data(df_oos, strat, initial_capital)
        ret_series_oos = equity_oos.pct_change().fillna(0.0)
        stats_oos = compute_performance_stats(ret_series_oos)
        
        results.append({
            "Strategy": strat,
            "IS_Return": stats_is["total_return"],
            "IS_Sharpe": stats_is["sharpe"],
            "IS_MaxDD": stats_is["max_drawdown"],
            "OOS_Return": stats_oos["total_return"],
            "OOS_Sharpe": stats_oos["sharpe"],
            "OOS_MaxDD": stats_oos["max_drawdown"],
        })

    # 3. 打印对比表
    res_df = pd.DataFrame(results)
    # 格式化输出
    print("\n>>> 综合复盘对比表 (IS vs OOS) <<<")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 60)
    
    # 简单结论打印
    print("简要解读提示：")
    print("1. 如果 IS 表现很好，但 OOS 表现很差（甚至亏损），可能存在过拟合。")
    print("2. 如果 OOS 表现与 IS 接近，说明策略风格较稳定，适应性尚可。")
    print("3. 注意 OOS 区间的市场风格（是牛是熊？），有些策略天然只适应特定行情。")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "dist" / "btc_data.csv"
    
    # 根据实际数据范围 (2024-11-18 ~ 2025-11-17)，选择中间点作为切分
    run_comprehensive_comparison(default_csv, split_date="2025-06-01")


if __name__ == "__main__":
    main()
