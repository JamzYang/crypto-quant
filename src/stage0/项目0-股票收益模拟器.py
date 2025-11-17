from __future__ import annotations

"""项目0：股票收益模拟器

目标：
1. 生成 1 年（252 个交易日）的模拟股价（价格随机游走）
2. 计算日收益率
3. 绘制价格曲线和收益率分布
4. 计算累计收益、最大回撤
5. 在控制台输出统计报告

用法：
    直接运行本脚本：
    python 项目0-股票收益模拟器.py
"""
import logging
import platform
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# 模块级 logger
logger = logging.getLogger(__name__)


# 设置中文字体，兼容不同操作系统
if platform.system() == "Darwin":
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
elif platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class StockSimulationConfig:
    """股票价格模拟配置

    属性:
        initial_price: 初始价格
        mu: 日均期望收益（例如 0.001 表示 0.1%）
        sigma: 日收益波动率（标准差），例如 0.02 表示 2%
        n_days: 模拟的交易日数量
        seed: 随机数种子，便于复现实验；为 None 时不固定种子
    """

    initial_price: float = 100.0
    mu: float = 0.001
    sigma: float = 0.02
    n_days: int = 252
    seed: int | None = 42


@dataclass
class StockSimulationResult:
    """股票收益模拟结果数据结构

    属性:
        prices: 模拟得到的价格序列，长度为 n_days
        daily_returns: 日简单收益率序列，长度为 n_days，首日为 NaN
        cumulative_return: 整个周期的累计收益率，例如 0.25 表示 25%
        max_drawdown: 最大回撤（负值），例如 -0.3 表示 -30%
        max_drawdown_start: 最大回撤开始点在序列中的索引
        max_drawdown_end: 最大回撤结束点在序列中的索引
    """

    prices: np.ndarray
    daily_returns: np.ndarray
    cumulative_return: float
    max_drawdown: float
    max_drawdown_start: int
    max_drawdown_end: int


def simulate_prices(config: StockSimulationConfig) -> np.ndarray:
    """根据给定参数模拟股票价格随机游走

    使用正态分布生成日收益率，然后通过累乘得到价格路径。

    Args:
        config: 股票模拟配置

    Returns:
        一维数组，长度为 config.n_days，表示每日收盘价
    """

    logger.info(
        "开始模拟股价: 初始价格=%.2f, mu=%.4f, sigma=%.4f, 天数=%d",
        config.initial_price,
        config.mu,
        config.sigma,
        config.n_days,
    )

    if config.seed is not None:
        logger.debug("设置随机种子: %d", config.seed)
        np.random.seed(config.seed)

    # 生成日收益率（简单收益率）
    daily_returns: np.ndarray = np.random.normal(
        loc=config.mu,
        scale=config.sigma,
        size=config.n_days,
    )

    # 价格随机游走：P_t = P_0 * Π(1 + r_i)
    prices: np.ndarray = config.initial_price * np.cumprod(1 + daily_returns)

    logger.info("股价模拟完成")
    return prices


def calculate_daily_returns(prices: np.ndarray) -> np.ndarray:
    """根据价格序列计算日简单收益率

    简单收益率定义为: r_t = (P_t - P_{t-1}) / P_{t-1}

    Args:
        prices: 价格序列

    Returns:
        日收益率序列，首个元素为 NaN，其余为有效值
    """

    if prices.ndim != 1:
        raise ValueError("prices 必须是一维数组")

    logger.info("开始计算日收益率，样本数=%d", prices.shape[0])

    daily_returns: np.ndarray = np.empty_like(prices, dtype=float)
    daily_returns[0] = np.nan
    daily_returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

    logger.info("日收益率计算完成")
    return daily_returns


def calculate_cumulative_return(daily_returns: np.ndarray) -> float:
    """计算整个周期的累计收益率

    Args:
        daily_returns: 日简单收益率序列，首个元素可以为 NaN

    Returns:
        累计收益率，例如 0.2 表示 20%
    """

    valid_returns: np.ndarray = daily_returns[~np.isnan(daily_returns)]
    if valid_returns.size == 0:
        raise ValueError("日收益率序列为空，无法计算累计收益")

    cumulative_return: float = float(np.prod(1 + valid_returns) - 1.0)
    logger.info("累计收益率=%.4f", cumulative_return)
    return cumulative_return


def calculate_max_drawdown(daily_returns: np.ndarray) -> Tuple[float, int, int]:
    """根据日收益率计算最大回撤

    最大回撤基于净值曲线定义：
        净值曲线 C_t = Π(1 + r_i)
        回撤 D_t = (C_t - max(C_0..C_t)) / max(C_0..C_t)

    Args:
        daily_returns: 日简单收益率序列，首个元素可以为 NaN

    Returns:
        一个三元组 (max_drawdown, start_index, end_index):
        - max_drawdown: 最大回撤（负值，例如 -0.25 表示 -25%）
        - start_index: 对应峰值所在索引
        - end_index: 对应谷值所在索引
    """

    valid_returns: np.ndarray = daily_returns[~np.isnan(daily_returns)]
    if valid_returns.size == 0:
        raise ValueError("日收益率序列为空，无法计算最大回撤")

    equity_curve: np.ndarray = np.cumprod(1 + valid_returns)
    running_max: np.ndarray = np.maximum.accumulate(equity_curve)
    drawdowns: np.ndarray = (equity_curve - running_max) / running_max

    end_index: int = int(np.argmin(drawdowns))
    max_drawdown: float = float(drawdowns[end_index])

    # 通过回溯找到对应的峰值位置
    start_index: int = int(np.argmax(equity_curve[: end_index + 1]))

    logger.info(
        "最大回撤=%.4f, 开始索引=%d, 结束索引=%d",
        max_drawdown,
        start_index,
        end_index,
    )
    return max_drawdown, start_index, end_index


def plot_price_and_returns(prices: np.ndarray, daily_returns: np.ndarray, output_path: str = "股票收益模拟器_图表.png") -> None:
    """绘制价格曲线和日收益率分布图

    左图为价格时间序列，右图为日收益率直方图。

    Args:
        prices: 模拟价格序列
        daily_returns: 日简单收益率序列
        output_path: 图表保存路径
    """

    logger.info("开始绘制图表，输出文件: %s", output_path)

    valid_returns: np.ndarray = daily_returns[~np.isnan(daily_returns)] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("股票收益模拟器", fontsize=16, fontweight="bold")

    # 价格曲线
    ax_price = axes[0]
    ax_price.plot(prices, color="#1f77b4", linewidth=2)
    ax_price.set_title("模拟价格曲线")
    ax_price.set_xlabel("交易日")
    ax_price.set_ylabel("价格")
    ax_price.grid(True, alpha=0.3)

    # 收益率分布
    ax_ret = axes[1]
    ax_ret.hist(valid_returns, bins=30, color="#ff7f0e", alpha=0.7, edgecolor="black")
    mean_ret: float = float(valid_returns.mean())
    ax_ret.axvline(mean_ret, color="red", linestyle="--", linewidth=2, label=f"均值={mean_ret:.2f}%")
    ax_ret.set_title("日收益率分布")
    ax_ret.set_xlabel("日收益率 (%)")
    ax_ret.set_ylabel("频数")
    ax_ret.legend()
    ax_ret.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=120)
    plt.close(fig)

    logger.info("图表已保存: %s", output_path)


def run_simulation(config: StockSimulationConfig) -> StockSimulationResult:
    """执行完整的股票收益模拟流程

    包含步骤：
    1. 模拟价格
    2. 计算日收益率
    3. 计算累计收益
    4. 计算最大回撤
    5. 绘制图表

    Args:
        config: 股票模拟配置

    Returns:
        封装模拟结果的数据结构
    """

    logger.info("===== 开始股票收益模拟 =====")

    prices: np.ndarray = simulate_prices(config)
    daily_returns: np.ndarray = calculate_daily_returns(prices)
    cumulative_return: float = calculate_cumulative_return(daily_returns)
    max_drawdown, dd_start, dd_end = calculate_max_drawdown(daily_returns)

    plot_price_and_returns(prices, daily_returns)

    logger.info("===== 股票收益模拟完成 =====")

    return StockSimulationResult(
        prices=prices,
        daily_returns=daily_returns,
        cumulative_return=cumulative_return,
        max_drawdown=max_drawdown,
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
    )


def print_report(result: StockSimulationResult) -> None:
    """在控制台输出模拟结果统计报告

    报告内容包括：
    - 平均日收益率
    - 日收益率标准差
    - 累计收益率
    - 最大回撤
    """

    valid_returns: np.ndarray = result.daily_returns[~np.isnan(result.daily_returns)]

    mean_return: float = float(valid_returns.mean())
    std_return: float = float(valid_returns.std())

    print("\n" + "=" * 60)
    print("股票收益模拟报告")
    print("=" * 60)
    print(f"样本天数: {valid_returns.size} 天")
    print(f"平均日收益率: {mean_return * 100:.3f}%")
    print(f"日收益率标准差: {std_return * 100:.3f}%")
    print(f"累计收益率: {result.cumulative_return * 100:.2f}%")
    print(f"最大回撤: {result.max_drawdown * 100:.2f}%")
    print(f"最大回撤开始索引: {result.max_drawdown_start}")
    print(f"最大回撤结束索引: {result.max_drawdown_end}")
    print("=" * 60 + "\n")


def main() -> None:
    """主函数：运行股票收益模拟器

    该函数用于命令行直接运行本脚本时的入口。
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("启动股票收益模拟器")

    config = StockSimulationConfig(
        initial_price=100.0,
        mu=0.001,
        sigma=0.02,
        n_days=252,
        seed=42,
    )

    result = run_simulation(config)
    print_report(result)


if __name__ == "__main__":
    main()
