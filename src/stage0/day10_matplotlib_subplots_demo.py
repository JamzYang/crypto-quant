from __future__ import annotations

"""Day10：Matplotlib 2x2 子图练习

本脚本对应阶段0学习计划中 Day10 的可视化练习，目标：
1. 使用 `plt.subplots` 创建 2x2 子图
2. 绘制价格曲线、收益率直方图、散点图和多资产价格曲线
3. 熟悉 Matplotlib 基本 API：plot/scatter/hist/legend 等

用法：
    直接运行本脚本：
    python day10_matplotlib_subplots_demo.py
"""

import logging
import platform
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# 模块级 logger
logger = logging.getLogger(__name__)


# 设置中文字体，兼容不同操作系统，与项目其他脚本保持一致
if platform.system() == "Darwin":
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
elif platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def generate_demo_data(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成 Day10 练习用的示例数据

    数据设计：
    - prices: 模拟价格曲线（随机游走）
    - returns: 根据 prices 计算的简单收益率，用于直方图
    - x, y: 散点图使用的两个序列，y 在 prices 基础上叠加噪声
    - x1, x2: 模拟两种资产（例如 BTC / ETH）的价格曲线

    Args:
        n_points: 序列长度，即模拟的数据点数量

    Returns:
        一个六元组 (prices, returns, x, y, x1, x2)
    """

    logger.info("生成示例数据，点数: %d", n_points)

    np.random.seed(42)

    # 时间索引
    x: np.ndarray = np.arange(n_points, dtype=float)

    # 模拟价格随机游走
    noise: np.ndarray = np.random.normal(loc=0.0, scale=1.0, size=n_points)
    prices: np.ndarray = 100.0 + np.cumsum(noise)

    # 简单收益率（相对第一天的价格，便于直观理解）
    returns: np.ndarray = (prices - prices[0]) / prices[0]

    # 散点图 y：在价格基础上叠加噪声
    scatter_noise: np.ndarray = np.random.normal(loc=0.0, scale=5.0, size=n_points)
    y: np.ndarray = prices + scatter_noise

    # 两条资产价格曲线，模拟 BTC / ETH
    btc_noise: np.ndarray = np.random.normal(loc=0.1, scale=1.0, size=n_points)
    eth_noise: np.ndarray = np.random.normal(loc=0.15, scale=1.2, size=n_points)
    x1: np.ndarray = 20000.0 + np.cumsum(btc_noise)
    x2: np.ndarray = 1500.0 + np.cumsum(eth_noise)

    return prices, returns, x, y, x1, x2


def plot_day10_subplots(
    prices: np.ndarray,
    returns: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    output_path: str = "day10_matplotlib_subplots.png",
) -> None:
    """按照学习计划绘制 2x2 子图示例

    子图布局：
    - 左上：价格曲线
    - 右上：收益率分布直方图
    - 左下：散点图
    - 右下：两条资产价格曲线

    Args:
        prices: 价格序列
        returns: 收益率序列
        x: 散点图 x 轴数据
        y: 散点图 y 轴数据
        x1: 资产1价格序列（例如 BTC）
        x2: 资产2价格序列（例如 ETH）
        output_path: 图表保存路径
    """

    logger.info("开始绘制 Day10 Matplotlib 2x2 子图，输出文件: %s", output_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Day10 - Matplotlib 可视化示例", fontsize=16, fontweight="bold")

    # 左上：价格曲线
    ax_price = axes[0, 0]
    ax_price.plot(prices, color="#1f77b4", linewidth=2)
    ax_price.set_title("价格曲线")
    ax_price.set_xlabel("时间")
    ax_price.set_ylabel("价格")
    ax_price.grid(True, alpha=0.3)

    # 右上：收益率分布直方图
    ax_hist = axes[0, 1]
    ax_hist.hist(returns, bins=30, color="#ff7f0e", alpha=0.7, edgecolor="black")
    ax_hist.set_title("收益率分布")
    ax_hist.set_xlabel("收益率 (相对初始价格)")
    ax_hist.set_ylabel("频数")
    ax_hist.grid(True, alpha=0.3)

    # 左下：散点图
    ax_scatter = axes[1, 0]
    ax_scatter.scatter(x, y, color="#2ca02c", alpha=0.6, edgecolors="none")
    ax_scatter.set_title("散点图示例")
    ax_scatter.set_xlabel("X")
    ax_scatter.set_ylabel("Y")
    ax_scatter.grid(True, alpha=0.3)

    # 右下：两条价格曲线
    ax_multi = axes[1, 1]
    ax_multi.plot(x1, label="BTC", color="#9467bd", linewidth=2)
    ax_multi.plot(x2, label="ETH", color="#d62728", linewidth=2)
    ax_multi.set_title("多资产价格曲线")
    ax_multi.set_xlabel("时间")
    ax_multi.set_ylabel("价格")
    ax_multi.legend()
    ax_multi.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=120)
    plt.close(fig)

    logger.info("Day10 图表已保存: %s", output_path)


def main() -> None:
    """脚本入口：生成示例数据并绘制 2x2 子图

    该函数用于命令行直接运行本脚本时的入口，方便快速验证 Day10 练习。
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("启动 Day10 Matplotlib 子图示例脚本")

    prices, returns, x, y, x1, x2 = generate_demo_data(n_points=120)
    plot_day10_subplots(prices, returns, x, y, x1, x2)


if __name__ == "__main__":
    main()
