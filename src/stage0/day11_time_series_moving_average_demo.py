from __future__ import annotations

import logging

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _configure_chinese_font() -> None:
    """配置 Matplotlib 中文字体

    思路：
    1. 从常见的中英文字体名称列表中，找到当前系统真实存在的字体
    2. 找到后设置为 sans-serif 的首选字体
    3. 兼容 macOS 和 Windows，最大化减少中文乱码和缺字告警
    """
    candidate_keywords: list[str] = [
        "PingFang",  # macOS 苹果系统中文字体族
        "Hiragino",  # macOS 常见中文字体
        "Heiti",  # macOS 黑体相关
        "Songti",  # macOS 宋体相关
        "STSong",
        "STHeiti",
        "Microsoft YaHei",  # Windows 常见中文字体
        "SimHei",  # Windows 黑体
        "SimSun",  # Windows 宋体
    ]

    chosen_font: str | None = None
    for font in font_manager.fontManager.ttflist:
        for keyword in candidate_keywords:
            if keyword in font.name:
                chosen_font = font.name
                break
        if chosen_font is not None:
            break

    if chosen_font is not None:
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [chosen_font, "DejaVu Sans"]
        logger.info("使用中文字体: %s", chosen_font)
    else:
        # 如果没有找到合适的中文字体，至少保证英文字体配置合理
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["DejaVu Sans"]

    # 解决坐标轴负号显示为方块的问题
    rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


class TimeSeriesMovingAverageDemo:
    """时间序列与移动平均示例

    这个类演示如何：
    1. 构造一段模拟的价格时间序列（带趋势 + 随机波动）
    2. 使用 pandas 计算简单移动平均（SMA）
    3. 使用 pandas 计算指数加权移动平均（EMA）
    4. 绘制价格与均线，帮助直观理解趋势和平滑
    """

    def __init__(self, seed: int = 42) -> None:
        """初始化示例类

        参数:
            seed: 随机种子，保证每次运行结果一致，方便对比和调试
        """
        self.seed: int = seed
        np.random.seed(self.seed)

    def generate_price_series(self, n_days: int = 200) -> pd.DataFrame:
        """生成模拟价格时间序列

        思想：
        - 先生成一串“日收益率”，包含一个小的正期望（长期略微向上）
        - 再通过累乘 (1 + 日收益率) 得到价格序列

        这样得到的数据更接近真实市场行为：价格非平稳，收益率更接近平稳。

        参数:
            n_days: 交易天数

        返回:
            包含日期索引和收盘价列 close 的 DataFrame
        """
        dates = pd.date_range(start="2024-01-01", periods=n_days, freq="B")

        # 模拟“日收益率”：均值为 0.0005（≈0.05%），标准差为 0.02（≈2% 波动）
        daily_returns = np.random.normal(loc=0.0005, scale=0.02, size=n_days)

        # 将日收益率转换为价格序列，起始价格设为 100
        prices = 100 * np.cumprod(1 + daily_returns)

        df = pd.DataFrame({"date": dates, "close": prices})
        df.set_index("date", inplace=True)
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """在价格数据上添加简单移动平均和指数加权移动平均

        新增列:
            - SMA_5: 5 日简单移动平均（短期）
            - SMA_20: 20 日简单移动平均（中期）
            - EMA_12: 12 日指数加权移动平均（更强调近期价格）

        参数:
            df: 至少包含 close 列的 DataFrame

        返回:
            原 DataFrame 的复制品，加入了三条均线
        """
        df_ma = df.copy()

        # 简单移动平均：对过去 window 天的收盘价做算术平均
        df_ma["SMA_5"] = df_ma["close"].rolling(window=5).mean()
        df_ma["SMA_20"] = df_ma["close"].rolling(window=20).mean()

        # 指数加权移动平均：越新的数据权重越大
        df_ma["EMA_12"] = df_ma["close"].ewm(span=12, adjust=False).mean()

        return df_ma

    def plot_price_and_averages(self, df: pd.DataFrame) -> None:
        """绘制价格与均线在同一张图上

        通过图形直观感受：
            - 原始价格曲线更“抖动”，噪音较多
            - 均线更加平滑，反映的是“趋势”和“中长期方向”
            - 短均线更贴近价格，长均线更迟钝但更稳定

        参数:
            df: 至少包含 close、SMA_5、SMA_20、EMA_12 的 DataFrame
        """
        plt.figure(figsize=(12, 6))

        # 绘制原始价格
        plt.plot(df.index, df["close"], label="收盘价 close", color="black", linewidth=1)

        # 绘制简单移动平均
        plt.plot(df.index, df["SMA_5"], label="5日简单均线 SMA_5", color="blue", linewidth=1.2)
        plt.plot(df.index, df["SMA_20"], label="20日简单均线 SMA_20", color="orange", linewidth=1.2)

        # 绘制指数加权移动平均
        plt.plot(df.index, df["EMA_12"], label="12日指数均线 EMA_12", color="green", linewidth=1.2)

        plt.title("模拟价格时间序列与移动平均示例")
        plt.xlabel("日期")
        plt.ylabel("价格")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_demo(self) -> None:
        """运行完整示例：生成价格 + 计算均线 + 绘图

        使用方式：
            if __name__ == "__main__":
                demo = TimeSeriesMovingAverageDemo()
                demo.run_demo()

        你可以在此基础上自由修改参数（如天数、波动率），观察曲线形状的变化。
        """
        logger.info("开始运行时间序列移动平均示例")
        df_price = self.generate_price_series(n_days=200)
        df_with_ma = self.add_moving_averages(df_price)
        self.plot_price_and_averages(df_with_ma)


if __name__ == "__main__":
    demo = TimeSeriesMovingAverageDemo(seed=42)
    demo.run_demo()
