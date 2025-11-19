"""双均线趋势跟随策略实现。

主要职责：
- 基于收盘价计算短期 / 长期移动平均线
- 根据均线关系生成持仓信号（0 = 空仓，1 = 满仓）
- 标记金叉 / 死叉位置，方便后续可视化与复盘

设计原则：
- 仅负责“信号生成”，不直接参与资金曲线与绩效计算。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MATrendConfig:
    """双均线策略配置。

    属性说明：
        short_window: 短期均线窗口长度（天数）。
        long_window: 长期均线窗口长度（天数）。
        price_col: 用于计算均线的价格列名。
    """

    short_window: int = 20
    long_window: int = 60
    price_col: str = "close"

    def validate(self) -> None:
        """校验参数合法性。"""
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("均线窗口必须为正整数")
        if self.short_window >= self.long_window:
            raise ValueError("短期均线窗口应严格小于长期均线窗口")


class MATrendStrategy:
    """双均线趋势跟随策略类。

    职责：
    - 根据配置计算短/长均线
    - 生成不含未来信息的持仓序列（position）
    - 标记金叉 / 死叉事件

    当前实现：
    - 只考虑“做多 / 空仓”两种状态，不做反向做空。
    """

    def __init__(self, config: MATrendConfig | None = None) -> None:
        """初始化策略。

        参数:
            config: 策略参数配置，如果为 None 则使用默认配置 (20, 60)。
        """
        self.config = config or MATrendConfig()
        self.config.validate()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于价格数据生成双均线信号。

        处理逻辑：
        1. 计算短期 / 长期简单移动均线；
        2. 当短期均线高于长期均线时，视为“看多”区间，signal = 1，否则 signal = 0；
        3. 为避免未来函数，实际持仓 position = signal 向后平移 1 个交易日；
        4. 当 signal 从 0 变为 1 时标记为金叉，从 1 变为 0 时标记为死叉。

        参数:
            df: 以日期为索引、至少包含价格列的 DataFrame。

        返回:
            附加以下列后的副本：
            - ma_short: 短期均线
            - ma_long: 长期均线
            - signal: 当期信号（0/1，仅用于逻辑判断）
            - position: 实际持仓（0/1，用于回测）
            - golden_cross: 是否为金叉点（bool）
            - death_cross: 是否为死叉点（bool）
        """
        if self.config.price_col not in df.columns:
            raise KeyError(f"数据中不存在价格列: {self.config.price_col}")

        data = df.copy()

        # 计算移动平均线
        data["ma_short"] = (
            data[self.config.price_col]
            .rolling(window=self.config.short_window, min_periods=1)
            .mean()
        )
        data["ma_long"] = (
            data[self.config.price_col]
            .rolling(window=self.config.long_window, min_periods=1)
            .mean()
        )

        # 生成原始信号：短期均线在长期均线之上则看多
        data["signal"] = 0
        data.loc[data["ma_short"] > data["ma_long"], "signal"] = 1

        # 为避免未来函数，实际持仓使用前一日信号
        data["position"] = data["signal"].shift(1).fillna(0).astype(int)

        # 标记金叉 / 死叉位置，方便画图与复盘
        prev_signal = data["signal"].shift(1).fillna(0)
        data["golden_cross"] = (prev_signal == 0) & (data["signal"] == 1)
        data["death_cross"] = (prev_signal == 1) & (data["signal"] == 0)

        return data
