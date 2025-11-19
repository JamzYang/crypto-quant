"""布林带均值回归策略实现。

主要职责：
- 基于收盘价计算布林带三条轨道（中轨、上轨、下轨）；
- 根据价格与布林带位置关系，生成均值回归方向的持仓信号；
- 仅负责信号生成，不直接参与资金曲线与绩效计算。

本实现采用「只做多」版本：
- 价格跌破下轨：认为价格明显偏离中枢，准备做多博弈回归中轨；
- 价格回到中轨及以上：认为本次回归目标已达到，准备平多离场；
- 其余时间信号保持不变，避免在中间噪音区间频繁反复横跳。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BollingerConfig:
    """布林带策略参数配置。

    属性说明：
        window: 布林带滚动窗口长度（天数），通常取 20。
        num_std: 上下轨与中轨之间的标准差倍数，通常取 2.0。
        price_col: 用于计算布林带的价格列名，默认使用收盘价 "close"。
    """

    window: int = 20
    num_std: float = 2.0
    price_col: str = "close"

    def validate(self) -> None:
        """校验参数合法性。"""
        if self.window <= 0:
            raise ValueError("布林带窗口长度必须为正整数")
        if self.num_std <= 0:
            raise ValueError("标准差倍数必须为正数")


class BollingerMeanReversionStrategy:
    """布林带均值回归策略类。

    职责：
    - 计算布林带三条轨道（中轨、上轨、下轨）；
    - 生成只做多方向的均值回归信号；
    - 输出用于回测的持仓序列以及开仓/平仓事件标记。

    当前实现的核心交易规则：
    - 当收盘价跌破下轨：视为价格严重偏离中枢，记为 signal = 1（准备做多）；
    - 当收盘价回到中轨及以上：视为均值回归目标完成，记为 signal = 0（准备平多）；
    - 其余时间保持前一时刻的 signal 不变，避免在噪音区间频繁进出；
    - 实际持仓 position 为 signal 向后平移一日，以规避未来函数。
    """

    def __init__(self, config: BollingerConfig | None = None) -> None:
        """初始化布林带均值回归策略。

        参数:
            config: 策略参数配置，如果为 None 则使用默认配置。
        """
        self.config: BollingerConfig = config or BollingerConfig()
        self.config.validate()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于价格数据生成布林带及均值回归信号。

        处理步骤概览：
        1. 使用滚动窗口计算中轨（移动平均）与滚动标准差；
        2. 得到上轨、下轨：中轨 ± num_std × 标准差；
        3. 构造原始信号 signal_raw：
           - 收盘价 < 下轨 → signal_raw = 1（准备做多）；
           - 收盘价 >= 中轨 → signal_raw = 0（准备平多 / 空仓）；
           - 其他情况 → signal_raw 为空；
        4. 对 signal_raw 做前向填充并用 0 初始化，得到平滑后的 signal；
        5. 实际持仓 position = signal 向后平移 1 日；
        6. 标记 enter_long / exit_long 事件，便于后续画图和复盘。

        参数:
            df: 以日期为索引、包含价格列的 DataFrame。

        返回:
            附加以下列后的 DataFrame 副本：
            - bb_mid: 布林带中轨；
            - bb_upper: 布林带上轨；
            - bb_lower: 布林带下轨；
            - signal: 当期信号（0/1，仅用于逻辑判断）；
            - position: 实际持仓（0/1，用于回测收益计算）；
            - enter_long: 是否在该日信号从 0 变为 1（开多）；
            - exit_long: 是否在该日信号从 1 变为 0（平多）。
        """
        if self.config.price_col not in df.columns:
            raise KeyError(f"数据中不存在价格列: {self.config.price_col}")

        data = df.copy()

        # 计算布林带三条轨道
        rolling = data[self.config.price_col].rolling(
            window=self.config.window,
            min_periods=1,
        )
        data["bb_mid"] = rolling.mean()
        data["bb_std"] = rolling.std(ddof=0)

        data["bb_upper"] = data["bb_mid"] + self.config.num_std * data["bb_std"]
        data["bb_lower"] = data["bb_mid"] - self.config.num_std * data["bb_std"]

        # 原始信号：仅在「极端偏离」与「回归完成」处更新，其余时间保持不变
        signal_raw = pd.Series(np.nan, index=data.index, dtype=float)
        price_series = data[self.config.price_col]

        signal_raw.loc[price_series < data["bb_lower"]] = 1.0
        signal_raw.loc[price_series >= data["bb_mid"]] = 0.0

        # 前向填充 + 初始空仓，得到平滑后的信号
        data["signal"] = signal_raw.ffill().fillna(0.0).astype(int)

        # 为避免未来函数，实际持仓使用前一日信号
        data["position"] = data["signal"].shift(1).fillna(0).astype(int)

        # 标记开仓 / 平仓事件
        prev_signal = data["signal"].shift(1).fillna(0).astype(int)
        data["enter_long"] = (prev_signal == 0) & (data["signal"] == 1)
        data["exit_long"] = (prev_signal == 1) & (data["signal"] == 0)

        # 不再需要的中间列可以根据需要在外部丢弃，这里保留以便调试
        return data
