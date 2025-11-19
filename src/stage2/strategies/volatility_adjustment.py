"""波动率调仓模块。

主要职责：
- 基于收益率的滚动标准差刻画「近期波动水平」；
- 将市场简单划分为低/中/高波动三个 regime；
- 根据波动 regime 对已有策略的基础仓位进行缩放（调仓），得到波动率调整后的仓位。

设计思路（与阶段 2 第 3 块对应）：
- 使用日收益率的滚动标准差作为波动率 proxy；
- 使用**固定阈值**划分低/中/高波动，避免在全样本上用未来数据估计分位数；
- 仅使用「当前及历史数据」计算波动率，并在第 t 日收盘后，用该波动率决定第 t+1 日的仓位系数，规避未来函数。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VolRegime = Literal["low", "medium", "high"]


@dataclass
class VolatilityAdjustmentConfig:
    """波动率调仓参数配置。

    属性说明：
        window: 计算滚动标准差的窗口长度（天数），例如 30；
        low_threshold: 低波动与中等波动的分界阈值（基于日收益率标准差）；
        high_threshold: 中等波动与高波动的分界阈值；
        ret_col: 日收益率列名，默认为 "ret"；
        base_position_col: 已有策略基础仓位列名，默认为 "position"；
        position_low: 低波动 regime 下的仓位系数（例如 1.0 = 满仓）；
        position_medium: 中等波动 regime 下的仓位系数（例如 0.6）；
        position_high: 高波动 regime 下的仓位系数（例如 0.3 或 0.0）。
    """

    window: int = 30
    low_threshold: float = 0.01
    high_threshold: float = 0.03
    ret_col: str = "ret"
    base_position_col: str = "position"
    position_low: float = 1.0
    position_medium: float = 0.6
    position_high: float = 0.3

    def validate(self) -> None:
        """校验参数合法性。"""
        if self.window <= 0:
            raise ValueError("波动率窗口长度必须为正整数")
        if self.low_threshold <= 0 or self.high_threshold <= 0:
            raise ValueError("波动率阈值必须为正数")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("低波动阈值应小于高波动阈值")
        if not (0.0 <= self.position_high <= self.position_medium <= self.position_low <= 1.0):
            raise ValueError("仓位系数应满足 0 <= high <= medium <= low <= 1")


class VolatilityAdjuster:
    """波动率调仓执行类。

    职责：
    - 在给定 DataFrame 上计算滚动波动率；
    - 基于阈值划分低/中/高波动 regime；
    - 在已有策略基础仓位列的基础上，生成波动率调整后的仓位列。

    使用方式：
    - 先通过其他策略（如双均线趋势）生成基础仓位列 base_position_col；
    - 然后调用 `adjust_positions`，得到包含以下新增列的 DataFrame：
      - vol: 日收益率滚动标准差；
      - vol_regime: 字符串标记 "low" / "medium" / "high"；
      - position_multiplier: 对应 regime 的仓位系数；
      - position_vol: 波动率调整后的仓位（基础仓位 × 仓位系数）。
    """

    def __init__(self, config: VolatilityAdjustmentConfig | None = None) -> None:
        """初始化波动率调仓模块。

        参数:
            config: 波动率调仓参数配置，如为 None 则使用默认配置。
        """
        self.config: VolatilityAdjustmentConfig = config or VolatilityAdjustmentConfig()
        self.config.validate()

    def adjust_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """在已有策略基础仓位之上叠加波动率调仓。

        参数:
            df: 包含日收益率列和基础仓位列的 DataFrame，索引为日期。

        返回:
            附加以下列后的 DataFrame 副本：
            - vol: 日收益率滚动标准差；
            - vol_regime: 低/中/高波动标签；
            - position_multiplier: 仓位系数；
            - position_vol: 波动率调整后的仓位（可为 0~1 之间的小数）。
        """
        if self.config.ret_col not in df.columns:
            raise KeyError(f"数据中不存在收益率列: {self.config.ret_col}")
        if self.config.base_position_col not in df.columns:
            raise KeyError(f"数据中不存在基础仓位列: {self.config.base_position_col}")

        data = df.copy()

        # 计算基于日收益率的滚动波动率
        rolling = data[self.config.ret_col].rolling(
            window=self.config.window,
            min_periods=1,
        )
        data["vol"] = rolling.std(ddof=0).fillna(0.0)

        # 划分波动率 regime
        regime_series = pd.Series("medium", index=data.index, dtype="object")
        regime_series.loc[data["vol"] <= self.config.low_threshold] = "low"
        regime_series.loc[data["vol"] >= self.config.high_threshold] = "high"
        data["vol_regime"] = regime_series.astype("category")

        # 根据 regime 决定仓位系数
        multiplier = pd.Series(
            self.config.position_medium,
            index=data.index,
            dtype=float,
        )
        multiplier.loc[data["vol_regime"] == "low"] = self.config.position_low
        multiplier.loc[data["vol_regime"] == "high"] = self.config.position_high
        data["position_multiplier"] = multiplier

        # 生成波动率调整后的仓位：基础仓位 × 仓位系数
        base_position = data[self.config.base_position_col].astype(float)
        data["position_vol"] = (base_position * data["position_multiplier"]).astype(float)

        logger.debug(
            "完成波动率调仓计算: window=%d, low_threshold=%.4f, high_threshold=%.4f",
            self.config.window,
            self.config.low_threshold,
            self.config.high_threshold,
        )

        return data
