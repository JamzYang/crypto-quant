"""数据加载与预处理模块

主要职责：
- 从本地 CSV 文件加载单标的 OHLCV 数据
- 统一时间索引与排序
- 提供基础收益率计算辅助函数

注意：
- 不依赖具体交易所，只假定有“日期列 + 收盘价列”。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ohlcv_csv(
    file_path: str | Path,
    date_col: str = "date",
    price_col: str = "close",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """从 CSV 中加载单标的 OHLCV 或价格数据。

    参数:
        file_path: CSV 文件路径，可以是相对路径或绝对路径。
        date_col: 日期列名称，默认假定为 "date"。
        price_col: 收盘价列名称，默认假定为 "close"。
        parse_dates: 是否将日期列解析为 pandas 的 datetime 类型。

    返回:
        以日期为索引、按时间升序排序的 DataFrame。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    df = pd.read_csv(path)

    # 尝试自动识别日期列，兼容常见命名
    if date_col not in df.columns:
        candidate_date_cols = [date_col, "timestamp", "datetime", "Date", "Timestamp"]
        found = None
        for col in candidate_date_cols:
            if col in df.columns:
                found = col
                break
        if found is None:
            raise ValueError(
                f"CSV 中找不到日期列: {date_col}，可用列为: {list(df.columns)}"
            )
        date_col = found

    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(by=date_col)
    df = df.set_index(date_col)

    # 尝试自动识别收盘价列
    if price_col not in df.columns:
        candidate_price_cols = [price_col, "Close", "close_price", "price"]
        found = None
        for col in candidate_price_cols:
            if col in df.columns:
                found = col
                break
        if found is None:
            raise ValueError(
                f"CSV 中找不到收盘价列: {price_col}，可用列为: {list(df.columns)}"
            )
        price_col = found

    # 统一列名，方便后续策略代码书写
    if price_col != "close":
        df = df.rename(columns={price_col: "close"})

    return df


def add_return_columns(
    df: pd.DataFrame,
    price_col: str = "close",
    log_return: bool = False,
) -> pd.DataFrame:
    """在数据中添加日收益率列。

    参数:
        df: 以日期为索引的价格数据。
        price_col: 价格列名称。
        log_return: 是否额外计算对数收益率。

    返回:
        带有简单收益率（列名: ret）以及可选对数收益率（列名: log_ret）的 DataFrame。
    """
    data = df.copy()
    if price_col not in data.columns:
        raise KeyError(f"DataFrame 中不存在价格列: {price_col}")

    # 简单日收益率
    data["ret"] = data[price_col].pct_change().fillna(0.0)

    # 可选的对数收益率
    if log_return:
        data["log_ret"] = np.log(data[price_col] / data[price_col].shift(1)).fillna(0.0)

    return data
