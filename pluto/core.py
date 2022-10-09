import datetime
import logging
from typing import Optional, Tuple

import numpy as np
from coretypes import Frame, bars_dtype
from omicron import tf
from omicron.extensions import find_runs
from omicron.models.security import Security
from omicron.models.stock import Stock

logger = logging.getLogger(__name__)


async def vanilla_score(bars: bars_dtype, code: str = None) -> Tuple:
    """对买入信号发出之后一段时间的表现进行评价。

    规则：
        1. bars中的第一根bar为信号发出日(T0)。如果此时未涨停，则以收盘价作为买入价
        2. 如果第一天收盘已涨停，则使用第二天的开盘价作为买入价
        3. 分别计算T1, T2, ..., T(len(bars)-1)日的累计涨跌幅
        4. 计算累计最大涨幅
        5. 计算出现累计最大跌幅（上涨之前）
        6. 计算中除情况2之外，都使用收盘价计算。

    Args:
        bars: 包含信号发出日的行情数据
        code: 股票代码

    Returns:
        包含每日累计涨跌幅，最大涨幅和最大跌幅的元组。
    """
    pass
