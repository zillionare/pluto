from typing import Tuple

import numpy as np
import talib as ta
from coretypes import BarsArray
from omicron.talib import pct_error


def dom_pressure(bars: BarsArray, win: int, convexity: float = None) -> float:
    """判断穹顶压力

    原理：
    当最后七个收盘价均线向下拐头时， 对窗口为win的均线进行拟合，
    如果拟合误差小于特定值，最后一个bar的收盘价低于穹顶，
    视为压力确认有效。close以低于ma为主，且至少一个bar的high高于ma,确认压力。

    Args:
    bars: 具有时间序列的BarsArray, 其中必须包含收盘价，最高价，传入长度至少为37
    win: 均线窗口，win的值不超过30，比如：当win=10，窗口为10的收盘价移动平均值的穹顶压力
    convexity: 穹顶弧度限制，该数为负数，越小弧度越明显

    Returns:
    返回float：最后七个bar的最高价冲过压力的数量/7,
    比如：最后七个bar中的第二,三个最高价冲过压力，返回2/7)
    最后七个bars没有穹顶压力，或传入数据不足37个，返回None。

    """
    assert win in (10, 20, 30), "传入均线窗口必须[10, 20, 30]中的数字！"
    if len(bars) < 37:
        return None

    if win == 10:
        convexity = -3e-3
    elif win == 20:
        convexity = -2e-3
    elif win == 30:
        convexity = -1e-3

    close = bars["close"]
    high = bars["high"]
    close = close.astype(np.float64)

    index = np.arange(7)
    ma = ta.MA(close, win)[-7:]
    cls = close[-7:]
    hig = high[-7:]
    z = np.polyfit(index, ma, 3)
    p = np.poly1d(z)
    ma_hat = p(index)
    error = pct_error(ma, ma_hat)
    max_ma_hat = np.nanmax(ma_hat)

    # 二阶导：
    coef = list(p)
    convex = index * 6 * coef[0] + 2 * coef[1]

    if (
        (np.count_nonzero(convex < 0) > 5)
        and (error < 3e-3)
        and (cls[-1] < max_ma_hat)
        and (np.mean(convex) < convexity)
    ):
        hig_break = hig > ma
        hig_break_num = np.count_nonzero(hig_break)
        cls_under_num = np.count_nonzero(cls < ma)

        # close以低于ma为主，且至少一个bar的high高于ma,确认压力
        if (cls_under_num > 3) and (hig_break_num >= 1) and (not hig_break[-1]):
            return np.count_nonzero(hig_break) / 7
