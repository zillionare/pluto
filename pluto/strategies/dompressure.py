from typing import Tuple

import numpy as np
import talib as ta
from coretypes import BarsArray
from omicron.talib import pct_error


def dom_pressure(bars: BarsArray, win: int) -> Tuple:
    """判断穹顶压力

    原理：
    当最后七个收盘价均线向下拐头时， 对win = (10, 20,30)的均线进行拟合，
    如果拟合误差小于3e-3，且当前处于右侧，则如果日线或者30分钟线上冲不过，
    视为压力确认有效。close以低于ma为主，且至少一个bar的high高于ma,确认压力。

    Args:
    bars: 具有时间序列的BarsArray, 其中必须包含收盘价，最高价，传入长度至少为37。
    win: 均线参数，当win=10，计算参数为10的收盘价移动平均值的穹顶压力。

    Returns:
    返回Tuple：第一个为判断最后七个bar是否出现了穹顶压力的booling值，
    第二个为最后七个bar的最高价是否冲过压力的booling list。
    如果没有最后七个bars没有穹顶压力，或传入数据不足37个，返回None。

    """

    if len(bars) < 37:
        return None

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
    argmax_ma_hat = np.argmax(ma_hat)

    # 二阶导：
    coef = list(p)
    convex = index * 6 * coef[0] + 2 * coef[1]

    if (np.count_nonzero(convex < 0) > 5) and (error < 3e-3) and (argmax_ma_hat <= 4):
        hig_break = hig > ma
        hig_break_num = np.count_nonzero(hig_break)
        cls_under_num = np.count_nonzero(cls < ma)

        # close以低于ma为主，且至少一个bar的high高于ma,确认压力
        if (cls_under_num > 3) and (hig_break_num >= 1) and (not hig_break[-1]):
            return True, hig_break
