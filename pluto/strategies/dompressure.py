from typing import Tuple

import numpy as np
import talib as ta
from coretypes import BarsArray

from pluto.core.metrics import convex_score


def dom_pressure(bars: BarsArray, win: int) -> Tuple:
    """判断穹顶压力

    原理：
    当最后七个收盘价均线向下拐头时， 对窗口为win的均线进行拟合，
    如果拟合误差小于特定值，最后一个bar的收盘价低于穹顶，
    视为压力确认有效。close以低于ma为主，且至少一个bar的high高于ma,确认压力。

    Args:
    bars: 具有时间序列的BarsArray, 其中必须包含收盘价，最高价，传入长度至少为37
    win: 均线窗口，win的值不超过30，比如：当win=10，窗口为10的收盘价移动平均值的穹顶压力

    Returns:
    返回float：最后七个bar的最高价冲过压力的数量/10,
    比如：最后十个bar中的第二,三个最高价冲过压力，返回2/10)
    最后七个bars没有穹顶压力，或传入数据不足37个，返回None。

    """
    assert len(bars) >= 30, "传入行情数据的不得少于30！"

    close = bars["close"]
    high = bars["high"]
    low = bars["low"]

    ma = ta.MA(close, win)[-10:]
    score = convex_score(ma)

    # 最后一个bar如果全部在均线之上（low > ma)则不满足条件；如果收盘价在ma上方不远，符合条件
    last_bar_status = (
        (close[-1] < ma[-1] * 1.01) and (low[-1] < ma[-1]) and ma[-1] < np.max(ma)
    )
    if score < -0.5 and last_bar_status and ma[-1] > ma[0]:
        breakouts = np.count_nonzero((high[-10:] >= ma))
        return breakouts / 10, score
