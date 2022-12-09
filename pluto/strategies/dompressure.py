import numpy as np
import talib as ta
from coretypes import BarsArray
from omicron.talib import moving_average, pct_error


def dom_pressure(bars: BarsArray, win: int) -> float:
    """判断穹顶压力

    原理：
    当最后七个收盘价均线向下拐头时， 对窗口为win的均线进行拟合，
    如果拟合误差小于特定值，最后一个bar的收盘价低于穹顶，
    视为压力确认有效。close以低于ma为主，且至少一个bar的high高于ma,确认压力。

    Args:
    bars: 具有时间序列的BarsArray, 其中必须包含收盘价，最高价，传入长度至少为37
    win: 均线窗口，win的值不超过30，比如：当win=10，窗口为10的收盘价移动平均值的穹顶压力

    Returns:
    返回float：最后七个bar的最高价冲过压力的数量/7,
    比如：最后七个bar中的第二,三个最高价冲过压力，返回2/7)
    最后七个bars没有穹顶压力，或传入数据不足37个，返回None。

    """
    assert win in (5, 10, 20, 30), "传入均线窗口必须[5, 10, 20, 30]中的数字！"
    if len(bars) < 37:
        return None

    if (win == 10) or (win == 5):
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
    # max_ma_hat = np.nanmax(ma_hat)

    # 二阶导：
    coef = list(p)
    convex = index * 6 * coef[0] + 2 * coef[1]

    if (
        (np.count_nonzero(convex <= 0) > 5)
        and (error < 3e-3)
        and (np.mean(convex) < convexity)
    ):
        # and (cls[-1] < max_ma_hat)
        hig_break = hig > ma
        hig_break_num = np.count_nonzero(hig_break)
        cls_under_num = np.count_nonzero(cls < ma)

        if win != 5:
            # close以低于ma为主，且至少一个bar的high高于ma,确认压力
            if (cls_under_num > 3) and (hig_break_num >= 1):  # and (not hig_break[-1])
                return np.count_nonzero(hig_break) / 7
        elif win == 5:
            if hig_break_num >= 3:  # and (not hig_break[-1])
                return np.count_nonzero(hig_break) / 7


def saucer_support(bars: BarsArray, win: int) -> float:
    """判断茶碟支撑

    如果以`win`为窗口的某段均线的最后7个点的连线构成凹曲线，并且
        1. 该期间内，收盘价以高于对应均线值为主，
        2. 至少存在一个bar,其最低价击穿（或者接近击穿）对应均线价，
        3. 最后一个bar收于对应均线价之上
    则认为该段均线对股价有支撑作用。

    Args:
        bars: 行情数据，长度必须大于win + 6
        win: 均线窗口

    Returns:
        最后七个bar中最低价冲击支撑的数量比
        比如：最后七个bar中的第二,三个最高价冲过压力，返回2/7)。如果不存在茶碟支撑，则返回None
    """
    assert len(bars) > win + 6, f"'bars'数组的长度必须大于{win + 6}"
    close = bars["close"]
    low = bars["low"]

    pct = close[1:] / close[:-1] - 1
    std = np.std(pct)

    ma = moving_average(close, win)[-7:]
    ma_hat = np.arange(7) * (ma[-1] - ma[0]) / 6 + ma[0]

    # 如果均线还存在交织的情况，认为不存在趋势
    interleave = ma_hat - ma
    if np.all(interleave >= 0) or np.all(interleave <= 0):
        return None

    # 自拐点之后，仍有股价击穿均线的情况，认为支撑不够强
    pmin = np.argmin(ma)
    if np.any(close[pmin:] < ma[pmin:]):
        return None

    # 存在对均线的测试，允许最低价高于ma的1%
    nbreaks = np.count_nonzero((low[-7:] * 0.99 < ma) & (close[-7:] >= ma))
    if nbreaks > 0:
        return nbreaks / 7

    return None
