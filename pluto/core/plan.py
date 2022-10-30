from typing import Dict, Iterable, List, Tuple

import numpy as np
from omicron.extensions import array_math_round, math_round
from omicron.talib import polyfit


def magic_numbers(close: float, opn: float, low: float) -> List[float]:
    """根据当前的收盘价、开盘价和最低价，猜测整数支撑、最低位支撑及开盘价支撑价格

    当前并未针对开盘价和最低价作特殊运算，将其返回仅仅为了使用方便考虑（比如，一次性打印出所有的支撑价）

    猜测支撑价主要使用整数位和均线位。

    Example:
        >>> guess_support_prices(9.3, 9.3, 9.1)
        [8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.3, 9.1]

    Args:
        close: 当前收盘价
        opn: 当前开盘价
        low: 当前最低价

    Returns:
        可能的支撑价格列表
    """
    numbers = []

    order = len(str(int(close))) - 1
    step = 10 ** (order - 1)
    half_step = step // 2 or 0.5

    lower = int((close * 0.9 // step) * step)
    upper = int((close // step) * step)

    if close >= 10:
        for i in range(lower, upper, step):
            numbers.append(i)
            numbers.append(i + half_step)
    else:
        for i in range(int(close * 9), int(close * 10)):
            if i / (10 * close) < 0.97 and i / (10 * close) > 0.9:
                numbers.append(i / 10)

    numbers.extend((opn, low))
    return numbers


def predict_next_ma(
    ma: Iterable[float], win: int, err_thresh: float = 3e-3
) -> Tuple[float, float]:
    """预测下一个均线值。

    对于短均线如(5, 10, 20, 30),我们使用二阶拟合，如果不能拟合，则返回None, None
    对于长均线，使用一阶拟合。

    Args:
        ma: 均线数据
        win: 均线窗口。
        err_thresh: 进行均线拟合时允许的误差。

    Returns:
        预测的均线值，及其它信息，比如对短均线(<=30)进行预测时，还可能返回顶点坐标。
    """
    if win <= 30 and len(ma) >= 7:
        ma = ma[-7:]
        err, (a, b, c), (vx, _) = polyfit(ma / ma[0], deg=2)
        if err < err_thresh:
            f = np.poly1d((a, b, c))
            ma_hat = f(np.arange(8))
            pred_ma = math_round(ma_hat[-1] * ma[0], 2)
            return pred_ma, (a, vx)
        else:
            return None, ()

    if win > 30 and len(ma) >= 3:
        ma = ma[-3:]
        _, (a, b) = polyfit(ma, deg=1)
        f = np.poly1d((a, b))
        ma_hat = f([0, 1, 2, 3])
        pred_ma = math_round(ma_hat[-1], 2)
        return pred_ma, (a,)

    return None, None


def ma_support_prices(mas: Dict[int, np.array], c0: float) -> Dict[int, float]:
    """计算下一周期的均线支撑价

    返回值中，如果某一均线对应值为负数，则表明该均线处于下降状态，不具有支撑力；如果为None, 则表明无法进行计算或者不适用（比如距离超过一个涨跌停）；否则返回正的支撑价。

    Args:
        mas: 移动均线序列，可包含np.NAN.键值必须为[5, 10, 20, 30, 60, 120, 250]中的一个。
        c0: 当前收盘价
    Returns:
        均线支撑价格.
    """
    # 判断短均线(5, 10, 20, 30)中有无向下拐头
    pred_prices = {}

    c0 = math_round(c0, 2)
    for win in (5, 10, 20, 30):
        ma = mas.get(win)
        if ma is None or c0 < math_round(ma[-1], 2):
            pred_prices[win] = None
            continue

        ma = ma[-7:]
        if np.count_nonzero(np.isnan(ma)) >= 1:
            pred_prices[win] = None
            continue

        pred_ma, (_, vx) = predict_next_ma(ma, win)
        if pred_ma is not None:
            vx = min(max(round(vx), 0), 6)

            if pred_ma < ma[vx] or pred_ma > c0:
                pred_prices[win] = -1
            else:
                gap = pred_ma / c0 - 1
                if abs(gap) < 0.099:
                    pred_prices[win] = pred_ma
                else:
                    pred_prices[win] = None
        elif ma[-1] < ma[-3]:
            pred_prices[win] = -1
        else:
            pred_prices[win] = None

    # 判断长均线走势及预期
    for win in (60, 120, 250):
        ma = mas.get(win)
        if ma is None or c0 < ma[-1]:
            pred_prices[win] = None
            continue

        pred_ma, (*_,) = predict_next_ma(ma, win)
        gap = pred_ma / c0 - 1

        if pred_ma > ma[-2]:
            pred_prices[win] = pred_ma
        elif pred_ma < ma[-2]:
            pred_prices[win] = -1
        else:
            pred_prices[win] = pred_ma

    return pred_prices
