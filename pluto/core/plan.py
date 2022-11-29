import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import talib as ta
from coretypes import BarsArray, FrameType
from omicron.extensions import math_round
from omicron.models.stock import Stock
from omicron.talib import polyfit


def magic_numbers(close: float, opn: float, low: float) -> List[float]:
    """根据当前的收盘价、开盘价和最低价，猜测整数支撑、最低位支撑及开盘价支撑价格

    当前并未针对开盘价和最低价作特殊运算，将其返回仅仅为了使用方便考虑（比如，一次性打印出所有的支撑价）

    猜测支撑价主要使用整数位和均线位。

    Example:
        >>> magic_numbers(9.3, 9.3, 9.1)
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
        for i in np.arange(lower, upper, step):
            numbers.append(round(i, 1))
            numbers.append(round(i + half_step, 1))
    else:
        for i in range(int(close * 9), int(close * 10)):
            if i / (10 * close) < 0.97 and i / (10 * close) > 0.9:
                numbers.append(i / 10)

    numbers.extend((opn, low))
    return numbers


def predict_next_ma(
    ma: Iterable[float], win: int, err_thresh: float = 3e-3
) -> Tuple[float, Tuple]:
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

    return None, ()


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

        pred_ma, extra = predict_next_ma(ma, win)
        if pred_ma is not None:
            vx, _ = extra
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


def name2code(name):
    """临时性用以计划的函数，需要添加到omicron"""
    result = Stock.fuzzy_match(name)
    if len(result) == 1:
        return list(result.keys())[0]
    else:
        return None


async def weekend_score(
    name: str, start: datetime.date
) -> Tuple[str, float, float, float, float]:
    """用以每周末给当周所选股进行评价的函数

    Args:
        name: 股票名
        start: 买入日期
    """
    code = name2code(name)
    if code is None:
        raise ValueError(f"股票名({name})错误")

    bars = await Stock.get_bars_in_range(code, FrameType.DAY, start)
    opn = bars["open"][0]
    close = bars["close"][-1]
    high = np.max(bars["high"])
    returns = close / opn - 1
    return name, opn, close, high, returns


async def group_weekend_score(names: str, start: datetime.date):
    """对一组同一时间买入的股票，计算到评估日（调用时）的表现"""
    results = []
    for name in names:
        results.append(await weekend_score(name, start))

    df = pd.DataFrame(results, columns=["股票", "开盘价", "周五收盘", "最高", "收盘收益"])
    return df.style.format(
        {
            "开盘价": lambda x: f"{x:.2f}",
            "周五收盘": lambda x: f"{x:.2f}",
            "最高": lambda x: f"{x:.2f}",
            "收盘收益": lambda x: f"{x:.1%}",
        }
    )


async def round_numbers(
    price: float, limit_prices: Tuple[float, float]
) -> Tuple[list, list]:
    """该函数列出当前传入行情数据下的整数支撑，整数压力。
    传入行情价格可以是日线级别，也可以是30分钟级别。
    整数的含义不仅是以元为单位的整数，还有以角为单位的整数。

    原理：
        支撑位整数是跌停价到传入价格之间的五档整数。
        压力位整数是传入价格价到涨停价之间的五档整数。
        除此之外，0.5，5和50的倍数也是常用支撑压力位。

    Example:
        >>> round_numbers(10.23, 10.2, 10.56, 10.11)
        ([9.9, 10.0, 10.1, 10.2], [10.6, 10.7, 10.8])

    Args:
        price: 传入的价格
        limit_prices: 传入需要计算整数支撑和压力的当天[跌停价，涨停价]

    Returns:
        返回有两个元素的Tuple, 第一个为支撑数列， 第二个为压力数列。
    """
    low_limit = limit_prices[0]
    high_limit = limit_prices[1]
    mean_limit = (low_limit + high_limit) / 2
    step = int(mean_limit) / 50  # 上下涨跌20%，再分10档，即2%左右为一档

    # 根据传入的价格，100以内保留一位小数，大于100只保留整数位
    if price < 10:
        step_ints = np.around(np.arange(low_limit, high_limit + step, step), 1)
        # 涨跌停价格之间0.5为倍数的所有数+十档
        int_low = low_limit - low_limit % 0.5 + 0.5
        five_times = np.around(np.arange(int_low, high_limit, 0.5), 1)
        total_int = np.append(step_ints, five_times)

    elif 10 <= price < 100:
        # 涨跌停价格之间0.5为倍数的所有数
        int_low = low_limit - low_limit % 0.5 + 0.5
        total_int = np.around(np.arange(int_low, high_limit, 0.5), 1)

    elif 100 <= price < 500:
        step_ints = np.around(np.arange(low_limit, high_limit + step, step), 0)
        # 涨跌停价格之间5为倍数的所有数
        int_low = low_limit - low_limit % 5 + 5
        five_times = np.around(np.arange(int_low, high_limit, 5), 1)
        total_int = np.append(step_ints, five_times)

    elif 500 <= price < 1000:
        # 涨跌停价格之间50为倍数的所有数
        int_low = low_limit - low_limit % 5 + 5
        total_int = np.around(np.arange(int_low, high_limit, 5), 1)

    else:
        # 涨跌停价格之间50为倍数的所有数
        int_low = low_limit - low_limit % 50 + 50
        total_int = np.around(np.arange(int_low, high_limit, 50), 1)

    total_int = total_int[(total_int <= high_limit) & (total_int >= low_limit)]
    total_int = np.append(low_limit, total_int)
    total_int = np.append(total_int, high_limit)
    total_int = np.unique(np.around(total_int, 2))

    support_list = total_int[total_int < price]
    resist_list = total_int[total_int > price]

    return support_list, resist_list


async def ma_sup_resist(code: str, bars: BarsArray) -> Tuple[dict, dict]:
    """均线支撑、压力位与当前k线周期同步，即当k线为日线时，
    使用日线均线计算；如果为30分钟，则使用30分钟均线；
    对超过涨跌停的支撑、压力位不显示；当有多个支撑位时，
    支撑位从上到下只显示3档；对压力位也是如此。

    包含wins = [5, 10, 20, 30, 60, 90, 120, 250]的均线；
    均线只有趋势向上才可做为支撑；
    只返回传输行情数据的最后一天的支撑均线， 压力均线。

    Args:
        code: 股票代码
        bars: 具有时间序列的行情数据，长度必须大于250。

    Returns：
        返回包含两个Dictionary类型的Tuple。
        第一个为均线支撑的Dictionary：keys是均线的win, values是(对应win的最后一个均线值， 均线值/最低价-1)；
        第二个为均线压力的Dictionary：keys是均线的win, values是(对应win的最后一个均线值， 均线值/最高价-1)。
    """
    assert len(bars) > 260, "Length of data must more than 260!"

    close = bars["close"]
    close = close.astype(np.float64)
    frame = bars["frame"][-1]
    low = bars["low"][-1]
    high = bars["high"][-1]
    open_ = bars["open"][-1]
    high_body = max(close[-1], open_)
    low_body = min(close[-1], open_)

    date = frame.item()
    limit_flag = await Stock.get_trade_price_limits(code, date, date)
    high_limit = limit_flag["high_limit"].item()
    low_limit = limit_flag["low_limit"].item()

    wins = [5, 10, 20, 30, 60, 90, 120, 250]
    ma_sups = {}
    ma_resist = {}
    for win in wins:
        all_ma = ta.MA(close, win)
        ma_trade = all_ma[-1] - all_ma[-5]
        ma = all_ma[-1]
        if (ma >= low_limit) and (ma <= low_body) and (ma_trade > 0):
            ma_sups[win] = ma, ma / low - 1
        elif (ma >= high_body) and (ma <= high_limit):
            ma_resist[win] = ma, ma / high - 1

    sorted_ma_sups = sorted(ma_sups.items(), key=lambda x: (x[1], x[0]), reverse=True)
    selected_ma_sups = dict(sorted_ma_sups[:3])

    sorted_ma_resist = sorted(ma_resist.items(), key=lambda x: (x[1], x[0]))
    selected_ma_resists = dict(sorted_ma_resist[:3])

    return selected_ma_sups, selected_ma_resists
