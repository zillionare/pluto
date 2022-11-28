import logging
from itertools import combinations
from typing import Iterable, Tuple

import numpy as np
from coretypes import BarsArray, bars_dtype
from omicron import tf
from coretypes import FrameType
from omicron.extensions import price_equal
from omicron.models.stock import Stock
from omicron.talib import peaks_and_valleys, moving_average

logger = logging.getLogger(__name__)


async def vanilla_score(
    bars: bars_dtype, code: str = None, frametype: FrameType = FrameType.DAY
) -> Tuple:
    """对买入信号发出之后一段时间的表现进行评价。

    规则：
        1. bars中的第一根bar为信号发出时间。如果此时未涨停，则以收盘价作为买入价，
           以信号发出日为T0日，分别计算T1, T2, ..., T(len(bars)-1)日的累计涨跌幅。
        2. 如果第一根bar已涨停(此处默认10%为涨停限制)，则使用第二天的开盘价作为买入价，
           以信号发出日为T0日，分别计算T2, T3, ..., T(len(bars)-2)日的累计涨跌幅
        3. 计算累计最大涨幅。
        4. 计算出现累计最大跌幅（上涨之前）
        5. 计算中除情况2之外，都使用收盘价计算。

    Args:
        bars: 包含信号发出日的行情数据
        code: 股票代码
        frametype: 传入带有时间序列数据的时间类型，
        只有两种时间类型可被接受：FrameType.DAY or FrameType.MIN30

    Returns:
        包含每日累计涨跌幅，最大涨幅和最大跌幅的元组。
    """
    returns = []
    max_returns = []
    mdds = []

    assert frametype in (
        FrameType.DAY,
        FrameType.MIN30,
    ), "'frametype' must be either FrameType.DAY or FrameType.MIN30!"
    if frametype == FrameType.DAY:
        assert (
            len(bars) >= 3
        ), "must provide a day frametype array with at least 3 length!"
        limit_flag = (
            await Stock.trade_price_limit_flags(
                code, bars["frame"][0].item(), bars["frame"][0].item()
            )
        )[0][0]
        # 如果检测当天涨停，第二天开盘价未涨停买入，第二天开始收盘价作为收益。
        if limit_flag & (
            (bars["open"][1] - bars["close"][0]) / bars["close"][0] < 0.099
        ):
            price_np = np.append(bars["open"][1], bars["close"][2:])
            returns = (price_np[1:] - price_np[0]) / price_np[0]
            max_return = np.nanmax(returns)
            max_returns.append(max_return)
            max_index = np.argmax(returns)
            # 防止涨停之前的最大跌幅为空值，取到最大值
            to_max = returns[: max_index + 1]
            mdd = np.nanmin(to_max)
            if mdd < 0:
                mdds.append(mdd)

        # 如果检测当天可以买进，则直接买入，后五天的收盘价作为收益，开盘涨停则不考虑
        elif not limit_flag:
            returns = (bars["close"][1:] - bars["close"][0]) / bars["close"][0]
            max_return = np.nanmax(returns)
            max_returns.append(max_return)
            max_index = np.argmax(returns)
            # 防止涨停之前的最大跌幅为空值，取到最大值
            to_max = returns[: max_index + 1]
            mdd = np.nanmin(to_max)
            if mdd < 0:
                mdds.append(mdd)

    elif frametype == FrameType.MIN30:
        assert (
            len(bars) >= 24
        ), "must prrovide a min30 framtype array with at least 24 length!"
        first_frame = bars["frame"][0].item()
        first_day = tf.day_shift(first_frame, 0)
        second_day = tf.day_shift(first_frame, 1)
        second_open_time = tf.combine_time(second_day, 10)
        second_day_end_index = np.where(
            bars["frame"] == tf.combine_time(second_day, 15)
        )[0].item()
        # 检测第二天开始日收盘价
        day_bars = bars[second_day_end_index:][::8]
        # 获取当天涨停价
        first_limit_price = (
            await Stock.get_trade_price_limits(code, first_day, first_day)
        )[0][1]
        # 获取第二天涨停价
        second_limit_price = (
            await Stock.get_trade_price_limits(code, second_day, second_day)
        )[0][1]
        # 检测点已涨停，第二天开盘未涨停,开盘价买入，从第三天收盘价开始计算收益率：
        if price_equal(bars["close"][0], first_limit_price) and (
            bars["open"][bars["frame"] == second_open_time].item() != second_limit_price
        ):
            price = np.append(
                bars["open"][bars["frame"] == second_open_time], day_bars["close"][1:]
            )
            returns = (price[1:] - price[0]) / price[0]
            max_return = np.nanmax(returns)
            max_returns.append(max_return)
            max_index = np.argmax(returns)
            # 防止涨停之前的最大跌幅为空值，取到最大值
            to_max = returns[: max_index + 1]
            mdd = np.nanmin(to_max)
            if mdd < 0:
                mdds.append(mdd)
        # 检测点未涨停，直接买入，第二天收盘价开始计算收益率：
        elif bars["close"][0] != first_limit_price:
            price = np.append(bars["close"][0], day_bars["close"])
            returns = (price[1:] - price[0]) / price[0]
            max_return = np.nanmax(returns)
            max_returns.append(max_return)
            max_index = np.argmax(returns)
            # 防止涨停之前的最大跌幅为空值，取到最大值
            to_max = returns[: max_index + 1]
            mdd = np.nanmin(to_max)
            if mdd < 0:
                mdds.append(mdd)

    return returns, max_returns, mdds


def parallel_score(mas: Iterable[float]) -> float:
    """求均线排列分数。

    返回值介于[0, 1]之间。如果为1，则最后一期均线值为全多头排列，即所有的短期均线都位于所有的长期均线之上；如果为0，则是全空头排列，即所有的短期均线都位于所有的长期均线之下。值越大，越偏向于多头排列；值越小，越偏向于空头排列。

    Args:
        mas: 移动平均线数组

    Returns:
        排列分数，取值在[0,1]区间内。
    """
    count = 0
    total = 0

    for a, b in combinations(mas, 2):
        total += 1
        if a >= b:
            count += 1

    return count / total


def last_wave(ts: np.array, max_win: int = 60):
    """返回顶点距离，以及波段涨跌幅

    Args:
        ts: 浮点数的时间序列
        max_win: 在最大为`max_win`的窗口中检测波段。设置这个值是出于性能考虑，但也可能存在最后一个波段长度大于60的情况。
    """
    ts = ts[-max_win:]
    pv = peaks_and_valleys(ts)
    prev = np.argwhere(pv != 0).flatten()[-2]
    return len(ts) - prev, ts[-1] / ts[prev] - 1


def adjust_close_at_pv(
    bars: BarsArray, flag: int
) -> Tuple[np.array, np.array, np.array]:
    """将close序列中的峰谷值替换成为对应的high/low。

        通过指定flag为（-1， 0， 1）中的任一个，以决定进行何种替换。如果flag为-1，则将close中对应谷处的数据替换为low；如果flag为1，则将close中对应峰处的数据替换为high。如果为0，则返回两组数据。

        最后，返回替换后的峰谷标志序列（1为峰，-1为谷）
    Args:
        bars: 输入的行情数据
        flag: 如果为-1,表明只替换low; 如果为1，表明只替换high；如果为0，表明全换

    Returns:
        返回替换后的序列: 最低点替换为low之后的序列，最高点替换为high之后的序列，以及峰谷标记
    """
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]

    pvs = peaks_and_valleys(close)
    last = pvs[-1]
    # 如果最后的bar是从高往下杀，低于昨收，peaks_and_valleys会判为-1，此时可能要用high代替close
    if high[-1] > close[-2] and last == -1:
        pvs[-1] = 1

    # 如果最后的bar是底部反转，高于昨收，peaks_and_valleys会判定为1，但此时仍然可能要用low代替close
    if low[-1] < close[-2] and last == 1:
        pvs[-1] = -1

    for p in np.argwhere(pvs == 1).flatten():  # 对p前后各2元素检查谁最大
        if p < 2:
            pvs[p] = 0
            i = np.argmax(high[:2])
            pvs[i] = 1
        elif p >= len(pvs) - 2:
            pvs[p] = 0
            i = np.argmax(high[-2:])
            pvs[i - 2] = 1
        else:
            i = np.argmax(high[p - 2 : p + 3])
            if i != 2:
                pvs[p] = 0
                pvs[p + i - 2] = 1

    for v in np.argwhere(pvs == -1).flatten():
        if v < 2:
            pvs[v] = 0
            i = np.argmin(low[:2])
            pvs[i] = -1
        elif v >= len(pvs) - 2:
            pvs[v] = 0
            i = np.argmin(low[-2:])
            pvs[i - 2] = -1
        else:
            i = np.argmin(low[v - 2 : v + 3])
            if i != 2:
                pvs[v] = 0
                pvs[v + i - 2] = -1

    if flag == -1:
        return np.where(pvs == -1, low, close), None, pvs
    elif flag == 0:
        return np.where(pvs == -1, low, close), np.where(pvs == 1, high, close), pvs
    else:
        return None, np.where(pvs == 1, high, close), pvs


def convex_short_signal(bars: BarsArray, ex_info=False):
    """评估曲线的升降性

    如果均线中间的点都落在端点连线上方，则该函数为凸函数；反之，则为凹函数。使用点到连线的差值的
    平均值来表明曲线的凹凸性。进一步地，我们将凹凸性引申为升降性，并且对已经走成直线的均线，我们
    使用平均涨跌幅来表明其升降性，从而使得在凹函数、凸函数和直线三种情况下，函数的返回值都能表明
    均线的未来升降趋势。
    Args:
        bars:    行情数据，以30分钟为宜。
        ex_info: 是否返回均线的详细评分信息

    Returns:
        如果出现空头信号，则返回1,否则返回0；如果ex_info为True，还将返回详细评估分数。
    """
    close = bars["close"]

    scores = []
    for win in (5, 10, 20):
        n = 10 if win == 20 else 7
        ma = moving_average(close, win)[-n:]

        #         coeff = np.polyfit((0, n - 1), (ma[0], ma[-1]), deg=1)
        #         ma_hat = np.poly1d(coeff)(np.arange(n))
        ma_hat = np.arange(n) * (ma[-1] - ma[0]) / (n - 1) + ma[0]

        score = np.mean(1 - ma[1 : n - 1] / ma_hat[1 : n - 1])
        if abs(score) < 1e-3:
            score = (ma[-1] / ma[0] - 1) / n

        scores.append(score * 100)

    scores = np.array(scores)
    if np.count_nonzero(scores < 0) >= 2 and scores[-1] < 0:
        flag = 1
    else:
        flag = 0

    if ex_info:
        return flag, scores
    else:
        return flag
