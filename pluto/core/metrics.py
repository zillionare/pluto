import logging
from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import numpy as np
import talib as ta
from coretypes import BarsArray, FrameType, bars_dtype
from numpy.typing import NDArray
from omicron import tf
from omicron.extensions import find_runs, price_equal, top_n_argpos
from omicron.models.stock import Stock
from omicron.talib import moving_average, peaks_and_valleys, rsi_watermarks

from pluto.strategies.dompressure import dom_pressure

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

    返回值介于[0, 1]之间。如果为1，则最后一期均线值为全多头排列，即所有的短期均线都位于所有的长期均线之上；
    如果为0，则是全空头排列，即所有的短期均线都位于所有的长期均线之下。
    值越大，越偏向于多头排列；值越小，越偏向于空头排列。

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

        通过指定flag为（-1， 0， 1）中的任一个，以决定进行何种替换。

        如果flag为-1，则将close中对应谷处的数据替换为low；如果flag为1，则将close中对应峰处的数据替换为high。

        如果为0，则返回两组数据。

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


def convex_signal(
    bars: BarsArray = None,
    wins=(5, 10, 20),
    mas: List[NDArray] = None,
    ex_info=False,
    thresh: float = 3e-3,
) -> Tuple[int, List[float]]:
    """根据均线的升降性，判断是否发出买入或者卖出信号

    调用者需要保证参与计算的均线，至少有10个以上的有效值（即非np.NaN).
    Args:
        bars: 行情数据。
        wins: 均线生成参数
        ex_info: 是否返回均线的详细评分信息
        thresh: 决定均线是按直线拟合还是按曲线拟合的阈值

    Returns:
        如果出现空头信号，则返回-1，多头信号则返回1，否则返回0；如果ex_info为True，还将返回详细评估分数。
    """
    if mas is None:
        assert bars is not None, "either 'bars' or 'mas' should be presented"

        mas = []
        close = bars["close"]
        close = close.astype(np.float64)
        for win in wins:
            ma = moving_average(close, win)
            mas.append(ma)

    scores = []
    for ma in mas:
        assert len(ma) >= 10, "length of moving average array should be at least 10."
        scores.append(convex_score(ma[-10:], thresh=thresh))

    scores = np.array(scores)

    # 如果均线为0，则表明未表态
    non_zero = np.count_nonzero(scores)

    if np.count_nonzero(scores > 0) == non_zero:
        flag = 1
    elif np.count_nonzero(scores < 0) == non_zero:
        flag = -1
    else:
        flag = 0

    if ex_info:
        return flag, scores
    else:
        return flag


def convex_score(ts: NDArray, n: int = 0, thresh: float = 1.5e-3) -> float:
    """评估时间序列`ts`的升降性

    如果时间序列中间的点都落在端点连线上方，则该函数为凸函数；反之，则为凹函数。使用点到连线的差值的
    平均值来表明曲线的凹凸性。进一步地，我们将凹凸性引申为升降性，并且对单调上升/下降（即直线)，我们
    使用平均涨跌幅来表明其升降性，从而使得在凹函数、凸函数和直线三种情况下，函数的返回值都能表明
    均线的未来升降趋势。
    Args:
        ts:  时间序列
        n: 用来检测升降性的元素个数。
        thresh: 当点到端点连线之间的平均相对差值小于此值时，认为该序列的几何图形为直线

    Returns:
        返回评估分数，如果大于0，表明为上升曲线，如果小于0，表明为下降曲线。0表明无法评估或者为横盘整理。
    """
    if n == 0:
        n = len(ts)
    elif n == 2:
        return (ts[1] / ts[0] - 1) / n
    elif n == 1:
        raise ValueError("'n' must be great than 1")

    ts = ts[-n:]
    ts_hat = np.arange(n) * (ts[-1] - ts[0]) / (n - 1) + ts[0]

    # 如果点在连线下方，则曲线向上，分数为正
    interleave = ts_hat - ts
    score = np.mean(ts_hat[1:-1] / ts[1:-1] - 1)

    slp = (ts[-1] / ts[0] - 1) / n
    if abs(score) < thresh and abs(slp) > 1.5e-3:
        # 如果convex_score小于阈值，且按直线算斜率又大于1.5e-3,认为直线是趋势
        score = slp
    if np.all(interleave >= 0) or np.all(interleave <= 0):
        return score * 100
    # 存在交织的情况，取最后一段
    else:
        _, start, length = find_runs(interleave >= 0)
        if length[-1] == 1:  # 前一段均为负，最后一个为零时，会被单独分为一段，需要合并
            n = length[-2] + 1
            begin = start[-2]
        else:
            n = length[-1]
            begin = start[-1]

        return convex_score(ts[begin:], n)


async def short_signal(
    bars: BarsArray, ex_info=True, upper_thresh: float = 0.03
) -> Tuple[int, Optional[dict]]:
    """通过穹顶压力、rsi高位和均线拐头来判断是否出现看空信号。

    Args:
        bars: 行情数据
        ex_info: 是否返回附加信息。这些信息可用以诊断
        upper_limit: 上影相对于实体最高部分的长度比

    """
    info = {}

    mas = []
    close = bars["close"]

    wins = (5, 10, 20)
    for win in wins:
        mas.append(moving_average(close, win)[-10:])

    # 检测多均线拐头或者下降压力
    flag, scores = convex_signal(mas=mas, ex_info=ex_info)
    info.update({"convex_scores": scores})
    if flag == -1:
        return flag, info

    # 检测穹顶压力
    for win, score in zip(wins, scores):
        if score < -0.3:  # todo: need to tune this parameter
            dp = dom_pressure(bars, win)
            info.update({"dom_pressure": dp, "win": win})
            if dp > 0:
                return -1, info

    # 检测8周期内是否出现RSI高位，并且已经触发回调
    # 先找长上影 这里需要增加 upper_limit:float = 0.03参数
    # 这个参数根据传入数据的时间单位不同而改变

    high = bars["high"]
    open_ = bars["open"]
    upper_line = high / np.maximum(close, open_) - 1
    iupper = np.where(upper_line >= upper_thresh)[0]
    condlist = [upper_line >= upper_thresh, upper_line < upper_thresh]
    choicelist = [high, close]
    new_price = np.select(condlist, choicelist)

    new_price = new_price.astype(np.float64)
    rsi = ta.RSI(new_price, 6)
    pivots = peaks_and_valleys(new_price)
    ipivots = np.where(pivots == 1)[0]
    ihigh_upper = np.intersect1d(iupper, ipivots)

    if len(ihigh_upper) > 0:
        dist = len(close) - 1 - ihigh_upper[-1]
        if dist <= 8:
            upper_rsi = rsi[ihigh_upper]
            upper_rsi = upper_rsi[~np.isnan(upper_rsi)]
            if (len(upper_rsi) == 1) and (upper_rsi[-1] >= 70):
                info.update({"top_rsi_dist": dist})
                return -1, info
            elif (len(upper_rsi) == 2) and (upper_rsi[1] >= upper_rsi[0]):
                info.update({"top_rsi_dist": dist})
                return -1, info
            elif len(upper_rsi) > 2:
                prev_mean_rsi = np.mean([upper_rsi[-2], upper_rsi[-3]])
                if upper_rsi[-1] >= prev_mean_rsi:
                    info.update({"top_rsi_dist": dist})
                    return -1, info

    # 其它情况
    return 0, info


def high_upper_lead(bars: NDArray, upper_thresh: float = 0.03) -> bool:
    """如果传入行情数据的最后一个bar有长上影，且长上影时触发过RSI高位，则认为会触发调整。

    RSI高位判断利用rsi_watermark函数求出。

    触发调整返回True，没有则返回False。

    Args:
        bars(NDArray): 长度不低于60，具有时间序列的行情数据
        upper_thresh(float): 上引线的长度最低值限制

    Returns:
        触发调整返回True，没有则返回False。
    """

    assert len(bars) >= 60, "The length of bars must not less than 60!"
    close = bars["close"]

    # 先判断当前是否有长上引
    hig = bars[-1]["high"]
    cls = bars[-1]["close"]
    opn = bars[-1]["open"]
    body_high = max(opn, cls)
    up_lead = hig / body_high - 1

    adj = False
    if up_lead >= upper_thresh:
        _, hrsi, crsi = rsi_watermarks(close)
        if hrsi is not None:
            if crsi >= hrsi:
                adj = True
    return adj
