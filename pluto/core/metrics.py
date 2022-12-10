import datetime
import logging
from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import talib as ta
from coretypes import BarsArray, FrameType, bars_dtype
from numpy.typing import NDArray
from omicron import tf
from omicron.extensions import find_runs, price_equal
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.talib import moving_average, peaks_and_valleys, rsi_watermarks
from plotly.subplots import make_subplots

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
    thresh: float = 1.5e-3,
) -> Tuple[int, List[float]]:
    """根据均线的升降性，判断是否发出买入或者卖出信号

    调用者需要保证参与计算的均线，至少有10个以上的有效值（即非np.NaN).
    Args:
        bars: 行情数据。
        wins: 均线生成参数
        ex_info: 是否返回均线的详细评分信息
        thresh: 决定均线是按直线拟合还是按曲线拟合的阈值

    Returns:
        如果出现空头信号，则返回-1或-2(-1表示全部为降性图形， -2表示部分为降性图形)，多头信号则返回1，否则返回0；如果ex_info为True，还将返回详细评估分数。
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
        score = convex_score(ma[-7:], thresh=thresh)
        scores.append(score)

    scores = np.array(scores)

    if np.count_nonzero(scores >= 0.05) == len(scores):
        flag = 1

    elif np.count_nonzero(scores <= -0.05) == len(scores):
        flag = -2

    elif (
        np.count_nonzero(scores <= -0.2) > 0
        and scores[0] < 0
        and np.count_nonzero(scores < 0) >= 2
    ):
        flag = -1
    else:
        flag = 0

    if ex_info:
        return flag, scores
    else:
        return flag


def convex_score(ts: NDArray, thresh: float = 1.5e-3) -> float:
    """评估时间序列`ts`的升降性

    如果时间序列中间的点都落在端点连线上方，则该函数为凸函数；反之，则为凹函数。使用点到连线的差值的
    平均值来表明曲线的凹凸性。进一步地，我们将凹凸性引申为升降性，并且对单调上升/下降（即直线)，我们
    使用平均涨跌幅来表明其升降性，从而使得在凹函数、凸函数和直线三种情况下，函数的返回值都能表明
    均线的未来升降趋势。

    Args:
        ts:  时间序列
        n: 用来检测升降性的元素个数。

    Returns:
        返回评估分数，如果大于0，表明为上升曲线，如果小于0，表明为下降曲线。0表明无法评估或者为横盘整理。
    """
    n = len(ts)

    if n < 5:
        return 0

    ts_hat = np.arange(n) * (ts[-1] - ts[0]) / (n - 1) + ts[0]

    # 如果点在连线下方，则曲线向上，分数为正
    interleave = ts_hat - ts

    # 当前序列不能再分段处理了
    if np.all(interleave >= 0) or np.all(interleave <= 0):
        score = np.mean(ts_hat[1:-1] / ts[1:-1] - 1)

        if abs(score) < thresh:
            # 弧度不明显，按直线处理
            score = (ts[-1] / ts[0] - 1) / (n - 1)

        return score * 100
    # 存在分段情况，取最后一段
    else:
        _, start, length = find_runs(interleave >= 0)
        if length[-1] == 1:  # 前一段均为负，最后一个为零时，会被单独分为一段，需要合并
            n = length[-2] + 1
            begin = start[-2]
        else:
            n = length[-1]
            begin = start[-1]

        if n >= len(ts) // 2:
            return convex_score(ts[begin:])
        else:
            # 无法识别的情况
            return 0


async def short_signal(
    bars: BarsArray, ex_info=True, upper_thresh: float = 0.015
) -> Tuple[int, Optional[dict]]:
    """通过穹顶压力、rsi高位和均线拐头来判断是否出现看空信号。

    Args:
        bars: 行情数据
        ex_info: 是否返回附加信息。这些信息可用以诊断
        upper_limit: 上影相对于实体最高部分的长度比

    Returns:
        返回类型为Tuple[int, Optional[dict]]:
        第一个代表当前bars最后一个数据风险类型，
        0，表示没有风险，-1表示具有一定风险，仅起到提醒作用（一般是具有高RSI长上影特征或半降性图形特征），
        -2表示风险较大应直接卖出（一般是穹顶压力或降性图形特征）；
        第二个为该类型的详情信息。

    """

    assert len(bars) >= 30, "传入行情数据长度不得少于30！"

    info = {}

    close = bars["close"]
    close = close.astype(np.float64)
    rsi = ta.RSI(close, 6)

    if rsi[-1] <= 30:
        return 0, None

    # 检测8周期内是否出现RSI高位，并且已经触发回调
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

    wins = (5, 10, 20)
    mas = []
    for win in wins:
        ma = moving_average(close, win)
        mas.append(ma[-10:])
        if win == 5:
            ma5 = ma

    # 检测穹顶压力
    low = bars["low"]
    dp_ma = ma5[-10:]
    score = convex_score(dp_ma)
    # 最后一个bar如果全部在均线之上（low > ma)则不满足条件；如果收盘价在ma上方不远，符合条件
    last_bar_status = (
        (close[-1] < dp_ma[-1] * 1.01)
        and (low[-1] < dp_ma[-1])
        and (dp_ma[-1] < np.max(dp_ma))
    )
    if score < -0.5 and last_bar_status and dp_ma[-1] > dp_ma[0]:
        breakouts = np.count_nonzero((high[-10:] >= dp_ma))
        if breakouts > 0:
            info.update({"dom_pressure": breakouts / 10, "win": 5, "dp_convex": score})
            return -2, info

    # 检测多均线拐头或者下降压力
    flag, scores = convex_signal(mas=mas, ex_info=ex_info)
    if flag == -2:
        info.update({"convex_scores": scores})
        return flag, info
    elif flag == -1 and np.all(rsi[-4:] > 30):
        info.update({"convex_scores": scores})
        return flag, info

    # 其它情况
    return 0, info


def high_upper_lead(bars: NDArray, upper_thresh: float = 0.015) -> bool:
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


async def evaluate(code: str, end: datetime, ex_info=True, upper_thresh: float = 0.015):
    """计算传入code, 以end为截止时间，bars[30: -5]中出现风险警报的数据

    Args:
        code: 股票代码
        end: 截止时间
        ex_info: 额外信息
        upper_thresh: 上影比例

    Returns:
        返回字典，keys为bars[30: -5]中出现风险提醒的index；
        values是相对应风险触发，相应RSI值，
        触发原因（如果是穹顶压力，增加压力值，win均线窗口值，升降性分数；
        如果是长上影线，增加距离风险点的距离；升降性原因，增加升降性分数），
        终止时间，后5天内的超额收益
    """
    bars = await Stock.get_bars(code, 120, FrameType.MIN30, end=end)
    results = {}
    if len(bars) < 120:
        return results

    close = bars["close"].astype(np.float64)
    rsi = ta.RSI(close, 6)

    for i in range(30, 115):
        xbars = bars[:i]
        sell, info = await short_signal(xbars, ex_info, upper_thresh)

        if sell < 0:
            sub_results = {}
            sub_results["fired_at"] = bars["frame"][i - 1]
            sub_results["rsi"] = rsi[i - 1]

            if sell == -1:
                #  由八周期内高RSI的长上影引起
                if list(info.keys()).count("top_rsi_dist") > 0:
                    sub_results["cause_reason"] = "upper_line"
                    sub_results["top_rsi_dist"] = info["top_rsi_dist"]

                elif list(info.keys()).count("convex_scores") > 0:
                    # 由convex_signal引起的卖点触发
                    sub_results["convex_scores"] = info["convex_scores"]
                    sub_results["cause_reason"] = "convex_remind"

            elif sell == -2:
                con = len(info)
                if con == 1:
                    # 由convex_signal引起的卖点触发
                    sub_results["convex_scores"] = info["convex_scores"]
                    sub_results["cause_reason"] = "convex"
                elif con == 3:
                    # 由穹顶压力引起
                    sub_results["cause_reason"] = "dompressure"
                    sub_results["dom_pressure"] = info["dom_pressure"]
                    sub_results["win"] = info["win"]
                    sub_results["dp_convex"] = info["dp_convex"]

            # 计算后五天收益
            close_ = close[i - 1 : i + 5]
            c0 = close_[0]
            imin = np.argmin(close_[1:])
            imax = np.argmax(close_[1:])
            cmin = np.min(close_[1:])
            cmax = np.max(close_[1:])

            if np.all(close_[1:] >= c0):  # 单调上涨，错误卖出
                profit = 1 - cmax / c0
            elif np.all(close_[1:] <= c0):  # 单调下跌，正确！
                profit = 1 - cmin / c0
            elif imin < imax:  # 先下跌再上涨
                profit = 1 - cmin / c0
            elif (imin > imax) and (cmax - c0 < abs(c0 - cmin)):  # 先上涨再下跌，上涨较小，判断正确
                profit = 1 - cmin / c0
            elif (imin > imax) and (cmax - c0 >= abs(c0 - cmin)):  # 先上涨再下跌，上涨较大
                profit = 1 - cmax / c0
            else:  # 一般为涨停，判断正确
                print(code, c0, close_[1:], "\n")
                profit = 0
            sub_results["end"] = end
            sub_results["profits"] = profit
            results[i - 1] = sub_results

    return results


async def plot_evaluate(
    code: str, end: datetime, ex_info=True, upper_thresh: float = 0.015
):
    """将卖点标记点在图上标记出来
    由convex_signal引起的卖点触发，绘制出卖出点的两点连线；
    由穹顶压力引起，绘制出相应均线的穹顶图形；
    由八周期内高RSI的长上影引起，标记出长上影所在点

    Args:
    code: 股票代码
    end: 传入行情数据的最后一个时间点
    ex_info： 是否需要额外信息
    upper_limit: 上影相对于实体最高部分的长度比
    """

    results = await evaluate(code, end, ex_info, upper_thresh)

    if len(results) > 0:
        name = await Security.alias(code)
        bars = await Stock.get_bars(code, 120, FrameType.MIN30, end=end)
        frame = bars["frame"]
        close = bars["close"]
        close = close.astype(np.float64)
        high = bars["high"]
        index = np.arange(len(bars))
        MA5 = ta.MA(close, 5)
        MA10 = ta.MA(close, 10)
        MA20 = ta.MA(close, 20)
        MA30 = ta.MA(close, 30)
        rsi = ta.RSI(close, 6)

        fig = make_subplots(
            rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True, shared_yaxes=False
        )
        fig.add_trace(
            go.Candlestick(
                x=index,
                close=close,
                high=high,
                open=bars["open"],
                low=bars["low"],
                increasing=dict(line=dict(color="red")),
                decreasing=dict(line=dict(color="green")),
                name="K线",
                text=frame,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=index, y=MA5, mode="lines", name="MA5", text=bars["frame"]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=index, y=MA10, mode="lines", name="MA10", text=bars["frame"]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=index, y=MA20, mode="lines", name="MA20", text=bars["frame"]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=index, y=MA30, mode="lines", name="MA30", text=bars["frame"]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=index, y=rsi, mode="lines", name="RSI6", text=bars["frame"]),
            row=2,
            col=1,
        )
        fig.add_hline(y=30, row=2, col=1)

        iframes = list(results.keys())
        for iframe in iframes:
            sub_result = results[iframe]
            cause_type = sub_result["cause_reason"]
            if cause_type == "convex":
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=index[iframe],
                    y0=0,
                    x1=index[iframe],
                    y1=1,
                    line=dict(color="blue", width=1),
                    opacity=0.5,
                )
                convex_scores = sub_result["convex_scores"]
                convex_scores = np.around(convex_scores, 6)
                fig.add_trace(
                    go.Scatter(
                        x=[index[iframe]],
                        y=[high[iframe] * 1.01],
                        mode="markers",
                        marker_symbol="triangle-down",
                        marker_line_color="midnightblue",
                        marker_color="blue",
                        marker_line_width=1,
                        marker_size=8,
                        name="降性",
                        text=f"{convex_scores},{frame[iframe]}",
                    ),
                    row=1,
                    col=1,
                )

            elif cause_type == "convex_remind":
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=index[iframe],
                    y0=0,
                    x1=index[iframe],
                    y1=1,
                    line=dict(color="purple", width=1),
                    opacity=1,
                )
                convex_scores = sub_result["convex_scores"]
                convex_scores = np.around(convex_scores, 6)
                fig.add_trace(
                    go.Scatter(
                        x=[index[iframe]],
                        y=[high[iframe] * 1.01],
                        mode="markers",
                        marker_symbol="triangle-down",
                        marker_line_color="midnightblue",
                        marker_color="purple",
                        marker_line_width=1,
                        marker_size=8,
                        name="降性",
                        text=f"{convex_scores},{frame[iframe]}",
                    ),
                    row=1,
                    col=1,
                )

            elif cause_type == "dompressure":
                dom_win = sub_result["win"]
                dp_convex = sub_result["dp_convex"]
                dp_convex = np.around(dp_convex, 6)
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=index[iframe],
                    y0=0,
                    x1=index[iframe],
                    y1=1,
                    line=dict(color="black", width=1),
                    opacity=0.5,
                )
                fig.add_trace(
                    go.Scatter(
                        x=index[iframe - 6 : iframe + 1],
                        y=ta.MA(close, dom_win)[iframe - 6 : iframe + 1],
                        mode="lines",
                        line=dict(width=3, color="black"),
                        text=f"{dp_convex}",
                        name=f"MA_{dom_win}穹顶",
                    ),
                    row=1,
                    col=1,
                )

            elif cause_type == "upper_line":
                dist = sub_result["top_rsi_dist"]
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=index[iframe],
                    y0=0,
                    x1=index[iframe],
                    y1=1,
                    line=dict(color="orange", width=1),
                    opacity=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[index[iframe - dist]],
                        y=[high[iframe - dist] * 1.02],
                        mode="markers",
                        marker_symbol="triangle-down",
                        marker_line_color="midnightblue",
                        marker_color="red",
                        marker_line_width=1,
                        marker_size=10,
                        name="最近的高RSI长上影线",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[index[iframe - dist]],
                        y=[rsi[iframe - dist] * 1.1],
                        mode="markers",
                        marker_symbol="triangle-down",
                        marker_line_color="midnightblue",
                        marker_color="red",
                        marker_line_width=1,
                        marker_size=10,
                        name="最近的高RSI长上影线",
                    ),
                    row=2,
                    col=1,
                )

        fig.update_layout(
            title=(f"{code}:{name}触发风险警报，蓝色降性，黑色穹顶，橙色上影"), width=1000, height=600
        )
        fig.update_yaxes(dict(domain=[0.4, 1]), row=1, col=1)
        fig.update_yaxes(dict(domain=[0.0, 0.4]), row=2, col=1)

        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(showspikes=True, spikethickness=2)
        fig.update_yaxes(showspikes=True, spikethickness=2)

        fig.show()

        return results


def hrsi_upline_confirm(
    bars: NDArray, upper_thresh: float = 0.015, hist_long: int = 16
) -> int:
    """计算传入行情数据最近的历史长度hist_long中出现过处于高位RSI带有长上影线的距离

    Args:
    bars: 具有时间序列的行情数据，可以是任意时间级别
    upper_thresh: K线上影部分相比较实体最高部分的比，默认参数适合MIN30的时间单位
    hist_long: 检测最后的历史数据长度

    Returns:
    dist: 返回dist是在规定历史长度中有高位RSI并带有长上影的距离。
        例如dist==1, 传入历史数据的倒数第二个满足条件。
        没有则返回None
    """

    assert len(bars) >= 60, "传入行情数据不得少于60！"
    close = bars["close"]
    close = close.astype(np.float64)
    rsi = ta.RSI(close, 6)

    # 检测16周期内是否出现RSI高位，并且已经触发回调
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
        if dist <= hist_long:
            upper_rsi = rsi[ihigh_upper]
            upper_rsi = upper_rsi[~np.isnan(upper_rsi)]
            if (len(upper_rsi) == 1) and (upper_rsi[-1] >= 70):
                return dist
            elif (len(upper_rsi) == 2) and (upper_rsi[1] >= upper_rsi[0]):
                return dist
            elif len(upper_rsi) > 2:
                prev_mean_rsi = np.mean([upper_rsi[-2], upper_rsi[-3]])
                if upper_rsi[-1] >= prev_mean_rsi:
                    return dist
