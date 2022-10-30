import logging
from itertools import combinations
from typing import Iterable, Tuple

import numpy as np
from coretypes import bars_dtype
from omicron import tf
from omicron.extensions import price_equal
from omicron.models.stock import Stock

logger = logging.getLogger(__name__)


async def vanilla_score(
    bars: bars_dtype, code: str = None, frametype: str = "day"
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
        只有两种时间类型可被接受：'day'，'min30'。

    Returns:
        包含每日累计涨跌幅，最大涨幅和最大跌幅的元组。
    """
    returns = []
    max_returns = []
    mdds = []

    assert (frametype == "day") or (
        frametype == "min30"
    ), "'frmetype' must choose string between 'day' and 'min30'!"
    if frametype == "day":
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

    elif frametype == "min30":
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
