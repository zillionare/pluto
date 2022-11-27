"""成交量相关特征的提取"""
from typing import Tuple

import numpy as np
from bottleneck import move_mean, move_sum
from coretypes import BarsArray
from omicron.extensions import top_n_argpos


def volume_feature(volume: np.array) -> Tuple:
    """提取成交量特征

    返回值：
        0： 最后4列的形态特征
    """
    pass


def describe_volume_morph(pattern: int) -> str:
    return {
        0: "未知成交量形态",
        1: "持续放量阳线",
        2: "持续缩量阳线",
        3: "间歇放量阳线",
        -1: "持续放量阴线",
        -2: "持续缩量阴线",
        -3: "间歇放量阴线",
    }.get(pattern, "未知成交量形态")


def morph_pattern(bars: BarsArray) -> int:
    """成交量的形态特征

    bars的长度必须大于5，当bars大于5时，只取最后5个周期进行计算。

    算法：
        1. 根据阳线、阴线，赋予成交量符号
        2. 计算形态特征
    返回类型：
        0:  未检测到模式
        1:  单调递增, 比如 0.9 1.2 1.3 1.4 （收阳）
        2:  单调递减, 比如 1.4 1.3 1.2 1 （收阳）
        3:  放量阳，缩量阴
        -1: 单调递增，都收阴
        -2: 单调递减，都收阴
        -3: 放量阴，缩量阳

    Args:
        bars: 行情数据

    Returns:
        形态特征
    """
    if len(bars) < 5:
        raise ValueError("bars must be at least 5 length")

    bars = bars[-5:]
    close = bars["close"]
    opn = bars["open"]
    vol = bars["volume"]

    yinyang = np.select((close > opn, close < opn), [1, -1], 0)[1:]
    vol = vol[1:] / vol[:-1]
    inc_dec = np.select((vol >= 1.1, vol <= 0.9), [1, 3], 2)

    flags = yinyang * inc_dec
    if np.all(flags == 1):
        return 1
    elif np.all(flags == 3):
        return 2
    elif np.all(flags == -1):
        return -1
    elif np.all(flags == -3):
        return -2
    elif np.all(flags == np.array([1, -3, 1, -3])) or np.all(
        flags == np.array([-3, 1, -3, 1])
    ):
        return 3
    elif np.all(flags == np.array([-1, 3, -1, 3])) or np.all(
        flags == np.array((3, -1, 3, -1))
    ):
        return -3
    else:
        return 0


def top_volume_direction(bars: BarsArray, n: int = 10) -> Tuple[float, float]:
    """计算`n`周期内，最大成交量的量比（带方向）及该笔成交量与之后的最大异向成交量的比值（带方向）。

    成交量方向：如果当前股价收阳则成交量方向为1，下跌则为-1。本函数用以发现某个时间点出现大笔买入（或者卖出），并且在随后的几个周期里，缩量下跌（或者上涨）的情形。主力往往会根据这个特征来判断跟风资金的意图，从而制定操作计划。

    计算方法：
        1. 找出最大成交量的位置
        2. 找出其后一个最大异向成交量的位置
        3. 计算最大成交量与之前`n`个成交量均值的量比及方向
        4. 计算最大成交量与之后的所有成交量中，最大反向成交量的量比

    args:
        bars: 行情数据
        n: 参与计算的周期。太长则影响到最大成交量的影响力。

    return:
        前一个元素表明最大成交量与之前`n`个成交量均值的量比，其符号表明是阳线还是阴线；后一个元素表明最大成交量与之后所有成交量中，最大异向成交量的量比。如果不存在异向成交量，则值为0。

    """

    bars = bars[-n:]
    volume = bars["volume"]

    flags = np.select(
        (bars["close"] > bars["open"], bars["close"] < bars["open"]), [1, -1], 0
    )

    pmax = np.argmax(volume)

    # 移除3个最大成交量后的成交量均值
    top_volume = np.sum(volume[top_n_argpos(volume, 3)])
    vmean = (np.sum(volume[-n:]) - top_volume) / n

    # 最大成交量及之后的成交量
    vol = (volume * flags)[pmax:]
    vmax = vol[0]

    if flags[pmax] == 1 and np.any(vol[1:] < 0):
        vr = [vmax / vmean, np.min(vol) / vmax]
    elif flags[pmax] == -1 and np.any(vol[1:] > 0):
        vr = [vmax / vmean, abs(np.max(vol) / vmax)]
    else:
        vr = [vmax / vmean, 0]

    return vr


def moving_net_volume(bars: BarsArray, win=5) -> np.array:
    """移动净余成交量
    args:
        bars: 行情数据
        win: 滑动窗口

    return:
        np.array: `win`周期内归一化（除以周期内成交量均值）的移动和

    """
    vol = bars["volume"]
    close = bars["close"]
    open_ = bars["open"]

    flags = np.select((close > open_, close < open_), [1, -1], 0)
    signed_vol = vol * flags

    return move_sum(signed_vol, win) / move_mean(vol, win)


def net_buy_volume(bars) -> float:
    """bars全部区间内的净买入量

    如果k线为阳线，则为买入量；如果为阴线，则为卖出量
    """
    volume = bars["volume"]
    close = bars["close"]
    open_ = bars["open"]

    flags = np.select((close > open_, close < open_), [1, -1], 0)
    signed_vol = volume * flags
    return np.sum(signed_vol) / np.mean(volume)
