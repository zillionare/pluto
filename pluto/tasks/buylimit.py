"""跟踪涨停板"""
import datetime

import numpy as np
from coretypes import FrameType
from omicron import tf
from omicron.models.security import Security
from omicron.models.stock import Stock

from pluto.core import is_main_board
from pluto.core.metrics import last_wave
from pluto.core.morph import n_red_candlestick
from pluto.store.buy_limit_pool import BuyLimitPoolStore


async def three_white_soldiers(
    end=None, min_total=1, scope_win=10, continuous_rng=(1, 3)
):
    blp = BuyLimitPoolStore()

    scope_end = tf.floor(end or datetime.datetime.now(), FrameType.DAY)
    scope_start = tf.day_shift(scope_end, scope_win=-scope_win)

    results = []
    for code, total, continuous, last in blp.find_all(scope_start, scope_end):
        if (
            total < min_total
            or continuous < continuous_rng[0]
            or continuous > continuous_rng[1]
        ):
            continue
        name = await Security.alias(code)

        if not await is_main_board(code):
            continue

        bars = await Stock.get_bars(code, 3, FrameType.DAY, end=end)
        if len(bars) < 3:
            continue

        till_now = tf.count_day_frames(last, end or datetime.datetime.now())
        if till_now <= 3:  # 涨停不能包含在阳线之中
            continue

        if not n_red_candlestick(bars, 3):
            continue

        bars = await Stock.get_bars(code, 60, FrameType.DAY, end=end)
        close = bars["close"]
        wave_len, wave_amp = last_wave(close)
        results.append((code, name, wave_len, wave_amp, wave_amp / wave_len))

    return np.array(results, dtype = [
        ("code", "U16"),
        ("name", "U16"),
        ("wave_len", "i4"),
        ("wave_amp", "f4"),
        ("daily_mean", "f4")
    ])
