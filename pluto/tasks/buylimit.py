"""跟踪涨停板"""
import datetime

from coretypes import FrameType
from omicron import tf
from omicron.models.security import Security
from omicron.models.stock import Stock

from pluto.core import is_main_board
from pluto.core.morph import n_red_candlestick
from pluto.store.buy_limit_pool import BuyLimitPoolStore


async def three_white_soldiers(
    end=None, min_total=2, scope_win=30, continuous_rng=(2, 3)
):
    blp = BuyLimitPoolStore()

    scope_end = tf.floor(end or datetime.datetime.now(), FrameType.DAY)
    scope_start = tf.day_shift(scope_end, scope_win=-30)

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
