######

# 此文件中的代码是实验性的，主要目的是以交互式方式在vscode中运行。

import asyncio
import os

import cfg4py
import numpy as np
import omicron
import pandas as pd
from coretypes import FrameType
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.plotting.candlestick import Candlestick
from omicron.talib import moving_average

from pluto.core import is_main_board
from pluto.core.metrics import convex_score

codes = []


async def init():
    global codes
    cfg4py.init(os.path.expanduser("~/zillionare/pluto"))
    await omicron.init()

    codes = await Security.select().types(["stock"]).eval()
    codes = [code for code in codes if await is_main_board(code)]


async def ascending_ma_lines():
    global codes
    results = []
    for code in codes:
        name = await Security.alias(code)
        bars = await Stock.get_bars(code, 30, FrameType.DAY, unclosed=False)
        if len(bars) < 30:
            continue
        close = bars["close"]
        scores = []
        mas = []
        for win in (5, 10, 20):
            ma = moving_average(close, win)[-10:]
            score = convex_score(ma)
            if score > 0:
                scores.append(score)
                mas.append(ma[-1])
        if len(scores) == 3:
            convergencey = np.max(mas) / np.min(mas) - 1
            results.append((name, code, *scores, convergencey))

    cols = "name,code,s5,s10,s20,convergencey".split(",")
    return pd.DataFrame(results, columns=cols)


async def main():
    await init()
    return await ascending_ma_lines()
