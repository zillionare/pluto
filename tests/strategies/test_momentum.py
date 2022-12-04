import datetime
import os
import shutil
import unittest

import cfg4py
import omicron
import pytest
from coretypes import FrameType
from omicron.models.stock import Stock

from pluto.strategies.momentum import MomemtumStrategy


@pytest.mark.skipif(os.getenv("IS_GITHUB"), reason="run at local only")
class MomemtumTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        os.environ["pluto_store_path"] = os.path.expanduser("~/tmp/pluto.zarr")
        cfg4py.init(os.path.expanduser("~/zillionare/pluto"))
        await omicron.init()
        try:
            shutil.rmtree(os.path.expanduser("~/tmp/pluto.zarr"))
        except FileNotFoundError:
            pass

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    async def test_extract_short_features(self):
        bars = await Stock.get_bars(
            "002782.XSHE", 70, FrameType.MIN30, end=datetime.datetime(2022, 11, 22, 10)
        )
        mom = MomemtumStrategy()
        actual = mom.extract_short_features(bars)
        print(actual)
        desc = mom.describe_short_features(actual)
        exp = [
            "RSI高水位差: 3.1",
            "距离RSI前高: 0 bars",
            "近期RSI摸高次数: 1",
            "前高: 2.27%",
            "3日内最大跌幅: -1.54%",
            "3日收阴率: 50.0%",
            "3日sharpe: 3.3",
            "最后bar涨幅: -0.51%",
            "frame序号: 0",
            "最大成交量比(负数为卖出): -8.4",
            "异向成交量比: 0.0",
            "是否涨停: False",
            "下方均线数: 4",
            "5_10_20多头指数: 100.0%",
            "10_20_30多头指数: 66.7%",
            "10_20_60多头指数: 66.7%",
            "5日均线走势: 0.61% 0.07% 0.59% -1",
            "10日均线走势: 0.08% 0.08% -0.26% 2",
            "20日均线走势: 0.08% 0.05% -0.29% 3",
            "30日均线走势: 0.03% 0.02% -0.20% 5",
            "60日均线走势: 0.03% 0.01% -0.17% 6",
        ]

        self.assertListEqual(exp, desc)
        bars = await Stock.get_bars(
            "603737.XSHG", 70, FrameType.MIN30, end=datetime.datetime(2022, 11, 25, 11)
        )
        actual = mom.extract_short_features(bars)
        desc = mom.describe_short_features(actual)
        exp = [
            "RSI高水位差: 7.0",
            "距离RSI前高: 0 bars",
            "近期RSI摸高次数: 2",
            "前高: 0.00%",
            "3日内最大跌幅: -0.90%",
            "3日收阴率: 50.0%",
            "3日sharpe: 3.8",
            "最后bar涨幅: 2.36%",
            "frame序号: 2",
            "最大成交量比(负数为卖出): 7.2",
            "异向成交量比: 0.0",
            "是否涨停: False",
            "下方均线数: 5",
            "5_10_20多头指数: 100.0%",
            "10_20_30多头指数: 100.0%",
            "10_20_60多头指数: 100.0%",
            "5日均线走势: 0.18% 0.01% 0.19% -1",
            "10日均线走势: 0.06% 0.01% 0.24% -1",
            "20日均线走势: 0.07% 0.01% -0.01% 1",
            "30日均线走势: 0.02% 0.02% -0.09% 3",
            "60日均线走势: 0.02% 0.00% 0.06% -1",
        ]
        self.assertListEqual(exp, desc)
