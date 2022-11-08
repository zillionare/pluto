import datetime
import os
import unittest

import numpy as np
import omicron
from coretypes import FrameType
from omicron.models.security import Security
from omicron.models.stock import Stock

from pluto.strategies.u_turn_board import TurnaroundStrategy
from tests import init_test_env


class TurnaroundStrategyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        init_test_env()
        await omicron.init()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    async def test_evaluate_long(self):
        bars = await Stock.get_bars_in_range(
            "002279.XSHE",
            FrameType.DAY,
            datetime.date(2022, 7, 1),
            datetime.date(2022, 10, 27),
        )
        actual = TurnaroundStrategy().evaluate_long(bars, 5, 5, None)
        exp = 2
        self.assertEqual(actual, exp)

        # distance is None:
        actual = TurnaroundStrategy().evaluate_long(bars, 30, 5, None)
        exp = None
        self.assertEqual(actual, exp)

    @pytest.mark.skipif(os.environ.get("IS_GITHUB"), reason="本测试不能在github上运行，缺乏数据")
    async def test_scan(self):
        frame = datetime.date(2022, 10, 26)
        codes = (
            await Security.select(frame)
            .types(["stock"])
            .exclude_st()
            .exclude_kcb()
            .exclude_cyb()
            .eval()
        )
        codes = codes[:100]
        actual = await TurnaroundStrategy().scan(codes, frame, 2, 1, None)
        exp = ["600079", "600100", "600455"]
        self.assertEqual(actual, exp)

    async def test_score(self):
        codes = ["600079", "600100", "600455"]
        frame = datetime.date(2022, 10, 26)
        actual = await TurnaroundStrategy().score(codes, frame, 2, 1, None)
        exp = [
            [
                datetime.date(2022, 10, 26),
                "人福医药",
                "600079.XSHG",
                -0.6654053926467896,
                1,
                [("881140", "化学制药"), ("884144", "化学制剂")],
            ],
            [
                datetime.date(2022, 10, 26),
                "同方股份",
                "600100.XSHG",
                -0.22471353877335787,
                1,
                [("881130", "计算机设备")],
            ],
            [
                datetime.date(2022, 10, 26),
                "博通股份",
                "600455.XSHG",
                1.7458714544773102,
                1,
                [("881178", "教育"), ("884179", "其他传媒")],
            ],
        ]
        self.assertEqual(actual, exp)

    async def test_backtest(self):
        start = datetime.date(2022, 10, 26)
        end = datetime.date(2022, 10, 26)
        actual = (await TurnaroundStrategy().backtest(start, end, 8, 1, None)).values
        exp = np.array(
            [
                [
                    datetime.date(2022, 10, 26),
                    "罗欣药业",
                    "002793.XSHE",
                    0.37641418166458607,
                    1,
                    list([("881140", "化学制药"), ("884144", "化学制剂")]),
                ],
                [
                    datetime.date(2022, 10, 26),
                    "立方制药",
                    "003020.XSHE",
                    0.7558655925095081,
                    1,
                    list([("881140", "化学制药"), ("884144", "化学制剂")]),
                ],
                [
                    datetime.date(2022, 10, 26),
                    "多瑞医药",
                    "301075.XSHE",
                    0.24968564976006746,
                    1,
                    list([("881140", "化学制药"), ("884144", "化学制剂")]),
                ],
                [
                    datetime.date(2022, 10, 26),
                    "拓新药业",
                    "301089.XSHE",
                    -2.1184219047427177,
                    1,
                    list([("881140", "化学制药"), ("884143", "原料药")]),
                ],
            ],
            dtype=object,
        )
        np.testing.assert_array_equal(actual, exp)
