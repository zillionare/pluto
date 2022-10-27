import unittest
from unittest import mock

import numpy as np

from pluto.core.metrics import vanilla_score


class MetricsTest(unittest.IsolatedAsyncioTestCase):
    async def test_vanilla_score(self):
        # 类型一：检测当天涨停，第二天未涨停
        code = "002750.XSHE"
        bars = np.array(
            [
                (
                    "2022-09-27T00:00:00",
                    8.55,
                    9.49,
                    8.55,
                    9.49,
                    19119568.0,
                    1.77904128e08,
                    6.158,
                ),
                (
                    "2022-09-28T00:00:00",
                    9.35,
                    9.85,
                    9.29,
                    9.36,
                    24557842.0,
                    2.35395644e08,
                    6.158,
                ),
                (
                    "2022-09-29T00:00:00",
                    9.26,
                    9.54,
                    8.96,
                    9.03,
                    14725071.0,
                    1.36067616e08,
                    6.158,
                ),
                (
                    "2022-09-30T00:00:00",
                    9.05,
                    9.93,
                    9.0,
                    9.93,
                    11580068.0,
                    1.12574827e08,
                    6.158,
                ),
            ],
            dtype=[
                ("frame", "<M8[s]"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        with mock.patch(
            "omicron.models.stock.Stock.trade_price_limit_flags",
            return_value=([True], [False]),
        ):
            actual = await vanilla_score(bars, code)
            exp = (
                [0.00106952, -0.0342246, 0.06203209],
                [0.062032085561497335],
                [-0.03422459893048131],
            )
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i], exp[i], decimal=3)

        # 类型二： 检测当天涨停，第二天开盘也涨停
        code = "002380.XSHE"
        bars = np.array(
            [
                (
                    "2022-06-07T00:00:00",
                    10.12,
                    11.04,
                    9.92,
                    11.04,
                    7192401.0,
                    7.66028270e07,
                    3.342714,
                ),
                (
                    "2022-06-08T00:00:00",
                    12.1,
                    12.14,
                    12.1,
                    12.14,
                    4939200.0,
                    5.99058860e07,
                    3.342714,
                ),
                (
                    "2022-06-09T00:00:00",
                    13.35,
                    13.35,
                    13.01,
                    13.35,
                    8613404.0,
                    1.14910770e08,
                    3.342714,
                ),
                (
                    "2022-06-10T00:00:00",
                    14.17,
                    14.69,
                    13.97,
                    14.69,
                    16849817.0,
                    2.43503344e08,
                    3.342714,
                ),
                (
                    "2022-06-13T00:00:00",
                    15.0,
                    16.16,
                    14.81,
                    16.16,
                    25108534.0,
                    3.92079273e08,
                    3.342714,
                ),
            ],
            dtype=[
                ("frame", "<M8[s]"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        with mock.patch(
            "omicron.models.stock.Stock.trade_price_limit_flags",
            return_value=([True], [False]),
        ):
            actual = await vanilla_score(bars, code)
            exp = (
                [0.00330579, 0.10330579, 0.21404959, 0.33553719],
                [0.33553719008264465],
                [],
            )
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i], exp[i], decimal=3)

        # 类型三： 检测当天未涨停
        code = "002380.XSHE"
        bars = np.array(
            [
                (
                    "2022-07-13T00:00:00",
                    12.12,
                    12.63,
                    12.05,
                    12.43,
                    7019231.0,
                    8.68321520e07,
                    3.353773,
                ),
                (
                    "2022-07-14T00:00:00",
                    12.52,
                    13.15,
                    12.46,
                    12.65,
                    11040618.0,
                    1.41468480e08,
                    3.353773,
                ),
                (
                    "2022-07-15T00:00:00",
                    12.49,
                    12.56,
                    12.2,
                    12.2,
                    8024077.0,
                    9.89049220e07,
                    3.353773,
                ),
                (
                    "2022-07-18T00:00:00",
                    12.58,
                    12.96,
                    12.46,
                    12.8,
                    10895999.0,
                    1.39033790e08,
                    3.353773,
                ),
                (
                    "2022-07-19T00:00:00",
                    12.87,
                    13.34,
                    12.65,
                    13.23,
                    16645821.0,
                    2.17165421e08,
                    3.353773,
                ),
            ],
            dtype=[
                ("frame", "<M8[s]"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        with mock.patch(
            "omicron.models.stock.Stock.trade_price_limit_flags",
            return_value=([False], [False]),
        ):
            actual = await vanilla_score(bars, code)
            exp = (
                [0.01769912, -0.01850362, 0.02976669, 0.06436042],
                [0.06436041834271929],
                [-0.018503620273531814],
            )
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i], exp[i], decimal=3)
