import datetime
import unittest
from unittest import mock

import numpy as np

from pluto.core.metrics import parallel_score, vanilla_score


class MetricsTest(unittest.IsolatedAsyncioTestCase):
    async def test_vanilla_score(self):
        # 类型一：日线，检测当日未涨停
        code = "002380.XSHE"
        bars = np.array(
            [
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
                (
                    "2022-07-20T00:00:00",
                    13.28,
                    13.28,
                    12.9,
                    13.04,
                    9927421.0,
                    1.29080920e08,
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
            actual = await vanilla_score(bars, code, frametype="day")
            exp = ([0.04918036, 0.08442621, 0.06885247], [0.08442621], [])
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i], exp[i], decimal=3)

        # 类型二： 日线，检测当日涨停
        code = "002380.XSHE"
        bars = np.array(
            [
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
                (
                    "2022-06-14T00:00:00",
                    16.65,
                    17.37,
                    15.0,
                    16.24,
                    41124181.0,
                    6.74687061e08,
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
            actual = await vanilla_score(bars, code, frametype="day")
            exp = ([0.14043753, 0.14608325], [0.14608325], [])
            for i in range(3):
                np.testing.assert_array_almost_equal(actual[i], exp[i], decimal=3)

        # 类型三： 30分钟线，检测时未涨停
        code = "002380.XSHE"
        with mock.patch(
            "omicron.tf.day_shift",
            side_effect=(datetime.date(2022, 7, 13), datetime.date(2022, 7, 14)),
        ):
            with mock.patch(
                "omicron.models.stock.Stock.get_trade_price_limits",
                side_effect=(
                    [(datetime.date(2022, 7, 13), 1.3, 10.88)],
                    [(datetime.date(2022, 7, 14), 13.67, 11.19)],
                ),
            ):
                bars = np.array(
                    [
                        (
                            "2022-07-13T11:30:00",
                            12.32,
                            12.44,
                            12.31,
                            12.39,
                            802700.0,
                            9939067.0,
                            3.353773,
                        ),
                        (
                            "2022-07-13T13:30:00",
                            12.4,
                            12.5,
                            12.33,
                            12.47,
                            1043900.0,
                            12983265.0,
                            3.353773,
                        ),
                        (
                            "2022-07-13T14:00:00",
                            12.47,
                            12.48,
                            12.42,
                            12.43,
                            514900.0,
                            6408176.0,
                            3.353773,
                        ),
                        (
                            "2022-07-13T14:30:00",
                            12.43,
                            12.6,
                            12.41,
                            12.45,
                            1136100.0,
                            14198468.0,
                            3.353773,
                        ),
                        (
                            "2022-07-13T15:00:00",
                            12.45,
                            12.48,
                            12.42,
                            12.43,
                            1003400.0,
                            12485592.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T10:00:00",
                            12.52,
                            13.14,
                            12.46,
                            12.92,
                            5904300.0,
                            75851763.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T10:30:00",
                            12.93,
                            12.94,
                            12.8,
                            12.86,
                            1535000.0,
                            19753274.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T11:00:00",
                            12.87,
                            12.87,
                            12.77,
                            12.84,
                            551700.0,
                            7073972.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T11:30:00",
                            12.85,
                            12.86,
                            12.79,
                            12.82,
                            271000.0,
                            3475846.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T13:30:00",
                            12.83,
                            12.84,
                            12.77,
                            12.82,
                            493700.0,
                            6318695.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T14:00:00",
                            12.82,
                            12.82,
                            12.74,
                            12.77,
                            491000.0,
                            6277095.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T14:30:00",
                            12.77,
                            12.78,
                            12.58,
                            12.64,
                            786700.0,
                            9967807.0,
                            3.353773,
                        ),
                        (
                            "2022-07-14T15:00:00",
                            12.65,
                            12.69,
                            12.61,
                            12.65,
                            1007200.0,
                            12750028.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T10:00:00",
                            12.49,
                            12.56,
                            12.34,
                            12.4,
                            2545600.0,
                            31630911.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T10:30:00",
                            12.41,
                            12.44,
                            12.31,
                            12.32,
                            1053400.0,
                            13012812.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T11:00:00",
                            12.31,
                            12.34,
                            12.26,
                            12.28,
                            888500.0,
                            10924709.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T11:30:00",
                            12.3,
                            12.32,
                            12.2,
                            12.31,
                            714900.0,
                            8761878.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T13:30:00",
                            12.32,
                            12.32,
                            12.24,
                            12.29,
                            476700.0,
                            5850614.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T14:00:00",
                            12.29,
                            12.3,
                            12.22,
                            12.28,
                            627800.0,
                            7698980.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T14:30:00",
                            12.28,
                            12.29,
                            12.23,
                            12.26,
                            548200.0,
                            6722689.0,
                            3.353773,
                        ),
                        (
                            "2022-07-15T15:00:00",
                            12.25,
                            12.33,
                            12.2,
                            12.2,
                            1169000.0,
                            14302329.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T10:00:00",
                            12.58,
                            12.95,
                            12.46,
                            12.78,
                            5389600.0,
                            68732486.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T10:30:00",
                            12.78,
                            12.88,
                            12.66,
                            12.66,
                            1206900.0,
                            15399565.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T11:00:00",
                            12.67,
                            12.73,
                            12.64,
                            12.69,
                            439500.0,
                            5580426.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T11:30:00",
                            12.69,
                            12.74,
                            12.68,
                            12.72,
                            295200.0,
                            3751783.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T13:30:00",
                            12.72,
                            12.79,
                            12.68,
                            12.7,
                            471000.0,
                            5996444.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T14:00:00",
                            12.7,
                            12.8,
                            12.68,
                            12.79,
                            529300.0,
                            6739336.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T14:30:00",
                            12.78,
                            12.86,
                            12.72,
                            12.84,
                            928200.0,
                            11859515.0,
                            3.353773,
                        ),
                        (
                            "2022-07-18T15:00:00",
                            12.83,
                            12.86,
                            12.78,
                            12.8,
                            1636300.0,
                            20974235.0,
                            3.353773,
                        ),
                        (
                            "2022-07-19T10:00:00",
                            12.87,
                            13.0,
                            12.65,
                            12.92,
                            4197600.0,
                            54163142.0,
                            3.353773,
                        ),
                        (
                            "2022-07-19T10:30:00",
                            12.92,
                            12.96,
                            12.8,
                            12.84,
                            1291000.0,
                            16597255.0,
                            3.353773,
                        ),
                        (
                            "2022-07-19T11:00:00",
                            12.83,
                            13.14,
                            12.82,
                            13.04,
                            2625800.0,
                            34222092.0,
                            3.353773,
                        ),
                        (
                            "2022-07-19T11:30:00",
                            13.04,
                            13.13,
                            13.01,
                            13.04,
                            1150200.0,
                            15051945.0,
                            3.353773,
                        ),
                        (
                            "2022-07-19T13:30:00",
                            13.05,
                            13.07,
                            12.94,
                            13.0,
                            983000.0,
                            12781171.0,
                            3.353773,
                        ),
                        (
                            "2022-07-19T14:00:00",
                            13.01,
                            13.31,
                            12.99,
                            13.17,
                            2679800.0,
                            35262736.0,
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

                actual = await vanilla_score(bars, code, frametype="min30")
                exp = (
                    [0.02098461, -0.01533499, 0.03309119],
                    [0.03309119],
                    [-0.01533499],
                )
                for i in range(3):
                    np.testing.assert_array_almost_equal(actual[i], exp[i], decimal=3)

        # 类型四： 30分钟线，检测时涨停
        with mock.patch(
            "omicron.models.stock.Stock.get_trade_price_limits",
            side_effect=(
                [(datetime.date(2022, 10, 10), 14.32, 11.72)],
                [(datetime.date(2022, 10, 11), 15.75, 12.89)],
            ),
        ):
            with mock.patch(
                "omicron.tf.day_shift",
                side_effect=(datetime.date(2022, 10, 10), datetime.date(2022, 10, 11)),
            ):
                bars_ = np.array(
                    [
                        (
                            "2022-10-10T10:00:00",
                            13.37,
                            14.32,
                            13.24,
                            14.32,
                            6187400.0,
                            8.63085590e07,
                            3.353773,
                        ),
                        (
                            "2022-10-10T10:30:00",
                            14.32,
                            14.32,
                            14.32,
                            14.32,
                            305700.0,
                            4.37762400e06,
                            3.353773,
                        ),
                        (
                            "2022-10-10T11:00:00",
                            14.32,
                            14.32,
                            14.32,
                            14.32,
                            99200.0,
                            1.42054400e06,
                            3.353773,
                        ),
                        (
                            "2022-10-10T11:30:00",
                            14.32,
                            14.32,
                            14.32,
                            14.32,
                            43000.0,
                            6.15760000e05,
                            3.353773,
                        ),
                        (
                            "2022-10-10T13:30:00",
                            14.32,
                            14.32,
                            14.32,
                            14.32,
                            100900.0,
                            1.44488800e06,
                            3.353773,
                        ),
                        (
                            "2022-10-10T14:00:00",
                            14.32,
                            14.32,
                            14.32,
                            14.32,
                            99700.0,
                            1.42770400e06,
                            3.353773,
                        ),
                        (
                            "2022-10-10T14:30:00",
                            14.32,
                            14.32,
                            14.32,
                            14.32,
                            268100.0,
                            3.83919200e06,
                            3.353773,
                        ),
                        (
                            "2022-10-10T15:00:00",
                            14.32,
                            14.32,
                            14.32,
                            14.32,
                            571600.0,
                            8.18501100e06,
                            3.353773,
                        ),
                        (
                            "2022-10-11T10:00:00",
                            14.9,
                            15.17,
                            14.52,
                            14.65,
                            8425400.0,
                            1.24763053e08,
                            3.353773,
                        ),
                        (
                            "2022-10-11T10:30:00",
                            14.66,
                            15.0,
                            14.64,
                            14.69,
                            3357700.0,
                            4.96986620e07,
                            3.353773,
                        ),
                        (
                            "2022-10-11T11:00:00",
                            14.66,
                            14.9,
                            14.54,
                            14.73,
                            1965200.0,
                            2.89275330e07,
                            3.353773,
                        ),
                        (
                            "2022-10-11T11:30:00",
                            14.74,
                            15.75,
                            14.6,
                            15.75,
                            4985200.0,
                            7.65381320e07,
                            3.353773,
                        ),
                        (
                            "2022-10-11T13:30:00",
                            15.75,
                            15.75,
                            15.75,
                            15.75,
                            1041800.0,
                            1.64083500e07,
                            3.353773,
                        ),
                        (
                            "2022-10-11T14:00:00",
                            15.75,
                            15.75,
                            15.75,
                            15.75,
                            1466900.0,
                            2.31039740e07,
                            3.353773,
                        ),
                        (
                            "2022-10-11T14:30:00",
                            15.75,
                            15.75,
                            15.75,
                            15.75,
                            1537700.0,
                            2.42187750e07,
                            3.353773,
                        ),
                        (
                            "2022-10-11T15:00:00",
                            15.75,
                            15.75,
                            15.75,
                            15.75,
                            345400.0,
                            5.44005000e06,
                            3.353773,
                        ),
                        (
                            "2022-10-12T10:00:00",
                            16.18,
                            16.47,
                            15.51,
                            15.86,
                            19099600.0,
                            3.06769935e08,
                            3.353773,
                        ),
                        (
                            "2022-10-12T10:30:00",
                            15.88,
                            17.0,
                            15.76,
                            16.8,
                            7086500.0,
                            1.16071283e08,
                            3.353773,
                        ),
                        (
                            "2022-10-12T11:00:00",
                            16.81,
                            16.88,
                            16.31,
                            16.42,
                            2614400.0,
                            4.33098160e07,
                            3.353773,
                        ),
                        (
                            "2022-10-12T11:30:00",
                            16.42,
                            16.49,
                            16.23,
                            16.27,
                            905100.0,
                            1.48070050e07,
                            3.353773,
                        ),
                        (
                            "2022-10-12T13:30:00",
                            16.25,
                            16.25,
                            15.8,
                            16.0,
                            1400400.0,
                            2.24104100e07,
                            3.353773,
                        ),
                        (
                            "2022-10-12T14:00:00",
                            16.0,
                            17.33,
                            15.9,
                            17.33,
                            5134700.0,
                            8.75839460e07,
                            3.353773,
                        ),
                        (
                            "2022-10-12T14:30:00",
                            17.33,
                            17.33,
                            17.33,
                            17.33,
                            526400.0,
                            9.12261600e06,
                            3.353773,
                        ),
                        (
                            "2022-10-12T15:00:00",
                            17.33,
                            17.33,
                            17.33,
                            17.33,
                            376500.0,
                            6.52474500e06,
                            3.353773,
                        ),
                        (
                            "2022-10-13T10:00:00",
                            17.6,
                            19.06,
                            17.58,
                            19.06,
                            22189100.0,
                            4.07764112e08,
                            3.353773,
                        ),
                        (
                            "2022-10-13T10:30:00",
                            19.06,
                            19.06,
                            19.06,
                            19.06,
                            1835000.0,
                            3.49755380e07,
                            3.353773,
                        ),
                        (
                            "2022-10-13T11:00:00",
                            19.06,
                            19.06,
                            19.06,
                            19.06,
                            256400.0,
                            4.88698400e06,
                            3.353773,
                        ),
                        (
                            "2022-10-13T11:30:00",
                            19.06,
                            19.06,
                            19.06,
                            19.06,
                            140100.0,
                            2.66981100e06,
                            3.353773,
                        ),
                        (
                            "2022-10-13T13:30:00",
                            19.06,
                            19.06,
                            19.06,
                            19.06,
                            181200.0,
                            3.45367200e06,
                            3.353773,
                        ),
                        (
                            "2022-10-13T14:00:00",
                            19.06,
                            19.06,
                            19.06,
                            19.06,
                            91900.0,
                            1.75174700e06,
                            3.353773,
                        ),
                        (
                            "2022-10-13T14:30:00",
                            19.06,
                            19.06,
                            19.06,
                            19.06,
                            111300.0,
                            2.12137800e06,
                            3.353773,
                        ),
                        (
                            "2022-10-13T15:00:00",
                            19.06,
                            19.06,
                            19.06,
                            19.06,
                            301000.0,
                            5.73713600e06,
                            3.353773,
                        ),
                        (
                            "2022-10-14T10:00:00",
                            20.06,
                            20.97,
                            19.34,
                            20.01,
                            30942993.0,
                            6.31107987e08,
                            3.353773,
                        ),
                        (
                            "2022-10-14T10:30:00",
                            20.01,
                            20.17,
                            19.58,
                            19.96,
                            3284219.0,
                            6.53043860e07,
                            3.353773,
                        ),
                        (
                            "2022-10-14T11:00:00",
                            19.97,
                            20.86,
                            19.81,
                            20.75,
                            3359457.0,
                            6.85313790e07,
                            3.353773,
                        ),
                        (
                            "2022-10-14T11:30:00",
                            19.67,
                            19.67,
                            19.63,
                            19.63,
                            105000.0,
                            2.06305400e06,
                            3.353773,
                        ),
                        (
                            "2022-10-14T13:30:00",
                            19.7,
                            20.73,
                            19.65,
                            20.51,
                            3051158.0,
                            6.10360510e07,
                            3.353773,
                        ),
                        (
                            "2022-10-14T14:00:00",
                            20.51,
                            20.51,
                            19.71,
                            19.78,
                            1736700.0,
                            3.48235070e07,
                            3.353773,
                        ),
                        (
                            "2022-10-14T14:30:00",
                            19.76,
                            19.93,
                            19.67,
                            19.68,
                            2321508.0,
                            4.58501780e07,
                            3.353773,
                        ),
                        (
                            "2022-10-14T15:00:00",
                            18.01,
                            18.01,
                            18.01,
                            18.01,
                            1274600.0,
                            2.29555460e07,
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
                actual = await vanilla_score(bars_, code, frametype="min30")
                exp = ([0.16308728, 0.27919462, 0.20872487], [0.27919462], [])
                for i in range(3):
                    np.testing.assert_array_almost_equal(actual[i], exp[i], decimal=3)

    def test_parallel_score(self):
        mas = [1, 1.1, 1.2, 1.3, 1.4]
        self.assertEqual(0, parallel_score(mas))

        mas = [1.4, 1.3, 1.2, 1.1, 1]
        self.assertEqual(1, parallel_score(mas))

        mas = [1.4, 1.3, 1.1, 1.2, 1]
        self.assertAlmostEqual(0.9, parallel_score(mas))
