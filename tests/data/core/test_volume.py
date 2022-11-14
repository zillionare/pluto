import unittest

import numpy as np
import omicron

from pluto.core.volume import morph_pattern, moving_net_volume, top_volume_direction
from tests import init_test_env


class TestVolumeFeatures(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        init_test_env()
        await omicron.init()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    def test_top_volume_direction(self):
        # 600163.XSHG 2021-09-02 11:30

        bars = np.array(
            [
                (5.72, 5.74, 5.5, 5.61, 6753800.0, 37871375.0, 2.352),
                (5.62, 5.6900005, 5.5899997, 5.62, 2434500.0, 13724468.0, 2.352),
                (5.63, 5.76, 5.63, 5.71, 5078400.0, 28906977.0, 2.352),
                (5.7, 5.79, 5.63, 5.79, 4571400.0, 26181890.0, 2.352),
                (5.79, 5.86, 5.75, 5.82, 4766500.0, 27690079.0, 2.352),
                (5.82, 5.83, 5.74, 5.76, 4448900.0, 25807039.0, 2.352),
                (5.76, 5.94, 5.63, 5.94, 12006400.0, 69975416.0, 2.352),
                (5.95, 6.0600004, 5.89, 6.0, 13042900.0, 78035291.0, 2.352),
                (5.99, 6.0600004, 5.98, 6.05, 5927400.0, 35724998.0, 2.352),
                (6.04, 6.15, 5.99, 6.15, 6730000.0, 40837283.0, 2.352),
            ],
            dtype=[
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        vd = top_volume_direction(bars, n=5)
        exp = [2.1, 0]
        np.testing.assert_array_almost_equal(exp, vd, 1)

        bars["volume"][5] *= 3
        vd = top_volume_direction(bars, n=5)
        exp = [-2.8, 0.97]
        np.testing.assert_array_almost_equal(exp, vd, 1)

    def test_moving_on_balance_vol(self):
        bars = np.array(
            [
                (1, 0, 3),
                (1, 0, 4),
                (0, 1, 5),
                (0, 1, 8),
                (0, 1, 9),
                (1, 0, 5),
                (1, 0, 4),
                (1, 0, 1),
                (1, 0, 5),
            ],
            dtype=[("open", "<f4"), ("close", "<f4"), ("volume", "<f8")],
        )

        actual = moving_net_volume(bars, 5)
        exp = [np.nan] * 4 + [2.5862069, 2.09677419, 2.09677419, 1.2962963, -1.25]
        np.testing.assert_array_almost_equal(actual, exp, 2)

        bars[1]["open"] = 0
        actual = moving_net_volume(bars, 5)
        exp = [np.nan] * 4 + [3.275, 2.741, 2.09677419, 1.2962963, -1.25]
        np.testing.assert_array_almost_equal(actual, exp, 2)

    async def test_morph_pattern(self):
        from coretypes import FrameType
        from omicron.models.stock import Stock

        bars = await Stock.get_bars("003001.XSHE", 5, FrameType.DAY)
        morph_pattern(bars)
