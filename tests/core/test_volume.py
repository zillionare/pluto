import datetime
import os
import unittest

import cfg4py
import numpy as np
import omicron
from coretypes import FrameType
from omicron.models.stock import Stock

from pluto.core.volume import moving_net_volume, top_volume_direction
from tests import init_test_env


class TestVolumeFeatures(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg4py.init(os.path.expanduser("~/zillionare/pluto"))
        os.environ["all_proxy"] = ""
        await omicron.init()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    def test_moving_net_volume(self):
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

    async def test_top_volume_direction(self):
        bars = await Stock.get_bars(
            "002782.XSHE", 30, FrameType.MIN30, end=datetime.datetime(2022, 11, 24, 15)
        )

        actual = top_volume_direction(bars)
        self.assertEqual(0, actual[0])

        bars = await Stock.get_bars(
            "002782.XSHE", 30, FrameType.MIN30, end=datetime.datetime(2022, 11, 24, 15)
        )

        vmax, vreverse = top_volume_direction(bars, n=24)
        self.assertAlmostEqual(-7.57, vmax, places=2)
        self.assertAlmostEqual(0.3, vreverse, places=2)

        bars = await Stock.get_bars(
            "002782.XSHE", 10, FrameType.DAY, end=datetime.date(2022, 11, 7)
        )
        vmax, vreverse = top_volume_direction(bars)
        self.assertAlmostEqual(2.76, vmax, 2)
        self.assertEqual(0, vreverse)
