import datetime
import os
import shutil
import unittest
from unittest import mock

import cfg4py
import omicron
import pytest
from freezegun import freeze_time

from pluto.config import get_config_dir
from pluto.store.touch_buy_limit_pool import TouchBuyLimitPoolStore


class TouchBuyLimitPoolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ[cfg4py.envar] = "DEV"
        cfg4py.init(get_config_dir())
        await omicron.init()
        try:
            shutil.rmtree("/tmp/pluto.zarr")
        except FileNotFoundError:
            pass

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @pytest.mark.skipif(os.environ.get("IS_GITHUB"), reason="本测试需要omicron数据，只能在本地运行")
    async def test_pooling(self):
        # this also tests get/query
        store = TouchBuyLimitPoolStore("/tmp/pluto.zarr")

        with mock.patch(
            "omicron.models.security.Query.eval", return_value=["000032.XSHE"]
        ):
            await store.pooling(datetime.date(2022, 10, 28))
            actual = await store.get(datetime.date(2022, 10, 28))
            self.assertEqual(actual[0]["name"], "深桑达A")
            self.assertEqual(
                actual[0]["date"].item().date(), datetime.date(2022, 10, 28)
            )
            self.assertAlmostEqual(actual[0]["upper_line"], 0.046, 2)

        with mock.patch(
            "omicron.models.security.Query.eval", return_value=["000032.XSHE"]
        ):  # already persisted
            actual = await store.pooling(datetime.date(2022, 10, 28))
            self.assertEqual(actual[0]["name"], "深桑达A")
            self.assertEqual(
                actual[0]["date"].item().date(), datetime.date(2022, 10, 28)
            )
            self.assertAlmostEqual(actual[0]["upper_line"], 0.046, 2)

            self.assertListEqual([20221028], store.pooled)

        now = datetime.datetime(2022, 11, 2)
        with mock.patch("arrow.now", return_value=now):
            with mock.patch(
                "omicron.models.security.Query.eval", return_value=["600640.XSHG"]
            ):  # should not be persisted
                actual = await store.pooling(now.date())
                self.assertEqual(actual[0]["name"], "国脉文化")
                self.assertAlmostEqual(actual[0]["upper_line"], 0.0225, 3)
                self.assertListEqual([20221028], store.pooled)

                actual = await store.query(now, "600640.XSHG", None)
                self.assertEqual(actual[0]["name"], "国脉文化")

                actual = await store.query(
                    datetime.date(2022, 10, 28),
                    "600640.XSHG",
                    end=datetime.date(2022, 10, 28),
                )
                self.assertTrue(len(actual) == 0)

                actual = await store.query(
                    datetime.date(2022, 10, 28), hit_flag=False, end=now
                )
                self.assertTrue(len(actual) == 0)

                actual = await store.query(now, hit_flag=True)
                self.assertTrue(len(actual) == 1)

        with mock.patch(
            "omicron.models.security.Query.eval", return_value=["002253.XSHE"]
        ):
            actual = await store.query(datetime.date(2022, 10, 24), hit_flag=False)
            self.assertTrue(len(actual) == 1)
            self.assertEqual(actual[0]["name"], "川大智胜")
