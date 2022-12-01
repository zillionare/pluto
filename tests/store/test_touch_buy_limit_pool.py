import datetime
import os
import shutil
import unittest
from unittest import mock

import arrow
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
            shutil.rmtree(os.path.expanduser("~/tmp/pluto.zarr"))
        except FileNotFoundError:
            pass

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @pytest.mark.skipif(os.environ.get("IS_GITHUB"), reason="本测试需要omicron数据，只能在本地运行")
    async def test_pooling(self):
        # this also tests get/query
        store = TouchBuyLimitPoolStore(os.path.expanduser("~/tmp/pluto.zarr"))

        with mock.patch(
            "omicron.models.security.Query.eval", return_value=["000032.XSHE"]
        ):
            await store.pooling(datetime.date(2022, 10, 28))
            actual = store.get(datetime.date(2022, 10, 28))
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

        with freeze_time("2022-11-02 11:03:00"):
            now = datetime.date(2022, 11, 2)
            with mock.patch(
                "omicron.models.security.Query.eval", return_value=["600640.XSHG"]
            ):  # should not be persisted
                actual = await store.pooling(now)
                self.assertEqual(actual[0]["name"], "国脉文化")
                self.assertAlmostEqual(actual[0]["upper_line"], 0.0378, 3)
                self.assertListEqual([20221028], store.pooled)

                # 没有持久化
                actual = await store.query(now, "600640.XSHG", None)
                self.assertEqual(0, len(actual))

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

        with mock.patch(
            "omicron.models.security.Query.eval", return_value=["002253.XSHE"]
        ):
            await store.pooling(datetime.date(2022, 10, 24))
            actual = await store.query(datetime.date(2022, 10, 24), hit_flag=False)
            self.assertTrue(len(actual) == 1)
            self.assertEqual(actual[0]["name"], "川大智胜")

        # 测试连续多天pooling
        with freeze_time("2022-11-03 14:30:00"):
            with mock.patch(
                "omicron.models.security.Query.eval",
                return_value=["600640.XSHG", "000505.XSHE"],
            ):
                for date in (datetime.date(2022, 10, 31), datetime.date(2022, 10, 28)):
                    await store.pooling(date)
                self.assertListEqual([20221028, 20221024, 20221031], store.pooled)
