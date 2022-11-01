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


class StoreTest(unittest.IsolatedAsyncioTestCase):
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
    async def test_store(self):
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

        # todo: 寻找一支在其它时间触及涨停的个股，通过store.pooled来验证当天的结果不存盘
        now = datetime.datetime(2022, 10, 28)
        with mock.patch("arrow.now", return_value=now):
            with mock.patch(
                "omicron.models.security.Query.eval", return_value=["003042.XSHE"]
            ):  # shold not be persisted
                actual = await store.pooling(now.date())
                self.assertEqual(actual[0]["name"], "中农联合")
                self.assertAlmostEqual(actual[0]["upper_line"], 0.079, 3)
                self.assertListEqual([20221028], store.pooled)

                self.assertListEqual([20221028], store.pooled)
