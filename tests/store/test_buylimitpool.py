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
from pluto.store.buy_limit_pool import BuyLimitPoolStore


class BuyLimitPoolTest(unittest.IsolatedAsyncioTestCase):
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

    @pytest.mark.skipif(os.environ.get("IS_GITHUB"), reason="本测试只能在本地运行")
    async def test_day_closed(self):
        store = BuyLimitPoolStore()

        with freeze_time("2022-10-10 13:00:00"):
            actual = store._day_closed(datetime.datetime.now().date())
            self.assertEqual(actual, datetime.date(2022, 9, 30))

        with freeze_time("2022-10-10 15:00:00"):
            actual = store._day_closed(datetime.datetime.now().date())
            self.assertEqual(actual, datetime.date(2022, 10, 10))

        with freeze_time("2022-10-7 15:00:00"):
            actual = store._day_closed(datetime.datetime.now().date())
            self.assertEqual(actual, datetime.date(2022, 9, 30))

    @pytest.mark.skipif(os.environ.get("IS_GITHUB"), reason="本测试只能在本地运行")
    async def test_all(self):
        store = BuyLimitPoolStore(os.path.expanduser("~/tmp/pluto.zarr"))

        secs = [
            "003007.XSHE",
            "002654.XSHE",
            "688136.XSHG",
            "300491.XSHE",
            "603232.XSHG",
            "600684.XSHG",
        ]
        with mock.patch("omicron.models.security.Query.eval", return_value=secs):
            start = datetime.date(2022, 11, 1)
            end = datetime.date(2022, 11, 2)
            for dt in (start, end):
                await store.pooling(dt)
                # ensure pooled only once
                await store.pooling(dt)
            query_results = store.find_all(start, end)
            exp_codes = {
                "688136.XSHG",
                "300491.XSHE",
                "603232.XSHG",
                "600684.XSHG",
                "003007.XSHE",
            }
            self.assertSetEqual(set(query_results["code"].tolist()), exp_codes)
            self.assertEqual(
                query_results[query_results["code"] == "688136.XSHG"]["total"], 1
            )
            self.assertEqual(
                query_results[query_results["code"] == "600684.XSHG"]["continuous"], 2
            )
            self.assertEqual(
                query_results[query_results["code"] == "600684.XSHG"]["last"],
                datetime.date(2022, 11, 2),
            )

        with mock.patch("omicron.models.security.Query.eval", return_value=secs):
            dt = datetime.date(2022, 11, 3)
            await store.pooling(dt)

            actual = store.find_all(datetime.date(2022, 11, 1), dt)
            self.assertEqual(
                actual[actual["code"] == "600684.XSHG"]["continuous"].item(), 3
            )

        actual = store.find_by_code(
            "600684.XSHG", datetime.date(2022, 11, 1), datetime.date(2022, 11, 3)
        )
        self.assertEqual(actual[1], 3)
