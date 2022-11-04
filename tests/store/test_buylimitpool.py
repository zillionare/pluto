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
            shutil.rmtree("/tmp/pluto.zarr")
        except FileNotFoundError:
            pass

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @pytest.mark.skipif(os.environ.get("IS_GITHUB"), reason="本测试只能在本地运行")
    async def test_adjust_timestamp(self):
        store = BuyLimitPoolStore()

        with freeze_time("2022-10-10 13:00:00"):
            actual = store._adjust_timestamp(datetime.datetime.now().date())
            self.assertEqual(actual, datetime.date(2022, 9, 30))

        with freeze_time("2022-10-10 15:00:00"):
            actual = store._adjust_timestamp(datetime.datetime.now().date())
            self.assertEqual(actual, datetime.date(2022, 10, 10))

        with freeze_time("2022-10-7 15:00:00"):
            actual = store._adjust_timestamp(datetime.datetime.now().date())
            self.assertEqual(actual, datetime.date(2022, 9, 30))

    @pytest.mark.skipif(os.environ.get("IS_GITHUB"), reason="本测试只能在本地运行")
    async def test_all(self):
        store = BuyLimitPoolStore("/tmp/pluto.zarr")

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
            raw_records = await store.pooling(start, end)
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
            start = datetime.date(2022, 11, 3)
            end = datetime.date(2022, 11, 3)
            records = await store.pooling(start, end)
            self.assertEqual(records["code"].item(), "600684.XSHG")
            self.assertEqual(records["date"].item(), 20221103)

            query = store.find_all(datetime.date(2022, 11, 1), end)
            print(query)
            self.assertEqual(
                query[query["code"] == "600684.XSHG"]["continuous"].item(), 3
            )

        actual = store.find_by_code(
            "600684.XSHG", datetime.date(2022, 11, 1), datetime.date(2022, 11, 3)
        )
        self.assertEqual(actual[1], 3)
