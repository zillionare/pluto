import datetime
import os
import unittest

import cfg4py
import omicron
import pytest
from freezegun import freeze_time

from pluto.config import get_config_dir
from pluto.store.buy_limit_pool import BuyLimitPoolStore


class StoreTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ[cfg4py.envar] = "DEV"
        cfg4py.init(get_config_dir())
        await omicron.init()

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
