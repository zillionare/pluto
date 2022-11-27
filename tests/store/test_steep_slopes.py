from pluto.store.steep_slopes_pool import SteepSlopesPool
import unittest
from unittest import mock
from freezegun import freeze_time
import cfg4py
import os
import omicron
import datetime
import shutil


class SteepSlopesPoolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        os.environ["pluto_store_path"] = "/tmp/pluto.zarr"
        cfg4py.init(os.path.expanduser("~/zillionare/pluto"))
        await omicron.init()
        try:
            shutil.rmtree("/tmp/pluto.zarr")
        except FileNotFoundError:
            pass

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @freeze_time("2022-11-15 15:31:00")
    async def test_pooling(self):
        ssp = SteepSlopesPool()
        codes = ["002780.XSHE", "000697.XSHE"]
        with mock.patch("omicron.models.security.Query.eval", return_value=codes):
            await ssp.pooling(dt=datetime.date(2022, 11, 16))
            actual = ssp.get(datetime.date(2022, 11, 16))
            self.assertEqual(5, len(actual))
            self.assertEqual("002780.XSHE", actual[0]["code"])
