import datetime
import os
import shutil
import unittest
from unittest import mock

import cfg4py
import omicron
from freezegun import freeze_time

from pluto.store.steep_slopes_pool import SteepSlopesPool


class SteepSlopesPoolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        os.environ["pluto_store_path"] = os.path.expanduser("~/tmp/pluto.zarr")
        cfg4py.init(os.path.expanduser("~/zillionare/pluto"))
        await omicron.init()
        try:
            shutil.rmtree(os.path.expanduser("~/tmp/pluto.zarr"))
        except FileNotFoundError:
            pass

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @freeze_time("2022-11-15 15:31:00")
    async def test_pooling(self):
        ssp = SteepSlopesPool()
        codes = ["002317.XSHE"]
        with mock.patch("omicron.models.security.Query.eval", return_value=codes):
            await ssp.pooling(dt=datetime.date(2022, 11, 15))
            actual = ssp.get(datetime.date(2022, 11, 15))
            self.assertEqual(4, len(actual))
            self.assertEqual("002317.XSHE", actual[0]["code"])
            self.assertAlmostEqual(0.0567, actual[0]["slp"], 3)
