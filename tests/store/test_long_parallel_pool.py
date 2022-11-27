from pluto.store.long_parallel_pool import LongParallelPool
import unittest
from unittest import mock
from freezegun import freeze_time
import cfg4py
import os
import omicron
import datetime
import shutil


class LongParallelPoolTest(unittest.IsolatedAsyncioTestCase):
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

    @freeze_time("2022-11-11 14:31:00")
    async def test_pooling(self):
        lpp = LongParallelPool()
        codes = ["600138.XSHG", "000697.XSHE"]
        with mock.patch("omicron.models.security.Query.eval", return_value=codes):
            await lpp.pooling(datetime.date(2022, 11, 10))
            actual = lpp.get(datetime.date(2022, 11, 10))
            self.assertEqual("600138.XSHG", actual[0]["code"])

            data = await lpp.scan_30m_frames()
            self.assertEqual(data[0][0], "600138.XSHG")
