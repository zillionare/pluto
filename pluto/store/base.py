import datetime
import logging
import os
from typing import Any, List

import arrow
import zarr
from omicron import tf
from omicron.models.security import Security

logger = logging.getLogger(__name__)


class ZarrStore(object):
    def __init__(self, path=None):
        cur_dir = os.path.dirname(__file__)
        self._store_path = (
            path
            or os.environ.get("pluto_store_path")
            or os.path.join(cur_dir, "pluto.zarr")
        )

        self._store = zarr.open(self._store_path, mode="a")

    def save(self, records: Any, key: str = None):
        """将`records` 存到`key`下面（替换式)

        Args:
            records: 要存储的数据
            key: 如果为None，则存到根下面。
        """
        if key is not None:
            key = f"{self.__class__.__name__.lower()}/{key}"
        else:
            key = f"{self.__class__.__name__.lower()}/"
        self._store[key] = records

    def append(self, records: Any, key: str = None):
        """向key所引用的数组增加数据"""
        if key is not None:
            key = f"{self.__class__.__name__.lower()}/{key}"
        else:
            key = f"{self.__class__.__name__.lower()}"

        if self._store.get(key):
            self._store[key].append(records)
        else:
            self._store[key] = records

    def get(self, key: str):
        key = f"{self.__class__.__name__.lower()}/{key}"
        return self._store[key]

    @property
    def data(self):
        key = f"{self.__class__.__name__.lower()}/"
        return self._store[key]

    @property
    def pooled(self) -> List[int]:
        """返回已进行涨停特征提取的交易日列表。

        注意这里返回的交易日为整数类型，即类似20221011。
        """
        try:
            pooled = self.data.attrs.get("pooled", [])
        except KeyError:
            pooled = []

        return pooled

    def _day_closed(self, timestamp: datetime.date) -> datetime.date:
        """给定`timestamp`，返回已结束的交易日"""
        now = datetime.datetime.now()
        if (
            tf.is_trade_day(timestamp)
            and timestamp == now.date()
            and datetime.datetime.now().hour < 15
        ):
            return tf.day_shift(timestamp, -1)
        else:
            return tf.day_shift(timestamp, 0)

    async def pooling(self, dt: datetime.date = None, **kwargs):
        """采集`dt`日的涨跌停数据并存盘。

        Args:
            dt: 统计日
        """
        dt = self._day_closed(dt or arrow.now().date())

        if getattr(self, "pooled", None) is not None:
            if tf.date2int(dt) in self.pooled:
                logger.info("%s already pooled", dt)
                return

        logger.info("building %s pool for %s...", self.__class__.__name__.lower(), dt)
        secs = (
            await Security.select()
            .types(["stock"])
            .exclude_cyb()
            .exclude_kcb()
            .exclude_st()
            .eval()
        )

        await self._do_pooling(secs, dt, **kwargs)
