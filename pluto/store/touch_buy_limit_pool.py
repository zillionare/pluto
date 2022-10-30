"""当天触及过涨停的个股池，记录其（30分钟bar）上影线、涨停bar前向量比，至当天收盘时的后向量比。
"""
import datetime
import logging
from typing import Optional, Tuple

import arrow
import numpy as np
from omicron import tf
from omicron.extensions import find_runs
from omicron.models.security import Security
from omicron.models.stock import Stock

from pluto.store.base import ZarrStore

logger = logging.getLogger(__name__)


class TouchBuyLimitPoolStore(ZarrStore):
    dtype = np.dtype(
        [
            ("name", "<U16"),
            ("code", "<U16"),
            ("date", "datetime64[S]"),
            ("shadow", "f4"),
            ("va", "f4"),
            ("vb", "f5"),
        ]
    )

    def __init__(self, path: str = None):
        super().__init__(path)

    def save(self, records):
        self._store.append(records)

    async def get(self, timestamp: datetime.date):
        start = tf.combine_time(timestamp, 0)
        end = tf.combine_time(timestamp, 15)
        idx = np.argwhere((self._store["date"] >= start) & (self._store["date"] < end))
        return self._store[idx]

    async def extract_features(
        self, code: str, end: datetime.date, n: int = 10
    ) -> Optional[Tuple]:
        """提取个股在[start, end]期间的涨跌停特征

        Args:
            code: 股票代码
            end: 截止时间
            n: 取多少个bar进行检查

        Returns:
            如果存在涨停，则返回(name, code, 总涨停次数，连续涨停次数，最后涨停时间，最后涨停距现在的bar数)
        """
        start = tf.day_shift(end, -1)

        try:
            flags = await Stock.get_trade_price_limits(code, start, end)
            if flags is None or len(flags) == 0:
                return None

            flags, _ = flags
            total = np.count_nonzero(flags)

            if total > 0:
                name = await Security.alias(code)
                if name.find("ST") != -1:
                    return None

                last = np.argwhere(flags).ravel()[-1]
                last_date = tf.day_shift(end, last - n + 1)

                v, _, length = find_runs(flags)
                pos = np.where(v == 1)
                cont = length[pos]
                cont = max(cont)
                till_now = n - last

                return name, code, total, cont, last_date, till_now
        except Exception as e:
            logger.exception(e)

    async def pooling(self, end: datetime.date = None):
        if end is None:
            end = self._adjust_timestamp(arrow.now().date())

        logger.info("building buy limit pool for %s(%s)...", end, self.win)
        secs = (
            await Security.select()
            .types(["stock"])
            .exclude_kcb()
            .exclude_cyb()
            .exclude_st()
            .eval()
        )
        result = []
        for sec in secs:
            r = await self.extract_buy_limit_features(sec, end, self.win)
            if r is not None:
                result.append(r)

        records = np.array(result, dtype=self.dtype)
        self.save(end, records)
        return records

    def status(self):
        return sorted(self._store.array_keys())

    async def query(self, timestamp: datetime.date, code: str):
        pool = await self.get(self._adjust_timestamp(timestamp))
        return pool[pool["code"] == code]
