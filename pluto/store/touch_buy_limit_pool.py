"""当天触及过涨停的个股池，记录其（30分钟bar）上影线、涨停bar前向量比，至当天收盘时的后向量比。
"""
import datetime
import logging
from types import FrameType
from typing import Optional, Tuple

import arrow
import numpy as np
from omicron import tf
from omicron.extensions import find_runs, price_equal
from omicron.models.security import Security
from omicron.models.stock import Stock

from pluto.store.base import ZarrStore

logger = logging.getLogger(__name__)


class TouchBuyLimitPoolStore(ZarrStore):
    dtype = np.dtype(
        [
            ("name", "<U16"),
            ("code", "<U16"),
            ("date", "datetime64[s]"),
            ("upper_line", "f4"),
        ]
    )

    def __init__(self, path: str = "传入一个path,用于测试"):
        super().__init__(path)

    def save(self, records):
        self._store.append(records)

    def _adjust_timestamp(self, timestamp: datetime.date) -> datetime.date:
        if tf.is_trade_day(timestamp) and datetime.datetime.now().hour < 15:
            return tf.day_shift(timestamp, -1)
        else:
            return tf.day_shift(timestamp, 0)

    async def get(self, timestamp: datetime.date):
        start = tf.combine_time(timestamp, 0)
        end = tf.combine_time(timestamp, 15)
        idx = np.argwhere((self._store["date"] >= start) & (self._store["date"] < end))
        return self._store[idx]

    async def extract_touch_buy_limit_features(
        self, code: str, end: datetime.date
    ) -> Optional[Tuple]:
        """提取个股在[end]期间冲涨停特征, 只记录当天第一次涨停时间

        Args:
            code: 股票代码
            end: 截止时间

        Returns:
            如果存在涨停，则返回(name, code, 涨停时间(30MIN为单位), 上引线百分比)
        """

        try:
            flags = await Stock.get_trade_price_limits(code, end, end)
            if flags is None or len(flags) == 0:
                return None

            high_limit = flags["high_limit"][0]
            end_time = tf.combine_time(end, 15)
            bar = await Stock.get_bars(code, 8, FrameType.MIN30, end_time)
            close = bar["close"][-1]
            open = bar["close"][-1]
            for i in range(len(bar)):
                limit_signal = price_equal(bar["high"][i], high_limit)
                if limit_signal and (not price_equal(close, high_limit)):
                    name = await Security.alias(code)
                    upper_line = high_limit / max(close, open) - 1
                    return (name, code, bar["frame"][i].item(), upper_line)

        except Exception as e:
            logger.exception(e)

    async def pooling(self, end: datetime.date = None):
        if end is None:
            end = self._adjust_timestamp(arrow.now().date())

        logger.info("building buy limit pool on %s...", end)
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
            r = await self.extract_touch_buy_limit_features(sec, end)
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
