"""当天触及过涨停的个股池，记录其（30分钟bar）上影线、涨停bar前向量比，至当天收盘时的后向量比。
"""
import datetime
import logging
from typing import Optional, Tuple

import arrow
import numpy as np
from coretypes import FrameType
from omicron import tf
from omicron.extensions import price_equal
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

    def __init__(self, path: str = None):
        super().__init__(path)

    def save(self, records):
        date = records[-1]["date"].item().date()
        try:
            records = np.append(self.store, records)
        except KeyError:  # the very first time
            pass
        super().save("/", records)
        pooled = self.store.attrs.get("pooled", [])
        pooled.append(tf.date2int(date))
        self.store.attrs["pooled"] = pooled

    def _adjust_timestamp(self, timestamp: datetime.date) -> datetime.date:
        """避免存入非交易日数据"""
        if not tf.is_trade_day(timestamp):
            return tf.day_shift(timestamp, 0)

    async def get(self, timestamp: datetime.date):
        if tf.date2int(timestamp) not in self.pooled:
            return await self.pooling(timestamp)
        else:
            start = tf.combine_time(timestamp, 0)
            end = tf.combine_time(timestamp, 15)
            idx = np.argwhere(
                (self.store["date"] >= start) & (self.store["date"] < end)
            ).flatten()
            return self.store[idx]

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
            prices = await Stock.get_trade_price_limits(code, end, end)
            if len(prices) == 0:
                return None

            high_limit = prices["high_limit"][0]
            start = tf.combine_time(end, 10)
            end_time = tf.combine_time(end, 15)
            bars = await Stock.get_bars_in_range(
                code, FrameType.MIN30, start=start, end=end_time
            )
            close = bars["close"][-1]
            opn = bars["open"][0]
            for i in range(len(bars)):
                limit_signal = price_equal(bars["high"][i], high_limit)
                if limit_signal and (not price_equal(close, high_limit)):
                    name = await Security.alias(code)
                    upper_line = high_limit / max(close, opn) - 1
                    return (name, code, bars["frame"][i].item(), upper_line)

        except Exception as e:
            logger.exception(e)

    async def pooling(self, end: datetime.date = None):
        if end is None:
            end = self._adjust_timestamp(arrow.now().date())

        if tf.date2int(end) in self.pooled:
            return await self.get(end)

        logger.info("building touch buy limit pool on %s...", end)
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

        if end != arrow.now().date():
            self.save(records)

        return records

    @property
    def pooled(self):
        try:
            pooled = self.store.attrs.get("pooled", [])
        except KeyError:
            pooled = []

        return pooled

    async def query(self, timestamp: datetime.date, code: str):
        pool = await self.get(self._adjust_timestamp(timestamp))
        return pool[pool["code"] == code]
