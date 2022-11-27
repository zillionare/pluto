"""当天触及过涨停的个股池，记录其（30分钟bar）上影线、涨停bar前向量比，至当天收盘时的后向量比。
"""
import datetime
import logging
from typing import Optional, Tuple

import arrow
import numpy as np
from coretypes import FrameType
from numpy.typing import NDArray
from omicron import tf
from omicron.extensions import math_round, price_equal
from omicron.models.security import Security
from omicron.models.stock import Stock

from pluto.store.base import ZarrStore

logger = logging.getLogger(__name__)

"""touch_buy_limit_pool的存储结构，包括name, code, date, upper_line, max_adv, hit等字段
"""
touch_pool_dtype = np.dtype(
    [
        ("name", "<U16"),  # 股票名
        ("code", "<U16"),  # 股票代码
        ("date", "datetime64[s]"),  # 触及涨停日期
        ("upper_line", "f4"),  # 上影线
        ("max_adv", "f4"),  # 最大涨幅
        ("hit", "i4"),  # 是否完全触及
    ]
)


class TouchBuyLimitPoolStore(ZarrStore):
    def __init__(self, path: str = None, thresh=0.985):
        """
        Args:
            thresh: 当股份超过high_limit * thresh时即计入统计
            path: 存储位置
        """
        self.thresh = thresh
        super().__init__(path)

    def save(self, date: datetime.date, records):
        if len(records) == 0:
            return

        try:
            if tf.date2int(date) in self.pooled:
                return
        except KeyError:
            pass

        logger.info("save pool for day %s", date)
        super().append(records)

        pooled = self.data.attrs.get("pooled", [])
        pooled.append(tf.date2int(date))
        self.data.attrs["pooled"] = pooled

    def get(self, timestamp: datetime.date):
        if tf.date2int(timestamp) not in self.pooled:
            return None

        start = tf.combine_time(timestamp, 0)
        end = tf.combine_time(timestamp, 15)
        idx = np.argwhere(
            (self.data["date"] >= start) & (self.data["date"] < end)
        ).flatten()
        return self.data[idx]

    async def extract_touch_buy_limit_features(
        self, code: str, end: datetime.date
    ) -> Optional[Tuple]:
        """提取个股在[end]期间冲涨停特征, 只记录当天第一次涨停时间

        Args:
            code: 股票代码
            end: 截止时间

        Returns:
            如果存在涨停，则返回(name, code, 涨停时间(30MIN为单位), 上引线百分比,最大涨幅,是否触板)
        """
        try:
            prices = await Stock.get_trade_price_limits(code, end, end)
            if len(prices) == 0:
                return None

            high_limit = math_round(prices["high_limit"][0].item(), 2)
            start = tf.combine_time(tf.day_shift(end, -1), 15)
            end_time = tf.combine_time(end, 15)
            bars = await Stock.get_bars_in_range(
                code, FrameType.MIN30, start=start, end=end_time
            )

            frames = bars["frame"]
            if frames[0].item().date() != tf.day_shift(end, -1):
                return None

            c1 = math_round(bars["close"][0], 2)
            bars = bars[1:]

            close = math_round(bars["close"][-1], 2)
            opn = math_round(bars["open"][0], 2)
            idx = np.argmax(bars["high"])
            high = math_round(bars["high"][idx], 2)
            if high >= high_limit * self.thresh and not price_equal(close, high_limit):
                name = await Security.alias(code)
                upper_line = high / max(close, opn) - 1
                max_adv = high / c1 - 1
                if price_equal(high, high_limit):
                    hit_flag = True
                else:
                    hit_flag = False

                return (
                    name,
                    code,
                    bars["frame"][idx].item(),
                    upper_line,
                    max_adv,
                    hit_flag,
                )

        except Exception as e:
            logger.exception(e)

        return None

    async def pooling(self, end: datetime.date = None):
        end = end or datetime.datetime.now().date()

        if tf.date2int(end) in self.pooled:
            logger.info("%s already pooled.", end)
            return self.get(end)

        logger.info(
            "building touch buy limit pool on %s, currently pooled: %s",
            end,
            len(self.pooled),
        )
        secs = (
            await Security.select()
            .types(["stock"])
            .exclude_kcb()
            .exclude_cyb()
            .exclude_st()
            .eval()
        )
        result = []
        for i, sec in enumerate(secs):
            if (i + 1) % 500 == 0:
                logger.info("progress update: %s/%s", i + 1, len(secs))
            r = await self.extract_touch_buy_limit_features(sec, end)
            if r is not None:
                result.append(r)

        records = np.array(result, dtype=touch_pool_dtype)
        if end == self._day_closed(end):
            self.save(end, records)

        return records

    async def query(
        self, start: datetime.date, code: str = None, hit_flag=True, end=None
    ) -> NDArray[touch_pool_dtype]:
        """查询某日触板股票情况

        Args:
            start: 起始日期
            code: 如果未传入，则返回当天所有触板股票
            hit_flag: 如果为None，则返回当天完全触板及尝试触板的股票.
            end: 结束日期，如果不传入，则为最后一个交易日。

        Returns:
            类型为_dtype的numpy structured array
        """
        end = end or arrow.now().date()
        results = np.array([], dtype=touch_pool_dtype)

        for date in tf.get_frames(start, end, FrameType.DAY):
            date = tf.int2date(date)
            pool = self.get(date)

            if pool is None:
                continue

            if code is not None:
                pool = pool[pool["code"] == code]

            if hit_flag is not None:
                results = np.append(results, pool[(pool["hit"] == hit_flag)])
            else:
                results = np.append(results, pool)

        return results
