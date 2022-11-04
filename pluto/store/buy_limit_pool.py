"""涨停板数据统计类

使用方法：
在使用之前，请先创建一个实例，然后调用pooling方法生成原始数据，然后可进行查询。如下面的例子所示：
```python
# path是数据文件存储位置。如果不传入，将依次使用pluto_store_path环境变量指定的值，如仍未指定，将使用当前目录下的pluto.zarr
store = BuyLimitPoolStore(path)
await store.pooling(start, end)
print(store.find_all(start, end))
print(store.pooled)
```
以下为输出示例:
```
[
    ('300491.XSHE', 1, 1, '2022-11-01')
    ('600684.XSHG', 3, 3, '2022-11-03')
    ('003007.XSHE', 2, 2, '2022-11-02')
    ('688136.XSHG', 1, 1, '2022-11-01')
    ('603232.XSHG', 1, 1, '2022-11-02')
]

# 已经进行了统计的交易日
20221101, 20221102, 20221103
```
"""
import datetime
import logging
from typing import List, Tuple

import arrow
import numpy as np
from coretypes import FrameType
from numpy.typing import NDArray
from omicron import tf
from omicron.extensions import find_runs
from omicron.models.security import Security
from omicron.models.stock import Stock

from pluto.store.base import ZarrStore

logger = logging.getLogger(__name__)

buylimitquery_dtype = np.dtype(
    [("code", "<U16"), ("total", "i4"), ("continuous", "i4"), ("last", "datetime64[D]")]
)


class BuyLimitPoolStore(ZarrStore):
    dtype = np.dtype([("code", "<U16"), ("date", "i8")])

    def __init__(self, path: str = None):
        super().__init__(path)

    def save(self, records, dates: List[int]):
        if len(records) == 0:
            return

        logger.info("save pool from %s~%s", dates[0], dates[-1])
        super().append(records)

        pooled = self.data.attrs.get("pooled", [])
        pooled.extend(dates)
        self.data.attrs["pooled"] = pooled

    @property
    def pooled(self):
        """返回已进行涨停特征提取的交易日列表"""
        try:
            pooled = self.data.attrs.get("pooled", [])
        except KeyError as e:
            logger.exception(e)
            pooled = []

        return pooled

    def _adjust_timestamp(self, timestamp: datetime.date) -> datetime.date:
        if tf.is_trade_day(timestamp) and datetime.datetime.now().hour < 15:
            return tf.day_shift(timestamp, -1)
        else:
            return tf.day_shift(timestamp, 0)

    async def pooling(self, start: datetime.date, end: datetime.date = None):
        if end is None:
            end = self._adjust_timestamp(arrow.now().date())

        logger.info("building buy limit pool from %s - %s...", start, end)
        secs = await Security.select().types(["stock"]).exclude_st().eval()
        records = []
        frames = tf.get_frames(start, end, FrameType.DAY)
        for sec in secs:
            flags = await Stock.trade_price_limit_flags(sec, start, end)
            if flags is None or len(flags) == 0:
                continue

            flags, _ = flags
            if len(flags) != len(frames):
                logger.warning(
                    "%s: length of flags(%s) is not same size as frames(%s)",
                    sec,
                    len(flags),
                    len(frames),
                )
                continue

            for frame, flag in zip(frames, flags):
                if flag and frame not in self.pooled:
                    records.append((sec, frame))

        records = np.array(records, dtype=self.dtype)
        self.save(records, frames)
        return records

    def count_continous(self, records, frames: List[int]) -> int:
        """找出最长的连续板个数"""
        flags = np.isin(frames, records["date"])
        v, _, length = find_runs(flags)
        return max(length[v])

    def _calc_stats(self, records, frames):
        total = len(records)
        last = np.max(records["date"])
        continuous = self.count_continous(records, frames)
        return total, continuous, tf.int2date(last)

    def find_all(
        self, start: datetime.date, end: datetime.date = None
    ) -> NDArray[buylimitquery_dtype]:
        """找出`[start, end]`区间所有涨停的个股，返回代码、涨停次数、最长连续板数和最后涨停时间

        Args:
            start: 起始时间
            end: 结束时间

        Raises:
            ValueError: 如果指定区间存在一些交易日未进行过pooling操作，则抛出此错误

        Returns:
            返回代码、涨停次数、最长连续板数和最后涨停时间
        """
        frames = tf.get_frames(start, end, FrameType.DAY)
        missed = set(frames) - set(self.pooled)
        if len(missed) > 0:
            raise ValueError(f"data not ready for frames, run pooling first: {missed}")

        start = tf.date2int(start)
        end = tf.date2int(end)

        idx = np.argwhere((self.data["date"] >= start) & (self.data["date"] <= end))
        records = self.data[idx.flatten()]

        results = []
        for code in set(records["code"]):
            sub = records[records["code"] == code]
            results.append((code, *self._calc_stats(sub, frames)))

        return np.array(results, buylimitquery_dtype)

    def find_by_code(
        self, code: str, start: datetime.date, end: datetime.date = None
    ) -> Tuple[int, int, datetime.date]:
        """查找个股`code`在区间[`start`, `end`]里的涨停数据

        Args:
            code: 股票代码
            start: 起始日期
            end: 结束日期

        Raises:
            ValueError: 如果指定区间存在一些交易日未进行过pooling操作，则抛出此错误

        Returns:
            返回涨停次数、最长连续板数和最后涨停时间
        """
        end = end or arrow.now().date()
        frames = tf.get_frames(start, end, FrameType.DAY)
        missed = set(frames) - set(self.pooled)
        if len(missed) > 0:
            raise ValueError(f"data not ready for frames, run pooling first: {missed}")

        start = tf.date2int(start)
        end = tf.date2int(end)

        idx = np.argwhere((self.data["date"] >= start) & (self.data["date"] <= end))
        records = self.data[idx.flatten()]
        return self._calc_stats(records[records["code"] == code], frames)
