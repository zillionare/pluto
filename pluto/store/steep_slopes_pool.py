import datetime
import logging
from collections import defaultdict
from typing import List

import numpy as np
from coretypes import FrameType
from numpy.typing import NDArray
from omicron import tf
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.talib import moving_average

from pluto.core.metrics import convex_score, parallel_score
from pluto.store.base import ZarrStore

logger = logging.getLogger(__name__)
ssp_dtype = np.dtype([("code", "U16"), ("slp", "f4"), ("win", "i4")])


class SteepSlopesPool(ZarrStore):
    def __init__(self, path: str = None):
        super().__init__(path)

    def save(self, date: datetime.date, records):
        if len(records) == 0:
            return

        logger.info("saving %s records for %s", len(records), date)

        date = tf.date2int(date)
        super().save(records, key=f"{date}")

        pooled = self.data.attrs.get(f"pooled", [])
        pooled.append(date)
        self.data.attrs["pooled"] = pooled

    def get(self, dt: datetime.date = None, win: int = None) -> NDArray[ssp_dtype]:
        if dt is not None:
            result = super().get(f"{tf.date2int(dt)}")
        else:
            try:
                dt = self.pooled[-1]
                result = super().get(f"{dt}")
            except IndexError:
                return None

        if result is None:
            return None

        # convert zarr to numpy array
        result = result[:]
        if win is None:
            return result

        return result[result["win"] == win]

    async def _do_pooling(self, secs: List[str], dt: datetime.date, n: int = 30):
        results = defaultdict(list)

        for i, code in enumerate(secs):
            if (i + 1) % 500 == 0:
                logger.info("progress update: %s/%s", i + 1, len(secs))
            bars = await Stock.get_bars(code, 70, FrameType.DAY, end=dt)
            if len(bars) < 10:
                continue

            close = bars["close"]

            # 尽管可以后期过滤，但当天涨幅过大的仍没有必要选取，它们在后面应该仍有机会被重新发现
            if close[-1] / close[-2] - 1 > 0.07:
                continue

            last_mas = []
            mas = {}
            for win in (10, 20, 30, 60):
                if len(bars) < win + 10:
                    break

                ma = moving_average(close, win)[-10:]
                last_mas.append(ma[-1])
                mas[win] = ma

            # 如果均线不为多头，则仍然不选取
            try:
                if parallel_score(last_mas) < 5 / 6:
                    continue
            except ZeroDivisionError:
                pass

            for win in (10, 20, 30, 60):
                ma = mas.get(win)
                if ma is not None:
                    score = convex_score(ma)
                    if score < 0:
                        continue

                    results[win].append((code, score))
            if len(results) < 4:  # 只有所有趋势都向上的，才计入
                continue

        # 对10, 20, 30 60均线，每种取前30支
        records = []
        for win in (10, 20, 30, 60):
            recs = results.get(win)
            if recs is None or len(recs) == 0:
                continue

            recs = sorted(recs, key=lambda x: x[1], reverse=True)
            for rec in recs[:n]:
                records.append((*rec, win))

        records = np.array(records, dtype=ssp_dtype)
        self.save(dt, records)
