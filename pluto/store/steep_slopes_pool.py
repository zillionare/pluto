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
from pluto.core.metrics import last_wave

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

        min_ma_wave_len = {
            10: 6,
            20: 5,
            30: 4,
            60: 3
        }

        for i, code in enumerate(secs):
            if (i + 1) % 500 == 0:
                logger.info("progress update: %s/%s", i + 1, len(secs))
            bars = await Stock.get_bars(code, 70, FrameType.DAY, end=dt)
            if len(bars) < 70:
                continue

            close = bars["close"]

            _, last_wave_amp = last_wave(close)
            if last_wave_amp > 0.3:
                continue

            slopes = {}
            for win in (10, 20, 30, 60):
                ma = moving_average(close, win)[-10:]
                pmin = np.argmin(ma)
                if 10 - pmin < min_ma_wave_len.get(win, 6):
                    break

                # 要求均线必须向上
                score = convex_score(ma/ma[0], thresh = 1e-2)
                if score < 3e-1:
                    logger.debug(f"{code} convex not meet: {win} {score}")
                    continue


                slopes[win] = (code, score)
            if len(slopes) < 4: # 只有所有趋势都向上的，才计入
                continue

            for win in (10, 20, 30, 60):
                results[win].append(slopes.get(win))

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
