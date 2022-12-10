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
from omicron.talib import moving_average, polyfit

from pluto.core.metrics import convex_score, last_wave, parallel_score
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

        pooled = self.data.attrs.get("pooled", [])
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

    async def _do_pooling(self, secs: List[str], dt: datetime.date, n: int = 100):
        results = defaultdict(list)

        for i, code in enumerate(secs):
            if (i + 1) % 500 == 0:
                logger.info("progress update: %s/%s", i + 1, len(secs))
            bars = await Stock.get_bars(code, 70, FrameType.DAY, end=dt)
            if len(bars) < 70:
                continue

            close = bars["close"]

            wave_len, last_wave_amp = last_wave(close)
            if (last_wave_amp > 0.5 and wave_len < 40) or (
                last_wave_amp > 0.3 and wave_len < 10
            ):
                continue

            # 5日线需要向上
            ma5 = moving_average(close, 5)[-10:]
            if convex_score(ma5) < 0.2:
                continue

            wins = (10, 20, 30, 60)
            for win in wins:
                ma = moving_average(close, win)[-10:]

                # 要求均线必须向上
                ts = ma / ma[0]
                err, (slp, _) = polyfit(ts, deg=1)
                if err < np.std(ts) / 2:
                    results[win].append((code, slp))

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
