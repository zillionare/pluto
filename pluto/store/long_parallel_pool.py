import datetime
import logging
from typing import List, Tuple

import numpy as np
import talib as ta
from coretypes import FrameType
from numpy.typing import NDArray
from omicron import tf
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.notify.dingtalk import ding
from omicron.talib import moving_average

from pluto.core.ma import predict_next_price
from pluto.core.metrics import last_wave, parallel_score
from pluto.store.base import ZarrStore

logger = logging.getLogger(__name__)
momentum_feature_dtype = np.dtype(
    [
        ("code", "U16"),  # 股票代码
        ("pred5", "f4"),  # 用5日线预测的股价涨跌幅
        ("pred10", "f4"),  # 用10日线预测的股价涨跌幅
        ("rsi", "f4"),  # 当前RSI值
        ("dist", "i4"),  # 距离RSI高点位置
        ("gap2year", "f4"),  # 到年线距离
        ("ps", "f4"),  # 均线排列分数
        ("wave_len", "i4"),  # 最后一个波段涨幅
        ("wave_amp", "f4"),  # 最后一个波段涨跌幅
        ("max_adv", "f4"),  # 近5日日线最大涨幅
    ]
)


class LongParallelPool(ZarrStore):
    def __init__(self, path: str = None):
        super().__init__(path)

    def save(self, date: datetime.date, records):
        if len(records) == 0:
            return

        logger.info("saving %s records for %s", len(records), date)

        date = tf.date2int(date)
        super().save(records, key=str(date))

        pooled = self.data.attrs.get("pooled", [])
        pooled.append(date)
        self.data.attrs["pooled"] = pooled

    async def pooling(self, end: datetime.date = None):
        """采集`end`日线多头数据并存盘

        Args:
            end: 结束日期
        """
        end = self._day_closed(end or datetime.datetime.now().date())

        if tf.date2int(end) in self.pooled:
            logger.info("%s already pooled", end)
            return await self.get(end)

        logger.info(
            "building long parallel pool on %s, currently pooled: %s",
            end,
            len(self.pooled),
        )
        secs = (
            await Security.select()
            .types(["stock"])
            .exclude_st()
            .exclude_kcb()
            .exclude_cyb()
            .eval()
        )

        result = []
        for i, code in enumerate(secs):
            if (i + 1) % 500 == 0:
                logger.info("progress update: %s/%s", i + 1, len(secs))
            bars = await Stock.get_bars(code, 260, FrameType.DAY, end=end)
            if len(bars) < 60:
                continue

            close = bars["close"]
            returns = close[-10:] / close[-11:-1] - 1
            # 最近10天上涨都小于3.5%，暂不关注
            if np.max(returns) < 0.035:
                continue

            mas = []
            for win in (5, 10, 20, 60, 120, 250):
                if len(close) < win:
                    break

                ma = moving_average(close, win)[-1]
                mas.append(ma)

            if len(mas) == 6:
                gap2year = close[-1] / mas[-1] - 1
            else:
                gap2year = None

            mas = np.array(mas)
            # 短均线（5，10，20）必须多头
            ps = parallel_score(mas[:3])
            if ps != 1:
                continue

            ps = parallel_score(mas)

            # 去掉正处于40日内RSI高位的
            rsi = ta.RSI(close[-60:].astype("f8"), 6)
            dist = 40 - np.nanargmax(rsi[-40:])
            if dist == 1 and rsi[-1] >= 85:
                continue

            # 预测下一个收盘价
            pred5, _ = predict_next_price(bars, win=5)
            pred10, _ = predict_next_price(bars, win=10)

            # 波段涨幅和长度
            wave_len, wave_amp = last_wave(close)

            result.append(
                (
                    code,
                    pred5,
                    pred10,
                    rsi[-1],
                    dist,
                    gap2year,
                    ps,
                    wave_len,
                    wave_amp,
                    np.max(returns),
                )
            )

        records = np.array(result, dtype=momentum_feature_dtype)
        self.save(end, records)

    def get(self, date: datetime.date = None) -> NDArray[momentum_feature_dtype]:
        try:
            if date is None:
                date = self.pooled[-1]
        except Exception as e:
            logger.exception(e)
            return None

        return super().get(tf.date2int(date))

    async def filter_long_parallel(
        self,
        max_gap2year: float = None,
        wave_amp_rng: Tuple = None,
        date: datetime.date = None,
    ):
        date = self._day_closed(date or datetime.datetime.now().date())
        results = self.get(date)
        if max_gap2year is not None:
            idx = np.argwhere(results["gap2year"] <= max_gap2year).flatten()
            if len(idx) > 0:
                results = results[idx]
            else:
                return []

        if wave_amp_rng is not None:
            idx = np.argwhere(
                (results["wave_amp"] <= wave_amp_rng[1])
                & (results["wave_amp"] >= wave_amp_rng[0])
            ).flatten()
            if len(idx) > 0:
                results = results[idx]
            else:
                return []

        return results

    async def scan_30m_frames(self) -> List[Tuple]:
        """在日线多头个股中，寻找30分钟也多头的股票。

        Returns:
            返回一个数组，其中每一行由以下各列构成：code, name, change, pred5, pred10, rsi, dist, gap2year, ps, wave_len, wave_amp, max_adv
        """
        end = tf.floor(datetime.datetime.now(), FrameType.MIN30)
        data = []

        filtered = await self.filter_long_parallel(0.2, [0.05, 0.25])
        for (
            code,
            pred5,
            pred10,
            rsi,
            dist,
            gap2year,
            ps,
            wave_len,
            wave_amp,
            max_adv,
        ) in filtered:
            bars = await Stock.get_bars(code, 60, FrameType.MIN30, end=end)
            if len(bars) < 60:
                continue

            close = bars["close"]
            today = bars[-1]["frame"].item().date()
            prev_day = tf.day_shift(today, -1)
            c1 = bars[bars["frame"] == tf.combine_time(prev_day, 15)][0]["close"]
            # 今日涨幅
            change = close[-1] / c1 - 1
            name = await Security.alias(code)

            mas = []
            for win in (5, 10, 20, 60):
                ma = moving_average(close, win)[-1]
                mas.append(ma)

            mas = np.array(mas)
            # 上方有均线压制则不考虑
            if not (np.all(close[-1] > mas) and parallel_score(mas) == 1):
                continue

            # 从现在起，还可以上涨多少？
            pred5 = pred5 / close[-1] - 1
            pred10 = pred10 / close[-1] - 1
            data.append(
                (
                    code,
                    name,
                    change,
                    pred5,
                    pred10,
                    rsi,
                    dist,
                    gap2year,
                    ps,
                    wave_len,
                    wave_amp,
                    max_adv,
                )
            )

        msg = [
            "30分钟多头选股选出以下股票：",
            "-----------------------",
            " ".join([item[1] for item in data]),
        ]

        await ding("\n".join(msg))
        return data
