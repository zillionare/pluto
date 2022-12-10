"""尾盘快速选股，使用30分钟多头，或者涨停股、触及涨停股30分钟当天出现RSI低位（至少3周期确认）"""
import datetime
import logging
from typing import List

import numpy as np
import pandas as pd
import talib as ta
from coretypes import BarsArray, FrameType
from empyrical import sharpe_ratio
from numpy.typing import NDArray
from omicron import tf
from omicron.extensions import price_equal, smallest_n_argpos, top_n_argpos
from omicron.models.stock import Stock
from omicron.talib import moving_average, polyfit
from talib import RSI

from pluto.core.metrics import adjust_close_at_pv, parallel_score
from pluto.core.volume import top_volume_direction
from pluto.store.base import ZarrStore
from pluto.store.buy_limit_pool import BuyLimitPoolStore
from pluto.store.touch_buy_limit_pool import TouchBuyLimitPoolStore

logger = logging.getLogger(__name__)


def pct_change_by_m30(bars: BarsArray):
    """根据30分钟行情计算出当天涨跌幅"""
    today = bars[-1]["frame"].item().date()
    prev_day = tf.day_shift(today, -1)
    c1 = bars[bars["frame"] == tf.combine_time(prev_day, 15)][0]["close"]

    return bars["close"][-1] / c1 - 1


class MomemtumStrategy:
    def __init__(self, path: str = None):
        self.store = ZarrStore(path)

    def describe_short_features(self, features: List):
        """"""
        fmt = [
            "RSI高水位差: {:.1f}",  # 正值表明已超出高水位
            "距离RSI前高: {} bars",
            "近期RSI摸高次数: {}",  # 3周期内RSI摸高次数
            "前高: {:.2%}",  # 当前股份距前高百分位。0表明正在创新高
            "3日内最大跌幅: {:.2%}",
            "3日收阴率: {:.1%}",
            "3日sharpe: {:.1f}",
            "最后bar涨幅: {:.2%}",
            "frame序号: {}",
            "最大成交量比(负数为卖出): {:.1f}",
            "异向成交量比: {:.1f}",
            "是否涨停: {}",
            "下方均线数: {}",
            "5_10_20多头指数: {:.1%}",
            "10_20_30多头指数: {:.1%}",
            "10_20_60多头指数: {:.1%}",
            "5日均线走势: {:.2%} {:.2%} {:.2%} {:.0f}",
            "10日均线走势: {:.2%} {:.2%} {:.2%} {:.0f}",
            "20日均线走势: {:.2%} {:.2%} {:.2%} {:.0f}",
            "30日均线走势: {:.2%} {:.2%} {:.2%} {:.0f}",
            "60日均线走势: {:.2%} {:.2%} {:.2%} {:.0f}",
        ]

        if len(features) != 36:
            raise ValueError(
                f"length of features {len(features)} should math to formatters"
            )

        msg = []
        for i in range(0, 16):
            msg.append(fmt[i].format(features[i]))

        for i in range(0, 5):
            msg.append(fmt[16 + i].format(*features[16 + i * 4 : 16 + (i + 1) * 4]))

        return msg

    def extract_short_features(self, bars: BarsArray):
        """从30分钟bars中提取看空相关特征"""
        assert len(bars) >= 70, "size of bars must be at least 70."
        features = []
        close = bars["close"]
        returns = close[-24:] / close[-25:-1] - 1

        # 当前rsi与rsi高水位差值
        _, hclose, pvs = adjust_close_at_pv(bars[-60:], 1)
        rsi = ta.RSI(hclose.astype("f8"), 6)

        hrsi = top_n_argpos(rsi, 3)
        hrsi_mean = np.mean(rsi[hrsi])
        rsi_gap = rsi[-1] - hrsi_mean
        features.append(rsi_gap)

        # 当前距离rsi前高位置，以及3个bar以内有多少次高点：如果当前为最高，则距离为零
        dist = len(hclose) - hrsi - 1
        count = np.count_nonzero(dist < 2)
        features.extend((np.min(dist), count))

        # 最近的峰顶压力位
        peaks = np.argwhere(pvs == 1).flatten()
        if len(peaks) > 0 and peaks[-1] == 59:
            peaks = peaks[:-1]
        price_at_peaks = hclose[peaks]
        if len(price_at_peaks) > 0:
            gaps = price_at_peaks / close[-1] - 1
            gaps = gaps[gaps > 0]
            if len(gaps) > 0:
                peak_pressure = gaps[-1]
            else:
                peak_pressure = 0  # 创新高

        else:
            peak_pressure = -1  # 找不到顶，但也没创新高

        features.append(peak_pressure)

        # 3日内（24bars) 最大跌幅
        features.append(np.min(returns))

        # 3日内（24bars）阴线比率
        bulls = np.count_nonzero((bars["close"] > bars["open"])[-24:])
        features.append(bulls / 24)

        # 3日内(24 bars)的sharpe (不用sortino是因为sortino有可能为np.inf)
        # rf 必须为0，因为我们使用的是30分钟bar
        features.append(sharpe_ratio(returns))

        # 当前bar的序号 10:00 -> 0, 15:00 -> 8
        # 如果尾盘拉涨，一般要卖
        features.append(returns[-1])
        last_frame = bars[-1]["frame"].item()
        i = 0 if last_frame.minute == 0 else 1
        ilf = {600: 0, 630: 1, 660: 2, 690: 3, 810: 4, 840: 5, 870: 6, 900: 7}.get(
            last_frame.hour * 60 + i * 30
        )
        features.append(ilf)
        # 最大成交方向及力度
        vmax, vreverse = top_volume_direction(bars, 24)
        features.extend((vmax, vreverse))

        # 当前是否已涨停？涨停的情况下总是不卖
        # 如果close == high并且涨幅超9.5%，则认为已涨停
        df = pd.DataFrame(bars)
        day_bars = df.resample("1D", on="frame").agg({"close": "last", "high": "max"})
        c0 = day_bars["close"][-1]
        c1 = day_bars["close"][-2]
        h0 = day_bars["high"][-1]

        zt = price_equal(c0, h0) and c1 / c0 - 1 >= 0.095
        features.append(zt)

        # 均线走势
        mas = []
        maline_features = []
        for win in (5, 10, 20, 30, 60):
            ma = moving_average(close, win)[-10:]
            mas.append(ma[-1])

            err, (a, b, _), (vx, _) = polyfit(ma / ma[0], deg=2)
            maline_features.extend((err, a, b, np.clip(vx, -1, 10)))

        # 当前股价与均线关系, 0表示位于所有均线之下
        flag = np.count_nonzero(close[-1] >= np.array(mas))
        features.append(flag)

        # 5, 10, 20多头指数
        features.append(parallel_score(mas[:3]))

        # 10, 20, 30多头指数
        features.append(parallel_score(mas[1:4]))

        # 10, 20, 60多头指数
        features.append(parallel_score((mas[1], mas[2], mas[4])))

        # 加入均线走势特征
        features.extend(maline_features)

        return features

    async def add_short_sample(
        self, code: str, frame: datetime.datetime, label: int = 1
    ) -> int:
        """向`short`训练集中增加样本

        Args:
            code: 股票代码
            frame: 行情所属时间
            label: 样本标签，取值分别为-1（无操作），0（减半仓），1（清仓）

        Returns:
            datastore中的记录条数
        """
        bars = await Stock.get_bars(code, 70, FrameType.MIN30, end=frame)
        if len(bars) < 70:
            raise ValueError(f"size of bars {len(bars)} is less than 70")

        feature = self.extract_short_features(bars)
        logger.debug(
            "%s@%s\n:%s", code, frame, "\n".join(self.describe_short_features(feature))
        )
        key = "train/short/data"
        feature.append(label)
        self.store.append(np.array([feature]), key)
        data_size = len(self.store.get(key))

        meta_key = "train/short/meta"
        self.store.append([f"{code}:{tf.time2int(frame)}"], meta_key)
        meta_size = len(self.store.get(meta_key))

        if data_size != meta_size:
            raise ValueError("存储出错，元记录个数不等于数据个数。")

        return data_size

    def get_train_data(self, is_long):
        if not is_long:
            return self.store.get("train/short/data")
