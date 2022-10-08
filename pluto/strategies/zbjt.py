"""
2022年9月7日，中百集团在经过4个月横盘，下打、涨跌试盘、缩量回踩之后，连拉4个板。
研究表明，从技术形态上讲，具有以下特点：
1. 较长时间的横盘整理及均线粘合。
2. 有下打洗盘动作。
3. 5日内有涨停整理，及缩量回踩月线（未及），收下影线。
4. 5， 10， 20多头排列，且上方无长均线压制。
5. 启动时，行业板块向上，且未见顶。
6. 9月8日11：30启动并拉涨停。启动前，30分钟RSI在(0.5std, -0.5*std)条件约束下底背离
7. 启动时，当天大盘较平稳，最终收涨0.09%。
"""
import datetime

import cfg4py
from coretypes import bars_dtype
from omicron.talib import moving_average

from pluto.store import BuyLimitPoolStore
from pluto.strategies.base import Strategy


class StrategyZBJT(Strategy):
    name = "zbjt-strategy"
    desc = "2022年9月7日，中百集团在经过4个月横盘，下打、涨跌试盘、缩量回踩之后，连拉4个板。"

    def __init__(self, **kwargs):
        super().__init__()

        cfg = cfg4py.get_instance()
        self._buylimit_store = BuyLimitPoolStore(cfg.pluto.store_path)

    async def backtest(self, start: datetime.date, end: datetime.date, params: dict):
        pass

    async def extract_features(self, bars: bars_dtype, code: str):
        if len(bars) < 60:
            return None

        end = bars["frame"][-1]

        buy_limit_rec = await self._buylimit_store.query(end, code)
        # 近期必须有涨停
        if buy_limit_rec is None:
            return

        *_, total, continuous, last_date, till_now = buy_limit_rec
        # 不做连板高位股
        if continuous > 1:
            return None

        mas = []
        # 均线多头
        for win in (5, 10, 20):
            ma = moving_average(bars["close"], win)[-1].item()
            mas.append(ma)

        if not (mas[0] >= mas[1] and mas[1] >= mas[2]):
            return None

        # 上方无长均线压制

        # 区间统计
        # i_last_date = len(bars) - till_now
        # bars_ = bars[:i_last_date][-60:]
        # high = bars_["high"]
        # low = bars_["low"]
        # close = bars_["close"]
        # opn = bars_["open"]

        # amp = max(high) / min(low) - 1
        # amp = max(high) / min(low) - 1

    async def evaluate_long(self, code: str, bars: bars_dtype):
        pass
