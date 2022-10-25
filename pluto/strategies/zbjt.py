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
from coretypes import BarsArray
from omicron.talib import moving_average

from pluto.store import BuyLimitPoolStore
from pluto.strategies.base import BaseStrategy


class StrategyZBJT(BaseStrategy):
    name = "zbjt-strategy"
    desc = "2022年9月7日，中百集团在经过4个月横盘，下打、涨跌试盘、缩量回踩之后，连拉4个板。"

    def __init__(self, **kwargs):
        super().__init__()

        cfg = cfg4py.get_instance()
        self._buylimit_store = BuyLimitPoolStore(cfg.pluto.store_path)

    async def backtest(self, start: datetime.date, end: datetime.date, params: dict):
        pass

    async def extract_features(self, code: str, bars: BarsArray):
        if len(bars) < 60:
            return None

        end = bars["frame"][-1]
        c0 = bars["close"][-1]
        low0 = bars["low"][-1]

        buy_limit_rec = await self._buylimit_store.query(end, code)
        # 近期必须有涨停
        if buy_limit_rec is None:
            return

        *_, total, continuous, _, till_now = buy_limit_rec
        # 不做连板高位股
        if continuous > 1:
            return None

        # 均线多头
        ma5 = moving_average(bars["close"], 5)
        ma10 = moving_average(bars["close"], 5)
        ma20 = moving_average(bars["close"], 5)
        if not (
            ma5[-1].item() >= ma10[-1].item() and ma20[-1].item() >= ma20[-1].item()
        ):
            return None

        # 上方无长均线压制
        for win in (60, 120, 250):
            ma = moving_average(bars["close"], win)[-1].item()
            if ma > c0:
                return None

        # 区间统计
        i_last_date = len(bars) - till_now
        bars_ = bars[:i_last_date][-60:]
        high = bars_["high"]
        low = bars_["low"]
        close = bars_["close"][-1].item()
        opn = bars_["open"][-1].item()

        amp = max(high) / min(low) - 1
        adv = close / opn - 1

        # 20日线支撑？如果touch_20在[0, 1.5%]之间，说明支撑较强;3%以上表明未考验支撑。-0.5%以下说明没有支撑。其它情况表明是否存在支撑不显著。
        touch_20 = low0 / ma20 - 1
        return total, till_now, amp, adv, touch_20

    async def evaluate_long(self, code: str, bars: BarsArray):
        features = self.extract_features(code, bars)
        if features is None:
            return False
        else:
            _, till_now, amp, adv, touch_20 = features
            if till_now > 5 or adv > 0.1 or amp > 0.25 or touch_20 < -0.005:
                return False

        return True
