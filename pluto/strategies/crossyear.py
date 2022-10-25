import datetime
from typing import List, Tuple

import arrow
from boards.board import ConceptBoard, IndustryBoard
from coretypes import BarsArray, FrameType
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.plotting.candlestick import Candlestick
from omicron.talib import moving_average, valley_detect

from pluto.core.volume import describe_volume_morph, morph_pattern, net_buy_volume
from pluto.strategies.base import BaseStrategy


class StrategyCrossYear(BaseStrategy):
    name = "cross-year-strategy"
    desc = "个股在由下上攻年线时，更容易出现涨停"

    def extract_features(self, code: str, bars: BarsArray) -> Tuple:
        if len(bars) < 260:
            return None

        close = bars["close"]
        # opn = bars["open"]

        # 底部距离、上涨幅度
        dist, adv = valley_detect(close)

        ma250 = moving_average(close, 250, padding=False)
        c0 = close[-1]

        # 距年线距离
        gap = ma250[-1] / c0 - 1

        # morph pattern
        morph = morph_pattern(bars[-5:])

        # 净买入量
        nbv = net_buy_volume(bars[-5:])

        return (dist, adv, gap, morph, nbv)

    def evaluate_long(self, code: str, bars: BarsArray):
        if bars[-1]["close"] < bars[-1]["open"]:
            return False, None

        features = self.extract_features(code, bars)

        if features is None:
            return False, None

        dist, adv, gap, morph, nbv = features
        if dist is None or gap > 0.11 or gap < 0:
            return False, None

        return True, (dist, adv, gap, morph, nbv)

    def _belong_to_boards(
        self,
        code: str,
        ib,
        cb,
        with_industry: List[str] = None,
        with_concepts: List[str] = None,
    ):
        symbol = code.split(".")[0]

        if ib:
            industries = ib.get_boards(symbol)
            if set(industries).intersection(set(with_industry)):
                return True

        if cb:
            concepts = cb.get_boards(symbol)
            if set(concepts).intersection(with_concepts):
                return True

        return False

    async def backtest(self, start: datetime.date, end: datetime.date):
        return await super().backtest(start, end)

    async def scan(
        self,
        end: datetime.date = None,
        industry: List[str] = None,
        concepts: List[str] = None,
    ):
        end = end or arrow.now().date()
        codes = (
            await Security.select(end)
            .types(["stock"])
            .exclude_cyb()
            .exclude_kcb()
            .exclude_st()
            .eval()
        )

        if industry is not None:
            ib = IndustryBoard()
            ib.init()

            industry = ib.normalize_board_name(industry)
        else:
            ib = None

        if concepts is not None:
            cb = ConceptBoard()
            cb.init()
            concepts = cb.normalize_board_name(concepts)
        else:
            cb = None

        for code in codes:
            bars = await Stock.get_bars(code, 260, FrameType.DAY)
            fired, features = self.evaluate_long(code, bars)
            if not fired or (
                not self._belong_to_boards(code, ib, cb, industry, concepts)
            ):
                continue

            name = await Security.alias(code)
            dist, adv, gap, morph, nbv = features
            morph = describe_volume_morph(morph)

            cs = Candlestick(
                bars,
                title=f"{name} 底部跨度:{dist} 底部涨幅:{adv:.1%} 年线距离:{gap:.1%} 净买入量{nbv:.1f} {morph}",
            )
            cs.plot()
