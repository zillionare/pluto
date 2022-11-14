import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from coretypes import BarsArray, FrameType
from omicron.models.security import Security
from omicron.models.stock import Stock
from sklearn.metrics.pairwise import cosine_similarity

from pluto.core.metrics import vanilla_score
from pluto.strategies.base import BaseStrategy


class TheForceStrategy(BaseStrategy):
    # we mimic a neural network here, which eigens are `trained` weights, cosine_similarity is the function learned from training
    # todo: to mimic a full connected layer, let the result of function `predict` will fall into categories, instead of the raw similarity
    # by doing this (the mimic), we avoid building huge training dataset

    force_eigens = [
        [1, 1.4, 2]
        # negative eigen goes here
    ]
    returns_eigens = [
        [0, 0.02, 0.03]
        # negative eigen goes here
    ]

    def __init__(self, thresh: float = 0.95):
        """

        Args:
            thresh: 进行特征断言时需要满足的最小相似度阈值。
        """
        self.thresh = thresh

    def extract_features(self, bars: BarsArray) -> Tuple[np.array, np.array]:
        """从k线数据中提取本策略需要的特征"""
        if len(bars) < 4:
            raise ValueError("size of bars must be at least 4")

        bars = bars[-4:]
        vol = bars["volume"]
        close = bars["close"]

        vol = vol[1:] / vol[:-1]
        returns = close[1:] / close[:-1] - 1

        return np.vstack((vol, returns))

    def predict(self, bars: BarsArray) -> int:
        """

        Args:
            bars: 行情数据，不得小于4个bar。

        Returns:
            如果大于零，表明符合买入模式中的某一个。如果小于零，表明符合卖出中的某一个。等于零表明未响应特征。
        """
        vol_features, returns_features = self.extract_features(bars)

        # want row-wise sim result only
        sim_vol = cosine_similarity(self.force_eigens, vol_features)
        sim_returns = cosine_similarity(self.returns_eigens, returns_features)

        sim = np.hstack((sim_vol, sim_returns))

        # 判断是否某一行全部大于thresh
        mask = (sim >= self.thresh).all(axis=1)
        return np.any(mask)

    async def backtest(self, start: datetime.date, end: datetime.date):
        """回测"""
        codes = (
            await Security.select()
            .types(["stock"])
            .exclude_st()
            .exclude_cyb()
            .exclude_kcb()
            .eval()
        )
        results = []
        for code in codes:
            bars = await Stock.get_bars_in_range(code, FrameType.DAY, start, end)
            name = await Security.alias(code)
            for i in range(4, len(bars) - 3):
                xbars = bars[:i]
                fired = self.predict(xbars)

                if fired:
                    ybars = bars[i - 1 : i + 3]
                    #                 close = ybars["close"]

                    #                 t0 = ybars["frame"][0].item().date()
                    #                 c0 = round(ybars["close"][0].item(), 2)

                    _, max_returns, mdds = await vanilla_score(ybars, code)

                    if len(max_returns):
                        if len(mdds):
                            results.append(
                                (
                                    name,
                                    ybars[0]["frame"].item().date(),
                                    max_returns[0],
                                    mdds[0],
                                )
                            )
                        else:
                            results.append(
                                (
                                    name,
                                    ybars[0]["frame"].item().date(),
                                    max_returns[0],
                                    None,
                                )
                            )

        return pd.DataFrame(results, columns=["name", "frame", "max_return", "mdd"])

    async def scan(self):
        """以最新行情扫描市场，以期发现投资机会"""
        codes = (
            await Security.select()
            .types(["stock"])
            .exclude_st()
            .exclude_cyb()
            .exclude_kcb()
            .eval()
        )
        results = []
        for code in codes:
            bars = await Stock.get_bars(code, 4, FrameType.DAY)
            if len(bars) < 4:
                continue
            signal = self.predict(bars)
            if signal > 0:
                name = await Security.alias(code)
                results.append((name, code, signal))

        return results
