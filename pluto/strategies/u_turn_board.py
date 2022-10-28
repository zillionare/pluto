import datetime
import logging as log
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from boards.board import IndustryBoard
from coretypes import BarsArray, FrameType
from omicron import tf
from omicron.models.security import Security
from omicron.models.stock import Stock
from zigzag import peak_valley_pivots

IndustryBoard.init()
ib = IndustryBoard()


class TurnaroundStrategy(object):
    """检测除科创板，除创业板股票中包含发生底部反转股票最多的板块"""

    def evaluate_long(
        self,
        bars: BarsArray,
        period_limit: float,
        distance_limit: int,
        thresh: Tuple[float, float] = None,
    ) -> int:
        """底部反转条件设置

        条件与方法：
            1. 使用peaks_and_valleys，检出最末一段标志为-1，1（默认使用自适应算法）
            2. 上述最末段的涨幅大于period_limit%，最低点距今不超过distance_limit天

            上述数字部分可以通过参数设置。

        Args:
            bars: 包含最后底部信号发出日的行情数据
            period_limit: 底部信号发出日到最后一天的涨幅限制，即不大于period_limit%
            distance_limit: 底部信号发出日到最后一天的距离限制，即不大于distance_limit个单位距离
            thresh: 反转参数，默认为两个涨跌幅标准差。

        Returns:
            返回满足条件的底部信号发出日到最后一天的实际距离，如果不满足所设条件，返回None。

        """

        assert len(bars) > 59, "must provide an array with at least 60 length!"
        close = bars["close"].astype(np.float64)
        if thresh is None:
            std = np.std(close[-59:] / close[-60:-1] - 1)
            thresh = (2 * std, -2 * std)

        pivots = peak_valley_pivots(close, thresh[0], thresh[1])
        flags = pivots[pivots != 0]
        period_increase = None
        lowest_distance = None
        distance = None
        if (flags[-2] == -1) and (flags[-1] == 1):
            length = len(pivots)
            last_valley_index = np.where(pivots == -1)[0][-1]
            period_increase = (close[-1] - close[last_valley_index]) / close[
                last_valley_index
            ]
            lowest_distance = length - 1 - last_valley_index
            if (
                (period_increase >= period_limit * 0.01)
                and (lowest_distance <= distance_limit)
                and (lowest_distance > 0)
            ):
                distance = lowest_distance
        return distance

    async def scan(
        self,
        codes: List[str],
        dt: datetime.date,
        period_limit: float,
        distance_limit: int,
        thresh: Tuple,
    ) -> List[str]:
        """遍历`dt`日codes中指定的股票，并调用evaluate_long找出发出买入信号的股票代码

        Args:
            codes: 股票代码列表
            dt: 指定日期
            period_limit: 底部信号发出日到最后一天的涨幅限制，即不大于period_limit%
            distance_limit: 底部信号发出日到最后一天的距离限制，即不大于distance_limit个单位距离
            thresh: 反转参数，默认为两个涨跌幅标准差。

        Returns:
            返回发出底部反转信号的股票列表
        """
        signal_codes = []
        num = 0
        for code in codes:
            if num % 100 == 0:
                log.info(f"遍历第{num}只股票")
            num += 1
            bar = await Stock.get_bars_in_range(
                code, FrameType.DAY, start=tf.day_shift(dt, -59), end=dt
            )
            if len(bar) < 60:
                continue
            distance = self.evaluate_long(bar, period_limit, distance_limit, thresh)
            if distance is not None:
                signal_codes.append(code[:6])
                log.info(f"满足条件的股票：{code}: {await Security.alias(code)}")
        return signal_codes

    async def score(
        self,
        codes: List[str],
        signal_date: datetime.date,
        period_limit: float,
        distance_limit: int,
        thresh: Tuple,
    ) -> List[list]:
        """以signal_date当日收盘价买入，次日收盘价卖出的收益。
            如果买入当天股票涨停，则无法买入。

        Args:
            codes: 发出底部反转信号的股票列表
            signal_date: 买入日期
            period_limit: 底部信号发出日到最后一天的涨幅限制，即不大于period_limit%
            distance_limit: 底部信号发出日到最后一天的距离限制，即不大于distance_limit个单位距离
            thresh: 反转参数，默认为两个涨跌幅标准差。

        Returns:
            returns: 返回包含每个发出信号股票买入日期，股票名称，代码，收益率%，距离，所属板块的列

        """
        returns = []
        for code in codes:
            code = tuple(Stock.fuzzy_match(code).keys())
            if len(code) < 1:
                continue
            code = code[0]
            bar = await Stock.get_bars_in_range(
                code,
                FrameType.DAY,
                start=tf.day_shift(signal_date, -59),
                end=tf.day_shift(signal_date, 1),
            )
            if len(bar) < 61:
                continue
            # 判断当日是否涨停，涨停则无法买入
            limit_flag = (
                await Stock.trade_price_limit_flags(code, signal_date, signal_date)
            )[0][0]
            if not limit_flag:
                return_ = (bar["close"][-1] - bar["close"][-2]) / bar["close"][-2]
                distance = self.evaluate_long(
                    bar[:-1], period_limit, distance_limit, thresh
                )
                board_names = []
                for board_code in ib.get_boards(code[:6]):
                    board_names.append((board_code, ib.get_name(board_code)))
                name = await Security.alias(code)
                returns.append(
                    [signal_date, name, code, return_ * 100, distance, board_names]
                )
        return returns

    async def backtest(
        self,
        start: datetime.date,
        end: datetime.date,
        period_limit: float,
        distance_limit: int,
        thresh: Tuple,
    ) -> pd.DataFrame:
        """在[start, end]区间对除科创板，除创业板有底部反转的所有股票筛选，
        选出出现频率最高的行业板块，买入此板块下筛选出来的股票，并计算次日收益

        Args:
            start: 筛选底部反转板块起始时间
            end: 筛选底部反转终止时间
            period_limit: 底部信号发出日到最后一天的涨幅限制，即不大于period_limit%
            distance_limit: 底部信号发出日到最后一天的距离限制，即不大于distance_limit个单位距离
            thresh: 反转参数，默认为两个涨跌幅标准差。

        Returns:
            返回包含每个发出信号股票买入日期，股票名称，代码，未涨停的此日收益率%，到反转的距离，所属板块的表格

        """
        results = []
        for frame in tf.get_frames(start, end, FrameType.DAY):
            frame = tf.int2date(frame)
            log.info(f"遍历时间：{frame}")
            codes = (
                await Security.select(frame)
                .types(["stock"])
                .exclude_st()
                .exclude_kcb()
                .exclude_cyb()
                .eval()
            )
            fired = await self.scan(codes, frame, period_limit, distance_limit, thresh)
            belong_boards = defaultdict(int)
            for scaned_code in fired:
                for board in ib.get_boards(scaned_code):
                    belong_boards[board] += 1
            if len(belong_boards) < 1:
                continue
            sort_pd = pd.DataFrame(belong_boards.items()).sort_values(
                by=1, ascending=False, ignore_index=True
            )
            most_boards = sort_pd[sort_pd[1] == sort_pd[1][0]][0].values
            selected_codes = np.array([])
            for most_board in most_boards:
                log.info(f"出现最多的板块：{most_board}: {ib.get_name(most_board)}")
                board_members = ib.get_members(most_board)
                # 买入选出出现最多板块的成分股
                selected_code = np.intersect1d(board_members, fired)
                selected_codes = np.append(selected_codes, selected_code)
            # 去除不同板块的重复股票
            selected_codes = np.unique(selected_codes)
            log.info(
                f"板块中符合条件的股票：{tuple(list((Stock.fuzzy_match(x)).items())[0][1][:2] for x in selected_codes)}"
            )
            # 计算次日收益率
            result = await self.score(
                selected_codes, frame, period_limit, distance_limit, thresh
            )
            results += result
        results = pd.DataFrame(results, columns="日期,名称,代码,收益率%,距离,板块".split(","))
        return results
