import asyncio
import datetime
import logging as log
from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import cfg4py
import numpy as np
import omicron
import pandas as pd
import plotly.graph_objects as go
import talib as ta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from coretypes import BarsArray, FrameType, bars_dtype
from IPython.display import display
from numpy.random import choice
from numpy.typing import NDArray
from omicron import tf
from omicron.extensions import find_runs, price_equal
from omicron.models.board import Board, BoardType
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.notify import dingtalk
from omicron.talib import moving_average, peaks_and_valleys, rsi_watermarks
from plotly.subplots import make_subplots
from traderclient import TraderClient

pd.options.display.max_rows = None
pd.options.display.max_columns = None
Board.init("192.168.100.101")

cfg = cfg4py.init("/home/belva/zillionare/config")


async def W_R(bars, up_thresh=90):
    """
    30分钟线：WR1<= WR2<20, 且发生拐头，超买信号；
    反之，WR1>= WR2>=90, 且发生拐头后，拐头后两个均小于90大于20，超卖信号。

    找出传入行情数据的超买超卖的index，-1代表超卖即底部，1代表超买即顶部。
    0代表什么都不是
    """

    close = bars["close"].astype(np.float64)
    high = bars["high"].astype(np.float64)
    low = bars["low"].astype(np.float64)
    wr10 = 100 + (ta.WILLR(high, low, close, 7))
    index = np.arange(len(close))

    # 买点---------------------------------------------------------
    # 下降通过中，前面出现连续跌的情况
    new_buys = []
    after_returns = []
    for b in index:
        before_buy = wr10[b - 4 : b]
        if (
            np.all(before_buy <= 20) and 20 < wr10[b]
        ):  # and np.count_nonzero(before_buy<=down_thresh)>0
            new_buys.append(b)

            # 计算其后续收益
            if b + 8 < len(close) - 1:
                after_return = close[b + 8] / close[b] - 1
            else:
                after_return = close[-1] / close[b] - 1
            after_returns.append(after_return)

    # 卖点，实时卖出，不用high替换close
    new_sells = []
    for b in index:
        before_sell = wr10[b - 4 : b]
        if (
            np.all(before_sell >= 80) and wr10[b] < 80
        ):  # and np.count_nonzero(before_buy<=down_thresh)>0
            new_sells.append(b)

    # 高RSI卖点
    isell_hrsi = []
    ihigh_wr10 = np.where(wr10 >= up_thresh)[0]
    rsi = ta.RSI(close, 6)
    ih_rsi = np.where(rsi >= 80)[0]
    isell_hrsi = np.intersect1d(ihigh_wr10, ih_rsi)
    iall_sells = []
    iall_sells = np.where(wr10 >= 95)[0]

    return new_buys, after_returns, new_sells, isell_hrsi, iall_sells, wr10


async def plot_wr(code: str):
    """将卖点标记点在图上标记出来
    由convex_signal引起的卖点触发，绘制出卖出点的两点连线；
    由穹顶压力引起，绘制出相应均线的穹顶图形；
    由八周期内高RSI的长上影引起，标记出长上影所在点

    Args:
    code: 股票代码
    end: 传入行情数据的最后一个时间点
    ex_info： 是否需要额外信息
    upper_limit: 上影相对于实体最高部分的长度比
    """

    name = await Security.alias(code)
    bars = await Stock.get_bars(code, 120, FrameType.MIN30)
    new_buys, _, new_sells, isell_hrsi, iall_sells, wr10 = await W_R(bars)
    frame = bars["frame"]
    close = bars["close"]
    close = close.astype(np.float64)
    high = bars["high"].astype(np.float64)
    low = bars["low"].astype(np.float64)
    index = np.arange(len(bars))
    MA5 = ta.MA(close, 5)

    # rsi = ta.RSI(close, 6)
    fig = make_subplots(
        rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True, shared_yaxes=False
    )
    fig.add_trace(
        go.Candlestick(
            x=index,
            close=close,
            high=high,
            open=bars["open"],
            low=bars["low"],
            increasing=dict(line=dict(color="red")),
            decreasing=dict(line=dict(color="green")),
            name="K线",
            text=frame,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=index, y=MA5, mode="lines", name="MA5", text=bars["frame"]),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=index, y=wr10, mode="lines", name="WR10", text=bars["frame"]),
        row=2,
        col=1,
    )

    fig.add_hline(y=95, row=2, col=1)
    fig.add_hline(y=20, row=2, col=1)
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=index[-1],
        y0=0,
        x1=index[-1],
        y1=1,
        line=dict(
            color="grey",
            width=2,
        ),
    )

    if len(new_buys) > 0:
        fig.add_trace(
            go.Scatter(
                x=index[new_buys],
                y=low[new_buys] * 0.985,
                mode="markers",
                marker_symbol="triangle-up",
                marker_line_color="midnightblue",
                marker_color="yellow",
                marker_line_width=1,
                marker_size=10,
                name="前面连续跌买点",
                text=frame[new_buys],
            ),
            row=1,
            col=1,
        )

    if len(new_sells) > 0:
        fig.add_trace(
            go.Scatter(
                x=index[new_sells],
                y=high[new_sells] * 1.01,
                mode="markers",
                marker_symbol="triangle-down",
                marker_line_color="midnightblue",
                marker_color="pink",
                marker_line_width=1,
                marker_size=9,
                name="连涨卖点",
                text=frame[new_sells],
            ),
            row=1,
            col=1,
        )

    if len(isell_hrsi) > 0:
        fig.add_trace(
            go.Scatter(
                x=index[isell_hrsi],
                y=high[isell_hrsi] * 1.015,
                mode="markers",
                marker_symbol="triangle-down",
                marker_line_color="midnightblue",
                marker_color="green",
                marker_line_width=1,
                marker_size=9,
                name="rsi>80卖点",
                text=frame[isell_hrsi],
            ),
            row=1,
            col=1,
        )

    if len(iall_sells) > 0:
        fig.add_trace(
            go.Scatter(
                x=index[iall_sells],
                y=high[iall_sells] * 1.017,
                mode="markers",
                marker_symbol="triangle-down",
                marker_line_color="midnightblue",
                marker_color="black",
                marker_line_width=1,
                marker_size=9,
                name="所有卖点",
                text=frame[iall_sells],
            ),
            row=1,
            col=1,
        )

    fig.update_layout(title=(f"{code}:{name}触发低位WR买点"), width=1000, height=600)
    fig.update_yaxes(dict(domain=[0.4, 1]), row=1, col=1)
    fig.update_yaxes(dict(domain=[0.0, 0.4]), row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(showspikes=True, spikethickness=2)
    fig.update_yaxes(showspikes=True, spikethickness=2)
    fig.show()
    return


async def buy_wr(code: str, up_thresh: int = 90):
    bars = await Stock.get_bars(code, 120, FrameType.MIN30)
    if len(bars) < 120:
        return None

    new_buys, after_returns, new_sells, isell_hrsi, iall_sells, wr10 = await W_R(
        bars, up_thresh
    )

    # 只算前面连跌买点, 规定检测点最后一个的下一个wr值要大于前一个, 买卖点不能重合
    iaall_sells = sorted(new_sells + list(isell_hrsi) + list(iall_sells))
    iaall_sells = np.array(iaall_sells)

    if (
        (len(new_buys) > 2)
        and (np.all(iaall_sells != len(bars) - 2))
        and (np.all(iaall_sells != len(bars) - 1))
    ):
        after_returns = np.array(after_returns)
        if (
            (new_buys[-1] == len(bars) - 2)
            and (np.all(after_returns[:-1] > 0))
            and (np.any(after_returns[:-1] > 0.01))
            and (np.all(np.diff(new_buys) > 8))
            and (wr10[-1] >= wr10[-2])
        ):
            return -1, after_returns


async def sell_wr(code: str, up_thresh: int = 90):
    bars = await Stock.get_bars(code, 120, FrameType.MIN30)
    if len(bars) < 120:
        return None

    _, after_returns, new_sells, isell_hrsi, iall_sells, _ = await W_R(bars, up_thresh)

    # 卖出点
    if len(isell_hrsi) > 0:
        if isell_hrsi[-1] == len(bars) - 1:
            return 1, after_returns
    elif len(new_sells) > 0:
        if new_sells[-1] == len(bars) - 1:
            return 1, after_returns
    elif len(iall_sells) > 0:
        if iall_sells[-1] == len(bars) - 1:
            return 1, after_returns

    return None


# 模拟盘-----------------------------------
# 只出现所有卖点iall_sells，首次只卖一半，如果下一个close更低，则全部卖出，要不就等下一个卖出信号；遇到高位RSI的WR，连涨WR全部卖出。
def account_init():
    url = "http://192.168.100.130:8000/api/trade/v0.1"
    acct = "gw_anzhi"
    token = "25f050e5-95c6-4e26-8cb3-145e7636eb4d"
    client = TraderClient(url, acct, token)

    try:
        result = client.info()
        if result is None:
            log.error("failed to get information")
            return None
        return client
    except Exception as e:
        print(e)
        return False


# 特定时间筛选出来，满足买入条件的股票，返回股票详情
async def scan_buy():
    result_pd = pd.DataFrame()
    codes = (
        await Security.select()
        .types(["stock"])
        .exclude_cyb()
        .exclude_kcb()
        .exclude_st()
        .eval()
    )
    num = 0
    total_num = len(codes)
    for code in codes:
        if num % 100 == 0:
            log.info(f"遍历第{num}/{total_num}支股票")
        num += 1

        result = await buy_wr(code)
        if result is None:
            continue
        if result[0] == -1:
            name = await Security.alias(code)
            log.info(f"{name}{code}触发买点。")
            log.info(f"历史买点八个时间单位后收益{np.around(result[1], 4)}")
            board_name = await Board.board_info_by_security(
                code, _btype=BoardType.INDUSTRY
            )
            name = await Security.alias(code)
            now = datetime.datetime.now()
            code_info = pd.DataFrame(
                [(code, name, now, board_name)],
                columns=["代码,名称,检测买入时间,所属板块".split(",")],
            )
            result_pd = pd.concat([result_pd, code_info])
            # plot = await plot_wr(code)

    log.info("筛选出来可买股票：")
    display(result_pd)
    return result_pd


# 开盘时运行函数
async def market_buy():
    """跌停不买"""
    log.info("函数(market_buy)运行")
    client = account_init()

    selected_code_pd = await scan_buy()
    if len(selected_code_pd) == 0:
        log.info("此时间点，没有满足条件股票池！")
        return None
    selected_code_pd = selected_code_pd.reset_index(drop=True)

    # 买卖
    codes = selected_code_pd["代码"].values.flatten().tolist()
    latest_prices = await Stock.get_latest_price(codes)
    ave_holds = (client.available_money / sum(latest_prices)) // 100

    # 本金足够可以全买, 不够全买则不买
    if ave_holds > 0:
        for code, last_price in zip(codes, latest_prices):
            # 判断买入点是否涨停
            buy_day = datetime.date.today()
            limit_prices = await Stock.get_trade_price_limits(code, buy_day, buy_day)
            high_limit = limit_prices["high_limit"][0]
            low_limit = limit_prices["low_limit"][0]
            if price_equal(high_limit, last_price) or price_equal(
                low_limit, last_price
            ):
                log.info(f"{code} 出现涨跌停，无法买入！")
                continue
            # 未涨停的全部买入
            buy = client.market_buy(code, volume=ave_holds * 100)
            name = await Security.alias(code)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 发送钉钉消息
            msg = {
                "title": " 买入信号trader",
                "text": f"#### 买入股票：{code} {name}  \n  #### 触发时间：{now}  \n  #### 买入价：{last_price} \n  #### 成交数量：{ave_holds*100}/{ave_holds*100*last_price}￥",
            }

            await dingtalk.ding(msg)
            buy_info = pd.DataFrame.from_dict(buy, orient="index").T
            log.info("买入股票详情：")
            display(buy_info)
            log.info("------------------------------------------------------")

    else:
        log.info("可用本金无法买入全部满足条件股票")
    return


async def market_sell(rsik_limit: float = -0.03):
    """跌停不卖;在有持仓，且可卖的情况下，卖出，
    后续根据卖出情况，调整每次卖出数量，或者分批卖出
    """
    # log.info('函数运行时间(market_sell):'+ str(datetime.datetime.now()))
    client = account_init()
    position = client.positions()

    # 空仓返回None
    if len(position) == 0:
        log.info("空仓")
        return None
    available_share = position["sellable"]
    availbale_position = position[available_share > 0]

    # 没有可卖的，返回None
    if len(availbale_position) == 0:
        log.info("持有股票均不可卖出！")
        return None

    # 有可卖股票
    for p in availbale_position:
        code = p["security"]
        alia = p["alias"]
        last_price = await Stock.get_latest_price([code])
        cost_price = p["price"]

        # 跌停不卖
        sell_day = datetime.date.today()
        limit_prices = await Stock.get_trade_price_limits(code, sell_day, sell_day)
        low_limit = limit_prices["low_limit"][0]
        if price_equal(low_limit, last_price[0]):
            log.info(f"{code}{alia}跌停，不进行交易")
            continue

        # 卖出条件
        sell_result = await sell_wr(code)

        # 满足风控, 或者满足条件，卖出
        if (last_price / cost_price - 1 < rsik_limit) or (sell_result is not None):
            aval_share = p["sellable"]
            sell = client.market_sell(code, volume=aval_share)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = {
                "title": " 卖出信号trader",
                "text": f"### WR高位卖出：{code} {alia}  \n  ### 触发时间：{now}  \n  ### 卖出价：{last_price} \n  ### 成交数量：{aval_share}/{aval_share*last_price}￥",
            }

            if last_price / cost_price - 1 < rsik_limit:
                log.info(f"{code}{alia}触发风控卖出")
                msg = {
                    "title": " 卖出信号trader",
                    "text": f"### 风控卖出：{code} {alia}  \n  ### 触发时间：{now}  \n  ### 卖出价：{last_price} \n  ### 成交数量：{aval_share}/{aval_share*last_price}￥",
                }

            await dingtalk.ding(msg)
            log.info(f"卖出股票详情：{sell}")
            log.info("卖出后持仓: ")
            display(pd.DataFrame(client.positions()))
            await plot_wr(code)
    return


# # 添加运行任务
# # await main(scheduler)
# asyncio.run(main(scheduler))


# # await market_sell()

# # 打印任务列表
# # scheduler.print_jobs()
# # 任务开始
# # scheduler.start()

# # # 暂停任务
# # scheduler.pause()
# # # 删除所有任务
# # scheduler.remove_all_jobs()
# # scheduler.shutdown(wait=False)
