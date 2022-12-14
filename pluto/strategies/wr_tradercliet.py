import datetime
import logging as log

import cfg4py
import easytrader
import numpy as np
import pandas as pd
import talib as ta
from coretypes import FrameType
from omicron import tf
from omicron.extensions import price_equal
from omicron.models.board import Board, BoardType
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.notify import dingtalk
from traderclient import TraderClient
import os

Board.init("192.168.100.101")

cfg = cfg4py.init(os.expanduser("~/zillionare/config"))


async def W_R(bars, up_thresh=90):
    close = bars["close"].astype(np.float64)
    high = bars["high"].astype(np.float64)
    low = bars["low"].astype(np.float64)
    wr10 = 100 + (ta.WILLR(high, low, close, 7))
    rsi = ta.RSI(close, 6)
    index = np.arange(len(close))

    # 买点---------------------------------------------------------
    # 下降通过中，前面出现连续跌的情况， 去掉筛选之后下一个继续跌的
    new_buys = []
    after_returns = []
    for b in index:
        if b < index[-1]:
            before_buy = wr10[b - 4 : b]
            if (
                np.all(before_buy <= 35)
                and (35 < wr10[b])
                and (close[b + 1] >= close[b])
            ):
                new_buys.append(b)

                # 计算其后续收益
                if b + 9 <= len(close) - 1:
                    after_return = close[b + 9] / close[b + 1] - 1
                else:
                    after_return = close[-1] / close[b + 1] - 1
                after_returns.append(after_return)

    # 卖点------------------------------------------------------
    sell_flag = False
    before_sell = wr10[-5:-1]
    if (np.all(before_sell >= 80) and wr10[b] < 80) or (
        (wr10[-1] >= up_thresh) and (rsi[-1] >= 80)
    ):
        sell_flag = True

    return new_buys, after_returns, sell_flag, wr10


async def buy_wr(code: str, up_thresh: int = 90):
    now = datetime.datetime.now()
    end = tf.floor(now, FrameType.MIN30)
    # 除去11：25， 14：50买入的情况，这两个时间按照实时价格计算
    if (now.hour == 11 and now.minute > 20) or (now.hour == 14 and now.minute > 40):
        end = now

    bars = await Stock.get_bars(code, 120, FrameType.MIN30, end)
    if len(bars) < 120:
        return None

    new_buys, after_returns, _, wr10 = await W_R(bars, up_thresh)

    if len(new_buys) < 2:
        return None

    if wr10[-1] < 80:
        after_returns = np.array(after_returns)
        if (
            (new_buys[-1] == len(bars) - 2)
            and (np.all(after_returns[:-1] > 0.01))
            and (np.all(np.diff(new_buys) > 8))
        ):
            return -1, after_returns


async def sell_wr(code: str, up_thresh: int = 90):
    bars = await Stock.get_bars(code, 120, FrameType.MIN30)
    if len(bars) < 120:
        return None

    _, after_returns, sell_flag, _ = await W_R(bars, up_thresh)

    # 卖出点
    if sell_flag:
        return 1, after_returns

    return None


# 模拟盘
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


# 雪球组合
def account_xq():
    # 指定使用雪球
    user = easytrader.use("xq")

    # 初始化信息
    cookies = "Hm_lvt_1db88642e346389874251b5a1eded6e3=1671676265; device_id=90f8ad87786e14b5e0550130f5b04540; xq_a_token=1bb1c98e85f4e29dedf6a39bd73612422cb3339e; xqat=1bb1c98e85f4e29dedf6a39bd73612422cb3339e; xq_r_token=650a3b0633c69be757f0a9cf170845ebf2c499ff; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjcyOTcwNjg5MzcsImlzcyI6InVjIiwiZXhwIjoxNjc0MjAxNjI1LCJjdG0iOjE2NzE2NzYyOTAzNjgsImNpZCI6ImQ5ZDBuNEFadXAifQ.nBeQgTWx0a_QXsov11JPy4kfg3-IBKUXWd_jb9zJHQ6--e0QtnNkV9XkcGm7s5LLzj1_ecQtBRG5dcFcw5N1zrX0o7IUEYAQjI6jSLDL3BxAMkssNuARhH3WYKyTlKwfO_riAm-tdgQEESoaT6bXrkMEy2NvxT86PcCp5bjXP7lWpA1ifzSJBHOSFyol-TKm5ep4YQ3IQTnFVS84RJSUKC6UmV5NoROMP3aOUs6PX3Lufaj6k6WiGVpcv32vpQfjfnK90-7WzQfP1ZL2YTI-AqoqhZeg-ldcHp8tgpvhRsC1odbcYmS3Kzlud3JPwy9EVmCWZ704atCGwIsfGsBITA; u=7297068937; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1671679840; remember=1; xq_is_login=1; s=c7112ixy35; bid=822457e0c816e90c707221bbd69d3b4a_lbygwvam; __utma=1.1446526309.1671676310.1671676310.1671679840.2; __utmc=1; __utmz=1.1671676310.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); acw_tc=276077af16716782093591004e8ca079685368659aa6ebcd9fb638a52d5d84; __utmb=1.1.10.1671679840; __utmt=1"
    portfolio_code = "ZH3198089"

    user.prepare(cookies=cookies, portfolio_code=portfolio_code, portfolio_market="cn")

    return user


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

    log.info(f"筛选出来可买股票：{result_pd.values}")
    return result_pd


# 开盘时运行函数
async def market_buy(money_percent: float = 1):
    """跌停不买"""
    log.info("函数(market_buy)运行")
    note = {
        "title": " WR买入检测开始trader",
        "text": "#### WR低位买入检测开始",
    }

    await dingtalk.ding(note)

    client = account_init()
    user = account_xq()

    selected_code_pd = await scan_buy()
    if len(selected_code_pd) == 0:
        log.info("此时间点，没有满足条件股票池！")
        return None
    selected_code_pd = selected_code_pd.reset_index(drop=True)

    # 买卖
    codes = selected_code_pd["代码"].values.flatten().tolist()
    latest_prices = await Stock.get_latest_price(codes)
    ave_holds = (client.available_money * money_percent / sum(latest_prices)) // 100

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

            try:
                if code[0] == "6":
                    code_xq = "SH" + str(code[:6])
                    user.adjust_weight(
                        code_xq, (ave_holds * 100 * last_price) // 10000
                    ),
                else:
                    code_xq = "SZ" + str(code[:6])
                    user.adjust_weight(code_xq, (ave_holds * 100 * last_price) // 10000)
            except Exception as e:
                print(e)

            ave_holds = int(ave_holds)
            name = await Security.alias(code)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 发送钉钉消息
            msg = {
                "title": " WR买入信号trader",
                "text": f"#### WR买入股票：{code} {name}  \n  #### 触发时间：{now}  \n  #### 买入价：{last_price} \n  #### 成交数量：{ave_holds*100}/{ave_holds*100*last_price}¥",
            }

            await dingtalk.ding(msg)
            log.info(f"买入股票详情：{buy}")
            log.info("------------------------------------------------------")

    else:
        log.info("可用本金无法买入全部满足条件股票")
    return


async def market_sell(risk_limit: float = -0.03):
    """跌停不卖;在有持仓，且可卖的情况下，卖出，
    后续根据卖出情况，调整每次卖出数量，或者分批卖出
    """
    # log.info('函数运行时间(market_sell):'+ str(datetime.datetime.now()))
    client = account_init()
    user = account_xq()
    position = client.positions()

    # 空仓返回None
    if len(position) == 0:
        # log.info("空仓")
        return None
    available_share = position["sellable"]
    availbale_position = position[available_share > 0]

    # 没有可卖的，返回None
    if len(availbale_position) == 0:
        # log.info("持有股票均不可卖出！")
        return None

    # 有可卖股票
    for p in availbale_position:
        code = p["security"]
        alia = p["alias"]
        last_price = (await Stock.get_latest_price([code]))[0]
        cost_price = p["price"]

        # 跌停不卖
        sell_day = datetime.date.today()
        limit_prices = await Stock.get_trade_price_limits(code, sell_day, sell_day)
        low_limit = limit_prices["low_limit"][0]
        if price_equal(low_limit, last_price):
            log.info(f"{code} {alia}跌停，不进行交易")
            continue

        # 卖出条件
        sell_result = await sell_wr(code)

        # 满足风控, 或者满足条件，卖出
        if (last_price / cost_price - 1 < risk_limit) or (sell_result is not None):
            aval_share = p["sellable"]
            sell = client.market_sell(code, volume=aval_share)

            try:
                if code[0] == "6":
                    code_xq = "SH" + str(code[:6])
                    user.adjust_weight(code_xq, 0)
                else:
                    code_xq = "SZ" + str(code[:6])
                    user.adjust_weight(code_xq, 0)
            except Exception as e:
                print(e)

            aval_share = int(aval_share)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = {
                "title": " WR卖出信号trader",
                "text": f"#### WR高位卖出: {code} {alia}  \n  #### 触发时间：{now}  \n  #### 卖出价：{last_price} \n  #### 成交数量：{aval_share}/{aval_share*last_price}¥",
            }

            if last_price / cost_price - 1 < risk_limit:
                log.info(f"{code}{alia}触发风控卖出")
                msg = {
                    "title": " WR卖出信号trader",
                    "text": f"#### WR风控卖出: {code} {alia}  \n  #### 触发时间：{now}  \n  #### 卖出价：{last_price} \n  #### 成交数量：{aval_share}/{aval_share*last_price}¥",
                }

            await dingtalk.ding(msg)
            log.info(f"卖出股票详情：{sell}")
    return
