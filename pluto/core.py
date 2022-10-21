import datetime
import logging
from typing import Optional, Tuple

import numpy as np
from coretypes import Frame, bars_dtype
from omicron import tf
from omicron.extensions import find_runs
from omicron.models.security import Security
from omicron.models.stock import Stock

logger = logging.getLogger(__name__)


async def vanilla_score(bars: bars_dtype, code: str = None) -> Tuple:
    """对买入信号发出之后一段时间的表现进行评价。

    规则：
        1. bars中的第一根bar为信号发出日(T0)。如果此时未涨停，则以收盘价作为买入价
        2. 如果第一天收盘已涨停，则使用第二天的开盘价作为买入价
        3. 分别计算T1, T2, ..., T(len(bars)-1)日的累计涨跌幅
        4. 计算累计最大涨幅
        5. 计算出现累计最大跌幅（上涨之前）
        6. 计算中除情况2之外，都使用收盘价计算。
        7. 时间单位为日，分钟单位数据不可计算。

    Args:
        bars: 包含信号发出日的行情数据
        code: 股票代码

    Returns:
        包含每日累计涨跌幅，最大涨幅和最大跌幅的元组。
    """
    assert len(bars)>=2, "must provide an array with at least 2 length!"

    returns=[]
    max_returns=[]
    mdds=[]

    limit_flag = (await Stock.trade_price_limit_flags(code, bars['frame'][0].item(), bars['frame'][0].item()))[0][0]
    # 如果检测当天涨停，第二天开盘价未涨停买入，第二天开始收盘价作为收益。
    if (limit_flag
        &((bars['open'][1]-bars['close'][0])/bars['close'][0]<0.099)):
        price_np = np.append(bars['open'][1],bars['close'][1:])
        returns = (price_np[1:]-price_np[0])/price_np[0]
        max_return = np.nanmax(returns)
        max_returns.append(max_return)
        max_index = np.argmax(returns)
        # 防止涨停之前的最大跌幅为空值，取到最大值
        to_max = returns[:max_index+1]
        mdd = np.nanmin(to_max)
        if (mdd<0): 
            mdds.append(mdd)
        
    
    # 如果检测当天可以买进，则直接买入，后五天的收盘价作为收益，开盘涨停则不考虑
    elif not limit_flag:
        returns = (bars['close'][1:]-bars['close'][0])/bars['close'][0]
        max_return = np.nanmax(returns)
        max_returns.append(max_return)
        max_index = np.argmax(returns)
        # 防止涨停之前的最大跌幅为空值，取到最大值
        to_max = returns[:max_index+1]
        mdd = np.nanmin(to_max)
        if (mdd<0): 
            mdds.append(mdd)
    return returns, max_returns, mdds

