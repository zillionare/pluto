import datetime
import logging
from abc import ABCMeta, abstractmethod
from typing import Final, Optional, Union

from coretypes import Frame
from omicron import tf
from pyemit import emit
from traderclient import TraderClient

# 归类与回测相关的事件，比如回测开始、结束等
E_BACKTEST: Final = "BACKTEST"

# 归类与策略相关的事件，比如买入，卖出等。
E_STRATEGY: Final = "STRATEGY"

logger = logging.getLogger(__name__)


class BaseStrategy(object, metaclass=ABCMeta):
    """所有Strategy的基类。

    本基类实现了以下功能：
    1. 所有继承本基类的子类策略，都可以通过[alpha.strategies.get_all_strategies][] 获取到所有策略的列表，包括name和description。这些信息可用于在控制台上显示策略的列表。
    2. 策略实现`buy`， `sell`功能，调用结果会通过事件通知发送出来，以便回测框架、控制台来刷新和更新进度。
    3. 在实盘和回测之间无缝切换。

    本模块依赖于zillionare-backtest和zillionare-trader-client库。

    Args:

    Raises:
        TaskIsRunningError:
        NotImplementedError:
    """

    name = "base-strategy"
    alias = "base for all strategies"
    desc = "Base Strategy Class"
    version = "NA"

    def __init__(
        self, broker: Optional[TraderClient] = None, mdd: float = 0.1, sl: float = 0.05
    ):
        """

        Args:
            broker : 交易代理
            mdd : 止损前允许的最大回撤
            sl : 止损前允许的最大亏损
        """
        # 当前持仓 code -> {security: , shares: , sellable: sl: }
        self._positions = {}
        self._principal = 0
        self._broker = None

        # 用于策略止损的参数
        self.thresholds = {"mdd": mdd, "sl": sl}

        if broker:
            self.broker = broker
            cash = self.broker.available_money
            if cash is None:
                raise ValueError("Failed to get available money from server")

        self._bt = None

    @property
    def broker(self) -> TraderClient:
        """交易代理"""
        if self._bt is None:
            return self._broker
        else:
            return self._bt._broker

    @property
    def cash(self) -> float:
        """可用资金"""
        return self.broker.available_money

    @property
    def principal(self) -> float:
        """本金"""
        return self._principal if self._bt is None else self._bt._principal

    @property
    def positions(self):
        return self.broker.positions

    async def notify(self, event: str, msg: dict):
        """通知事件。

        在发送消息之前，总是添加账号信息，以便区分。

        Args:
            msg: dict
        """
        assert event in (
            "started",
            "progress",
            "failed",
            "finished",
        ), f"Unknown event: {event}, event must be one of ('started', 'progress', 'failed', 'finished')"

        msg.update(
            {
                "event": event,
                "account": self.broker._account,
                "token": self.broker._token,
            }
        )

        channel = E_BACKTEST if self._bt is not None else E_STRATEGY
        await emit.emit(channel, msg)

    async def update_progress(self, current_frame: Frame):
        """更新回测进度

        此函数只在日线级别上触发进度更新。

        Args:
            current_frame : 最新的回测时间
        """
        if self._bt is None:
            logger.warning("Backtest is not running, can't update progress")
            return

        last_frame = tf.day_shift(current_frame, -1)
        if self._bt._last_frame is None:
            self._bt._last_frame = last_frame
        else:
            msg = f"frame rewinded: {self._bt._last_frame} -> {last_frame}"
            assert last_frame >= self._bt._last_frame, msg
            self._bt._last_frame = last_frame

        info = self.broker.info()

        await self.notify(
            "progress",
            {
                "frame": last_frame,
                "info": info,
            },
        )

    async def buy(
        self,
        code: str,
        shares: int,
        order_time: Optional[Frame] = None,
        price: Optional[float] = None,
    ):
        """买入股票

        Args:
            code: 股票代码
            shares: 买入数量。应该为100的倍数。如果不为100的倍数，会被取整到100的倍数。
            price: 买入价格,如果为None，则以市价买入
            order_time: 下单时间,仅在回测时需要，实盘时，即使传入也会被忽略
        """
        logger.info("buy: %s %s %s", code, shares, order_time)
        if price is None:
            self.broker.market_buy(code, shares, order_time=order_time)
        else:
            self.broker.buy(code, price, shares, order_time=order_time)

    async def sell(
        self,
        code: str,
        shares: float,
        order_time: Optional[Frame] = None,
        price: Optional[float] = None,
    ):
        """卖出持仓股票

        Args:
            code : 卖出的证券代码
            shares : 如果在(0, 1]之间，则为卖出持仓的比例，或者股数。
            order_time : 委卖时间，在实盘时不必要传入
            price : 卖出价。如果为None，则为市价卖出
        """
        assert (
            shares >= 100 or 0 < shares <= 1
        ), f"shares should be in (0, 1] or multiple of 100, get {shares}"
        if self._bt is not None:
            broker = self._bt._broker
            if type(order_time) == datetime.date:
                order_time = tf.combine_time(order_time, 14, 56)
        else:
            broker = self.broker

        sellable = broker.available_shares(code)
        if sellable == 0:
            logger.warning("%s has no sellable shares", code)
            return

        if 0 < shares <= 1:
            volume = sellable * shares
        else:
            volume = min(sellable, shares)

        logger.info("sell: %s %s %s", code, volume, order_time)

        if price is None:
            broker.market_sell(code, volume, order_time=order_time)
        else:
            broker.sell(code, price, volume, order_time=order_time)

    @abstractmethod
    async def backtest(self, start: datetime.date, end: datetime.date):
        """调用子类的回测函数以启动回测

        Args:
            start: 回测起始时间
            end: 回测结束时间
        """
        raise NotImplementedError

    def check_required_params(self, params: Union[None, dict]):
        """检查策略参数是否完整

        一些策略在回测时往往需要传入特别的参数。派生类应该实现这个方法，以确保在回测启动前，参数都已经传入。
        Args:
            params: 策略参数
        """
        raise NotImplementedError
