from typing import Tuple

import numpy as np
from omicron.extensions import array_math_round
from omicron.talib import moving_average, polyfit


def reverse_ma(ma, close):
    return len(close) * ma - sum(close[1:])


def predict_next_price(bars, win=10) -> Tuple[float, float]:
    """通过均线预测下一个bar的收盘价
    
        Returns:
            预测的下一个收盘价和均线值。如果无法预测，返回None,None
    """
    close = bars["close"]

    if len(close) < win + 6:
        return None, None

    # make moving_average faster by cutting off unnecessary close
    close = close[-win - 6 :]
    ma = array_math_round(moving_average(close, win)[-7:], 2)
    err, (a, b, c), _ = polyfit(ma / ma[0], deg=2)

    if err < 3e-3:
        f = np.poly1d((a, b, c))
        ma_hat = f(np.arange(8))[-1] * ma[0]
        return reverse_ma(ma_hat, close[-win:]), ma_hat.item()

    return None, None
