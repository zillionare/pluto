from math import copysign

import numpy as np


def morning_star(bars):
    """潞安环能 10.27涨停"""
    close = bars["close"]
    opn = bars["open"]
    low = bars["low"]

    vector = []

    # 前阴后阳
    vector.append(1, close[-1] - opn[-1])
    vector.append(1, close[-3] - opn[-3])
    vector.append(close[-2] / opn[-2] - 1)

    # 底部位置关系
    vector.extend(np.sign(low[-2:] - low[-3:-1]))

    # 收盘价关系
    vector.extend(close[-2:] / close[-3:-1] - 1)

    # rsi与极低值差值及距离

    # 下影线长？
