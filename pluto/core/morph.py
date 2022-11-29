import numpy as np


def n_red_candlestick(bars, n):
    """n连阳"""
    assert len(bars) >= n
    bars = bars[-n:]

    return np.all(bars["close"] >= bars["open"])
