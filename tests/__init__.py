"""Unit test package for pluto."""
"""Data and utilities for testing."""
import logging
import os
from typing import List, Union

import cfg4py
import numpy as np
import pandas as pd
from coretypes import Frame, FrameType, bars_dtype

cfg = cfg4py.get_instance()
logger = logging.getLogger()


def init_test_env():
    os.environ[cfg4py.envar] = "DEV"
    src_dir = os.path.dirname(__file__)
    config_path = os.path.normpath(os.path.join(src_dir, "../pluto/config"))

    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)-1.1s %(name)s:%(funcName)s:%(lineno)s | %(message)s"
    formatter = logging.Formatter(fmt=fmt)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)

    return cfg4py.init(config_path, False)


def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


def load_bars_from_file(name, ext: str = "csv", sep="\t"):
    file = os.path.join(data_dir(), f"{name}.{ext}")
    df = pd.read_csv(file, sep=sep, parse_dates=True)

    df["frame"] = pd.to_datetime(df["frame"])
    df.set_index("frame", inplace=True)
    dtype = [
        ("frame", "<M8[ms]"),
        ("open", "f4"),
        ("high", "f4"),
        ("low", "f4"),
        ("close", "f4"),
        ("volume", "f8"),
        ("amount", "f8"),
        ("factor", "f4"),
    ]

    return np.array(df.to_records(), dtype=dtype)


def lines2bars(lines: List[str], start: Frame = None, end: Frame = None):
    """将CSV记录转换为Bar对象

    header: date,open,high,low,close,money,volume,factor
    lines: 2022-02-10 10:06:00,16.87,16.89,16.87,16.88,4105065.000000,243200.000000,121.719130

    """
    if isinstance(lines, str):
        lines = [lines]

    if lines[0].startswith("date,"):
        lines = lines[1:]

    from io import StringIO

    import pandas as pd

    buf = StringIO("\n".join(lines))
    df = pd.read_csv(buf, names=bars_dtype.names, parse_dates=["frame"])

    if start is not None:
        df = df[df["frame"] >= start]
    if end is not None:
        df = df[df["frame"] <= end]

    return df.to_records(index=False).astype(bars_dtype)


def read_csv(fname, start=None, end=None):
    """start, end是行计数，从1开始，以便于与编辑器展示的相一致。
    返回[start, end]之间的行
    """
    path = os.path.join(data_dir(), fname)
    with open(path, "r") as f:
        lines = f.readlines()

    if start is None:
        start = 1  # skip header
    else:
        start -= 1

    if end is None:
        end = len(lines)

    return lines[start:end]


def bars_from_csv(
    code: str, ft: Union[str, FrameType], start: Frame = None, end: Frame = None
):
    ft = FrameType(ft)

    fname = f"{code}.{ft.value}.csv"

    return lines2bars(read_csv(fname), start, end)
