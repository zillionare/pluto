"""Console script for pluto."""

import datetime
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any

import arrow
import cfg4py
import fire
import httpx
from omicron import tf
from prettytable import PrettyTable

from pluto.store.buy_limit_pool import BuyLimitPoolStore
from pluto.store.long_parallel_pool import LongParallelPool
from pluto.store.steep_slopes_pool import SteepSlopesPool
from pluto.store.touch_buy_limit_pool import TouchBuyLimitPoolStore

logger = logging.getLogger(__name__)

cfg4py.init(os.path.expanduser("~/zillionare/pluto"))

pools = {
    "涨停": BuyLimitPoolStore(),
    "触及涨停": TouchBuyLimitPoolStore(),
    "上升均线": SteepSlopesPool(),
    "多头排列": LongParallelPool(),
}


def _day_closed(timestamp):
    now = datetime.datetime.now()
    if (
        tf.is_trade_day(timestamp)
        and timestamp == now.date()
        and datetime.datetime.now().hour < 15
    ):
        return tf.day_shift(timestamp, -1)
    else:
        return tf.day_shift(timestamp, 0)


def _parse_as_str_array(args: Any):
    if args is None:
        return None
    elif isinstance(args, str):
        arr = re.split(r"[,，]", args)
    elif hasattr(args, "__iter__"):
        arr = args
    elif isinstance(args, int):
        arr = [args]

    return [str(item) for item in arr]


def _save_proc_info(port, proc):
    path = os.path.expanduser("~/zillionare/pluto")
    file = os.path.join(path, "proc")
    with open(file, "w") as f:
        f.writelines(json.dumps({"port": port, "proc": proc}))


def _read_proc_info():
    path = os.path.expanduser("~/zillionare/pluto")
    file = os.path.join(path, "proc")
    try:
        with open(file, "r") as f:
            info = json.load(f)
            return info
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)

    return None


def _port():
    info = _read_proc_info()
    return info.get("port")


def is_service_alive(port: int = None) -> bool:
    if port is None:
        info = _read_proc_info()
        if info is None:
            raise ValueError("请指定端口")

        port = info["port"]

    try:
        resp = httpx.get(f"http://localhost:{port}/", trust_env=False)
    except httpx.NetworkError:
        return False

    return resp.status_code == 200


def status(port: int = None) -> bool:
    if not is_service_alive(port):
        print("------ pluto服务未运行 ------")
        return

    print("------ pluto服务正在运行 ------")

    x = PrettyTable()

    x.field_names = ["pool", "total", "latest"]
    for name, pool in pools.items():
        try:
            latest = sorted(pool.pooled)[-1]
        except Exception:
            x.add_row([name, "NA", "NA"])
            continue

        x.add_row([name, len(pool.pooled), latest])
    print(x)


def stop():
    info = _read_proc_info()
    if info is None:
        print("未发现正在运行的pluto服务")
        return

    proc = info["proc"]
    try:
        os.kill(proc, signal.SIGKILL)
    except ProcessLookupError:
        sys.exit()
    if not is_service_alive():
        print("pluto")
    else:
        print("停止pluto服务失败，请手工停止。")


def serve(port: int = 2712):
    if is_service_alive(port):
        print("pluto正在运行中，忽略此命令。")
        return

    proc = subprocess.Popen([sys.executable, "-m", "pluto", "serve", f"{port}"])

    for _ in range(30):
        if is_service_alive(port):
            _save_proc_info(port=port, proc=proc.pid)
            break
        else:
            time.sleep(1)


def pooling(pool: str, date: str = None):
    """启动`pool`（比如涨停池)的统计"""
    cmd = {"涨停": "blp", "触及涨停": "tblp", "上升均线": "ssp", "多头排列": "lpp"}.get(pool)

    if cmd is None:
        print("参数必须为(涨停，触及涨停，上升均线，多头排列)中的任一个。")
        return

    if not is_service_alive():
        print("服务未运行，或者配置端口错误")
        return
    port = _port()
    url = f"http://localhost:{port}/pluto/pools/pooling"
    rsp = httpx.post(url, json={"cmd": cmd, "end": date})
    if rsp.status_code == 200:
        print(f"统计{pool}的任务已创建！")


def show(pool_name: str, date: str = None):
    if pool_name not in pools:
        print(f"{pool_name}错误。支持的类型有{','.join(pools.keys())}")
        return

    pool = pools.get(pool_name)
    x = PrettyTable()
    x.field_names = []

    dt = arrow.get(date or arrow.now()).date()
    x.add_rows(pool.get(dt))
    print(x)


def strategy(name: str, **kwargs):
    if not is_service_alive():
        print("服务未运行，或者配置端口错误")
        return
    port = _port()
    url = f"http://localhost:{port}/pluto/strategies/{name}"
    rsp = httpx.post(url, json=kwargs, timeout=120)
    print(rsp.json())


def restart(port: int = 2712):
    stop()
    serve(port)


def main():
    fire.Fire(
        {
            "serve": serve,
            "status": status,
            "stop": stop,
            "restart": restart,
            "pooling": pooling,
            "show": show,
            "strategy": strategy,
        }
    )


if __name__ == "__main__":
    main()
