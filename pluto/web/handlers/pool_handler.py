import arrow
import numpy as np
from omicron.models.security import Security
from sanic import Blueprint, response
from sanic.exceptions import SanicException
import asyncio

from pluto.store.buy_limit_pool import BuyLimitPoolStore
from pluto.store.long_parallel_pool import LongParallelPool
from pluto.store.steep_slopes_pool import SteepSlopesPool
from pluto.store.touch_buy_limit_pool import TouchBuyLimitPoolStore

pools = Blueprint("pools", url_prefix="/pools")


@pools.route("/pooling", methods=["POST"])
async def pooling(request):
    params = request.json
    cmd = params.get("cmd")

    end = params.get("end")

    if cmd is None or cmd not in ("blp", "tblp", "ssp", "lpp"):
        msg = "必须提供命令参数: blp, tblp, ssp, lpp"
        raise SanicException(msg, status_code=401)

    if end is not None:
        end = arrow.get(end).date()
    if cmd == "blp":
        pool = BuyLimitPoolStore()
    elif cmd == "tblp":
        pool = TouchBuyLimitPoolStore()
    elif cmd == "ssp":
        pool = SteepSlopesPool()
    elif cmd == "lpp":
        pool = LongParallelPool()

    asyncio.create_task(pool.pooling(end))
    return response.text(f"task 'pooling {cmd}' is scheduled and running")


@pools.route("/buylimit/find_all")
async def buylimit_find_all(request):
    params = request.json

    start = arrow.get(params.get("start")).date()
    end = arrow.get(params.get("end")).date()

    total_min = params.get("total_min", 1)
    total_max = params.get("total_max", 10)

    continuous_min = params.get("continuous_min", 1)
    continuous_max = params.get("continuous_max", 3)

    till_now = params.get("till_now", 10)


@pools.route("/steep_slopes_pool")
async def steep_slopes(request):
    params = request.args

    win = params.get("win")
    if win is None:
        raise SanicException("必须指定均线'win'")

    dt = params.get("dt")
    if dt is not None:
        dt = arrow.get(dt).date()
    pool = SteepSlopesPool()
    records = pool.get(dt=dt, win=int(win))
    if records is None:
        return response.json([])

    names = [await Security.alias(code) for code in records[:]["code"]]

    results = np.empty(
        shape=(len(records[:]),),
        dtype=[("name", "U16"), ("code", "U16"), ("slp", "U16")],
    )
    results["name"] = names
    results["code"] = records[:]["code"]
    results["slp"] = [f"{slp:.2%}" for slp in records[:]["slp"]]
    # serialized = orjson.dumps(results[:].tolist(),option=orjson.OPT_SERIALIZE_NUMPY)
    return response.json(results[:].tolist())
