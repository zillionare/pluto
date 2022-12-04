from functools import partial

import arrow
import orjson
from omicron import tf
from sanic import Blueprint, response
from sanic.exceptions import SanicException

from pluto.store.buy_limit_pool import BuyLimitPoolStore

json_dumps = partial(orjson.dumps, option=orjson.OPT_SERIALIZE_NUMPY)
command = Blueprint("command", url_prefix="/command")


@command.route("/", methods=["POST"])
async def handle_command(request):
    body = request.body.decode("utf-8")

    params = {}
    for line in body.split("\n"):
        key, value = line.split(":")
        params[key] = value

    cmd = params.get("cmd")
    if cmd is None:
        raise SanicException("bad params, 'cmd' not found")

    del params["cmd"]
    if cmd == "buylimit":
        return handle_blp_query(**params)


def handle_blp_query(start: str, end: str = None):
    blp = BuyLimitPoolStore()

    try:
        start = arrow.get(start).date()
        if end is not None:
            end = arrow.get(end).date()

        result = blp.find_all(start, end)
        cols = result.dtype.names
        resp = {
            "data": result.tolist(),
            "cols": [{"title": v, "targets": i} for i,v in enumerate(cols)]
        }
        return response.json(body=resp, dumps=json_dumps)
    except KeyError:
        raise SanicException("日期错误，或者未进行pooling")
