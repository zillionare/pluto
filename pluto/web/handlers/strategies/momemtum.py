import arrow
from sanic import Blueprint, response
from sanic.exceptions import SanicException
import orjson
from functools import partial
import logging
from pluto.strategies.momentum import MomemtumStrategy

logger = logging.getLogger(__name__)
json_dumps = partial(orjson.dumps, option=orjson.OPT_SERIALIZE_NUMPY)
momemtum = Blueprint("momemtum", url_prefix="/momentum")


@momemtum.route("/", methods=["POST"])
async def handle_cmd(request):
    params = request.json

    cmd = params.get("cmd")
    commands = "add_short_sample"
    if not cmd in commands:
        raise SanicException(f"cmd must be one of {commands}, given {cmd}")

    if cmd == "add_short_sample":
        del params["cmd"]
        return await add_short_sample(**params)


async def add_short_sample(code: str, frame: str, label: str):
    try:
        frame = arrow.get(frame).naive
        label = int(label)
    except Exception as e:
        logger.exception(e)
        logger.warning("wrong params: %s %s %s", code, frame, label)
        raise SanicException(f"bad params: {code} {frame} {label}", status_code=401)
    mom = MomemtumStrategy()
    try:
        size = await mom.add_short_sample(code, frame, label)
        return response.json({
            "size": size
        })
    except Exception as e:
        logger.exception(e)
        raise SanicException(str(e))
