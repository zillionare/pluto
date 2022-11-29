from functools import partial

import orjson
from omicron import tf
from sanic import Blueprint, response

json_dumps = partial(orjson.dumps, option=orjson.OPT_SERIALIZE_NUMPY)
calendar = Blueprint("calendar", url_prefix="/calendar")


@calendar.route("/days")
async def get_days(request):
    return response.json(tf.day_frames, dumps=json_dumps)
