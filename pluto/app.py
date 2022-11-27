"""Main module."""
"""Main module."""
import logging
import os

import cfg4py
import omicron
import pkg_resources
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sanic import Sanic

from pluto.store.buy_limit_pool import pooling_latest
from pluto.store.long_parallel_pool import LongParallelPool
from pluto.store.steep_slopes_pool import SteepSlopesPool
from pluto.store.touch_buy_limit_pool import TouchBuyLimitPoolStore
from pluto.web.blueprints import bp

application = Sanic("pluto")

logger = logging.getLogger(__name__)
ver = pkg_resources.get_distribution("zillionare-pluto").parsed_version


# @application.route('/')
# async def index(request):
#     return response.json({
#         "greetings": "welcome to zillionare-pluto",
#         "version": str(ver)
#     })


def serve_static_files(app):
    # set static path
    app_dir = os.path.dirname(__file__)
    app.static("/", os.path.join(app_dir, "web/static/index.html"))
    app.static("dist", os.path.join(app_dir, "web/static/dist"))
    app.static("pages", os.path.join(app_dir, "web/static/pages"))
    app.static("data", os.path.join(app_dir, "web/static/data"))


async def init(app, loop):
    await omicron.init()
    lpp = LongParallelPool()
    tblp = TouchBuyLimitPoolStore()

    lpp = LongParallelPool()
    tblp = TouchBuyLimitPoolStore()
    ssp = SteepSlopesPool()

    scheduler = AsyncIOScheduler(event_loop=loop)
    scheduler.add_job(lpp.pooling, "cron", hour=15, minute=2)
    scheduler.add_job(pooling_latest, "cron", hour=15, minute=5)
    scheduler.add_job(tblp.pooling, "cron", hour=15, minute=8)
    scheduler.add_job(ssp.pooling, "cron", hour=15, minute=8)

    scheduler.start()


def start(port: int = 2712):
    cfg4py.init(os.path.expanduser("~/zillionare/pluto"))

    application.register_listener(init, "before_server_start")
    application.blueprint(bp)
    serve_static_files(application)

    application.run(
        host="0.0.0.0",
        port=port,
        register_sys_signals=True,
        workers=1,
        single_process=True,
    )
    logger.info("pluto serve stopped")


if __name__ == "__main__":
    start()
