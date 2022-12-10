import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
from apscheduler.job import Job
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from coretypes import FrameType
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.notify.mail import mail_notify
from omicron.talib import moving_average, polyfit
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

_local_observer = None


class Dispatcher(FileSystemEventHandler):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def dispatch(self, event):
        if not isinstance(event, FileModifiedEvent):
            return

        if event.src_path.endswith("ma.csv"):
            schedule_watch_ma(self.scheduler)


def start_monitors(scheduler: AsyncIOScheduler):
    global _local_observer
    _local_observer = Observer(scheduler)

    watch_dir = os.path.expanduser("~/zillionare/pluto/monitors/")
    _local_observer.schedule(Dispatcher(scheduler), watch_dir, recursive=False)
    _local_observer.start()
    schedule_watch_ma(scheduler)


def schedule_watch_ma(scheduler: AsyncIOScheduler):
    jobs: List[Job] = scheduler.get_jobs()
    for job in jobs:
        if job.name.startswith("watch_ma"):
            job.remove()

    cfg = os.path.expanduser("~/zillionare/pluto/monitors/ma.csv")
    if not os.path.exists(cfg):
        logger.warning("%s not found", cfg)
        return

    tasks = []
    with open(cfg, "r") as f:
        for line in f.readlines():
            try:
                code, win, ft = line.strip().split("\t")
                ft = FrameType(ft)
                tasks.append((code, int(win), ft))
            except Exception as e:
                logger.exception(e)

    scheduler.add_job(
        watch_ma,
        "cron",
        args=(tasks,),
        name=f"watch_ma:{code}:{win}:{ft.value}",
        hour=14,
        minute=30,
    )


async def watch_ma(watch_list: List):
    logger.info("checking if ma is reached")
    report = []

    for code, win, ft in watch_list:
        bars = await Stock.get_bars(code, win + 10, ft)
        if len(bars) < win + 10:
            return

        close = bars["close"]
        low = bars["low"]
        opn = bars["open"]

        ma = moving_average(close, win)[-10:]

        if opn[-1] >= ma[-1] and low[-1] < ma[-1] * 1.02:
            name = await Security.alias(code)
            today = bars[-1]["frame"].item().date()
            err, (a, _) = polyfit(ma / ma[0], deg=1)

            if err >= np.std(ma / ma[0]) / 2:
                msg = f"{name} {today} 触及{win}{ft.value}均线,但均线已不呈向上支撑直线，建议删除监控"
                logger.info(msg)
                report.append(msg)
                continue

            msg = f"{name} {today} 触及{win}{ft.value}均线"
            logger.info(msg)
            report.append(msg)

    if len(report):
        mail_notify("触及均线监控", body="\n".join(report))
