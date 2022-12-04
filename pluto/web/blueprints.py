from sanic import Blueprint

from pluto.web.handlers.calendar import calendar
from pluto.web.handlers.pool_handler import pools
from pluto.web.handlers.strategies import strategies
from pluto.web.handlers.command import command

bp = Blueprint.group(pools, strategies, calendar, command, url_prefix="/pluto")
