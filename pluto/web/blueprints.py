from sanic import Blueprint

from pluto.web.handlers.pool_handler import pools
from pluto.web.handlers.strategies import strategies

bp = Blueprint.group(pools, strategies, url_prefix="/pluto")
