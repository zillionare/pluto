from pluto.web.handlers.pool_handler import pools
from pluto.web.handlers.strategies import strategies
from sanic import Blueprint

bp = Blueprint.group(pools, strategies, url_prefix="/pluto")
