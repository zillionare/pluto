from sanic import Blueprint

from pluto.web.handlers.strategies.momemtum import momemtum

strategies = Blueprint.group(momemtum, url_prefix="/strategies")
