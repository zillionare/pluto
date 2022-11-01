import os
from typing import Any

import zarr


class ZarrStore(object):
    def __init__(self, path=None):
        cur_dir = os.path.dirname(__file__)
        self._store_path = (
            path
            or os.environ.get("pluto_store_path")
            or os.path.join(cur_dir, "pluto.axzarr")
        )

        self._store = zarr.open(self._store_path, mode="a")

    def save(self, key: str, value: Any):
        key = f"{self.__class__.__name__.lower()}/{key}"
        self._store[key] = value

    def get(self, key: str):
        key = f"{self.__class__.__name__.lower()}/{key}"
        return self._store[key]

    @property
    def store(self):
        key = f"{self.__class__.__name__.lower()}"
        return self._store[key]
