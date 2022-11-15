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

    def save(self, records: Any, key: str = None):
        """将`records` 存到`key`下面（替换式)

        Args:
            records: 要存储的数据
            key: 如果为None，则存到根下面。
        """
        if key is not None:
            key = f"{self.__class__.__name__.lower()}/{key}"
        else:
            key = f"{self.__class__.__name__.lower()}/"
        self._store[key] = records

    def append(self, records: Any, key: str = None):
        """向key所引用的数组增加数据"""
        if key is not None:
            key = f"{self.__class__.__name__.lower()}/{key}"
        else:
            key = f"{self.__class__.__name__.lower()}"

        if self._store.get(key):
            self._store[key].append(records)
        else:
            self._store[key] = records

    def get(self, key: str):
        key = f"{self.__class__.__name__.lower()}/{key}"
        return self._store[key]

    @property
    def data(self):
        key = f"{self.__class__.__name__.lower()}/"
        return self._store[key]
