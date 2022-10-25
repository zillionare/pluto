import os

import zarr


class ZarrStore(object):
    def __init__(self, path=None):
        cur_dir = os.path.dirname(__file__)
        self._store_path = (
            path
            or os.environ.get("pluto_store_path")
            or os.path.join(cur_dir, "pluto.zarr")
        )

        self._store = zarr.open(self._store_path, mode="a")

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value
