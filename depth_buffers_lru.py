import numpy as np
from collections import OrderedDict

class CameraDepths:
    def __init__(self, width: int, height: int, max_depths: int):
        self.width = width
        self.height = height
        self.max_depths = max_depths
        self._data = OrderedDict()

    def add_depth(self, ID: tuple[int, int], depth_buffer: np.ndarray):
        if depth_buffer.shape != (self.height, self.width):
            raise ValueError(
                f"Depth buffer shape {depth_buffer.shape} does not match "
                f"({self.height}, {self.width})"
            )

        if ID in self._data:
            self._data.move_to_end(ID)
            self._data[ID] = depth_buffer
            return

        if len(self._data) == self.max_depths:
            self._data.popitem(last=False)

        self._data[ID] = depth_buffer

    def get(self, ID: tuple[int, int]):
        if ID not in self._data:
            return None
        self._data.move_to_end(ID)
        return self._data[ID]

    def get_latest(self):
        if not self._data:
            return None
        last_key = next(reversed(self._data))
        return self._data[last_key]

    def get_all(self):
        return list(self._data.values())

    def __len__(self):
        return len(self._data)
