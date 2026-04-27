import gc
from typing import Any, Callable
import torch


class ModelManager:
    def __init__(self):
        self._current_model: Any = None
        self._current_name: str | None = None

    def load(self, name: str, loader_fn: Callable[[], Any]) -> Any:
        if self._current_name == name:
            return self._current_model
        self.unload()
        self._current_model = loader_fn()
        self._current_name = name
        return self._current_model

    def unload(self):
        if self._current_model is not None:
            del self._current_model
            gc.collect()
            torch.cuda.empty_cache()
        self._current_model = None
        self._current_name = None
