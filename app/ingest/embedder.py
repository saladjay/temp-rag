"""Embedder 适配器：复用 CloudEmbeddingService。"""
from __future__ import annotations
from typing import Sequence
import numpy as np


class CloudEmbedder:
    def __init__(self, service=None):
        self._svc = service
        self._dim = None
        self.last_usage = None

    def _svc_(self):
        if self._svc is None:
            from app.services import CloudEmbeddingService
            self._svc = CloudEmbeddingService()
        return self._svc

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self._svc_().get_dimension()
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        svc = self._svc_()
        vecs = svc.encode(list(texts))
        self.last_usage = getattr(svc, "last_usage", None)
        return vecs
