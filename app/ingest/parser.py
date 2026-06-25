"""Parser 适配器：复用 MinerUService，懒加载。"""
from __future__ import annotations


class MinerUParser:
    def __init__(self, service=None):
        self._svc = service

    def parse(self, file_path: str) -> str:
        if self._svc is None:
            from app.services import MinerUService
            self._svc = MinerUService()
        result = self._svc.parse_file(file_path)
        # MinerUResult.markdown 为正文
        return result.get("markdown") or ""
