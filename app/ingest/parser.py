"""Parser 适配器：MinerU 云端解析（二进制文档）或本地纯文本直读。"""
from __future__ import annotations
from pathlib import Path


class MinerUParser:
    """走 MinerU 云端解析（PDF/docx/xlsx 等二进制文档 → markdown）。懒加载。"""

    def __init__(self, service=None):
        self._svc = service

    def parse(self, file_path: str) -> str:
        if self._svc is None:
            from app.services import MinerUService
            self._svc = MinerUService()
        result = self._svc.parse_file(file_path)
        # MinerUResult.markdown 为正文
        return result.get("markdown") or ""


class LocalTextParser:
    """本地纯文本直读：直接读取 .md/.txt 等已解析好的文本文件，不依赖 MinerU。

    merged/ 下的数据本就是 MinerU 已解析的 .md/.txt 产物，二次喂回 MinerU 既浪费
    又强依赖云端；纯文本入库（或 MinerU 不可用）时用这个。
    """

    def __init__(self, encoding: str = "utf-8"):
        self._encoding = encoding

    def parse(self, file_path: str) -> str:
        path = Path(file_path)
        # utf-8 优先；Windows 中文历史文件可能是 gbk，回退一次
        try:
            return path.read_text(encoding=self._encoding)
        except UnicodeDecodeError:
            return path.read_text(encoding="gbk")
