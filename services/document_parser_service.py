"""文档解析服务模块

使用 MinerU 或其他接口解析文件内容
"""

import httpx
import base64
from typing import Dict, Any, Optional
from pathlib import Path

from ..config import settings


class DocumentParserService:
    """文档解析服务"""

    def __init__(self):
        """初始化服务"""
        self.mineru_url = settings.mineru_url
        self.auth_token = settings.mineru_auth_token
        self.timeout = settings.mineru_timeout

        # 创建 HTTP 客户端
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def parse_file(
        self,
        file_path: str,
        file_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """解析文件内容

        Args:
            file_path: 文件路径
            file_bytes: 文件字节数据（可选，如果不提供则读取文件）

        Returns:
            解析结果字典，包含：
            - content: 文本内容
            - metadata: 元数据（页数、作者等）
            - tables: 表格数据（如果有）
            - error: 错误信息（如果有）
        """
        if not self.mineru_url or not self.auth_token:
            return {
                "success": False,
                "error": "MinerU 服务未配置"
            }

        # 如果没有提供字节数据，读取文件
        if file_bytes is None:
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"读取文件失败: {e}"
                }

        # 准备请求
        files = {
            "file": (Path(file_path).name, file_bytes, "application/octet-stream")
        }

        headers = {
            "Authorization": f"Bearer {self.auth_token}"
        }

        try:
            # 调用 MinerU 接口
            response = await self.client.post(
                f"{self.mineru_url}/file_parse",
                files=files,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "file_path": file_path,
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "tables": result.get("tables", []),
                    "images": result.get("images", []),
                }
            else:
                return {
                    "success": False,
                    "error": f"MinerU 解析失败: {response.status_code}",
                    "status_code": response.status_code
                }

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "请求超时"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"解析异常: {e}"
            }

    async def parse_files(
        self,
        file_paths: list[str]
    ) -> Dict[str, Any]:
        """批量解析文件

        Args:
            file_paths: 文件路径列表

        Returns:
            批量解析结果
        """
        results = {
            "success": True,
            "total": len(file_paths),
            "parsed": 0,
            "failed": 0,
            "documents": [],
            "errors": []
        }

        for file_path in file_paths:
            result = await self.parse_file(file_path)
            if result.get("success"):
                results["parsed"] += 1
                results["documents"].append(result)
            else:
                results["failed"] += 1
                results["errors"].append({
                    "file_path": file_path,
                    "error": result.get("error", "Unknown error")
                })

        results["success"] = results["failed"] == 0
        return results


def create_fallback_parser() -> 'DocumentParserService':
    """创建回退解析器（不依赖 MinerU）

    用于测试环境没有 MinerU 服务的情况
    """
    import pypdf
    from io import BytesIO

    class FallbackParser:
        """回退解析器"""

        async def parse_file(
            self,
            file_path: str,
            file_bytes: Optional[bytes] = None
        ) -> Dict[str, Any]:
            """解析文件（使用 pypdf）

            Args:
                file_path: 文件路径
                file_bytes: 文件字节数据（可选）

            Returns:
                解析结果
            """
            if file_bytes is None:
                try:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"读取文件失败: {e}"
                    }

            # 尝试使用 pypdf 解析 PDF
            try:
                from pypdf import PdfReader
                pdf_file = BytesIO(file_bytes)
                reader = PdfReader(pdf_file)

                # 提取文本内容
                content_parts = []
                for page in reader.pages:
                    try:
                        text = page.extract_text()
                        if text:
                            content_parts.append(text)
                    except Exception:
                        continue

                content = "\n\n".join(content_parts)

                # 提取元数据
                metadata = reader.metadata or {}
                page_count = len(reader.pages)

                return {
                    "success": True,
                    "file_path": file_path,
                    "content": content,
                    "metadata": {
                        "page_count": page_count,
                        "author": metadata.get("/Author", ""),
                        "title": metadata.get("/Title", Path(file_path).name),
                        "creator": metadata.get("/Creator", ""),
                    },
                    "tables": [],
                    "images": [],
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"PDF 解析失败: {e}"
                }

    return FallbackParser()


# 创建单例
_parser_service: Optional[DocumentParserService] = None
_fallback_parser: Optional[Any] = None


def get_document_parser(use_mineru: bool = True) -> Any:
    """获取文档解析器实例

    Args:
        use_mineru: 是否使用 MinerU 服务

    Returns:
        解析器实例
    """
    global _parser_service, _fallback_parser

    if use_mineru and settings.mineru_url:
        if _parser_service is None:
            _parser_service = DocumentParserService()
        return _parser_service
    else:
        if _fallback_parser is None:
            _fallback_parser = create_fallback_parser()
        return _fallback_parser
