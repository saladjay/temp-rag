"""MinerU文档解析服务模块 - 调用远程MinerU API进行文档解析"""
from typing import Optional, Dict, Any, Union, TypedDict
from pathlib import Path
import httpx
from app.config import settings


class MinerUResult(TypedDict):
    """MinerU文档解析结果类型"""
    markdown: str
    success: bool
    message: str
    data: Optional[Dict[str, Any]]


class MinerUService:
    """MinerU文档解析服务类 - 调用远程MinerU API"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: int = 300
    ):
        """初始化MinerU文档解析服务

        Args:
            api_url: MinerU API地址
            auth_token: Basic认证token
            timeout: 请求超时时间(秒)
        """
        self.api_url = api_url or settings.mineru_url
        self.auth_token = auth_token or settings.mineru_auth_token
        self.timeout = timeout

        if not self.api_url:
            raise ValueError("mineru_url is not configured")
        if not self.auth_token:
            raise ValueError("mineru_auth_token is not configured")

        self._headers = {
            "Authorization": f"Basic {self.auth_token}",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "accept": "application/json"
        }

    def parse_file(
        self,
        file_path: Union[str, Path],
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
        parse_method: str = "auto",
        return_images: bool = False,
        return_content_list: bool = False,
        return_layout: bool = False,
        is_json_md_dump: bool = True,
        return_info: bool = False,
        output_dir: str = "output"
    ) -> MinerUResult:
        """解析文档文件

        Args:
            file_path: 文件路径
            file_bytes: 文件字节内容（可选，与file_path二选一）
            file_name: 文件名（当使用file_bytes时必填）
            parse_method: 解析方法 (auto, ocr, txt)
            return_images: 是否返回图片
            return_content_list: 是否返回内容列表
            return_layout: 是否返回布局信息
            is_json_md_dump: 是否返回JSON+Markdown
            return_info: 是否返回解析信息
            output_dir: 输出目录

        Returns:
            解析结果

        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 参数错误
        """
        if file_bytes is None:
            if not file_path:
                raise ValueError("Either file_path or file_bytes must be provided")
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            file_name = file_name or file_path.name

        if not file_bytes:
            raise ValueError("file_bytes is empty")
        if not file_name:
            raise ValueError("file_name must be provided when using file_bytes")

        # 构建multipart表单数据
        files = {
            "file": (file_name, file_bytes)
        }

        data = {
            "parse_method": parse_method,
            "return_images": str(return_images).lower(),
            "return_content_list": str(return_content_list).lower(),
            "return_layout": str(return_layout).lower(),
            "is_json_md_dump": str(is_json_md_dump).lower(),
            "return_info": str(return_info).lower(),
            "output_dir": output_dir
        }

        response = httpx.post(
            self.api_url,
            headers=self._headers,
            files=files,
            data=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, response: dict) -> MinerUResult:
        """解析API响应

        Args:
            response: API响应JSON数据

        Returns:
            解析结果
        """
        if "data" in response:
            # 标准格式: {"data": {...}, "success": true, ...}
            data = response["data"]
            markdown = data.get("content", data.get("markdown", ""))
            success = response.get("success", True)
            message = response.get("message", "")
            return {
                "markdown": markdown,
                "success": success,
                "message": message,
                "data": data
            }
        elif "markdown" in response:
            # 简单格式: {"markdown": "..."}
            return {
                "markdown": response["markdown"],
                "success": True,
                "message": "",
                "data": response
            }
        elif "content" in response:
            # 内容格式: {"content": "..."}
            return {
                "markdown": response["content"],
                "success": True,
                "message": "",
                "data": response
            }
        else:
            # 整个响应就是内容
            return {
                "markdown": str(response),
                "success": True,
                "message": "",
                "data": None
            }

    def parse_file_with_retry(
        self,
        file_path: Union[str, Path],
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
        parse_method: str = "auto",
        return_images: bool = False,
        return_content_list: bool = False,
        return_layout: bool = False,
        is_json_md_dump: bool = True,
        return_info: bool = False,
        output_dir: str = "output",
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> MinerUResult:
        """带重试机制的文档解析

        Args:
            file_path: 文件路径
            file_bytes: 文件字节内容
            file_name: 文件名
            parse_method: 解析方法
            return_images: 是否返回图片
            return_content_list: 是否返回内容列表
            return_layout: 是否返回布局信息
            is_json_md_dump: 是否返回JSON+Markdown
            return_info: 是否返回解析信息
            output_dir: 输出目录
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)

        Returns:
            解析结果
        """
        import time

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.parse_file(
                    file_path, file_bytes, file_name,
                    parse_method, return_images, return_content_list,
                    return_layout, is_json_md_dump, return_info, output_dir
                )
            except httpx.HTTPError as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                continue
            except Exception as e:
                raise e

        raise last_error if last_error else RuntimeError("Failed to parse file")

    def get_markdown_only(
        self,
        file_path: Union[str, Path],
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None
    ) -> str:
        """只获取解析后的Markdown内容

        Args:
            file_path: 文件路径
            file_bytes: 文件字节内容
            file_name: 文件名

        Returns:
            Markdown文本
        """
        result = self.parse_file(
            file_path=file_path,
            file_bytes=file_bytes,
            file_name=file_name,
            parse_method="auto",
            return_images=False,
            return_content_list=False,
            return_layout=False,
            is_json_md_dump=True,
            return_info=False
        )
        return result["markdown"]


class MockMinerUService(MinerUService):
    """Mock MinerU文档解析服务，用于测试"""

    def __init__(self, fixed_seed: int = 42):
        """初始化Mock服务

        Args:
            fixed_seed: 随机种子
        """
        object.__setattr__(self, 'api_url', 'mock://')
        object.__setattr__(self, 'auth_token', 'mock-token')
        object.__setattr__(self, 'timeout', 300)
        object.__setattr__(self, '_headers', {})

    def parse_file(
        self,
        file_path: Union[str, Path],
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
        parse_method: str = "auto",
        return_images: bool = False,
        return_content_list: bool = False,
        return_layout: bool = False,
        is_json_md_dump: bool = True,
        return_info: bool = False,
        output_dir: str = "output"
    ) -> MinerUResult:
        """Mock文档解析"""
        fn = str(file_path) if file_path else (file_name or "unknown")
        markdown = f"""# Mock解析结果

这是对文件 `{fn}` 的模拟解析结果。

- 文件名: {fn}
- 解析方法: {parse_method}
- 解析时间: {__import__('datetime').datetime.now().isoformat()}

## 内容摘要

这是模拟的文档内容。实际使用时会调用真实的MinerU API进行文档解析。

## 正文内容

这里是文档的正文内容模拟...

---

*此内容由MockMinerUService生成*
"""
        return {
            "markdown": markdown,
            "success": True,
            "message": "",
            "data": {
                "content": markdown,
                "file_name": fn,
                "parse_method": parse_method
            }
        }

    def parse_file_with_retry(self, file_path, file_bytes=None, file_name=None, parse_method="auto", return_images=False, return_content_list=False, return_layout=False, is_json_md_dump=True, return_info=False, output_dir="output", max_retries=3, retry_delay=2.0):
        """Mock重试机制"""
        return self.parse_file(
            file_path, file_bytes, file_name,
            parse_method, return_images, return_content_list,
            return_layout, is_json_md_dump, return_info, output_dir
        )

    def get_markdown_only(self, file_path, file_bytes=None, file_name=None):
        """Mock获取Markdown"""
        return self.parse_file(file_path, file_bytes, file_name)["markdown"]