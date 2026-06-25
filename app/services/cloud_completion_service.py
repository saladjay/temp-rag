"""云端文本补全服务模块 - 调用云端LLM补全API"""
from typing import List, Optional, Dict, Iterator, TypedDict
import httpx
import requests
import json
from app.config import settings


class CompletionResult(TypedDict):
    """文本补全结果类型"""
    text: str
    finish_reason: Optional[str]
    usage: Optional[Dict[str, int]]


class CloudCompletionService:
    """云端文本补全服务类 - 调用远程补全API"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 60,
        auth_token: Optional[str] = None
    ):
        """初始化云端文本补全服务

        Args:
            api_url: 补全API地址
            model_name: 模型名称
            timeout: 请求超时时间(秒)
            auth_token: 认证令牌
        """
        self.api_url = api_url or settings.cloud_completion_url
        self.model_name = model_name or settings.cloud_completion_model
        self.timeout = timeout
        self.auth_token = auth_token or settings.cloud_auth_token

        if not self.api_url:
            raise ValueError("cloud_completion_url is not configured")

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Basic {self.auth_token}"

        self._client = httpx.Client(timeout=self.timeout, headers=headers)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> CompletionResult:
        """执行文本补全

        Args:
            prompt: 提示文本
            max_tokens: 最大生成token数
            temperature: 温度参数(0-2)
            top_p: 核采样参数(0-1)
            stream: 是否流式输出

        Returns:
            补全结果

        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 响应格式错误
        """
        if stream:
            raise NotImplementedError("Streaming not implemented yet")

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "model": self.model_name
        }

        response = self._client.post(self.api_url, json=payload)
        response.raise_for_status()
        return self._parse_completion_response(response.json())

    def _parse_completion_response(self, response: dict) -> CompletionResult:
        """解析API响应，提取补全结果

        Args:
            response: API响应JSON数据

        Returns:
            补全结果

        Raises:
            ValueError: 响应格式错误
        """
        # 支持多种响应格式
        if "choices" in response:
            # OpenAI 风格: {"choices": [{"text": "...", "finish_reason": "stop"}], "usage": {...}}
            choice = response["choices"][0]
            text = choice.get("text", choice.get("message", {}).get("content", ""))
            finish_reason = choice.get("finish_reason")
            usage = response.get("usage")
        elif "output" in response:
            # 简单格式: {"output": "文本", "finish_reason": "stop"}
            text = response["output"]
            finish_reason = response.get("finish_reason")
            usage = response.get("usage")
        elif "text" in response:
            # 最简格式: {"text": "文本"}
            text = response["text"]
            finish_reason = response.get("finish_reason")
            usage = response.get("usage")
        elif "result" in response:
            # 结果格式: {"result": "文本"}
            text = response["result"]
            finish_reason = response.get("finish_reason")
            usage = response.get("usage")
        else:
            raise ValueError(f"Unexpected response format: {response}")

        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": usage
        }

    def complete_with_retry(
        self,
        prompt: str,
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> CompletionResult:
        """带重试机制的文本补全

        Args:
            prompt: 提示文本
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            stream: 是否流式输出
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)

        Returns:
            补全结果
        """
        import time

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.complete(prompt, max_tokens, temperature, top_p, stream)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                continue
            except Exception as e:
                raise e

        raise last_error if last_error else RuntimeError("Failed to complete text")

    def complete_stream(
        self,
        prompt: str,
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Iterator[str]:
        """流式文本补全

        Args:
            prompt: 提示文本
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数

        Yields:
            生成的文本片段
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "model": self.model_name
        }

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self.api_url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            yield self._parse_stream_chunk(chunk)
                        except Exception:
                            continue

    def _parse_stream_chunk(self, chunk: dict) -> str:
        """解析流式响应的数据块

        Args:
            chunk: 数据块

        Returns:
            文本片段
        """
        if "choices" in chunk:
            choice = chunk["choices"][0]
            return choice.get("text", choice.get("delta", {}).get("content", ""))
        elif "text" in chunk:
            return chunk["text"]
        return ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> CompletionResult:
        """对话式补全

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            stream: 是否流式输出

        Returns:
            补全结果
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "model": self.model_name
        }

        response = self._client.post(self.api_url, json=payload)
        response.raise_for_status()
        return self._parse_completion_response(response.json())

    def close(self) -> None:
        """关闭HTTP客户端"""
        self._client.close()

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.close()


class MockCloudCompletionService(CloudCompletionService):
    """Mock云端文本补全服务，用于测试"""

    def __init__(self, fixed_seed: int = 42):
        """初始化Mock服务

        Args:
            fixed_seed: 随机种子
        """
        object.__setattr__(self, 'api_url', 'mock://')
        object.__setattr__(self, 'model_name', 'mock-completion-model')
        object.__setattr__(self, 'timeout', 60)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> CompletionResult:
        """Mock文本补全"""
        # 简单的prompt响应模拟
        responses = [
            "这是一个模拟的回答。",
            "我是云端大模型服务的模拟实现。",
            "根据您的问题，我提供以下回答。",
            "这是基于预设模板生成的回复。"
        ]
        text = responses[hash(prompt) % len(responses)]
        return {
            "text": text,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(text), "total_tokens": len(prompt) + len(text)}
        }

    def complete_stream(self, prompt: str, max_tokens: int = 320, temperature: float = 0.7, top_p: float = 0.9) -> Iterator[str]:
        """Mock流式补全"""
        result = self.complete(prompt, max_tokens, temperature, top_p)
        text = result["text"]
        # 模拟逐字输出
        for char in text:
            yield char

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> CompletionResult:
        """Mock对话补全"""
        last_msg = messages[-1].get("content", "") if messages else ""
        return self.complete(last_msg, max_tokens, temperature, top_p, stream)

    def close(self) -> None:
        """Mock关闭方法"""
        pass


class GLMCompletionService(CloudCompletionService):
    """智谱AI GLM 文本补全服务 - 调用智谱API"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 60,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """初始化GLM文本补全服务

        Args:
            api_url: GLM API地址
            model_name: GLM模型名称
            timeout: 请求超时时间(秒)
            auth_token: 认证令牌（兼容Basic认证）
            api_key: API Key（用于智谱API认证）
            api_secret: API Secret（用于智谱API认证）
        """
        # 如果没有提供api_url，使用默认的智谱API地址
        default_url = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
        api_url = api_url or default_url

        # 设置模型名称
        model_name = model_name or "glm-4-7"

        # 设置认证信息
        if auth_token:
            # 使用Basic认证
            headers = {"Authorization": f"Basic {auth_token}"}
        elif api_key and api_secret:
            # 使用智谱API认证
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        else:
            # 如果没有提供认证信息，尝试从环境变量获取
            env_token = getattr(settings, 'glm_auth_token', None)
            env_api_key = getattr(settings, 'glm_api_key', None)
            env_api_secret = getattr(settings, 'glm_api_secret', None)

            if env_token:
                headers = {"Authorization": f"Basic {env_token}"}
            elif env_api_key and env_api_secret:
                headers = {
                    "Authorization": f"Bearer {env_api_key}",
                    "Content-Type": "application/json"
                }
            else:
                raise ValueError("未配置GLM认证信息，请提供auth_token或api_key/api_secret")

        super().__init__(
            api_url=api_url,
            model_name=model_name,
            timeout=timeout,
            auth_token=None  # 已经在headers中设置了
        )

        # 智谱特定的headers
        self.glm_headers = headers
        self.api_key = api_key or getattr(settings, 'glm_api_key', None)
        self.api_secret = api_secret or getattr(settings, 'glm_api_secret', None)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> CompletionResult:
        """执行文本补全

        Args:
            prompt: 提示文本
            max_tokens: 最大生成token数
            temperature: 温度参数(0-2)
            top_p: 核采样参数(0-1)
            stream: 是否流式输出

        Returns:
            补全结果

        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 响应格式错误
        """
        if stream:
            raise NotImplementedError("Streaming not implemented yet for GLM service")

        # 构建GLM API请求格式
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream
        }

        # 使用requests而不是httpx，因为智谱API的认证方式更适合
        try:
            response = requests.post(
                self.api_url,
                headers=self.glm_headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return self._parse_glm_response(response.json())
        except Exception as e:
            raise Exception(f"GLM API调用失败: {str(e)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> CompletionResult:
        """对话式补全

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数
            stream: 是否流式输出

        Returns:
            补全结果
        """
        if stream:
            raise NotImplementedError("Streaming not implemented yet for GLM service")

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.glm_headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return self._parse_glm_response(response.json())
        except Exception as e:
            raise Exception(f"GLM API调用失败: {str(e)}")

    def _parse_glm_response(self, response: dict) -> CompletionResult:
        """解析GLM API响应

        Args:
            response: GLM API响应JSON数据

        Returns:
            补全结果

        Raises:
            ValueError: 响应格式错误
        """
        # GLM API 响应格式: {"id": "chatcmpl-xxx", "object": "chat.completion",
        #                     "created": 1234567890, "model": "glm-4-7",
        #                     "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."},
        #                                 "finish_reason": "stop"}], "usage": {...}}

        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason")
            usage = response.get("usage")
        else:
            raise ValueError(f"GLM API响应格式错误: {response}")

        return {
            "text": content,
            "finish_reason": finish_reason,
            "usage": usage
        }

    def close(self) -> None:
        """关闭GLM服务"""
        # GLM服务没有需要特别关闭的资源
        pass