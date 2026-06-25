"""问题生成服务模块

使用 LLM 生成 5W1H 问题 (What, Who, Where, When, Why, How)
"""

import httpx
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import settings


# 5W1H 问题模板
QUESTION_TEMPLATES = """
你是一个专业的文档分析专家。请基于提供的文档内容，生成 5 个不同维度的探索性问题。

## 文档信息
文件名: {filename}
文档类型: {doc_type}
内容摘要: {content_summary}

## 问题维度
请从以下 5 个维度生成问题：

1. **What (内容)**
   - 文档的主要议题是什么？
   - 讨论了哪些核心内容？
   - 有哪些关键信息点？

2. **Who (主体)**
   - 涉及哪些人员、部门或组织？
   - 谁是决策者？
   - 谁是执行者？

3. **Where (地点/范围)**
   - 涉及哪些地点或区域？
   - 影响范围如何？
   - 有哪些特定场景？

4. **When (时间)**
   - 讨论了哪些时间节点？
   - 有哪些截止日期或里程碑？
   - 相关时间线是什么？

5. **Why (原因)**
   - 为什么需要讨论这些内容？
   - 背景和原因是什么？
   - 解决什么问题？

6. **How (方式)**
   - 具体如何实施？
   - 使用什么方法和流程？
   - 有哪些具体措施？

## 输出格式
请以 JSON 格式返回，每个问题包含问题内容、维度和优先级：

```json
{{
  "questions": [
    {
      "dimension": "What",
      "question": "问题内容",
      "priority": "high"
    },
    ...
  ]
}
```

注意事项：
1. 优先级分为：high, medium, low
2. 问题应该具体且有深度
3. 尽量覆盖文档的不同方面
4. 避免重复
"""


class QuestionGeneratorService:
    """问题生成服务"""

    def __init__(self):
        """初始化服务"""
        self.completion_url = settings.cloud_completion_url
        self.auth_token = settings.cloud_auth_token
        self.model = settings.cloud_completion_model
        self.timeout = settings.cloud_completion_timeout

        # 创建 HTTP 客户端
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def generate_questions(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """基于文档内容生成问题

        Args:
            file_path: 文件路径
            content: 文档内容
            metadata: 文档元数据

        Returns:
            生成结果字典
        """
        if not self.completion_url or not self.auth_token:
            return {
                "success": False,
                "error": "补全服务未配置"
            }

        # 准备上下文信息
        filename = Path(file_path).name
        doc_type = self._detect_doc_type(filename)
        content_summary = self._generate_summary(content)

        # 构建 prompt
        prompt = QUESTION_TEMPLATES.format(
            filename=filename,
            doc_type=doc_type,
            content_summary=content_summary[:2000]  # 限制长度
        )

        # 准备请求
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的文档分析专家，擅长从不同维度分析文档内容并生成有深度的问题。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }

        try:
            # 调用 LLM
            response = await self.client.post(
                f"{self.completion_url}/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()

                # 解析问题
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

                    # 尝试解析 JSON
                    questions = self._parse_questions_response(content)

                    return {
                        "success": True,
                        "file_path": file_path,
                        "content_summary": content_summary,
                        "questions": questions,
                        "raw_response": content
                    }

                return {
                    "success": False,
                    "error": "LLM 响应格式异常"
                }

            else:
                return {
                    "success": False,
                    "error": f"LLM 调用失败: {response.status_code}",
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
                "error": f"生成问题失败: {e}"
            }

    def _detect_doc_type(self, filename: str) -> str:
        """检测文档类型

        Args:
            filename: 文件名

        Returns:
            文档类型
        """
        ext = Path(filename).suffix.lower()

        type_map = {
            ".pdf": "PDF 文档",
            ".docx": "Word 文档",
            ".doc": "Word 文档",
            ".pptx": "PowerPoint 演示",
            ".ppt": "PowerPoint 演示",
            ".xlsx": "Excel 表格",
            ".xls": "Excel 表格",
            ".txt": "纯文本文档",
            ".md": "Markdown 文档",
        }

        return type_map.get(ext, "未知类型")

    def _generate_summary(self, content: str) -> str:
        """生成内容摘要

        Args:
            content: 文档内容

        Returns:
            摘要文本
        """
        if not content:
            return "无内容"

        # 简单摘要：取前 2000 字
        lines = content.split("\n")
        summary_lines = []
        total_chars = 0

        for line in lines:
            if total_chars + len(line) > 2000:
                break
            if line.strip():
                summary_lines.append(line)
                total_chars += len(line)

        return "\n".join(summary_lines)

    def _parse_questions_response(
        self,
        response: str
    ) -> List[Dict[str, Any]]:
        """解析 LLM 响应中的问题

        Args:
            response: LLM 响应内容

        Returns:
            问题列表
        """
        # 尝试解析 JSON
        try:
            # 清理响应
            cleaned = response.strip()

            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*?\}', cleaned, re.DOTALL)

            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                questions = data.get("questions", [])

                # 验证问题格式
                valid_questions = []
                for q in questions:
                    if isinstance(q, dict):
                        valid_questions.append({
                            "dimension": q.get("dimension", ""),
                            "question": q.get("question", ""),
                            "priority": q.get("priority", "medium")
                        })

                return valid_questions

        except (json.JSONDecodeError, Exception):
            pass

        # 回退：解析文本格式
        questions = []
        lines = response.split("\n")

        dimension_map = {
            "what": "What",
            "who": "Who",
            "where": "Where",
            "when": "When",
            "why": "Why",
            "how": "How"
        }

        current_dimension = None

        for line in lines:
            line = line.strip()

            # 检测维度标记
            for key, value in dimension_map.items():
                if value in line or key in line.lower():
                    current_dimension = value
                    break

            # 提取问题（以数字或 - 开头）
            if line and (line[0].isdigit() or line.startswith(('-', '•', '*'))):
                question_text = re.sub(r'^[-•*\d.]+\s*', '', line).strip()

                if question_text and len(question_text) > 10:
                    questions.append({
                        "dimension": current_dimension or "",
                        "question": question_text,
                        "priority": "medium"
                    })

        # 限制数量
        return questions[:6]


# 创建单例
_question_service: Optional[QuestionGeneratorService] = None


def get_question_generator() -> QuestionGeneratorService:
    """获取问题生成服务单例

    Returns:
        QuestionGeneratorService 实例
    """
    global _question_service

    if _question_service is None:
        _question_service = QuestionGeneratorService()

    return _question_service
