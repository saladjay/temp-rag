"""API 请求/响应模型。"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., min_length=1)
    kb_names: Optional[list[str]] = Field(None, description="限定知识库，缺省全查")


class SourceItem(BaseModel):
    doc_id: str
    segment_id: str
    doc_name: str
    score: float
    source: str
