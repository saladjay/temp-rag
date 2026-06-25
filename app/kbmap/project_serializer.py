"""kb_project 的 JSONL → 自然语言序列化。

原文件每行一个项目 JSON（带 <!-- chunk --> 标记行），序列化成自然语言描述
作为 embedding 输入（spec §3.3-3）。
"""
import json
from pathlib import Path

# 序列化字段顺序（影响语义权重，重要的在前）
_FIELDS_ORDER = [
    "立项年份", "技术领域", "项目名称", "项目编号",
    "承担单位", "项目负责人",
    "主要研究内容", "预期成果指标", "实际产出成果",
]


def serialize_project(obj: dict) -> str:
    """把单个项目 dict 序列化成自然语言描述，跳过空字段。

    例：「2019年立项的【桥梁工程】类项目《大跨径...》（编号 JT2019YB15），
    由广东省公路建设有限公司承担，项目负责人熊锋。主要研究内容：…」
    """
    year = obj.get("立项年份", "")
    field_obj = obj.get("技术领域", "")
    name = obj.get("项目名称", "")
    number = obj.get("项目编号", "")
    org = obj.get("承担单位", "")
    leader = obj.get("项目负责人", "")

    parts: list[str] = []
    # 头部：年份 + 领域 + 项目名 + 编号
    head = ""
    if year:
        head += f"{year}年立项的"
    if field_obj:
        head += f"【{field_obj}】类项目"
    if name:
        head += f"《{name}》"
    if number:
        head += f"（编号 {number}）"
    if head:
        parts.append(head)

    if org:
        parts.append(f"由{org}承担")
    if leader:
        parts.append(f"项目负责人{leader}")

    # 正文：其余字段按顺序拼「字段名：值」
    body_fields = [f for f in _FIELDS_ORDER if f not in
                   ("立项年份", "技术领域", "项目名称", "项目编号", "承担单位", "项目负责人")]
    body_parts = []
    for f in body_fields:
        v = obj.get(f)
        if v and str(v).strip():
            body_parts.append(f"{f}：{v}")
    if parts:
        head_text = "，".join(parts) + "。"
    else:
        head_text = ""
    body_text = "；".join(body_parts)
    return (head_text + body_text).strip()


def iter_jsonl_chunks(path: Path) -> list[str]:
    """读 kb_project 文件：跳过 <!-- chunk --> 标记行和空行，每行 JSON
    序列化成一个 chunk 文本。返回顺序与文件一致。"""
    path = Path(path)
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("<!--"):
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunks.append(serialize_project(obj))
    return chunks
