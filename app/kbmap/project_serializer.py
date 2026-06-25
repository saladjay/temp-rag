"""kb_project 的 JSONL → 自然语言序列化。

原文件每行一个项目 JSON（带 <!-- chunk --> 标记行），序列化成自然语言描述
作为 embedding 输入（spec §3.3-3）。
"""
import json
import re
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


_CHUNK_MARKER_RE = re.compile(r"<!--\s*chunk_idx=\d+[^>]*-->")
# 字段提取：容忍 "k":"v" 与 json.dumps 的 "k": "v"（冒号两侧空格）
_FIELD_RE = re.compile(r'"([^"\\]+)"\s*:\s*"([^"]*)"')


def _split_units(raw_text: str) -> list[str]:
    """按 chunk_idx 标记切成单元正文，剥掉 --- 分隔行与首尾空白。"""
    parts = _CHUNK_MARKER_RE.split(raw_text)
    units: list[str] = []
    for body in parts[1:]:  # parts[0] 是首个标记前的内容（通常为空），跳过
        cleaned = "\n".join(
            ln for ln in body.splitlines() if ln.strip() != "---"
        ).strip()
        if cleaned:
            units.append(cleaned)
    return units


def parse_projects(raw_text: str) -> list[dict]:
    """容错解析 kb_project 文件文本。每个逻辑项目一个 dict（字段名→值）。

    用字段正则提取，不依赖 json.loads，故未闭合的切断记录也能恢复：
    切断单元（{ 开头但不闭合）+ 紧随其后的裸续片会被合并成一个逻辑项目，
    其 实际产出成果 字段值由续片补全。返回顺序与文件一致。
    """
    units = _split_units(raw_text)
    blobs: list[str] = []
    for u in units:
        if u.startswith("{"):
            blobs.append(u)
        elif blobs:
            # 裸续片：回接到当前（最后一个）逻辑项目
            blobs[-1] += u
        # 没有前置 { 单元的孤立续片：丢弃
    projects: list[dict] = []
    for blob in blobs:
        # 切断记录的 实际产出成果 未闭合（无结尾 "）→ 补 " 让正则能匹配到值尾
        if not blob.endswith('"'):
            blob = blob + '"'
        d = {k: v for k, v in _FIELD_RE.findall(blob)}
        if d:
            projects.append(d)
    return projects
