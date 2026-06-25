"""kb_project 的 JSONL → 自然语言序列化。

原文件每行一个项目 JSON（带 <!-- chunk --> 标记行），序列化成自然语言描述
作为 embedding 输入（spec §3.3-3）。
"""
import json
import re
from pathlib import Path

def serialize_project(d: dict) -> str:
    """把（已清洗的）项目 dict 序列化成自然语言文本。

    头部一句散文（技术领域打头），随后主要研究内容/关键技术问题/成果分句。
    字段值原样输出（不做改写/概括，保证确定性）；空字段与弃用字段不出现。
    成果需先经 clean_project 清洗，本函数只负责排版。
    """
    def g(key: str) -> str:
        return (d.get(key) or "").strip()

    tech, name = g("技术领域"), g("项目名称")
    number, category = g("项目编号"), g("业务类别")
    year, org, leader = g("立项年份"), g("承担单位"), g("项目负责人")
    content = g("主要研究内容")
    problem = g("拟解决的关键技术问题")
    achievements = g("实际产出成果")

    sentences: list[str] = []

    # 头部：[技术领域]领域的科研项目《项目名称》（编号 X，业务类别），年份立项，由X承担，项目负责人X。
    head_left = f"{tech}领域的科研项目" if tech else ("科研项目" if name else "")
    name_part = f"《{name}》" if name else ""
    paren_inner = "，".join(p for p in [f"编号 {number}" if number else "", category] if p)
    paren = f"（{paren_inner}）" if paren_inner else ""
    core = (head_left + name_part + paren).strip()
    head_bits: list[str] = []
    if core:
        head_bits.append(core)
    if year:
        head_bits.append(f"{year}年立项")
    if org:
        head_bits.append(f"由{org}承担")
    if leader:
        head_bits.append(f"项目负责人{leader}")
    if head_bits:
        sentences.append("，".join(head_bits) + "。")

    if content:
        sentences.append(f"主要研究内容：{content}。")
    if problem:
        sentences.append(f"拟解决的关键技术问题：{problem}。")
    if achievements:
        sentences.append(f"已产出成果：{achievements}。")

    return "".join(sentences)


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


def _parse_blob(blob: str) -> dict:
    """解析一个逻辑项目文本（可能含续片回接）。
    优先 json.loads：完整记录可精确还原转义引号/换行/制表；
    失败（切断记录未闭合）时回退字段正则（best-effort，丢弃尾字段）。
    """
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 回退：切断记录未闭合 → 补 " 让最后一个字段值闭合，再字段正则提取
    if not blob.endswith('"'):
        blob = blob + '"'
    return {k: v for k, v in _FIELD_RE.findall(blob)}


def parse_projects(raw_text: str) -> list[dict]:
    """容错解析 kb_project 文件文本。每个逻辑项目一个 dict（字段名→值）。

    优先 json.loads（完整记录精确还原转义）；切断单元（{ 开头但不闭合）+
    紧随其后的裸续片合并成一个逻辑项目，json.loads 失败时回退字段正则恢复。
    返回顺序与文件一致。
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
        d = _parse_blob(blob)
        if d:
            projects.append(d)
    return projects


_ACHIEVEMENT_NAME_RE = re.compile(r"成果名称：([^；]+)")


def _clean_achievements(value: str) -> str:
    """剥 '成果类型：X，成果名称：' 套话 → 取名称 → 去重（保序）→ 全留，；连接。
    无套话模式时返回去空白后的原文（不丢数据）。"""
    if not value:
        return ""
    names = _ACHIEVEMENT_NAME_RE.findall(value)
    if not names:
        return value.strip()
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            unique.append(n)
    return "；".join(unique)


def _clean_garbage(value: str) -> str:
    """清 (项)/(本)/(篇) 等单位标记与 \\t,; 计数噪声，折叠空白。"""
    if not value:
        return ""
    s = re.sub(r"\([^)]*\)", " ", value)   # 去 (项)(本)(篇)
    s = re.sub(r"[\t,;]+", " ", s)          # 去制表/逗号/分号噪声
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_project(d: dict) -> dict:
    """清洗：实际产出成果（剥套话+去重全留）、预期成果指标（清垃圾）。
    返回新 dict，不改入参。"""
    out = dict(d)
    out["实际产出成果"] = _clean_achievements(d.get("实际产出成果", ""))
    out["预期成果指标"] = _clean_garbage(d.get("预期成果指标", ""))
    return out
