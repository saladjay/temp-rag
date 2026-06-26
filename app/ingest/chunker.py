"""结构感知切块（Structural Chunking）。

清洗 coreagent 导出噪声 → 忽略其多为截断的边界 → 按文档自身结构
（标题/章节/条/段落）+ 字数护栏重切 → 派确定性 segment_id。
设计见 docs/superpowers/specs/2026-06-25-chunker-structural-design.md。
"""
import hashlib
from app.config import settings
from dataclasses import dataclass
from .interfaces import Chunk


@dataclass
class RawChunk:
    """coreagent 原始 chunk（标记行之后的未清洗文本）。"""
    idx: int
    chunk_id: str
    text: str


def assign_segment_id(doc_id: str, ordinal: int, text: str) -> str:
    """确定性 segment id：sha256(doc_id|ordinal|text[:64])[:16]。"""
    raw = f"{doc_id}|{ordinal}|{text[:64]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


import re

# coreagent chunk 标记：<!-- chunk_idx=N chunk_id=hex ... -->
_RAW_CHUNK_RE = re.compile(r"<!--\s*chunk_idx=(\d+)\s+chunk_id=([0-9a-fA-F]+)[^>]*-->")


def parse_raw_chunks(raw_text: str) -> list[RawChunk]:
    """按 chunk 标记拆分。无标记时整文件作单个 RawChunk（chunk_id='nomarker'）。"""
    matches = list(_RAW_CHUNK_RE.finditer(raw_text))
    if not matches:
        return [RawChunk(idx=0, chunk_id="nomarker", text=raw_text)]
    out: list[RawChunk] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        out.append(RawChunk(idx=int(m.group(1)), chunk_id=m.group(2),
                            text=raw_text[start:end]))
    return out


# 噪声正则
_IMG_TAG_RE = re.compile(r"<img\s[^>]*?/?>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _strip_leading_meta_table(text: str) -> str:
    """剥离开头的 coreagent 注入元表（|字段|值| 头）。文档自带的表不剥。"""
    t = text.lstrip()
    if not t.startswith("|字段|值|"):
        return text
    lines = t.splitlines()
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        i += 1
    return "\n".join(lines[i:])


def clean_span(text: str) -> str:
    """剥离版式噪声：img 标签（URL）、<图片内容> 包装标记、开头注入元表、归一化空行。
    注意：只剥 <图片内容|start>/<图片内容|end> 标记本身，保留中间的 OCR 文字
    （图片里的发文机关/印章等正文）。"""
    # Word 目录(TOC)噪声：含 _Toc 书签的整行是正文标题的重复，无检索价值，删整行
    if "_Toc" in text:
        text = "\n".join(ln for ln in text.splitlines() if "_Toc" not in ln)
    text = _IMG_TAG_RE.sub("", text)
    text = text.replace("<图片内容|start>", "").replace("<图片内容|end>", "")
    text = _strip_leading_meta_table(text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


# 硬边界正则（行首）
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}\s", re.MULTILINE)
_NUM_CN_RE = re.compile(r"^[ \t]*[一二三四五六七八九十]+、", re.MULTILINE)
_NUM_CN_PAREN_RE = re.compile(r"^[ \t]*（[一二三四五六七八九十]+）", re.MULTILINE)
_NUM_AR_RE = re.compile(r"^[ \t]*\d+[、.][^0-9]", re.MULTILINE)  # 排除"2021年"这种
_CLAUSE_RE = re.compile(r"^[ \t]*第[一二三四五六七八九十百千\d]+条", re.MULTILINE)
_APPENDIX_RE = re.compile(r"^[ \t]*(附件|附表)[0-9一二三四五六七八九十A-Za-z]", re.MULTILINE)

_BOUNDARY_RES = (_HEADING_RE, _NUM_CN_RE, _NUM_CN_PAREN_RE, _NUM_AR_RE, _CLAUSE_RE, _APPENDIX_RE)


def find_hard_boundaries(full_text: str) -> list[int]:
    """返回硬边界行首 offset（升序、去重、不含 0）。"""
    positions: set[int] = set()
    for rx in _BOUNDARY_RES:
        for m in rx.finditer(full_text):
            positions.add(m.start())
    return sorted(p for p in positions if p > 0)


def _looks_like_heading(line: str) -> bool:
    """该行是否是硬边界起始行（标题/编号/条款/附件）。"""
    stripped = line.lstrip()
    if not stripped:
        return False
    return any(rx.match(stripped) for rx in _BOUNDARY_RES)


@dataclass
class SpanInterval:
    """清洗后某原 chunk_id 在 full_text 中的文本区间。"""
    chunk_id: str
    start: int
    end: int


@dataclass
class Section:
    """结构切分产物（未经字数护栏）。"""
    heading: str | None
    text: str
    start: int   # 在 full_text 中的起始 offset


def reconstruct(raw_chunks: list[RawChunk]) -> tuple[str, list[SpanInterval]]:
    """清洗每个 raw chunk，按顺序拼回成完整文本；忽略 coreagent 边界作切点。
    返回 (full_text, span_intervals)，区间用于后续 source_chunk_ids 追溯。"""
    cleaned = [(rc.chunk_id, clean_span(rc.text)) for rc in raw_chunks]
    cleaned = [(cid, t) for cid, t in cleaned if t]  # 丢清洗后为空的
    full = "\n".join(t for _, t in cleaned)
    intervals: list[SpanInterval] = []
    pos = 0
    for cid, t in cleaned:
        intervals.append(SpanInterval(cid, pos, pos + len(t)))
        pos += len(t) + 1  # +1 为 "\n" 分隔符
    return full, intervals


def split_by_structure(full_text: str) -> list[Section]:
    """按硬边界切 full_text 成 sections。每段 heading 取首行（若是边界起始）。"""
    if not full_text:
        return []
    bounds = find_hard_boundaries(full_text)
    cut_points = [0] + bounds + [len(full_text)]
    sections: list[Section] = []
    for i in range(len(cut_points) - 1):
        s, e = cut_points[i], cut_points[i + 1]
        raw_slice = full_text[s:e]
        chunk_text = raw_slice.strip()
        if not chunk_text:
            continue
        # sec.start 必须指向 strip 后文本的真实起点（剥掉前导空白），
        # 否则 _source_chunk_ids_for 的区间会左偏，追溯错源 chunk_id。
        leading_ws = len(raw_slice) - len(raw_slice.lstrip())
        start_offset = s + leading_ws
        first_line = chunk_text.split("\n", 1)[0].strip()
        heading = first_line if _looks_like_heading(first_line) else None
        sections.append(Section(heading=heading, text=chunk_text, start=start_offset))
    return sections


_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])")


def _split_oversized(sec: Section, max_size: int) -> list[Section]:
    """对超过 max_size 的 section：先按段落，再按句子，最后硬切。保留 heading 在首块。"""
    text = sec.text
    if len(text) <= max_size:
        return [sec]
    # 1) 按段落
    parts = [p for p in text.split("\n\n") if p.strip()]
    sep = "\n\n"  # 段落路径用 \n\n 聚合（还原原结构）
    if len(parts) == 1:
        # 2) 按句子
        parts = [p for p in _SENT_SPLIT_RE.split(text) if p]
        sep = ""  # 句子路径用空串聚合（保证拼接等于原文）
    # 3) 聚合到 max_size
    out: list[Section] = []
    buf = ""

    def _flush():
        nonlocal buf
        if buf:
            out.append(Section(sec.heading if not out else None, buf, sec.start))
            buf = ""

    for p in parts:
        if len(p) > max_size:
            # 单段/单句仍超：硬切
            _flush()
            for i in range(0, len(p), max_size):
                out.append(Section(None, p[i:i + max_size], sec.start + i))
            continue
        sep_len = len(sep) if buf else 0
        if len(buf) + sep_len + len(p) <= max_size:
            buf = (buf + sep + p) if buf else p
        else:
            _flush()
            buf = p
    _flush()
    return out if out else [sec]


def _merge_undersized(sections: list[Section], min_size: int) -> list[Section]:
    """小于 min_size 的 section 并入前一个（首段孤立则保留）。
    有 heading 的 section 不并入前一个（结构边界不可跨越）。"""
    out: list[Section] = []
    for sec in sections:
        if out and len(sec.text) < min_size and sec.heading is None:
            prev = out[-1]
            out[-1] = Section(prev.heading, prev.text + "\n\n" + sec.text, prev.start)
        else:
            out.append(sec)
    return out


def apply_size_guardrails(sections: list[Section], max_size: int, min_size: int) -> list[Section]:
    """大切/小并。先切超限，再并过小。"""
    split: list[Section] = []
    for sec in sections:
        split.extend(_split_oversized(sec, max_size))
    return _merge_undersized(split, min_size)


def _source_chunk_ids_for(intervals: list[SpanInterval], start: int, end: int) -> list[str]:
    """返回与 [start, end) 文本区间相交的原 chunk_id（去重、保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for iv in intervals:
        if iv.end > start and iv.start < end and iv.chunk_id not in seen:
            seen.add(iv.chunk_id)
            out.append(iv.chunk_id)
    return out


def _chunk_jsonl(doc_id: str, doc_name: str, kb: str, raw_text: str) -> list[Chunk]:
    """kb_project 旁路：每行 JSON 一个 chunk，自然语言序列化后作 embedding 输入。

    复用 app.kbmap.project_serializer 的 parse_projects（容错，含切断项目恢复）
    + clean_project + serialize_project 管线，与 kbmap.embed 共用单一源。
    ordinal 按 emitted chunk 递增 → segment_id 确定性。
    """
    from app.kbmap.project_serializer import parse_projects, clean_project, serialize_project
    chunks: list[Chunk] = []
    ordinal = 0
    for obj in parse_projects(raw_text):
        text = serialize_project(clean_project(obj))
        if not text:
            continue
        chunks.append(Chunk(
            segment_id=assign_segment_id(doc_id, ordinal, text),
            text=text, doc_id=doc_id, doc_name=doc_name, kb=kb,
            heading=None, source_chunk_ids=[], ordinal=ordinal,
        ))
        ordinal += 1
    return chunks


def _is_index_file(doc_name: str) -> bool:
    """coreagent 导出的目录/索引文件（*_index.md）：内容是文档标题列表，
    按结构标题切分会碎成空标题段，无检索意义 → 整篇作一个 section 交护栏。"""
    return doc_name.lower().endswith("_index.md")


def chunk_document(doc_id: str, doc_name: str, kb: str, raw_text: str) -> list[Chunk]:
    """主入口：根据 kb 选策略。kb_project 走 jsonl；其他走结构感知管线。"""
    if kb == "kb_project":
        return _chunk_jsonl(doc_id, doc_name, kb, raw_text)

    raw_chunks = parse_raw_chunks(raw_text)
    full, intervals = reconstruct(raw_chunks)
    if not full:
        return []
    # 短文档整篇化：清洗后整篇 ≤ chunk_whole_doc_max 时，整篇作一个 chunk，
    # 跳过结构切分（短文切碎反而损 embedding 信号 + 上下文）。
    if len(full) <= settings.chunk_whole_doc_max:
        return [Chunk(
            segment_id=assign_segment_id(doc_id, 0, full),
            text=full, doc_id=doc_id, doc_name=doc_name, kb=kb,
            heading=None, source_chunk_ids=[iv.chunk_id for iv in intervals], ordinal=0,
        )]
    sections = ([Section(heading=None, text=full, start=0)]
                if _is_index_file(doc_name)
                else split_by_structure(full))
    sections = apply_size_guardrails(
        sections, settings.chunk_target_max, settings.chunk_target_min
    )
    chunks: list[Chunk] = []
    for ordinal, sec in enumerate(sections):
        seg_id = assign_segment_id(doc_id, ordinal, sec.text)
        src = _source_chunk_ids_for(intervals, sec.start, sec.start + len(sec.text))
        chunks.append(Chunk(
            segment_id=seg_id, text=sec.text, doc_id=doc_id, doc_name=doc_name,
            kb=kb, heading=sec.heading, source_chunk_ids=src, ordinal=ordinal,
        ))
    return chunks


class StructuralChunker:
    """结构感知切块器：Chunker protocol 适配，委托给 chunk_document。

    主 spec pipeline 通过 .chunk(text, doc_id, doc_name, kb) 调用；
    护栏（chunk_target_max/min）由 chunk_document 从 settings 读。
    """

    def chunk(self, text: str, doc_id: str, doc_name: str = "", kb: str = "") -> list[Chunk]:
        return chunk_document(doc_id=doc_id, doc_name=doc_name or doc_id, kb=kb, raw_text=text)
