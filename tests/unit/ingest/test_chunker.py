from app.config import settings
from app.ingest.chunker import Chunk, assign_segment_id


def test_config_chunker_defaults():
    assert settings.chunker_backend == "structural"
    assert settings.chunk_target_max == 1500
    assert settings.chunk_target_min == 120


def test_assign_segment_id_is_deterministic():
    a = assign_segment_id("doc1", 0, "一段文本内容")
    b = assign_segment_id("doc1", 0, "一段文本内容")
    assert a == b
    assert len(a) == 16


def test_assign_segment_id_differs_by_inputs():
    d = assign_segment_id("doc1", 0, "abc")
    assert assign_segment_id("doc1", 1, "abc") != d   # 序号不同
    assert assign_segment_id("doc2", 0, "abc") != d   # doc_id 不同
    assert assign_segment_id("doc1", 0, "xyz") != d   # 文本不同


def test_chunk_dataclass_fields():
    c = Chunk(segment_id="x", text="t", doc_id="d", doc_name="n",
              kb="kb_policy_national", heading="一、总则",
              source_chunk_ids=["a", "b"], ordinal=0)
    assert c.segment_id == "x"
    assert c.source_chunk_ids == ["a", "b"]
    assert c.heading == "一、总则"


from app.ingest.chunker import parse_raw_chunks, RawChunk


def test_parse_raw_chunks_splits_by_markers():
    raw = ("<!-- chunk_idx=0 chunk_id=aaa123 -->\n第一段\n"
           "<!-- chunk_idx=1 chunk_id=bbb456 -->\n第二段\n"
           "<!-- chunk_idx=2 chunk_id=ccc789 occurTime=1 -->\n第三段")
    chunks = parse_raw_chunks(raw)
    assert len(chunks) == 3
    assert chunks[0].idx == 0 and chunks[0].chunk_id == "aaa123"
    assert "第一段" in chunks[0].text
    assert chunks[1].idx == 1 and chunks[1].chunk_id == "bbb456"
    assert chunks[2].chunk_id == "ccc789" and "第三段" in chunks[2].text


def test_parse_raw_chunks_no_markers_returns_single_chunk():
    chunks = parse_raw_chunks("没有任何标记的纯文本")
    assert len(chunks) == 1
    assert chunks[0].idx == 0
    assert "纯文本" in chunks[0].text


from app.ingest.chunker import clean_span


def test_clean_span_strips_image_blocks_and_img_tags():
    text = ('前言\n<图片内容|start>\n<img src="https://x/abc123def456.jpeg"/>\n'
            '图片说明\n<图片内容|end>\n正文部分\n<img src="https://y/xyz.png"/>')
    cleaned = clean_span(text)
    assert "<图片内容" not in cleaned
    assert "<img" not in cleaned
    assert "https://" not in cleaned
    assert "前言" in cleaned and "正文部分" in cleaned


def test_clean_span_strips_leading_meta_table_only():
    # 开头的注入元表（|字段|值|）被剥；文档自带的普通表保留
    text = ("|字段|值|\n|---|---|\n|发布日期|2021-11|\n\n正文开始\n"
            "|列1|列2|\n|---|---|\n|a|b|")
    cleaned = clean_span(text)
    assert "字段" not in cleaned      # 注入表头被剥
    assert "发布日期" not in cleaned
    assert "正文开始" in cleaned
    assert "|列1|列2|" in cleaned     # 文档自带表保留
    assert "|a|b|" in cleaned


def test_clean_span_normalizes_blank_lines():
    text = "行1\n\n\n\n\n行2"
    assert "\n\n\n" not in clean_span(text)


from app.ingest.chunker import find_hard_boundaries, _looks_like_heading


def test_find_hard_boundaries_detects_all_kinds():
    text = ("抬头内容\n"
            "# 一级标题\n正文A\n"
            "一、第一章\n正文B\n"
            "（二）项\n正文C\n"
            "第3条\n条款内容\n"
            "1. 阿拉伯项\n正文D\n"
            "附件1\n附件内容")
    positions = find_hard_boundaries(text)
    # 6 个硬边界（# 一级标题 / 一、 / （二） / 第3条 / 1. / 附件1）
    assert len(positions) == 6
    assert positions == sorted(positions)
    assert 0 not in positions


def test_find_hard_boundaries_empty_when_no_structure():
    assert find_hard_boundaries("纯散文没有任何标题或编号。") == []


def test_looks_like_heading():
    assert _looks_like_heading("# 标题")
    assert _looks_like_heading("一、总则")
    assert _looks_like_heading("（三）具体要求")
    assert _looks_like_heading("第5条")
    assert _looks_like_heading("2. 子项")
    assert _looks_like_heading("附件1")
    assert not _looks_like_heading("这是普通正文句子。")
    assert not _looks_like_heading("")


from app.ingest.chunker import reconstruct, split_by_structure, SpanInterval, Section


def test_reconstruct_concatens_and_tracks_intervals():
    rcs = [
        RawChunk(0, "id0", "段落甲。"),
        RawChunk(1, "id1", "段落乙。"),  # 与甲是截断连续
        RawChunk(2, "id2", "二、新章节\n内容"),  # 结构边界
    ]
    full, intervals = reconstruct(rcs)
    assert "段落甲" in full and "段落乙" in full and "二、新章节" in full
    assert len(intervals) == 3
    # 区间互不重叠且覆盖（首区间 start=0）
    assert intervals[0].start == 0
    assert intervals[0].chunk_id == "id0"
    assert intervals[0].end <= intervals[1].start
    assert intervals[-1].end == len(full)


def test_reconstruct_drops_empty_cleaned_spans():
    rcs = [RawChunk(0, "id0", "<图片内容|start><图片内容|end>"), RawChunk(1, "id1", "有效文本")]
    full, intervals = reconstruct(rcs)
    assert len(intervals) == 1  # 清洗后空的被丢
    assert intervals[0].chunk_id == "id1"


def test_split_by_structure_cuts_at_hard_boundaries():
    full = "抬头说明\n一、第一章\n正文甲\n二、第二章\n正文乙"
    sections = split_by_structure(full)
    # 抬头 / 一、 / 二、 三段
    assert len(sections) == 3
    assert sections[0].heading is None and "抬头说明" in sections[0].text
    assert sections[1].heading == "一、第一章"
    assert sections[2].heading == "二、第二章"
    # start offset 升序
    starts = [s.start for s in sections]
    assert starts == sorted(starts)


from app.ingest.chunker import apply_size_guardrails, Section


def _sec(text, heading=None, start=0):
    return Section(heading=heading, text=text, start=start)


def test_guardrails_split_oversized_by_paragraph():
    long_para = "句一。" * 400  # 1200 字，无段落分隔
    sec = _sec(long_para)
    out = apply_size_guardrails([sec], max_size=500, min_size=50)
    assert all(len(s.text) <= 500 for s in out)
    assert len(out) > 1
    # 不丢内容（拼接后等于原文）
    assert "".join(s.text for s in out) == long_para


def test_guardrails_split_uses_paragraph_boundaries_first():
    text = "段一有够多字。" * 50 + "\n\n" + "段二有够多字。" * 50  # 两个大段
    out = apply_size_guardrails([_sec(text)], max_size=300, min_size=20)
    # 应在段落 \n\n 处优先切
    assert len(out) >= 2
    assert all(len(s.text) <= 300 for s in out)


def test_guardrails_merge_undersized_into_previous():
    secs = [_sec("正常长度的段落，足够超过下限。"), _sec("短。"), _sec("另一正常段落内容。")]
    out = apply_size_guardrails(secs, max_size=1500, min_size=30)
    # "短。" 应被并入前一段，不再单独存在
    assert not any(s.text == "短。" for s in out)
    assert len(out) < len(secs)


def test_guardrails_keeps_solo_short_section():
    # 孤段且短：保留（不强行合并到虚无）
    out = apply_size_guardrails([_sec("孤短。")], max_size=1500, min_size=120)
    assert len(out) == 1


import json
from app.ingest.chunker import chunk_document


def test_chunk_document_full_pipeline_cleans_and_splits():
    raw = ("<!-- chunk_idx=0 chunk_id=aaa -->\n"
           "<图片内容|start><img src=\"https://x/hash.jpeg\"/><图片内容|end>\n"
           "抬头说明文字。\n"
           "<!-- chunk_idx=1 chunk_id=bbb -->\n"
           "一、总则\n本章节讲总体要求，内容比较丰富。\n"
           "<!-- chunk_idx=2 chunk_id=ccc -->\n"
           "二、附则\n第二条的具体规定。")
    chunks = chunk_document("doc1", "测试.doc", "kb_policy_national", raw)
    assert len(chunks) >= 2
    # 无噪声
    for c in chunks:
        assert "<图片内容" not in c.text and "<img" not in c.text
        assert "https://" not in c.text
    # segment_id 确定性、长度 16
    for c in chunks:
        assert len(c.segment_id) == 16
    # source_chunk_ids 非空（追溯原 chunk_id）
    all_src = {cid for c in chunks for cid in c.source_chunk_ids}
    assert "aaa" in all_src or "bbb" in all_src
    # heading 至少有一个被识别
    assert any(c.heading for c in chunks)


def test_chunk_document_is_idempotent():
    raw = "<!-- chunk_idx=0 chunk_id=a -->\n一、章\n正文内容。\n二、章\n正文。"
    a = chunk_document("docX", "x.doc", "kb_regulation", raw)
    b = chunk_document("docX", "x.doc", "kb_regulation", raw)
    assert [c.segment_id for c in a] == [c.segment_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_chunk_document_truncation_heals_into_paragraph():
    # 两块被 coreagent 在句中截断（无结构边界）：重建后合成一段连贯文本
    raw = ("<!-- chunk_idx=0 chunk_id=a -->\n这是一段连续的散文内容被截"
           "<!-- chunk_idx=1 chunk_id=b -->\n断了，但其实属于同一句话。")
    chunks = chunk_document("docT", "t.doc", "kb_policy_national", raw)
    # 拼起来应是连贯的一句（不被句中切开）
    joined = "".join(c.text for c in chunks)
    assert "同一句话" in joined
    assert "被截断了" in joined or "被截\n断了" in joined or "被截断" in joined.replace("\n","")


def test_chunk_document_kb_project_uses_jsonl_serializer():
    obj1 = {"立项年份": "2019", "项目名称": "桥A", "技术领域": "桥梁", "承担单位": "X公司",
            "项目负责人": "张三", "主要研究内容": "研究一"}
    obj2 = {"立项年份": "2020", "项目名称": "隧B", "技术领域": "隧道", "承担单位": "Y公司",
            "项目负责人": "李四", "主要研究内容": "研究二"}
    raw = ("<!-- chunk_idx=0 chunk_id=p0 -->\n" + json.dumps(obj1, ensure_ascii=False) + "\n"
           "<!-- chunk_idx=1 chunk_id=p1 -->\n" + json.dumps(obj2, ensure_ascii=False))
    chunks = chunk_document("docP", "项目.xlsx", "kb_project", raw)
    # 一项目一 chunk
    assert len(chunks) == 2
    assert "桥A" in chunks[0].text and "张三" in chunks[0].text
    assert "隧B" in chunks[1].text and "李四" in chunks[1].text
    # 序号递增
    assert chunks[0].ordinal == 0 and chunks[1].ordinal == 1


def test_source_chunk_ids_correct_despite_leading_whitespace():
    """heading 前有空行/缩进时，final chunk 的 source_chunk_ids 仍正确追溯。"""
    from app.ingest.chunker import chunk_document
    # raw: chunk0 = 抬头（含尾部空行），chunk1 = 缩进的标题+正文
    raw = ("<!-- chunk_idx=0 chunk_id=abc0 -->\n抬头。\n\n"
           "<!-- chunk_idx=1 chunk_id=def1 -->\n  一、章节\n正文内容。")
    chunks = chunk_document("docS", "s.doc", "kb_regulation", raw)
    # 找到含"一、章节"的 chunk，其 source_chunk_ids 必须含 def1（不能因偏移漏掉或错配）
    heading_chunk = next(c for c in chunks if c.heading and "章节" in c.heading)
    assert "def1" in heading_chunk.source_chunk_ids
    # 抬头 chunk 的 source_chunk_ids 含 abc0
    preamble = next(c for c in chunks if "抬头" in c.text)
    assert "abc0" in preamble.source_chunk_ids
