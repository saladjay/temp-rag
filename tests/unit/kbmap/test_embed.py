from pathlib import Path
import numpy as np
from app.services.cloud_embedding_service import MockCloudEmbeddingService

from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.embed import extract_file_text, embed_files
from tests.fixpaths import FIXTURE_MERGED


def test_extract_md_file_text_uses_title_and_first_chunk():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    # 找一个 kb_policy_national 的文件（十四五规划 或 交通强国纲要）
    entry = next(f for f in m.files if f.kb == "kb_policy_national")
    txt = extract_file_text(entry, FIXTURE_MERGED)
    # 标题（doc_name）必须出现在文本里
    assert entry.doc_name in txt


def test_extract_kb_project_uses_serializer():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    entry = next(f for f in m.files if f.kb == "kb_project")
    txt = extract_file_text(entry, FIXTURE_MERGED)
    # 用的是第一个项目序列化结果，应含「测试项目一」
    assert "测试项目一" in txt


def test_embed_files_returns_one_vector_per_file():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(dimension=1024)
    vecs = embed_files(m, FIXTURE_MERGED, embedder)
    # 每个 file 一条向量
    assert len(vecs) == len(m.files)
    for v in vecs.values():
        assert len(v) == 1024
    # mock 基于 hash(text) 确定性：相同文本同向量；不同文本大概率不同
    unique_texts = {extract_file_text(f, FIXTURE_MERGED) for f in m.files}
    if len(unique_texts) > 1:
        # 至少有一对向量不同
        vs = list(vecs.values())
        assert any(vs[0] != vs[i] for i in range(1, len(vs)))


def test_embed_files_is_deterministic_same_input_same_output():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    e1 = MockCloudEmbeddingService(dimension=1024)
    e2 = MockCloudEmbeddingService(dimension=1024)
    v1 = embed_files(m, FIXTURE_MERGED, e1)
    v2 = embed_files(m, FIXTURE_MERGED, e2)
    # 同进程内 mock 确定性：两次结果逐位相同
    for path in v1:
        assert v1[path] == v2[path]


def test_extract_kb_project_falls_back_to_doc_name_when_no_json(tmp_path):
    # kb_project 文件若没有可解析的 JSON 行，extract_file_text 应回退到 doc_name
    from app.kbmap.manifest import FileEntry
    from app.kbmap.embed import extract_file_text
    # 造一个只有注释/空行、无 JSON 的 kb_project 文件
    f = tmp_path / "空.xlsx.md"
    f.write_text("<!-- chunk_idx=0 chunk_id=x -->\n只有说明没有JSON\n", encoding="utf-8")
    entry = FileEntry(path="空.xlsx.md", kb="kb_project", doc_name="空.xlsx")
    txt = extract_file_text(entry, tmp_path)
    assert txt == "空.xlsx"  # 回退到 doc_name
