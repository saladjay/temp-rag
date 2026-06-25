from pathlib import Path
from app.kbmap.scanner import (
    scan_kb_dirs, strip_uuid_prefix, DEFAULT_DIR_TO_KB,
)

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "kbmap" / "merged"


def test_strip_uuid_prefix():
    assert strip_uuid_prefix("aaa1_政策文件") == "政策文件"
    assert strip_uuid_prefix("集团规划") == "集团规划"


def test_strip_uuid_prefix_keeps_underscore_kb_names_without_hex_prefix():
    # KB 名本身含下划线、无 UUID 前缀时，必须原样返回，否则会被误剥导致目录被跳过
    assert strip_uuid_prefix("操作指引_常见问题") == "操作指引_常见问题"
    assert strip_uuid_prefix("政策_会议_讲话汇编") == "政策_会议_讲话汇编"


def test_default_dir_to_kb_covers_all_needed():
    assert DEFAULT_DIR_TO_KB["政策文件"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["政策二"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["规划政策文件"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["政策_会议_讲话汇编"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["集团规划"] == "kb_policy_group"
    assert DEFAULT_DIR_TO_KB["操作指引_常见问题"] == "kb_ops"
    assert DEFAULT_DIR_TO_KB["科小星-操作文档"] == "kb_ops"
    assert DEFAULT_DIR_TO_KB["科研管理制度"] == "kb_regulation"
    assert DEFAULT_DIR_TO_KB["模版"] == "kb_template"
    assert DEFAULT_DIR_TO_KB["集团历史项目-v2"] == "kb_project"


def test_scan_assigns_kb_by_default_mapping():
    inv = scan_kb_dirs(FIXTURE)
    by_doc = {f.doc_name: f.kb for f in inv.files}
    assert by_doc["十四五规划"] == "kb_policy_national"
    assert by_doc["集团科技创新纲要.pdf"] == "kb_policy_group"
    assert by_doc["项目导出.xlsx"] == "kb_project"


def test_scan_dedups_by_content_hash():
    inv = scan_kb_dirs(FIXTURE)
    # 十四五规划.md 在两个目录内容相同 → 只保留一条，另一条进 duplicates
    docs = [f for f in inv.files if f.doc_name == "十四五规划"]
    assert len(docs) == 1
    # duplicates 的主 path 命中保留的那条
    assert any("政策二" in dup for dups in inv.duplicates.values() for dup in dups)


def test_scan_only_collects_md_files():
    inv = scan_kb_dirs(FIXTURE)
    for f in inv.files:
        assert f.path.endswith(".md")


def test_scan_dedups_by_body_text_ignoring_chunk_markers(tmp_path):
    # 同一文档从不同导出得到的两份：正文相同、<!-- chunk --> 标记不同 → 应判为重复
    root = tmp_path / "merged"
    (root / "aaa1_政策文件").mkdir(parents=True)
    (root / "bbb2_政策二").mkdir(parents=True)
    body = "这是同一份文档的正文。\n第二行内容。\n"
    (root / "aaa1_政策文件" / "文档X.md").write_text(
        "<!-- chunk_idx=0 chunk_id=AAA occurTime=1 -->\n" + body, encoding="utf-8")
    (root / "bbb2_政策二" / "文档X.md").write_text(
        "<!-- chunk_idx=0 chunk_id=BBB occurTime=2 -->\n" + body, encoding="utf-8")
    inv = scan_kb_dirs(root)
    docs = [f for f in inv.files if f.doc_name == "文档X"]
    assert len(docs) == 1  # 正文哈希相同 → 去重后只剩一份
    assert any("政策二" in dup for dups in inv.duplicates.values() for dup in dups)
