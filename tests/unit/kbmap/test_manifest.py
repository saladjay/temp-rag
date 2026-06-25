from pathlib import Path
import pytest
from pydantic import ValidationError

from app.kbmap.scanner import scan_kb_dirs, KbInventory
from app.kbmap.manifest import (
    Manifest, build_manifest_from_inventory, load_manifest, save_manifest,
    validate_manifest, DEFAULT_KB_DEFINES,
)

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "kbmap" / "merged"


def test_default_kb_defines_cover_six_kbs():
    assert set(DEFAULT_KB_DEFINES.keys()) == {
        "kb_policy_national", "kb_policy_group", "kb_ops",
        "kb_regulation", "kb_template", "kb_project",
    }
    # kb_project 走 jsonl 切块器
    assert DEFAULT_KB_DEFINES["kb_project"].chunker == "jsonl"
    assert DEFAULT_KB_DEFINES["kb_project"].serializer == "project_natural_language"


def test_build_manifest_from_inventory():
    inv = scan_kb_dirs(FIXTURE)
    m = build_manifest_from_inventory(inv)
    assert m.version == 1
    assert m.embedding_model == "bge-m3"
    assert "kb_policy_national" in m.kb_defines
    # 每条 file 的 kb 必须在 kb_defines
    for f in m.files:
        assert f.kb in m.kb_defines


def test_save_load_roundtrip(tmp_path):
    inv = scan_kb_dirs(FIXTURE)
    m = build_manifest_from_inventory(inv)
    p = tmp_path / "kb_manifest.yaml"
    save_manifest(m, p)
    m2 = load_manifest(p)
    assert m2 == m


def test_validate_rejects_unknown_kb(tmp_path):
    inv = scan_kb_dirs(FIXTURE)
    m = build_manifest_from_inventory(inv)
    # 篡改：把第一个 file 的 kb 改成不存在的
    m.files[0] = m.files[0].model_copy(update={"kb": "kb_nonexistent"})
    with pytest.raises(ValueError, match="kb_nonexistent"):
        validate_manifest(m)


def test_validate_rejects_bad_collection_name():
    # kb_defines 的 key 必须形如 kb_<小写字母数字下划线>
    m = Manifest(
        version=1, embedding_model="bge-m3",
        kb_defines={"Bad-Name": DEFAULT_KB_DEFINES["kb_ops"]},
        files=[],
    )
    with pytest.raises(ValueError, match="collection 名"):
        validate_manifest(m)
