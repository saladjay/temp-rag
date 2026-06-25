from app.kbmap.project_serializer import serialize_project, iter_jsonl_chunks
from tests.fixpaths import FIXTURE_MERGED  # 见 Step 2 注释


def test_serialize_project_basic():
    obj = {
        "立项年份": "2019",
        "项目编号": "JT2019YB15",
        "项目名称": "悬索桥缆索研究",
        "承担单位": "广东省公路建设有限公司",
        "项目负责人": "熊锋",
        "技术领域": "桥梁工程",
        "主要研究内容": "缆索腐蚀防护研究",
        "预期成果指标": "1项发明专利",
    }
    s = serialize_project(obj)
    # 关键事实必须出现在序列化文本里（embedding 输入靠这些召回）
    assert "2019" in s
    assert "JT2019YB15" in s
    assert "悬索桥缆索研究" in s
    assert "广东省公路建设有限公司" in s
    assert "熊锋" in s
    assert "桥梁工程" in s
    assert "缆索腐蚀防护研究" in s


def test_serialize_project_skips_empty_fields():
    obj = {"立项年份": "2019", "项目名称": "X", "承担单位": ""}
    s = serialize_project(obj)
    assert "承担单位" not in s  # 空字段不出现


def test_iter_jsonl_chunks_one_project_per_chunk():
    # fixture 里 项目导出.xlsx.md 有 2 个 JSON 行
    path = FIXTURE_MERGED / "ddd4_集团历史项目-v2" / "项目导出.xlsx.md"
    chunks = iter_jsonl_chunks(path)
    assert len(chunks) == 2
    assert "测试项目一" in chunks[0]
    assert "测试项目二" in chunks[1]
