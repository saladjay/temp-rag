from app.kbmap.project_serializer import serialize_project, iter_jsonl_chunks
from tests.fixpaths import FIXTURE_MERGED  # 见 Step 2 注释

from app.kbmap.project_serializer import parse_projects


def test_parse_projects_complete_records():
    raw = ('<!-- chunk_idx=0 chunk_id=a occurTime=1 -->\n'
           '{"立项年份":"2019","项目名称":"完整项目","技术领域":"桥梁"}\n\n---\n\n'
           '<!-- chunk_idx=1 chunk_id=b occurTime=2 -->\n'
           '{"立项年份":"2020","项目名称":"另一项目","技术领域":"隧道"}\n\n---\n\n')
    projects = parse_projects(raw)
    assert len(projects) == 2
    assert projects[0]["项目名称"] == "完整项目"
    assert projects[1]["技术领域"] == "隧道"


def test_parse_projects_recovers_truncated_record_with_fragments():
    # 切断：实际产出成果 中途截断（无闭合 " 与 }），后跟 2 个裸续片
    trunc = ('{"立项年份":"2020","项目名称":"切断项目",'
             '"实际产出成果":"成果类型：专利，成果名称：P2 ；')
    raw = ('<!-- chunk_idx=0 chunk_id=a -->\n'
           '{"立项年份":"2019","项目名称":"完整项目"}\n\n---\n\n'
           '<!-- chunk_idx=1 chunk_id=b -->\n' + trunc + '\n\n---\n\n'
           '<!-- chunk_idx=2 chunk_id=c -->\n'
           '成果类型：论文，成果名称：P3 ；\n\n---\n\n'
           '<!-- chunk_idx=3 chunk_id=d -->\n'
           '成果类型：指南，成果名称：P4 ；\n\n---\n\n')
    projects = parse_projects(raw)
    assert len(projects) == 2
    assert projects[1]["项目名称"] == "切断项目"
    # 切断项目的 实际产出成果 应含母项目成果 + 全部续片（回接恢复）
    ach = projects[1]["实际产出成果"]
    assert "P2" in ach and "P3" in ach and "P4" in ach


def test_parse_projects_tolerates_json_dumps_spacing():
    # test_chunker.py 用 json.dumps 造数据，字段间是 "： "（冒号后空格），必须兼容
    import json
    obj = {"立项年份": "2019", "项目名称": "间距项目"}
    raw = '<!-- chunk_idx=0 chunk_id=a -->\n' + json.dumps(obj, ensure_ascii=False)
    projects = parse_projects(raw)
    assert len(projects) == 1
    assert projects[0]["项目名称"] == "间距项目"


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
