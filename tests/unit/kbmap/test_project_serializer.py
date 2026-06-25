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


def test_parse_projects_preserves_escaped_quotes_in_values():
    # 值内含 JSON 转义引号 \" —— 必须完整保留，不能在第一个 \" 处截断（real-data 回归）
    raw = ('<!-- chunk_idx=0 chunk_id=a -->\n'
           '{"立项年份":"2022","项目名称":"含转义引号项目",'
           '"项目概况":"《规划》中\\"第8纵\\"的重要组成部分"}')
    projects = parse_projects(raw)
    assert len(projects) == 1
    assert '"第8纵"' in projects[0]["项目概况"]   # 转义引号解码为真实引号，整段保留
    assert "重要" in projects[0]["项目概况"]


def test_parse_projects_decodes_json_escapes():
    # 值内含 \n \t —— 解码为真实换行/制表（clean_project 依赖真实字符，非字面 \n）
    raw = ('<!-- chunk_idx=0 chunk_id=a -->\n'
           '{"项目名称":"X","主要研究内容":"行一\\n行二\\t缩进"}')
    mc = parse_projects(raw)[0]["主要研究内容"]
    assert "\n" in mc and "\t" in mc     # 真实换行/制表
    assert "\\n" not in mc                # 不是字面 backslash-n


from app.kbmap.project_serializer import clean_project


def test_clean_project_strips_achievement_boilerplate_and_dedupes():
    d = {"实际产出成果":
         "成果类型：专利，成果名称：自锁式行走轮 ；"
         "成果类型：专利，成果名称：自锁式行走轮 ；"   # 重复
         "成果类型：论文，成果名称：除湿微机控制系统 ；"}
    out = clean_project(d)
    assert out["实际产出成果"] == "自锁式行走轮；除湿微机控制系统"  # 剥套话+去重+；连接
    assert "成果类型" not in out["实际产出成果"]
    assert "成果名称" not in out["实际产出成果"]


def test_clean_project_preserves_non_boilerplate_achievements():
    # 无 成果名称： 套话模式时，原样返回去空白文本（不丢数据）
    d = {"实际产出成果": "  自由文本成果描述  "}
    assert clean_project(d)["实际产出成果"] == "自由文本成果描述"


def test_clean_project_cleans_garbage_in_expected_indicator():
    d = {"预期成果指标": "1项专利授权发明专利(项)\t\t,;\n9篇形成研究报告数(篇)\t\t,;"}
    out = clean_project(d)["预期成果指标"]
    assert "(项)" not in out and "(篇)" not in out
    assert "\t" not in out


def test_clean_project_does_not_mutate_input():
    d = {"实际产出成果": "成果类型：专利，成果名称：A ；"}
    original = dict(d)
    clean_project(d)
    assert d == original  # 原 dict 不被修改


def test_serialize_project_excludes_contact_and_budget_fields():
    obj = {
        "技术领域": "桥梁工程", "项目名称": "某项目", "项目编号": "JT0001",
        "立项年份": "2019", "承担单位": "某公司", "项目负责人": "张三",
        "项目负责人电话": "13900000000", "项目负责人电子邮箱": "secret@example.com",
        "项目预算总经费": "99999", "项目总决算": "88888",
    }
    s = serialize_project(obj)
    # 联系/经费字段不进入序列化文本
    assert "13900000000" not in s
    assert "secret@example.com" not in s
    assert "99999" not in s
    assert "88888" not in s
    # 主题字段在
    assert "桥梁工程" in s and "某项目" in s and "张三" in s


def test_serialize_project_head_is_prose_with_category_in_parens():
    obj = {"技术领域": "桥梁工程", "项目名称": "缆索研究", "项目编号": "JT1",
           "业务类别": "重点科技项目", "立项年份": "2019",
           "承担单位": "某公司", "项目负责人": "熊锋"}
    s = serialize_project(obj)
    assert s.startswith("桥梁工程领域的科研项目《缆索研究》（编号 JT1，重点科技项目）")
    assert "2019年立项" in s
    assert "由某公司承担" in s
    assert "项目负责人熊锋" in s
    assert s.endswith("。")


def test_iter_jsonl_chunks_recovers_truncated_project_from_file(tmp_path):
    trunc = ('{"立项年份":"2020","项目名称":"切断项目",'
             '"实际产出成果":"成果类型：专利，成果名称：P2 ；')
    content = ('<!-- chunk_idx=0 chunk_id=a -->\n'
               '{"立项年份":"2019","项目名称":"完整项目"}\n\n---\n\n'
               '<!-- chunk_idx=1 chunk_id=b -->\n' + trunc + '\n\n---\n\n'
               '<!-- chunk_idx=2 chunk_id=c -->\n'
               '成果类型：论文，成果名称：P3 ；\n\n---\n\n')
    f = tmp_path / "项目导出.xlsx.md"
    f.write_text(content, encoding="utf-8")
    chunks = iter_jsonl_chunks(f)
    assert len(chunks) == 2                    # 切断项目被恢复，不是 1
    assert "切断项目" in chunks[1]
    assert "P2" in chunks[1] and "P3" in chunks[1]
    assert "成果类型" not in chunks[1]         # 已清洗


def test_clean_then_serialize_full_pipeline():
    d = {"技术领域": "桥梁工程", "项目名称": "X", "项目编号": "JT1", "立项年份": "2019",
         "承担单位": "某公司", "项目负责人": "张三",
         "实际产出成果": "成果类型：专利，成果名称：A ；成果类型：专利，成果名称：A ；成果类型：论文，成果名称：B ；"}
    s = serialize_project(clean_project(d))
    assert "已产出成果：A；B。" in s          # 去重+全留+句末句号
    assert "成果类型" not in s and "成果名称" not in s
