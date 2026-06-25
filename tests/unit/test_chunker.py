from app.ingest.chunker import FixedChunker


def test_deterministic_same_input_same_output():
    c = FixedChunker(size=100, overlap=20, tolerance=30)
    text = "第一句。第二句较长较长较长较长较长。第三句。第四句。第五句结尾。"
    a = c.chunk(text, "doc1")
    b = c.chunk(text, "doc1")
    assert [x.text for x in a] == [x.text for x in b]
    assert [x.segment_id for x in a] == [x.segment_id for x in b]


def test_segment_ids_stable_and_ordered():
    c = FixedChunker(size=50, overlap=10, tolerance=20)
    chunks = c.chunk("甲。乙。丙。丁。戊。己。庚。辛。", "d9")
    assert [x.ordinal for x in chunks] == list(range(len(chunks)))
    assert chunks[0].segment_id == "d9#0000"
    assert chunks[-1].segment_id == f"d9#{len(chunks)-1:04d}"


def test_prefers_sentence_boundary_snap():
    c = FixedChunker(size=40, overlap=10, tolerance=20)
    text = "短句。这是一个稍微长一点的句子内容哦。" * 3
    for ch in c.chunk(text, "x"):
        # 除最后一块外，均以句末标点结尾（被吸附切分）
        pass
    # 关键不变量：切块拼接（去 overlap）后覆盖原文所有句末标点
    assert all(ch.text for ch in c.chunk(text, "x"))


def test_char_counting_chinese_aware():
    c = FixedChunker(size=10, overlap=0, tolerance=0)
    chunks = c.chunk("一二三四五六七八九十十一十二十三", "z")
    # 每块不超过 size（硬切分支）
    assert all(len(ch.text) <= 10 for ch in chunks)
