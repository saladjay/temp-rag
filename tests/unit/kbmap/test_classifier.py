from pathlib import Path
from app.services.cloud_embedding_service import MockCloudEmbeddingService

from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.centroids import build_centroids
from app.kbmap.classifier import KBClassifier
from tests.fixpaths import FIXTURE_MERGED


def _build(tmp_path):
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(1024)
    build_centroids(m, FIXTURE_MERGED, embedder, tmp_path)
    return m, embedder


def test_classify_returns_top_k_with_scores(tmp_path):
    m, embedder = _build(tmp_path)
    clf = KBClassifier(tmp_path, embedder)
    res = clf.classify("桥梁工程研究", top_k=3)
    assert len(res) <= 3
    assert all(isinstance(k, str) for k, _ in res)
    assert all(isinstance(s, float) for _, s in res)
    # 分数按降序
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


def test_classify_is_deterministic(tmp_path):
    m, embedder = _build(tmp_path)
    clf1 = KBClassifier(tmp_path, MockCloudEmbeddingService(1024))
    clf2 = KBClassifier(tmp_path, MockCloudEmbeddingService(1024))
    r1 = clf1.classify("同一问题")
    r2 = clf2.classify("同一问题")
    assert r1 == r2  # 同进程 mock 确定性


def test_assign_file_returns_ranking(tmp_path):
    m, embedder = _build(tmp_path)
    clf = KBClassifier(tmp_path, embedder)
    # 任挑一个 fixture 文件
    entry = m.files[0]
    res = clf.assign_file(entry.path, FIXTURE_MERGED, top_k=2)
    assert len(res) <= 2
    # 分数降序
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
