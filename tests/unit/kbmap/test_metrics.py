import math
from app.services.cloud_embedding_service import MockCloudEmbeddingService
from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.metrics import cosine, compute_centroids, verify
from tests.fixpaths import FIXTURE_MERGED


def test_cosine_identical_vectors_is_one():
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6


def test_cosine_orthogonal_is_zero():
    assert abs(cosine([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6


def test_compute_centroids_returns_one_per_kb():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    from app.kbmap.embed import embed_files
    embedder = MockCloudEmbeddingService(dimension=1024)
    p2v = embed_files(m, FIXTURE_MERGED, embedder)
    cents = compute_centroids(m, p2v)
    # 每个 KB 一个质心
    assert set(cents.keys()) == {f.kb for f in m.files}
    for c in cents.values():
        assert len(c) == 1024


def test_verify_returns_report_structure():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(dimension=1024)
    rep = verify(m, FIXTURE_MERGED, embedder)
    # cohesion 对每个 KB 有值
    assert set(rep.cohesion.keys()) == {f.kb for f in m.files}
    # outliers 是列表（可能为空，fixture 小）
    assert isinstance(rep.outliers, list)
