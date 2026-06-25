import json
from pathlib import Path
import numpy as np
from app.services.cloud_embedding_service import MockCloudEmbeddingService

from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.centroids import build_centroids, load_centroids
from tests.fixpaths import FIXTURE_MERGED


def test_build_and_load_centroids_roundtrip(tmp_path):
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(dimension=1024)
    build_centroids(m, FIXTURE_MERGED, embedder, tmp_path)

    assert (tmp_path / "kb_centroids.npy").exists()
    assert (tmp_path / "kb_names.json").exists()

    mat, names = load_centroids(tmp_path)
    # 每个 KB 一行
    assert mat.shape[0] == len(names)
    assert mat.shape[1] == 1024
    assert set(names) == {f.kb for f in m.files}


def test_centroids_are_deterministic(tmp_path):
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    build_centroids(m, FIXTURE_MERGED, MockCloudEmbeddingService(1024), tmp_path / "a")
    build_centroids(m, FIXTURE_MERGED, MockCloudEmbeddingService(1024), tmp_path / "b")
    mat_a, names_a = load_centroids(tmp_path / "a")
    mat_b, names_b = load_centroids(tmp_path / "b")
    assert names_a == names_b
    assert np.allclose(mat_a, mat_b)
