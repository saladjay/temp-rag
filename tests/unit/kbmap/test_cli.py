import os
import subprocess
import sys
from pathlib import Path
import yaml

from tests.fixpaths import FIXTURE_MERGED

_MOCK_ENV = {**os.environ, "KBMAP_EMBEDDER": "mock"}


def _run(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "app.kbmap", *args],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )


def test_cli_scan_writes_manifest(tmp_path):
    out = tmp_path / "kb_manifest.yaml"
    r = _run("scan", "--root", str(FIXTURE_MERGED), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["embedding_model"] == "bge-m3"
    assert len(data["files"]) > 0
    # 每条 file 的 kb 在 kb_defines
    for f in data["files"]:
        assert f["kb"] in data["kb_defines"]


def test_cli_verify_writes_report(tmp_path):
    manifest = tmp_path / "kb_manifest.yaml"
    report = tmp_path / "kb_verify_report.md"
    _run("scan", "--root", str(FIXTURE_MERGED), "--out", str(manifest))
    # verify 用 mock embedder（通过 KBMAP_EMBEDDER=mock 环境变量切换）
    r = _run("verify", "--manifest", str(manifest),
             "--root", str(FIXTURE_MERGED), "--out", str(report), env=_MOCK_ENV)
    assert r.returncode == 0, r.stderr
    assert report.exists()
    txt = report.read_text(encoding="utf-8")
    assert "类内凝聚度" in txt


def test_cli_build_centroids_and_classify(tmp_path):
    manifest = tmp_path / "kb_manifest.yaml"
    cents_dir = tmp_path / "cents"
    _run("scan", "--root", str(FIXTURE_MERGED), "--out", str(manifest))
    r = _run("build-centroids", "--manifest", str(manifest),
             "--root", str(FIXTURE_MERGED), "--out-dir", str(cents_dir), env=_MOCK_ENV)
    assert r.returncode == 0, r.stderr
    assert (cents_dir / "kb_centroids.npy").exists()

    r2 = _run("classify", "桥梁工程研究", "--centroids-dir", str(cents_dir), env=_MOCK_ENV)
    assert r2.returncode == 0, r2.stderr
    # 输出含至少一个 KB 名
    assert "kb_" in r2.stdout
