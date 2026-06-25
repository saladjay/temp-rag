"""入库 HTTP 接口 `/ingest` 集成测试。"""
import pytest
from app.main import create_app
from starlette.testclient import TestClient


def test_ingest_endpoint_shape(monkeypatch):
    """桩件替换 run_ingest / _make_components，断言返回 {kb, chunks} 结构。"""
    from app.ingest import pipeline
    monkeypatch.setattr(pipeline, "run_ingest", lambda *a, **k: 3)
    monkeypatch.setattr(pipeline, "_make_components", lambda: (None, None, None, None))
    app = create_app()
    with TestClient(app) as tc:
        r = tc.post(
            "/api/v1/ingest",
            data={"kb": "faq"},
            files={"file": ("a.txt", "内容".encode("utf-8"), "text/plain")},
        )
    assert r.status_code == 200
    assert r.json() == {"kb": "faq", "chunks": 3}
