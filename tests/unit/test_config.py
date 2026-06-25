from app.config import settings


def test_new_stability_defaults():
    assert settings.gen_temperature == 0.0
    assert settings.milvus_ef == 128
    assert settings.stability_cache_enabled is True
    assert settings.stability_semantic_threshold == 0.98


def test_milvus_collection_prefix():
    assert settings.milvus_collection_prefix == "kb_"
