from app.config import settings


def test_kbmap_defaults():
    assert settings.kbmap_embed_batch_size == 32
    assert settings.kbmap_chunk_preview_chars == 500
    assert settings.kbmap_outlier_margin == 0.05
    assert settings.kbmap_cohesion_min == 0.5
    assert settings.kbmap_separation_max == 0.85
    assert settings.kbmap_merged_root == ""
