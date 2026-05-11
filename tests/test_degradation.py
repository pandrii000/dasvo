import numpy as np

from dasvo.degradation import PRESETS, apply_visual_degradation


def test_apply_degradation_preserves_shape_and_none():
    img = np.zeros((64, 48, 3), dtype=np.uint8)
    assert apply_visual_degradation(img, None) is img
    out = apply_visual_degradation(img, "none")
    assert out.shape == img.shape


def test_all_presets_run():
    img = np.random.randint(0, 255, (32, 24, 3), dtype=np.uint8)
    for p in PRESETS:
        out = apply_visual_degradation(img, p)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
