import torch

from esr.config import make_config
from esr.features import LogMel
from esr.models import AttentionPool, AudioModel, SimpleCNN, TimmCNN, build_model


def _cfg(**kw):
    return make_config("cnn_baseline", clip_seconds=1.0, n_mels=64, **kw)


def test_logmel_shape():
    front = LogMel.from_config(_cfg())
    x = front(torch.randn(2, 16000) * 0.1)
    assert x.shape == (2, 1, 64, 101)
    assert torch.isfinite(x).all()


def test_logmel_silence_is_finite():
    assert torch.isfinite(LogMel.from_config(_cfg())(torch.zeros(1, 16000))).all()


def test_logmel_set_stats_normalizes_and_is_saved():
    front = LogMel.from_config(_cfg())
    raw = front(torch.randn(4, 16000) * 0.1)
    front.set_stats(raw.mean(), raw.std())
    x = front(torch.randn(4, 16000) * 0.1)
    assert abs(x.mean().item()) < 0.5
    sd = front.state_dict()
    assert "mean" in sd and "std" in sd and sd["std"].item() > 0


def test_attention_pool_shape():
    assert AttentionPool(32, 5)(torch.randn(3, 32, 7)).shape == (3, 5)


def test_simple_cnn_logits_shape():
    assert SimpleCNN(5)(torch.randn(2, 1, 64, 101)).shape == (2, 5)


def test_timm_efficientnet_logits_shape_without_download():
    net = TimmCNN("efficientnet_b0", 5, pretrained=False)
    assert net(torch.randn(2, 1, 64, 101)).shape == (2, 5)


def test_build_model_end_to_end_from_waveform():
    for exp in ("cnn_baseline", "effnet_standard"):
        cfg = make_config(exp, clip_seconds=1.0, n_mels=64)
        model = build_model(cfg, n_classes=5, pretrained=False)
        assert isinstance(model, AudioModel)
        logits = model(torch.randn(2, 16000) * 0.1)
        assert logits.shape == (2, 5)


def test_audio_model_applies_spec_transform():
    model = build_model(_cfg(), 5)
    seen = {}

    def spy(x):
        seen["shape"] = tuple(x.shape)
        return x

    model(torch.randn(2, 16000), spec_transform=spy)
    assert seen["shape"] == (2, 1, 64, 101)
