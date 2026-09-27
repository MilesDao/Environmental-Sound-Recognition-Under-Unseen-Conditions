import math

import pytest
import torch

from esr.corruptions import (
    _band_pass, _fft_convolve, _fft_size, CORRUPTIONS, QUANT_BITS, SEEN, SNR_BAND_HZ, SPEED_FACTOR, UNSEEN,
    add_noise_at_snr, apply_corruption, colored_noise, random_train_corruption,
)

SR = 16000


def _tone(freq, n=SR, batch=2):
    t = torch.arange(n) / SR
    return (0.5 * torch.sin(2 * math.pi * freq * t)).repeat(batch, 1)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _db(x):
    return 10 * torch.log10(x.pow(2).mean(-1))


def test_registry_matches_spec():
    assert SEEN == ("white_noise", "reverb")
    assert UNSEEN == ("brown_noise", "telephone", "clipping", "speed", "quantize")
    assert set(CORRUPTIONS) == set(SEEN + UNSEEN)


@pytest.mark.parametrize("name", ["white_noise", "reverb", "brown_noise", "telephone", "clipping", "speed", "quantize"])
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_every_corruption_keeps_shape_and_is_finite(name, severity):
    x = _tone(440) + _tone(60)  # 60 Hz lies outside every telephone band, so every corruption must change x
    out = apply_corruption(x, name, severity, _gen())
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert not torch.allclose(out, x)


def test_unknown_corruption_raises():
    with pytest.raises(KeyError):
        apply_corruption(_tone(440), "nope", 1, _gen())


def test_white_noise_hits_target_snr():
    x = _tone(440)
    out = apply_corruption(x, "white_noise", 2, _gen())  # 10 dB
    snr = _db(x) - _db(out - x)
    assert torch.allclose(snr, torch.full_like(snr, 10.0), atol=0.3)


def test_snr_is_measured_on_audible_part_of_zero_padded_clip():
    x = torch.zeros(1, SR)
    x[:, : SR // 10] = _tone(440, n=SR // 10, batch=1)  # 0.1 s of tone, then zero padding
    out = apply_corruption(x, "white_noise", 2, _gen())  # 10 dB
    audible = slice(0, SR // 10)
    snr = _db(x[:, audible]) - _db((out - x)[:, audible])
    assert snr.item() == pytest.approx(10.0, abs=0.6)


def test_brown_noise_is_low_frequency_heavy():
    n = colored_noise((1, SR), 2.0, _gen())
    spec = torch.fft.rfft(n).abs().pow(2)[0]
    assert spec[1:200].sum() > 10 * spec[4000:].sum()


def test_noise_on_silent_clip_is_finite():
    out = apply_corruption(torch.zeros(2, SR), "white_noise", 3, _gen())
    assert torch.isfinite(out).all() and out.abs().max() > 0
    out = add_noise_at_snr(torch.zeros(1, SR), torch.zeros(1, SR), 0.0)
    assert torch.isfinite(out).all()


def test_reverb_keeps_loudness_and_silence_stays_silent():
    x = _tone(440)
    out = apply_corruption(x, "reverb", 3, _gen())
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), x.pow(2).mean(-1).sqrt(), rtol=1e-3)
    assert torch.isfinite(apply_corruption(torch.zeros(1, SR), "reverb", 1, _gen())).all()


def test_telephone_removes_out_of_band_tone():
    low, mid = _tone(100), _tone(1000)
    assert _db(apply_corruption(low, "telephone", 2, _gen())).max() < _db(low).max() - 30
    assert torch.allclose(_db(apply_corruption(mid, "telephone", 2, _gen())), _db(mid), atol=0.5)


def test_clipping_saturates_and_keeps_peak():
    x = _tone(440)
    out = apply_corruption(x, "clipping", 3, _gen())
    assert torch.allclose(out.abs().amax(-1), x.abs().amax(-1), atol=1e-5)
    assert (out.abs() > 0.99 * x.abs().amax()).float().mean() > 0.5


def test_same_seed_same_output():
    x = _tone(440)
    for name in CORRUPTIONS:
        assert torch.equal(apply_corruption(x, name, 2, _gen(7)), apply_corruption(x, name, 2, _gen(7)))


def test_random_train_corruption_probability_extremes():
    x = _tone(440, batch=6)
    assert torch.equal(random_train_corruption(x, 0.0, _gen()), x)
    out = random_train_corruption(x, 1.0, _gen())
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert all(not torch.allclose(out[i], x[i]) for i in range(6))


def test_fft_size_is_one_power_of_two_for_all_training_rir_lengths():
    # Random RT60 (0.2-0.8 s) must not create a new cuFFT plan per clip (CUFFT_INTERNAL_ERROR on Kaggle).
    sizes = {_fft_size(160000 + int(rt60 * SR) - 1) for rt60 in (0.2, 0.37, 0.55, 0.8, 1.0)}
    assert len(sizes) == 1
    (n,) = sizes
    assert n & (n - 1) == 0 and n >= 160000 + SR - 1


def test_fft_convolve_matches_direct_convolution():
    import numpy as np

    g = _gen(3)
    x, h = torch.randn(2, 1000, generator=g), torch.randn(2, 37, generator=g)
    expected = np.stack([np.convolve(x[i].numpy(), h[i].numpy())[:1000] for i in range(2)])
    assert np.allclose(_fft_convolve(x, h).numpy(), expected, atol=1e-4)


@pytest.mark.parametrize("name", ["white_noise", "reverb", "brown_noise", "telephone", "clipping", "speed", "quantize"])
def test_every_corruption_is_finite_on_silence(name):
    out = apply_corruption(torch.zeros(2, SR), name, 3, _gen())
    assert out.shape == (2, SR) and torch.isfinite(out).all()


def test_brown_noise_snr_is_measured_in_the_analysis_band():
    # v1 measured broadband RMS: 99.9 % of brown noise lies below 50 Hz, so "0 dB" was ~+29 dB in-band.
    x = _tone(440)
    out = apply_corruption(x, "brown_noise", 2, _gen())  # 10 dB inside 50-8000 Hz
    snr = _db(_band_pass(x, SNR_BAND_HZ, SR)) - _db(_band_pass(out - x, SNR_BAND_HZ, SR))
    assert torch.allclose(snr, torch.full_like(snr, 10.0), atol=0.5)


def test_added_noise_has_no_energy_outside_the_analysis_band():
    x = _tone(440)
    out = apply_corruption(x, "brown_noise", 3, _gen())
    spec = torch.fft.rfft(out - x).abs().pow(2)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / SR)
    outside = spec[..., freqs < SNR_BAND_HZ[0]].sum(-1)
    assert (outside < 1e-6 * spec.sum(-1)).all()


def test_speed_shifts_pitch_up_and_keeps_length():
    x = _tone(1000)
    out = apply_corruption(x, "speed", 3, _gen())
    peak_hz = torch.fft.rfft(out).abs().argmax(-1).float() * SR / x.size(-1)
    assert out.shape == x.shape
    assert torch.allclose(peak_hz, torch.full_like(peak_hz, 1000 * SPEED_FACTOR[3]), atol=10)
    assert out[:, -int(SR * 0.2):].abs().max() == 0  # the shortened clip is zero-padded at the end


@pytest.mark.parametrize("severity", [1, 2, 3])
def test_quantize_levels_and_error(severity):
    x = _tone(440, batch=1)
    out = apply_corruption(x, "quantize", severity, _gen())
    step = x.abs().max() / 2 ** (QUANT_BITS[severity] - 1)
    assert out.unique().numel() <= 2 ** QUANT_BITS[severity] + 1
    assert (out - x).abs().max() <= step / 2 + 1e-6


def test_random_train_corruption_draws_from_given_ranges(monkeypatch):
    import esr.corruptions as c

    seen = {"snr": [], "rt60": []}
    monkeypatch.setattr(c, "add_noise_at_snr", lambda x, n, snr, sr=16000: seen["snr"].append(snr) or x)
    monkeypatch.setattr(c, "apply_reverb", lambda x, rt60, sr, g: seen["rt60"].append(rt60) or x)
    c.random_train_corruption(_tone(440, batch=40), 1.0, _gen(), snr_db=(0.0, 1.0), rt60_s=(0.9, 1.0))
    assert seen["snr"] and seen["rt60"]
    assert all(0.0 <= s <= 1.0 for s in seen["snr"]) and all(0.9 <= r <= 1.0 for r in seen["rt60"])
