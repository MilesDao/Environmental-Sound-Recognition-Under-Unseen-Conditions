"""Acoustic corruptions that simulate unseen recording conditions.

SEEN corruptions may be used as training augmentation; UNSEEN ones are test-only.
All functions take a CPU torch.Generator so evaluation noise is identical for every model.
"""
import torch

SNR_DB = {1: 20.0, 2: 10.0, 3: 0.0}
RT60_S = {1: 0.3, 2: 0.6, 3: 1.0}
BAND_HZ = {1: (100.0, 5000.0), 2: (300.0, 3400.0), 3: (500.0, 2000.0)}
CLIP_FRAC = {1: 0.5, 2: 0.2, 3: 0.05}


def _rms(x):
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt()


def _active_rms(x):
    """RMS over non-zero samples, so zero padding of short clips does not dilute the signal level."""
    active = (x != 0).sum(dim=-1, keepdim=True).clamp_min(1)
    return (x.pow(2).sum(dim=-1, keepdim=True) / active).sqrt()


def colored_noise(shape, exponent, gen, device="cpu"):
    """Unit-RMS noise with power spectrum ~ 1/f**exponent (0 white, 1 pink, 2 brown)."""
    noise = torch.randn(tuple(shape), generator=gen)
    if exponent != 0:
        spec = torch.fft.rfft(noise, dim=-1)
        f = torch.arange(spec.size(-1), dtype=torch.float32)
        f[0] = 1.0
        noise = torch.fft.irfft(spec / f.pow(exponent / 2), n=shape[-1], dim=-1)
    noise = noise - noise.mean(dim=-1, keepdim=True)
    return (noise / _rms(noise).clamp_min(1e-8)).to(device)


def add_noise_at_snr(wav, noise, snr_db):
    # SNR is defined on the audible (non-padded) part; silent clips get noise relative to a
    # -60 dBFS floor instead of producing NaN/no-op.
    signal_rms = _active_rms(wav).clamp_min(1e-3)
    scale = signal_rms / (10 ** (snr_db / 20)) / _rms(noise).clamp_min(1e-8)
    return wav + noise * scale


def _fft_size(n):
    """Next power of two >= n. Random RIR lengths would otherwise make a new cuFFT plan per clip, which
    fills the plan cache and raises CUFFT_INTERNAL_ERROR on GPU; one fixed size means one cached plan."""
    return 1 << (n - 1).bit_length()


def _fft_convolve(x, h):
    n = _fft_size(x.size(-1) + h.size(-1) - 1)  # extra zero padding does not change the linear convolution
    y = torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(h, n=n), n=n)
    return y[..., : x.size(-1)]


def _synthetic_rir(rt60, sample_rate, gen, n):
    length = max(int(rt60 * sample_rate), 16)
    t = torch.arange(length, dtype=torch.float32) / sample_rate
    rir = torch.randn((n, length), generator=gen) * torch.exp(-6.9078 * t / rt60)  # -60 dB at t = rt60
    rir[:, 0] = 1.0  # direct path
    return rir / rir.norm(dim=-1, keepdim=True)


def apply_reverb(wav, rt60, sample_rate, gen):
    rir = _synthetic_rir(rt60, sample_rate, gen, wav.size(0)).to(wav.device)
    wet = _fft_convolve(wav, rir)
    return wet * (_rms(wav) / _rms(wet).clamp_min(1e-8))


def white_noise(wav, severity, gen, sample_rate=16000):
    return add_noise_at_snr(wav, colored_noise(wav.shape, 0.0, gen, wav.device), SNR_DB[severity])


def brown_noise(wav, severity, gen, sample_rate=16000):
    return add_noise_at_snr(wav, colored_noise(wav.shape, 2.0, gen, wav.device), SNR_DB[severity])


def reverb(wav, severity, gen, sample_rate=16000):
    return apply_reverb(wav, RT60_S[severity], sample_rate, gen)


def telephone(wav, severity, gen, sample_rate=16000):
    lo, hi = BAND_HZ[severity]
    spec = torch.fft.rfft(wav, dim=-1)
    freqs = torch.fft.rfftfreq(wav.size(-1), d=1.0 / sample_rate).to(wav.device)
    spec = spec * ((freqs >= lo) & (freqs <= hi))
    return torch.fft.irfft(spec, n=wav.size(-1), dim=-1)


def clipping(wav, severity, gen, sample_rate=16000):
    frac = CLIP_FRAC[severity]
    t = wav.abs().amax(dim=-1, keepdim=True) * frac
    return torch.maximum(torch.minimum(wav, t), -t) / frac


CORRUPTIONS = {
    "white_noise": white_noise,
    "reverb": reverb,
    "brown_noise": brown_noise,
    "telephone": telephone,
    "clipping": clipping,
}
SEEN = ("white_noise", "reverb")
UNSEEN = ("brown_noise", "telephone", "clipping")


def apply_corruption(wav, name, severity, gen, sample_rate=16000):
    if name not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption {name!r}; choose one of {sorted(CORRUPTIONS)}")
    return CORRUPTIONS[name](wav, severity, gen, sample_rate)


def random_train_corruption(wav, p, gen, sample_rate=16000):
    """Seen-family augmentation with continuous parameters, applied per clip with probability p."""
    if p <= 0:
        return wav
    out = wav.clone()
    for i in range(wav.size(0)):
        if torch.rand(1, generator=gen).item() >= p:
            continue
        x = wav[i:i + 1]
        if torch.rand(1, generator=gen).item() < 0.5:
            snr = 5.0 + 25.0 * torch.rand(1, generator=gen).item()
            out[i:i + 1] = add_noise_at_snr(x, colored_noise(x.shape, 0.0, gen, x.device), snr)
        else:
            rt60 = 0.2 + 0.6 * torch.rand(1, generator=gen).item()
            out[i:i + 1] = apply_reverb(x, rt60, sample_rate, gen)
    return out
