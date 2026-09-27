"""Acoustic corruptions that simulate unseen recording conditions.

SEEN corruptions may be used as training augmentation; UNSEEN ones are test-only.
All functions take a CPU torch.Generator so evaluation noise is identical for every model.
"""
import torch
import torch.nn.functional as F

PROTOCOL = 2  # bump whenever a test-time corruption changes; compare never mixes protocols
SNR_DB = {1: 20.0, 2: 10.0, 3: 0.0}
SNR_BAND_HZ = (50.0, 8000.0)  # = LogMel f_min..f_max: SNR counts only the noise the model can hear
RT60_S = {1: 0.3, 2: 0.6, 3: 1.0}
BAND_HZ = {1: (100.0, 5000.0), 2: (300.0, 3400.0), 3: (500.0, 2000.0)}
CLIP_FRAC = {1: 0.5, 2: 0.2, 3: 0.05}
SPEED_FACTOR = {1: 1.05, 2: 1.15, 3: 1.3}  # playback speed-up: pitch and tempo rise together
QUANT_BITS = {1: 8, 2: 6, 3: 4}  # uniform quantisation relative to the clip's peak


def _rms(x):
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt()


def _active_rms(x):
    """RMS over non-zero samples, so zero padding of short clips does not dilute the signal level."""
    active = (x != 0).sum(dim=-1, keepdim=True).clamp_min(1)
    return (x.pow(2).sum(dim=-1, keepdim=True) / active).sqrt()


def _band_pass(x, band, sample_rate):
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / sample_rate).to(x.device)
    return torch.fft.irfft(spec * ((freqs >= band[0]) & (freqs <= band[1])), n=x.size(-1), dim=-1)


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


def add_noise_at_snr(wav, noise, snr_db, sample_rate=16000):
    # The added noise is restricted to the log-mel band (SNR_BAND_HZ) and the SNR is measured there too:
    # non-padded samples for the signal, band-passed noise for the noise. v1 added broadband noise while
    # measuring broadband RMS, so 99.9 % of brown noise sat below 50 Hz and "0 dB" was ~+29 dB in-band;
    # even measuring the band while still adding broadband noise leaked 24-30 dB of out-of-band energy
    # into the log-mel bands through Hann-window sidelobes, pushing low mel bands 7-11 dB hot.
    # Silent clips get noise relative to a -60 dBFS floor instead of producing NaN/no-op.
    signal_rms = _active_rms(wav).clamp_min(1e-3)
    nb = _band_pass(noise, SNR_BAND_HZ, sample_rate)
    return wav + nb * (signal_rms / (10 ** (snr_db / 20)) / _rms(nb).clamp_min(1e-8))


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
    return add_noise_at_snr(wav, colored_noise(wav.shape, 0.0, gen, wav.device), SNR_DB[severity], sample_rate)


def brown_noise(wav, severity, gen, sample_rate=16000):
    return add_noise_at_snr(wav, colored_noise(wav.shape, 2.0, gen, wav.device), SNR_DB[severity], sample_rate)


def reverb(wav, severity, gen, sample_rate=16000):
    return apply_reverb(wav, RT60_S[severity], sample_rate, gen)


def telephone(wav, severity, gen, sample_rate=16000):
    return _band_pass(wav, BAND_HZ[severity], sample_rate)


def clipping(wav, severity, gen, sample_rate=16000):
    frac = CLIP_FRAC[severity]
    t = wav.abs().amax(dim=-1, keepdim=True) * frac
    return torch.maximum(torch.minimum(wav, t), -t) / frac


def speed(wav, severity, gen, sample_rate=16000):
    n = wav.size(-1)
    m = max(1, round(n / SPEED_FACTOR[severity]))
    fast = F.interpolate(wav[:, None], size=m, mode="linear", align_corners=False)[:, 0]
    return F.pad(fast, (0, n - m))


def quantize(wav, severity, gen, sample_rate=16000):
    step = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 2 ** (QUANT_BITS[severity] - 1)
    return torch.round(wav / step) * step


CORRUPTIONS = {
    "white_noise": white_noise,
    "reverb": reverb,
    "brown_noise": brown_noise,
    "telephone": telephone,
    "clipping": clipping,
    "speed": speed,
    "quantize": quantize,
}
SEEN = ("white_noise", "reverb")
UNSEEN = ("brown_noise", "telephone", "clipping", "speed", "quantize")


def apply_corruption(wav, name, severity, gen, sample_rate=16000):
    if name not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption {name!r}; choose one of {sorted(CORRUPTIONS)}")
    return CORRUPTIONS[name](wav, severity, gen, sample_rate)


def random_train_corruption(wav, p, gen, sample_rate=16000, snr_db=(5.0, 30.0), rt60_s=(0.2, 0.8)):
    """Seen-family augmentation with continuous parameters, applied per clip with probability p."""
    if p <= 0:
        return wav
    out = wav.clone()
    for i in range(wav.size(0)):
        if torch.rand(1, generator=gen).item() >= p:
            continue
        x = wav[i:i + 1]
        if torch.rand(1, generator=gen).item() < 0.5:
            snr = snr_db[0] + (snr_db[1] - snr_db[0]) * torch.rand(1, generator=gen).item()
            out[i:i + 1] = add_noise_at_snr(x, colored_noise(x.shape, 0.0, gen, x.device), snr, sample_rate)
        else:
            rt60 = rt60_s[0] + (rt60_s[1] - rt60_s[0]) * torch.rand(1, generator=gen).item()
            out[i:i + 1] = apply_reverb(x, rt60, sample_rate, gen)
    return out
