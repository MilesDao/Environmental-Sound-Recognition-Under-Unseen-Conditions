"""Training-time augmentations on waveforms and spectrograms."""
import torch


def mixup(wav, y, alpha, rng):
    if alpha <= 0:
        return wav, y
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(wav.size(0), device=wav.device)
    return lam * wav + (1 - lam) * wav[perm], lam * y + (1 - lam) * y[perm]


def _mask_along(x, max_width, axis):
    b, size = x.size(0), x.size(axis)
    max_width = min(max_width, size // 4)
    if max_width < 1:
        return x
    width = torch.randint(0, max_width + 1, (b,), device=x.device)
    start = (torch.rand(b, device=x.device) * (size - width + 1)).long()
    idx = torch.arange(size, device=x.device)
    mask = (idx[None] >= start[:, None]) & (idx[None] < (start + width)[:, None])
    shape = [b, 1, 1, 1]
    shape[axis] = size
    return x.masked_fill(mask.view(shape), 0.0)


def spec_augment(x, freq_width=24, time_width=100, n_masks=2):
    for _ in range(n_masks):
        x = _mask_along(x, freq_width, axis=2)
        x = _mask_along(x, time_width, axis=3)
    return x


def freq_mixstyle(x, p, alpha=0.3, eps=1e-6):
    """Frequency-wise MixStyle (Schmid et al., 2022): mix per-frequency mean/std across the batch."""
    if p <= 0 or torch.rand(1).item() > p:
        return x
    b = x.size(0)
    mu = x.mean(dim=(1, 3), keepdim=True)
    sig = (x.var(dim=(1, 3), keepdim=True) + eps).sqrt()
    x_norm = (x - mu) / sig
    lam = torch.distributions.Beta(alpha, alpha).sample((b, 1, 1, 1)).to(x.device, x.dtype)
    perm = torch.randperm(b, device=x.device)
    mu_mix = lam * mu + (1 - lam) * mu[perm]
    sig_mix = lam * sig + (1 - lam) * sig[perm]
    return x_norm * sig_mix + mu_mix


def make_spec_transform(cfg):
    fns = []
    if cfg.freq_mixstyle_p > 0:
        fns.append(lambda x: freq_mixstyle(x, cfg.freq_mixstyle_p))
    if cfg.spec_augment:
        fns.append(spec_augment)
    if not fns:
        return None

    def transform(x):
        for fn in fns:
            x = fn(x)
        return x

    return transform
