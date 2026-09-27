"""Log-mel spectrogram front end."""
import torch
import torchaudio
from torch import nn


class LogMel(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128,
                 f_min=50.0, f_max=8000.0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels, power=2.0,
        )
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.n_mels, cfg.f_min, cfg.f_max)

    @torch.no_grad()
    def set_stats(self, mean, std):
        self.mean.fill_(float(mean))
        self.std.fill_(max(float(std), 1e-5))

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = torch.log(self.mel(wav.float()) + 1e-6)
            x = (x - self.mean) / self.std
        return x.unsqueeze(1)
