"""Experiment configuration and Kaggle dataset discovery."""
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    data_root: str = ""
    out_dir: str = "/kaggle/working/outputs"
    experiment: str = "effnet_robust"
    model: str = "efficientnet_b0"  # "simple_cnn" or any timm model name
    pretrained: bool = True
    # audio / features
    sample_rate: int = 16000
    clip_seconds: float = 10.0
    n_mels: int = 128
    n_fft: int = 1024
    win_length: int = 400
    hop_length: int = 160
    f_min: float = 50.0
    f_max: float = 8000.0
    # optimisation
    batch_size: int = 32
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    # augmentation / robustness
    balanced_sampling: bool = True
    spec_augment: bool = True
    mixup_alpha: float = 0.5  # 0 disables mixup
    freq_mixstyle_p: float = 0.0
    corruption_aug_p: float = 0.0
    train_snr_db: tuple = (5.0, 30.0)  # white-noise SNR range of the seen-corruption augmentation
    train_rt60_s: tuple = (0.2, 0.8)  # reverb RT60 range of the seen-corruption augmentation
    eq_aug_p: float = 0.0  # random EQ (FilterAugment-style) per clip
    eq_max_db: float = 12.0
    # evaluation / bookkeeping
    severities: tuple = (1, 2, 3)
    max_train_clips: int = 0  # 0 = use all
    max_eval_clips: int = 0  # 0 = use all
    resume_from: str = ""

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severities"] = list(self.severities)
        return d


_EFFNET = dict(model="efficientnet_b0", pretrained=True, lr=5e-4)

EXPERIMENTS = {
    "cnn_baseline": dict(model="simple_cnn", pretrained=False, lr=1e-3),
    "effnet_standard": dict(_EFFNET),
    "effnet_robust": dict(_EFFNET, freq_mixstyle_p=0.7, corruption_aug_p=0.5),  # Experiment 1
    # v2: + random EQ, and seen-corruption ranges widened to cover severity 3 (0 dB SNR, RT60 1.0 s)
    "effnet_robust_v2": dict(_EFFNET, freq_mixstyle_p=0.7, corruption_aug_p=0.5, eq_aug_p=0.5,
                             train_snr_db=(0.0, 30.0), train_rt60_s=(0.2, 1.0)),
    # ablations: exactly one robustness component each (v1 ranges)
    "effnet_mixstyle": dict(_EFFNET, freq_mixstyle_p=0.7),
    "effnet_corrupt": dict(_EFFNET, corruption_aug_p=0.5),
    "effnet_eq": dict(_EFFNET, eq_aug_p=0.5),
}


def make_config(experiment: str, **overrides) -> Config:
    if experiment not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {experiment!r}; choose one of {sorted(EXPERIMENTS)}")
    kw = dict(EXPERIMENTS[experiment])
    kw.update(overrides)
    return Config(experiment=experiment, **kw)


def find_data_root(search_root: str = "/kaggle/input", max_depth: int = 5) -> str:
    """Return the directory containing FSD50K.ground_truth/, searching shallow paths first."""
    base = Path(search_root)
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + ["FSD50K.ground_truth", "vocabulary.csv"])
        hits = sorted(base.glob(pattern))
        if hits:
            return str(hits[0].parent.parent)
    raise FileNotFoundError(
        f"No FSD50K.ground_truth/vocabulary.csv under {search_root}. "
        "In Kaggle: Add Input -> search 'yousirui1/fsd50k' -> Add."
    )
