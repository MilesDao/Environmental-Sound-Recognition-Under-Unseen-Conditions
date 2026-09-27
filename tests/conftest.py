import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from esr.config import make_config

CLASSES = ["Bark", "Siren", "Guitar", "Rain", "Speech"]
SR = 16000


def _tone(freq, seconds):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    """Synthetic FSD50K with the real folder names and CSV schemas.

    Mirrors Kaggle's nesting: <input>/fsd50k/fsd50k/FSD50K.*
    dev: 12 train + 4 val clips; val has no clip of class 1 ("Siren") on purpose.
    eval: 8 clips covering all 5 classes. Durations include 0.3 s and 2.5 s.
    """
    root = tmp_path_factory.mktemp("kaggle_input") / "fsd50k" / "fsd50k"
    gt = root / "FSD50K.ground_truth"
    dev_dir = root / "FSD50K.dev_audio_16k"
    eval_dir = root / "FSD50K.eval_audio_16k"
    for d in (gt, dev_dir, eval_dir):
        d.mkdir(parents=True)

    pd.DataFrame({"i": range(5), "label": CLASSES, "mid": [f"/m/{i}" for i in range(5)]}).to_csv(
        gt / "vocabulary.csv", header=False, index=False
    )

    durations = [0.3, 0.8, 1.5, 2.5]
    dev_rows = []
    for k in range(16):
        c1, c2 = k % 5, (k + 2) % 5
        split = "val" if k >= 12 else "train"  # val clips k=12..15 have c1 in {2,3,4,0}
        multi = k % 2 == 0 and not (split == "val" and c2 == 1)
        wav = _tone(200 + 150 * c1, durations[k % 4])
        labels = CLASSES[c1]
        if multi:
            wav = wav + _tone(200 + 150 * c2, durations[k % 4])
            labels = f"{CLASSES[c1]},{CLASSES[c2]}"
        sf.write(dev_dir / f"{1000 + k}.wav", wav, SR, subtype="PCM_16")
        dev_rows.append((1000 + k, labels, "/m/x", split))
    pd.DataFrame(dev_rows, columns=["fname", "labels", "mids", "split"]).to_csv(gt / "dev.csv", index=False)

    eval_rows = []
    for k in range(8):
        c = k % 5
        sf.write(eval_dir / f"{2000 + k}.wav", _tone(200 + 150 * c, durations[k % 4]), SR, subtype="PCM_16")
        eval_rows.append((2000 + k, CLASSES[c], "/m/x"))
    pd.DataFrame(eval_rows, columns=["fname", "labels", "mids"]).to_csv(gt / "eval.csv", index=False)
    return root


@pytest.fixture
def tiny_cfg(fake_root, tmp_path):
    def _make(experiment="cnn_baseline", **kw):
        base = dict(
            data_root=str(fake_root), out_dir=str(tmp_path / "out"), clip_seconds=1.0, n_mels=64,
            batch_size=4, epochs=1, num_workers=0, amp=False, pretrained=False, warmup_epochs=0,
        )
        base.update(kw)
        return make_config(experiment, **base)
    return _make
