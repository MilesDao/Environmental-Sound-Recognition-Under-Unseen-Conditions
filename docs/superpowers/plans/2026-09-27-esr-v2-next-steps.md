# ESR v2: Protocol Fix, New Unseen Conditions, Robustness v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the findings of Experiment 1 into code and a Kaggle run plan:

- a corrected evaluation protocol (v2);
- two new unseen conditions;
- per-condition reporting;
- a new training augmentation (random EQ) with an `effnet_robust_v2` preset and ablation presets;
- evaluation of an already trained checkpoint attached from `/kaggle/input`.

**Architecture:** All changes stay inside the existing `esr/` package and CLI; there are no new modules.

- **Corruptions:** `esr/corruptions.py` gets band-limited SNR, the `speed` and `quantize` corruptions, and a `PROTOCOL = 2` constant. The constant is written into every `robustness.csv` row, so `compare` never mixes v1 and v2 numbers.
- **Evaluation:** `esr/engine.py` seeds evaluation corruptions per clip, which makes the protocol independent of batch size.
- **Training:** `esr/augment.py` gains `random_eq`. `Config` gains the `eq_aug_p`, `eq_max_db`, `train_snr_db` and `train_rt60_s` fields.
- **Presets:** `EXPERIMENTS` gains `effnet_robust_v2`, `effnet_mixstyle`, `effnet_corrupt` and `effnet_eq`.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.3, timm ≥ 1.0, numpy, pandas, pytest. There are no new dependencies.

**Spec:** `docs/EXPERIMENT.md` §1.4–§1.5 (Experiment 1 analysis and proposed next steps). It builds on `docs/superpowers/specs/2026-09-27-esr-unseen-conditions-design.md`, and the code is described in `docs/ARCHITECHTURE.md`.

## Global Constraints

- **Git:** do **not** run `git add`, `git commit` or `git push`. The user stages and commits. Every task ends by listing the changed files for the user.
- **Where tests run:** in WSL, not on the Windows host, because Windows Application Control blocks torch DLLs there:
  `wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/User/documentaries/USTH_codes/Environmental-Sound-Recognition-Under-Unseen-Conditions && ~/.venvs/esr/bin/python -m pytest -q"`.
  WSL has about 7 GB RAM, so never run EfficientNet at batch > 8 there.
- **Tests must stay offline:** no downloads, `pretrained=False`.
- **No GPU requirement:** every code path must run on CPU (tests) and on CUDA with AMP (Kaggle).
- **FFT sizes stay fixed:** any FFT inside the training loop or evaluation must use a length that is the same for every clip (clip length, or `_fft_size(...)`). Varying sizes caused `CUFFT_INTERNAL_ERROR` on Kaggle.
- **Features stay as they are:** log-mel, n_fft 1024, win 400, hop 160, 128 mels, **50–8000 Hz**, 10 s clips (160,000 samples).
- **Model selection** uses clean val mAP only.
- **No test-corruption leakage:** an UNSEEN corruption function is never called during training.
  - `random_eq` is a *different*, generic family (smooth ±dB curves, never a hard band-stop).
  - Its overlap with `telephone` must be stated in the docs (Task 6).
- **Unchanged v1 defaults:** `effnet_standard`, `cnn_baseline` and `effnet_robust` keep their v1 behaviour. New `Config` fields default to v1 values.
- **Experiment names:** `cnn_baseline`, `effnet_standard`, `effnet_robust`, `effnet_robust_v2`, `effnet_mixstyle`, `effnet_corrupt`, `effnet_eq`.
- **Protocol v2 conditions:**
  - SEEN = `("white_noise", "reverb")`.
  - UNSEEN = `("brown_noise", "telephone", "clipping", "speed", "quantize")`.
  - Severities `1, 2, 3`, which gives 1 + 2×3 + 5×3 = **22 conditions** per model.

## Review Focus

1. **Zero-padded or silent clips under the new corruptions** (FSD50K has 0.3 s clips, so up to 97 % of samples can be zeros). `speed`, `quantize` and band-limited noise must return finite audio → `test_every_corruption_is_finite_on_silence` (Task 1).
2. **A v1 `robustness.csv` (no `protocol` column) attached as a Kaggle input next to v2 results.** `compare` must skip it with a message, never average it in → `test_compare_uses_only_latest_protocol_and_writes_detail` (Task 3).
3. **Evaluating a checkpoint that sits in read-only `/kaggle/input/...`.** Outputs must go to `<out-dir>/<experiment>/`, never next to the checkpoint → `test_run_robustness_from_checkpoint_outside_out_dir` and `test_evaluate_accepts_checkpoint_path` (Task 2).
4. **Two models evaluated with different `--batch-size`.** Corrupted inputs must be identical → `test_predict_array_corruption_does_not_depend_on_batch_size` (Task 2).
5. **Device mismatch in `random_eq` on GPU.** The gain curve is built on CPU and must be moved with `.to(wav.device)` before multiplying. CPU tests cannot catch this, so the reviewer checks that line by eye; the Kaggle `--quick` run in the Run Plan is the real test (Task 4).

---

### Task 1: Protocol v2 corruptions — band-limited SNR, `speed`, `quantize`

**Files:**
- Modify: `esr/corruptions.py`
- Test: `tests/test_corruptions.py`, `tests/test_cli.py` (condition count)

**Interfaces:**
- Consumes: the existing `_rms`, `_active_rms`, `colored_noise`, `apply_corruption`.
- Produces:
  - `PROTOCOL: int = 2`
  - `SNR_BAND_HZ = (50.0, 8000.0)`
  - `SPEED_FACTOR = {1: 1.05, 2: 1.15, 3: 1.3}`
  - `QUANT_BITS = {1: 8, 2: 6, 3: 4}`
  - `_band_pass(x, band, sample_rate) -> Tensor`
  - `add_noise_at_snr(wav, noise, snr_db, sample_rate=16000)`
  - `speed(wav, severity, gen, sample_rate=16000)` and `quantize(wav, severity, gen, sample_rate=16000)`
  - `UNSEEN = ("brown_noise", "telephone", "clipping", "speed", "quantize")`

- [ ] **Step 1: Write the failing tests** (in `tests/test_corruptions.py`).

Change the import block at the top to:

```python
from esr.corruptions import (
    _band_pass, _fft_convolve, _fft_size, CORRUPTIONS, QUANT_BITS, SEEN, SNR_BAND_HZ, SPEED_FACTOR, UNSEEN,
    add_noise_at_snr, apply_corruption, colored_noise, random_train_corruption,
)
```

Replace `test_registry_matches_spec` and the parametrize list of `test_every_corruption_keeps_shape_and_is_finite`:

```python
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
```

Append:

```python
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
```

In `tests/test_cli.py`, add `from esr.corruptions import SEEN, UNSEEN` to the imports. In `test_train_evaluate_compare_end_to_end`, replace `len(pd.read_csv(csv)) == 6` with:

```python
    assert csv.exists() and len(pd.read_csv(csv)) == 1 + len(SEEN) + len(UNSEEN)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/User/documentaries/USTH_codes/Environmental-Sound-Recognition-Under-Unseen-Conditions && ~/.venvs/esr/bin/python -m pytest -q tests/test_corruptions.py"`

Expected: collection error `ImportError: cannot import name '_band_pass'`.

- [ ] **Step 3: Implement it** in `esr/corruptions.py`.

Replace the imports and constants at the top:

```python
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
```

Add the following after `_active_rms`, and replace `add_noise_at_snr`:

```python
def _band_pass(x, band, sample_rate):
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / sample_rate).to(x.device)
    return torch.fft.irfft(spec * ((freqs >= band[0]) & (freqs <= band[1])), n=x.size(-1), dim=-1)


def add_noise_at_snr(wav, noise, snr_db, sample_rate=16000):
    # SNR is defined on the audible part: non-padded samples for the signal, and the log-mel band for the
    # noise (v1 used broadband noise RMS, so 99.9 % of brown noise sat below 50 Hz and "0 dB" was ~+29 dB).
    # Silent clips get noise relative to a -60 dBFS floor instead of producing NaN/no-op.
    signal_rms = _active_rms(wav).clamp_min(1e-3)
    noise_rms = _rms(_band_pass(noise, SNR_BAND_HZ, sample_rate)).clamp_min(1e-8)
    return wav + noise * (signal_rms / (10 ** (snr_db / 20)) / noise_rms)
```

Replace `white_noise`, `brown_noise` and `telephone`, and add `speed` and `quantize` after `clipping`:

```python
def white_noise(wav, severity, gen, sample_rate=16000):
    return add_noise_at_snr(wav, colored_noise(wav.shape, 0.0, gen, wav.device), SNR_DB[severity], sample_rate)


def brown_noise(wav, severity, gen, sample_rate=16000):
    return add_noise_at_snr(wav, colored_noise(wav.shape, 2.0, gen, wav.device), SNR_DB[severity], sample_rate)


def telephone(wav, severity, gen, sample_rate=16000):
    return _band_pass(wav, BAND_HZ[severity], sample_rate)


def speed(wav, severity, gen, sample_rate=16000):
    n = wav.size(-1)
    m = max(1, round(n / SPEED_FACTOR[severity]))
    fast = F.interpolate(wav[:, None], size=m, mode="linear", align_corners=False)[:, 0]
    return F.pad(fast, (0, n - m))


def quantize(wav, severity, gen, sample_rate=16000):
    step = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 2 ** (QUANT_BITS[severity] - 1)
    return torch.round(wav / step) * step
```

Update the registry:

```python
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
```

In `random_train_corruption`, pass the sample rate through on the noise branch:

```python
            out[i:i + 1] = add_noise_at_snr(x, colored_noise(x.shape, 0.0, gen, x.device), snr, sample_rate)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/User/documentaries/USTH_codes/Environmental-Sound-Recognition-Under-Unseen-Conditions && ~/.venvs/esr/bin/python -m pytest -q"`

Expected: all tests pass. `test_white_noise_hits_target_snr` still passes, because white noise keeps 99.4 % of its power in-band, a shift of 0.03 dB.

- [ ] **Step 5: Hand-off.** Report the changed files to the user (`esr/corruptions.py`, `tests/test_corruptions.py`, `tests/test_cli.py`). Suggested message: `feat: protocol v2 corruptions (band-limited SNR, speed, quantize)`. Do not commit.

---

### Task 2: Batch-size-independent eval noise, `protocol` column, evaluate any checkpoint path

**Files:**
- Modify: `esr/engine.py` (`predict_array`), `esr/pipeline.py` (`run_robustness`), `esr/cli.py` (`--checkpoint` help text)
- Test: `tests/test_engine.py`, `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `PROTOCOL` and `apply_corruption` from Task 1.
- Produces:
  - `predict_array(...)` has the same signature. Clip `j` of the whole array is corrupted with the generator seed `seed * 100003 + j`.
  - `run_robustness(cfg, checkpoint="best.pt", waves=None, targets=None)`: `checkpoint` may be a file name inside `<out_dir>/<experiment>/` or an absolute path.
  - `robustness.csv` columns are `["experiment", "condition", "group", "severity", "mAP", "subset", "protocol"]`.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_engine.py`:

```python
def test_predict_array_corruption_does_not_depend_on_batch_size(setup):
    cfg, model, _, wav = setup
    waves = (wav.numpy() * 32767).astype(np.int16)
    dev = torch.device("cpu")
    a = predict_array(model, waves, dev, 3, False, corruption="white_noise", severity=2, seed=1)
    b = predict_array(model, waves, dev, 8, False, corruption="white_noise", severity=2, seed=1)
    assert np.allclose(a, b, atol=1e-5)
```

In `tests/test_pipeline.py`, add `from esr.corruptions import PROTOCOL` to the imports. Change the column assertion in `test_run_robustness_rows_and_groups` to:

```python
    assert list(df.columns) == ["experiment", "condition", "group", "severity", "mAP", "subset", "protocol"]
```

Then append:

```python
def test_run_robustness_from_checkpoint_outside_out_dir(tiny_cfg, tmp_path):
    trained = tiny_cfg(severities=(1,))
    run_training(trained)
    ckpt = Path(trained.out_dir) / trained.experiment / "best.pt"
    fresh = tiny_cfg(severities=(1,), out_dir=str(tmp_path / "elsewhere"))
    df = run_robustness(fresh, checkpoint=str(ckpt))
    assert (tmp_path / "elsewhere" / fresh.experiment / "robustness.csv").exists()
    assert not (ckpt.parent / "robustness.csv").exists()  # never write next to a (read-only) input checkpoint
    assert (df["protocol"] == PROTOCOL).all()
```

Append to `tests/test_cli.py`:

```python
def test_evaluate_accepts_checkpoint_path(fake_root, tmp_path):
    main(["train", "--experiment", "cnn_baseline", *_args(fake_root, tmp_path / "a"), "--epochs", "1"])
    ckpt = tmp_path / "a" / "cnn_baseline" / "best.pt"
    main(["evaluate", "--experiment", "cnn_baseline", *_args(fake_root, tmp_path / "b"), "--severities", "1",
          "--checkpoint", str(ckpt)])
    assert (tmp_path / "b" / "cnn_baseline" / "robustness.csv").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/User/documentaries/USTH_codes/Environmental-Sound-Recognition-Under-Unseen-Conditions && ~/.venvs/esr/bin/python -m pytest -q tests/test_engine.py tests/test_pipeline.py tests/test_cli.py"`

Expected: 4 failures.
- The batch-size test fails on `np.allclose`.
- The columns test fails because `protocol` is missing.
- The two checkpoint-path tests fail with `FileNotFoundError`.

- [ ] **Step 3: Implement it.**

In `esr/engine.py`, replace the corruption block of `predict_array`:

```python
        if corruption is not None:  # one generator per clip: identical noise for any batch_size
            wav = torch.cat([
                apply_corruption(wav[i:i + 1], corruption, severity,
                                 torch.Generator().manual_seed(seed * 100003 + start + i), sample_rate)
                for i in range(wav.size(0))
            ])
```

Also remove the now-unused `b, ` from the loop, so it reads `for start in range(0, len(waves), batch_size):`.

In `esr/pipeline.py`, change the import to `from .corruptions import PROTOCOL, SEEN, UNSEEN`. Replace the start of `run_robustness` up to `device = get_device()`:

```python
    exp_dir = Path(cfg.out_dir) / cfg.experiment
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = exp_dir / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found - call run_training(cfg) first "
                                "(or pass --checkpoint /kaggle/input/.../best.pt).")
    exp_dir.mkdir(parents=True, exist_ok=True)  # results always go to out_dir, even for an input checkpoint
    device = get_device()
```

Then extend the row dict and the column list:

```python
        rows.append({"experiment": cfg.experiment, "condition": name, "group": group,
                     "severity": severity, "mAP": m, "subset": subset, "protocol": PROTOCOL})
        ...
    df = pd.DataFrame(rows, columns=["experiment", "condition", "group", "severity", "mAP", "subset", "protocol"])
```

In `esr/cli.py`, change the `--checkpoint` line:

```python
    p.add_argument("--checkpoint", default="best.pt",
                   help="evaluate only: file name in <out-dir>/<experiment>/ or an absolute path "
                        "(e.g. a best.pt attached from /kaggle/input)")
```

- [ ] **Step 4: Run the whole suite.** Use the command from Task 1, Step 4. Expected: all tests pass.

- [ ] **Step 5: Hand-off.** Changed files: `esr/engine.py`, `esr/pipeline.py`, `esr/cli.py`, `tests/test_engine.py`, `tests/test_pipeline.py`, `tests/test_cli.py`. Suggested message: `feat: per-clip eval seeds, protocol column, evaluate attached checkpoints`.

---

### Task 3: Reporting — latest-protocol filter and per-condition table

**Files:**
- Modify: `esr/pipeline.py` (add `per_condition`), `esr/cli.py` (`compare`)
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: the `protocol` column from Task 2.
- Produces:
  - `per_condition(df) -> DataFrame` with columns `experiment, condition, group, severity, mAP, rel`, where `rel` is corrupted / clean mAP of the same experiment.
  - `compare(inputs, include_subset=False, out=None)` keeps only the highest protocol present. Files without the column count as protocol 1. When `out` is set, it also writes `<out stem>_by_condition.csv` next to `out`.

- [ ] **Step 1: Write the failing tests.**

In `tests/test_pipeline.py`, add `per_condition` to the `from esr.pipeline import ...` line, then append:

```python
def test_per_condition_is_relative_to_own_clean():
    df = pd.DataFrame([
        ("a", "clean", "clean", 0, 0.5), ("a", "telephone", "unseen", 3, 0.2),
        ("b", "clean", "clean", 0, 0.8), ("b", "telephone", "unseen", 3, 0.4),
    ], columns=["experiment", "condition", "group", "severity", "mAP"])
    d = per_condition(df)
    assert "clean" not in set(d["condition"])
    rel = d.set_index("experiment")["rel"]
    assert rel["a"] == pytest.approx(0.4) and rel["b"] == pytest.approx(0.5)
```

In `tests/test_cli.py`, change the import to `from esr.cli import build_config, compare, main, parse_args`, then append:

```python
def test_compare_uses_only_latest_protocol_and_writes_detail(tmp_path):
    cols = ["experiment", "condition", "group", "severity", "mAP"]
    rows = [("x", "clean", "clean", 0, 0.5), ("x", "telephone", "unseen", 1, 0.25)]
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    pd.DataFrame([("old_model", *r[1:]) for r in rows], columns=cols).to_csv(  # v1 file: no protocol column
        tmp_path / "old" / "robustness.csv", index=False)
    pd.DataFrame(rows, columns=cols).assign(subset=False, protocol=2).to_csv(
        tmp_path / "new" / "robustness.csv", index=False)
    out = tmp_path / "summary.csv"
    table = compare([str(tmp_path / "*" / "robustness.csv")], out=str(out))
    assert table["experiment"].tolist() == ["x"]
    detail = pd.read_csv(tmp_path / "summary_by_condition.csv")
    assert detail["experiment"].tolist() == ["x"] and detail["rel"].tolist() == pytest.approx([0.5])
```

- [ ] **Step 2: Run the tests to verify they fail.** Use the command from Task 2, Step 2. Expected:
  - `ImportError: cannot import name 'per_condition'`
  - the compare test fails because `old_model` is present in the table.

- [ ] **Step 3: Implement it.**

Append to `esr/pipeline.py`:

```python
def per_condition(df):
    """mAP and relative mAP (corrupted / clean of the same experiment) for every condition x severity."""
    clean = df[df["group"] == "clean"].groupby("experiment")["mAP"].mean()
    out = df[df["group"] != "clean"][["experiment", "condition", "group", "severity", "mAP"]].copy()
    out["rel"] = out["mAP"] / out["experiment"].map(clean)
    return out.sort_values(["experiment", "group", "condition", "severity"]).reset_index(drop=True)
```

In `esr/cli.py`, add `from pathlib import Path` to the imports. Then replace `compare`:

```python
def compare(inputs, include_subset=False, out=None):
    from .pipeline import per_condition, summarize

    paths = sorted({p for pattern in inputs for p in glob.glob(pattern, recursive=True)})
    if not paths:
        raise SystemExit(f"No robustness.csv found for {inputs}. Run evaluate first or add outputs as inputs.")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "subset" not in df:
            df["subset"] = False
        if "protocol" not in df:
            df["protocol"] = 1  # written before protocol versions existed
        df["source"] = p
        frames.append(df)
    allres = pd.concat(frames, ignore_index=True)
    if not include_subset:
        allres = allres[~allres["subset"].astype(bool)]
    if allres.empty:
        raise SystemExit("Only --quick (subset) results found; pass --include-subset to compare them anyway.")
    latest = allres["protocol"].max()
    stale = sorted(set(allres.loc[allres["protocol"] < latest, "experiment"]))
    if stale:
        print(f"Skipping protocol < {latest} results (re-run evaluate on them): {', '.join(stale)}")
    allres = allres[allres["protocol"] == latest]
    allres = allres.drop_duplicates(["experiment", "condition", "severity"], keep="last")
    for exp, src in allres.groupby("experiment")["source"].first().items():
        print(f"{exp}: {src}")
    table = summarize(allres)
    print(table.round(4).to_string(index=False))
    detail = per_condition(allres)
    print(detail.pivot_table(index=["group", "condition", "severity"], columns="experiment", values="rel")
          .round(3).to_string())
    if out:
        table.to_csv(out, index=False)
        detail.to_csv(Path(out).with_name(Path(out).stem + "_by_condition.csv"), index=False)
    return table
```

- [ ] **Step 4: Run the whole suite.** Expected: all tests pass.

- [ ] **Step 5: Hand-off.** Changed files: `esr/pipeline.py`, `esr/cli.py`, `tests/test_pipeline.py`, `tests/test_cli.py`. Suggested message: `feat: compare filters by protocol and writes per-condition table`.

---

### Task 4: Training augmentation — `random_eq` and configurable seen-corruption ranges

**Files:**
- Modify: `esr/augment.py` (add `random_eq`), `esr/corruptions.py` (`random_train_corruption` ranges), `esr/config.py` (4 fields), `esr/engine.py` (`train_one_epoch`)
- Test: `tests/test_augment.py`, `tests/test_corruptions.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `add_noise_at_snr(wav, noise, snr_db, sample_rate)` from Task 1.
- Produces:
  - `random_eq(wav, p, gen, sample_rate=16000, max_db=12.0, n_points=(3, 6)) -> Tensor` (same shape, RMS kept per clip).
  - `random_train_corruption(wav, p, gen, sample_rate=16000, snr_db=(5.0, 30.0), rt60_s=(0.2, 0.8))`.
  - New `Config` fields: `eq_aug_p: float = 0.0`, `eq_max_db: float = 12.0`, `train_snr_db: tuple = (5.0, 30.0)`, `train_rt60_s: tuple = (0.2, 0.8)`.
  - `train_one_epoch` calls, positionally:
    - `random_train_corruption(wav, cfg.corruption_aug_p, gen, cfg.sample_rate, cfg.train_snr_db, cfg.train_rt60_s)`
    - `random_eq(wav, cfg.eq_aug_p, gen, cfg.sample_rate, cfg.eq_max_db)`

- [ ] **Step 1: Write the failing tests.**

In `tests/test_augment.py`, add `random_eq` to the `from esr.augment import ...` line, then append:

```python
def test_random_eq_p0_is_identity():
    x = torch.randn(3, 16000)
    assert torch.equal(random_eq(x, 0.0, torch.Generator().manual_seed(0)), x)


def test_random_eq_changes_every_clip_keeps_rms_and_is_deterministic():
    x = torch.randn(4, 16000, generator=torch.Generator().manual_seed(1))
    a = random_eq(x, 1.0, torch.Generator().manual_seed(0))
    b = random_eq(x, 1.0, torch.Generator().manual_seed(0))
    assert torch.equal(a, b) and a.shape == x.shape
    assert all(not torch.allclose(a[i], x[i], atol=1e-3) for i in range(4))
    assert torch.allclose(a.pow(2).mean(-1), x.pow(2).mean(-1), rtol=1e-3)


def test_random_eq_gain_stays_within_max_db():
    x = torch.randn(2, 16000, generator=torch.Generator().manual_seed(2))
    y = random_eq(x, 1.0, torch.Generator().manual_seed(0), max_db=6.0)
    ratio_db = 10 * torch.log10(torch.fft.rfft(y).abs().pow(2) / torch.fft.rfft(x).abs().pow(2))
    assert (ratio_db.amax(-1) - ratio_db.amin(-1) <= 12.0 + 1e-2).all()


def test_random_eq_silent_clip_stays_finite():
    assert torch.isfinite(random_eq(torch.zeros(2, 16000), 1.0, torch.Generator().manual_seed(0))).all()
```

Append to `tests/test_corruptions.py`:

```python
def test_random_train_corruption_draws_from_given_ranges(monkeypatch):
    import esr.corruptions as c

    seen = {"snr": [], "rt60": []}
    monkeypatch.setattr(c, "add_noise_at_snr", lambda x, n, snr, sr=16000: seen["snr"].append(snr) or x)
    monkeypatch.setattr(c, "apply_reverb", lambda x, rt60, sr, g: seen["rt60"].append(rt60) or x)
    c.random_train_corruption(_tone(440, batch=40), 1.0, _gen(), snr_db=(0.0, 1.0), rt60_s=(0.9, 1.0))
    assert seen["snr"] and seen["rt60"]
    assert all(0.0 <= s <= 1.0 for s in seen["snr"]) and all(0.9 <= r <= 1.0 for r in seen["rt60"])
```

Append to `tests/test_engine.py`:

```python
def test_train_one_epoch_passes_v2_augmentation_settings(tiny_cfg, monkeypatch):
    import esr.engine as engine

    calls = []
    monkeypatch.setattr(engine, "random_train_corruption",
                        lambda wav, p, gen, sr, snr_db, rt60_s: calls.append(("corrupt", snr_db, rt60_s)) or wav)
    monkeypatch.setattr(engine, "random_eq", lambda wav, p, gen, sr, max_db: calls.append(("eq", p, max_db)) or wav)
    cfg = tiny_cfg(corruption_aug_p=0.5, eq_aug_p=0.5, eq_max_db=9.0, train_snr_db=(0.0, 30.0),
                   train_rt60_s=(0.2, 1.0))
    model = build_model(cfg, n_classes=5)
    loader = DataLoader(TensorDataset(torch.randn(4, cfg.clip_samples) * 0.1, torch.ones(4, 5)), batch_size=4)
    opt, sch, scaler = _opt(model, 1)
    train_one_epoch(model, loader, opt, sch, scaler, torch.device("cpu"), cfg,
                    np.random.default_rng(0), torch.Generator().manual_seed(0))
    assert calls == [("corrupt", (0.0, 30.0), (0.2, 1.0)), ("eq", 0.5, 9.0)]
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/User/documentaries/USTH_codes/Environmental-Sound-Recognition-Under-Unseen-Conditions && ~/.venvs/esr/bin/python -m pytest -q tests/test_augment.py tests/test_corruptions.py tests/test_engine.py"`

Expected:
- `ImportError: cannot import name 'random_eq'`
- a `TypeError` about the unexpected keyword `snr_db`
- `TypeError: Config.__init__() got an unexpected keyword argument 'eq_aug_p'`

- [ ] **Step 3: Implement it.**

In `esr/augment.py`, add `import numpy as np` above `import torch`, then add after `mixup`:

```python
def random_eq(wav, p, gen, sample_rate=16000, max_db=12.0, n_points=(3, 6)):
    """FilterAugment-style random EQ (Nam et al., 2022) on the waveform, per clip with probability p.

    Gain curve: piecewise linear in dB over log-frequency through 3-6 random anchors in [-max_db, max_db];
    loudness is restored afterwards. A smooth tilt, never a hard band-stop, so it is not the telephone test.
    The FFT length is the clip length, which is fixed, so cuFFT keeps a single plan.
    """
    if p <= 0:
        return wav
    n = wav.size(-1)
    logf = np.log(np.maximum(np.fft.rfftfreq(n, d=1.0 / sample_rate), 20.0))
    gains = np.ones((wav.size(0), logf.size), dtype=np.float32)
    for i in range(wav.size(0)):
        if torch.rand(1, generator=gen).item() >= p:
            continue
        k = int(torch.randint(n_points[0], n_points[1] + 1, (1,), generator=gen))
        inner = torch.rand(k - 2, generator=gen).sort().values.numpy()
        anchors = logf[0] + (logf[-1] - logf[0]) * np.concatenate([[0.0], inner, [1.0]])
        db = ((torch.rand(k, generator=gen) * 2 - 1) * max_db).numpy()
        gains[i] = 10 ** (np.interp(logf, anchors, db) / 20)
    out = torch.fft.irfft(torch.fft.rfft(wav, dim=-1) * torch.from_numpy(gains).to(wav.device), n=n, dim=-1)
    scale = (wav.pow(2).mean(-1, keepdim=True) / out.pow(2).mean(-1, keepdim=True).clamp_min(1e-16)).sqrt()
    return out * scale
```

In `esr/corruptions.py`, replace `random_train_corruption`:

```python
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
```

In `esr/config.py`, extend the augmentation block of `Config`:

```python
    freq_mixstyle_p: float = 0.0
    corruption_aug_p: float = 0.0
    train_snr_db: tuple = (5.0, 30.0)  # white-noise SNR range of the seen-corruption augmentation
    train_rt60_s: tuple = (0.2, 0.8)  # reverb RT60 range of the seen-corruption augmentation
    eq_aug_p: float = 0.0  # random EQ (FilterAugment-style) per clip
    eq_max_db: float = 12.0
```

In `esr/engine.py`, change the import to `from .augment import mixup, random_eq`, and replace the corruption block in `train_one_epoch`:

```python
        if cfg.corruption_aug_p > 0:
            wav = random_train_corruption(wav, cfg.corruption_aug_p, gen, cfg.sample_rate, cfg.train_snr_db,
                                          cfg.train_rt60_s)
        if cfg.eq_aug_p > 0:
            wav = random_eq(wav, cfg.eq_aug_p, gen, cfg.sample_rate, cfg.eq_max_db)
```

- [ ] **Step 4: Run the whole suite.** Expected: all tests pass. `test_to_dict_is_json_serializable` still passes, because tuples serialise as lists.

- [ ] **Step 5: Hand-off.** Changed files: `esr/augment.py`, `esr/corruptions.py`, `esr/config.py`, `esr/engine.py`, `tests/test_augment.py`, `tests/test_corruptions.py`, `tests/test_engine.py`. Suggested message: `feat: random EQ augmentation and configurable seen-corruption ranges`.

---

### Task 5: Experiment presets — `effnet_robust_v2` and ablations

**Files:**
- Modify: `esr/config.py` (`EXPERIMENTS`)
- Test: `tests/test_config.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: the `Config` fields from Task 4.
- Produces: seven preset names. The CLI `--experiment` choices come from `sorted(EXPERIMENTS)` and need no change.

- [ ] **Step 1: Write the failing tests.**

In `tests/test_config.py`, replace `test_experiments_are_exactly_the_three_in_the_spec` with:

```python
def test_experiment_presets():
    assert set(EXPERIMENTS) == {"cnn_baseline", "effnet_standard", "effnet_robust", "effnet_robust_v2",
                                "effnet_mixstyle", "effnet_corrupt", "effnet_eq"}
    assert make_config("cnn_baseline").model == "simple_cnn"
    knobs = ("freq_mixstyle_p", "corruption_aug_p", "eq_aug_p")
    assert all(getattr(make_config("effnet_standard"), k) == 0.0 for k in knobs)
    v1 = make_config("effnet_robust")  # must stay exactly the Experiment 1 recipe
    assert (v1.freq_mixstyle_p, v1.corruption_aug_p, v1.eq_aug_p) == (0.7, 0.5, 0.0)
    assert v1.train_snr_db == (5.0, 30.0) and v1.train_rt60_s == (0.2, 0.8)
    v2 = make_config("effnet_robust_v2")
    assert (v2.freq_mixstyle_p, v2.corruption_aug_p, v2.eq_aug_p) == (0.7, 0.5, 0.5)
    assert v2.train_snr_db == (0.0, 30.0) and v2.train_rt60_s == (0.2, 1.0)
    for name, knob in [("effnet_mixstyle", "freq_mixstyle_p"), ("effnet_corrupt", "corruption_aug_p"),
                       ("effnet_eq", "eq_aug_p")]:
        cfg = make_config(name)
        assert {k for k in knobs if getattr(cfg, k) > 0} == {knob}
        assert cfg.model == "efficientnet_b0" and cfg.pretrained
```

Append to `tests/test_pipeline.py`:

```python
def test_run_training_effnet_robust_v2_smoke(tiny_cfg):
    assert len(run_training(tiny_cfg("effnet_robust_v2"))["history"]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/User/documentaries/USTH_codes/Environmental-Sound-Recognition-Under-Unseen-Conditions && ~/.venvs/esr/bin/python -m pytest -q tests/test_config.py tests/test_pipeline.py"`

Expected:
- the set assertion fails
- `KeyError: Unknown experiment 'effnet_robust_v2'`

- [ ] **Step 3: Implement it.** Replace `EXPERIMENTS` in `esr/config.py`:

```python
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
```

- [ ] **Step 4: Run the whole suite.** Expected: all tests pass.

- [ ] **Step 5: Hand-off.** Changed files: `esr/config.py`, `tests/test_config.py`, `tests/test_pipeline.py`. Suggested message: `feat: effnet_robust_v2 and ablation presets`.

---

### Task 6: Docs and Kaggle notebook for v2

**Files:**
- Modify: `notebooks/kaggle_run.ipynb`, `README.md`, `docs/ARCHITECHTURE.md`, `docs/EXPERIMENT.md`
- Test: `tests/test_kaggle_runner.py` (existing; it must still pass)

**Interfaces:**
- Consumes: everything above.
- Produces: documentation only.

- [ ] **Step 1: Notebook.** Edit `notebooks/kaggle_run.ipynb` with the NotebookEdit tool.

  1. **Cell 0 (markdown):** replace the last sentence with:
     > Run one experiment per committed version (*Save Version → Save & Run All*): `cnn_baseline`, `effnet_standard`, `effnet_robust`, `effnet_robust_v2`, or the ablations `effnet_mixstyle`, `effnet_corrupt`, `effnet_eq`.
  2. **Cell 4 (code):** change the comment of `EXPERIMENT` to `# cnn_baseline | effnet_standard | effnet_robust | effnet_robust_v2 | effnet_mixstyle | effnet_corrupt | effnet_eq`.
  3. **Insert a markdown cell after cell 8:**
     ```
     ## 4b. Re-evaluate a checkpoint from an earlier version (protocol v2)
     Add that version's output as an input, then point `--checkpoint` at its `best.pt`. Results go to `/kaggle/working/outputs/<experiment>/`.
     ```
  4. **Insert a code cell after it:**
     ```python
     CKPT = "/kaggle/input/<earlier-version-output>/outputs/effnet_robust/best.pt"  # <- adjust
     !python scripts/evaluate.py --experiment effnet_robust --data-path $DATA_PATH --checkpoint $CKPT
     ```
  5. **Last code cell (plot):** append:
     ```python
     detail = pd.read_csv("/kaggle/working/summary_by_condition.csv")
     display(detail.pivot_table(index=["group", "condition", "severity"], columns="experiment", values="rel").round(3))
     ```

- [ ] **Step 2: README.md.**
  - Line 5: change the unseen list to "(brown noise, telephone band, clipping, speed-up, quantisation)".
  - In the `## Experiments` table, add these rows:

    | effnet_robust_v2 | EfficientNet-B0 (ImageNet) + attention pooling | FreqMixStyle + white-noise (0–30 dB)/reverb (RT60 ≤ 1 s) aug + random EQ |
    |---|---|---|
    | effnet_mixstyle / effnet_corrupt / effnet_eq | EfficientNet-B0 | one robustness component each (ablation) |

  - Under outputs, add: "`compare` also writes `summary_by_condition.csv` (relative mAP per condition × severity) and skips results from an older evaluation protocol."

- [ ] **Step 3: docs/ARCHITECHTURE.md.** Make these edits:
  - **§1:**
    - UNSEEN list → `brown_noise`, `telephone`, `clipping`, `speed`, `quantize`.
    - Add the four new presets to the experiments table.
  - **§5.2:**
    - "16 passes" → "22 passes".
    - Add `protocol` to the `robustness.csv` column list.
  - **§6.1:**
    - Add the rows `train_snr_db`, `train_rt60_s`, `eq_aug_p`, `eq_max_db`.
    - Replace the `EXPERIMENTS` code block with the new one from Task 5.
  - **§6.4:** add a `random_eq` paragraph: FilterAugment-style; piecewise-linear dB over log-frequency; ±12 dB; RMS restored; applied after the seen corruption and before mixup.
  - **§6.5:**
    - Add table rows `speed | UNSEEN | ×1.05 / ×1.15 / ×1.3 | linear resample, zero-pad the tail` and `quantize | UNSEEN | 8 / 6 / 4 bits | uniform steps relative to the clip peak`.
    - Add a note: "Protocol v2: noise SNR is measured on the 50–8000 Hz band (`SNR_BAND_HZ`); v1 used broadband RMS, which made brown noise ~29 dB milder in-band. `PROTOCOL` is written to every row, and `compare` keeps only the latest protocol."
  - **§6.8:** `predict_array` seeds every clip with `seed * 100003 + clip_index`.
  - **§6.10:**
    - `--checkpoint` accepts an absolute path.
    - `compare` writes `summary_by_condition.csv`.
  - **§14:**
    - Replace point 12 with: "Evaluation noise is seeded per clip (protocol v2), so it no longer depends on `batch_size`."
    - Extend point 16 with: "`random_eq` (training only) is a smooth ±12 dB tilt. It overlaps *in kind* with `telephone` (both change the spectral envelope), so always report `telephone` results for `effnet_robust_v2` with that caveat."
  - **§16:** mark M-1 as resolved in v2.
  - **§18:** make the `--experiment` choices list all seven presets.

- [ ] **Step 4: docs/EXPERIMENT.md.** Make these edits:
  - After the §1.3 heading line, insert: "*These numbers use evaluation protocol v1 (broadband SNR, per-batch noise seeds). They are not comparable with protocol v2 results; see Experiment 2.*"
  - Append a new section:

  ```markdown
  ---

  ## Experiment 2 (planned): baselines + protocol v2 + `effnet_robust_v2`

  Implementation plan: `docs/superpowers/plans/2026-09-27-esr-v2-next-steps.md`. Run order and decision gates: see the plan's "Kaggle Run Plan".
  Protocol v2 = band-limited SNR (50–8000 Hz), per-clip eval seeds, 22 conditions (UNSEEN adds `speed`, `quantize`).
  ```

- [ ] **Step 5: Run the whole suite.** Expected: all tests pass, including `tests/test_kaggle_runner.py`.

- [ ] **Step 6: Hand-off.** Changed files: `notebooks/kaggle_run.ipynb`, `README.md`, `docs/ARCHITECHTURE.md`, `docs/EXPERIMENT.md`. Suggested message: `docs: v2 protocol, presets and Kaggle runner`.

---

## Kaggle Run Plan (for the user, after Tasks 1–6 are pushed)

Budget: Kaggle gives 30 GPU-hours per week. One EfficientNet-B0 training run takes about 1 h 45 min, and the 22-condition evaluation about 40–60 min.

In every session:

- Start with `!git pull origin main`.
- Keep `--batch-size` at its default of 32 for training.
- Use *Save Version → Save & Run All*, so that `outputs/` survives the session.

| Session | Commands (`--data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k`) | GPU time |
|---|---|---|
| 0: smoke | `train.py --experiment effnet_robust_v2 --quick`, then `evaluate.py --experiment effnet_robust_v2 --quick --severities 3` | ~15 min |
| 1: baselines | Re-evaluate Experiment 1 with `--checkpoint /kaggle/input/<exp1-output>/outputs/effnet_robust/best.pt` (if that output was not saved, retrain `effnet_robust`); train and evaluate `effnet_standard`; train and evaluate `cnn_baseline` | ~5–6 h |
| 2: v2 | Train and evaluate `effnet_robust_v2` and `effnet_eq` | ~4.5 h |
| 3: ablation | Train and evaluate `effnet_mixstyle` and `effnet_corrupt` | ~4.5 h |
| 4: compare | Add the outputs of sessions 1–3 as inputs, then run `compare.py --out /kaggle/working/summary.csv` | CPU only |

**Decision gates**

- **After session 0:** if it raises any error, stop and debug. This is the only GPU test of the `random_eq` device transfer and of the new FFT paths.
- **After session 1:**
  - If `effnet_robust` has an unseen_rel (v2) less than 0.02 above `effnet_standard`, the v1 robustness recipe barely helps.
  - In that case, still run session 2, which tests whether EQ helps. But record the null result in `docs/EXPERIMENT.md` before moving on.
- **After session 2:**
  - Success means `effnet_robust_v2` improves `telephone` sev 2–3 `rel` over `effnet_robust` while clean mAP drops by at most 0.005.
  - If clean mAP drops by more than 0.01, lower `eq_aug_p` to 0.3 in the `effnet_robust_v2` preset (there is no CLI flag for it) before the ablation session.
- **After session 4:** record every result as Experiment 2 in `docs/EXPERIMENT.md` (§ layout as in Experiment 1).

**Out of scope for v2:**
- DDP on both T4s.
- EfficientNet-B2.
- Multi-seed runs.
- Real-noise corpora (ESC-50/MUSAN).
- Codec (MP3/Opus) corruption, which needs ffmpeg or `lameenc`.

Revisit these after Experiment 2 if the GPU budget remains.
