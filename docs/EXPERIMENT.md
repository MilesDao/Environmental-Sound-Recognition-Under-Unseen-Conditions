# Experiments Log

This log records every full run on Kaggle: its configuration, raw numbers, what the numbers mean, and what to do next.
Metrics are macro mAP over the 200 FSD50K classes. Unless stated otherwise, "relative" means corrupted mAP / clean mAP of the same model.
Code and protocol are described in [ARCHITECHTURE.md](ARCHITECHTURE.md).

---

## Experiment 1: `effnet_robust` (EfficientNet-B0 + robustness training)

**Date:** 2026-09-27 · **Hardware:** Kaggle GPU (single T4) · **Commit:** `ce393e6` (with the cuFFT fix)

### 1.1 Setup

| Item | Value |
|---|---|
| Model | EfficientNet-B0 (timm, ImageNet-pretrained, `in_chans=1`) + attention pooling |
| Input | 16 kHz audio, 10 s clips, 128-bin log-mel (n_fft 1024, win 400, hop 160, 50–8000 Hz) |
| Training data | Full FSD50K dev train split, balanced sampling |
| Optimiser | AdamW, lr 5e-4, weight decay 0.01, 1 warm-up epoch, then cosine decay; batch 32, 20 epochs, AMP |
| Generic augmentation | SpecAugment, waveform mixup (α = 0.5) |
| Robustness augmentation | Frequency-wise MixStyle (p = 0.7); seen-family corruption (p = 0.5): white noise at SNR 5–30 dB **or** reverb at RT60 0.2–0.8 s |
| Model selection | `best.pt` = the epoch with the highest val mAP |
| Evaluation | 10,231 eval clips; clean + 5 corruptions × 3 severities, fixed seed |

Commands:

```bash
!python scripts/train.py    --experiment effnet_robust --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
!python scripts/evaluate.py --experiment effnet_robust --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
!python scripts/compare.py  --out /kaggle/working/summary.csv
```

### 1.2 Training curve

| Epoch | Loss | Val mAP | | Epoch | Loss | Val mAP |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 0.1093 | 0.2500 | | 11 | 0.0352 | 0.5540 |
| 2 | 0.0536 | 0.4125 | | 12 | 0.0334 | 0.5581 |
| 3 | 0.0478 | 0.4569 | | 13 | 0.0326 | 0.5610 |
| 4 | 0.0451 | 0.4809 | | 14 | 0.0319 | 0.5643 |
| 5 | 0.0431 | 0.4989 | | 15 | 0.0313 | 0.5767 |
| 6 | 0.0406 | 0.5137 | | 16 | 0.0317 | 0.5740 |
| 7 | 0.0393 | 0.5213 | | **17** | 0.0304 | **0.5807** |
| 8 | 0.0384 | 0.5205 | | 18 | 0.0308 | 0.5774 |
| 9 | 0.0367 | 0.5336 | | 19 | 0.0301 | 0.5796 |
| 10 | 0.0364 | 0.5419 | | 20 | 0.0300 | 0.5799 |

Wall time: epoch 1 took 507 s (it includes the pretrained-weight download and a cold disk cache). Later epochs took about 300 s each. The whole run took about **1 h 42 min**.

### 1.3 Robustness results (eval set, `best.pt` = epoch 17)

*These numbers use evaluation protocol v1 (broadband SNR, per-batch noise seeds). They are not comparable with
protocol v2 results; see Experiment 2.*

| Condition | Split | Sev 1 | Sev 2 | Sev 3 | Mean | Relative (sev 1 / 2 / 3) |
|---|---|---:|---:|---:|---:|---|
| clean | – | **0.5565** | | | | 1.00 |
| white_noise | seen | 0.5381 | 0.5133 | 0.4377 | 0.4964 | 0.97 / 0.92 / 0.79 |
| reverb | seen | 0.5129 | 0.5010 | 0.4698 | 0.4946 | 0.92 / 0.90 / 0.84 |
| brown_noise | unseen | 0.5498 | 0.5456 | 0.5386 | 0.5447 | 0.99 / 0.98 / 0.97 |
| telephone | unseen | 0.4957 | 0.3867 | 0.2351 | 0.3725 | 0.89 / 0.69 / **0.42** |
| clipping | unseen | 0.5464 | 0.5052 | 0.3906 | 0.4807 | 0.98 / 0.91 / 0.70 |

Summary (`compare.py`):

| Experiment | Clean | Seen | Unseen | Seen rel. | Unseen rel. |
|---|---:|---:|---:|---:|---:|
| effnet_robust | 0.5565 | 0.4955 | 0.4660 | 0.890 | 0.837 |
| *(unseen without brown_noise, see §1.4.3)* | | | *0.4266* | | *0.767* |

### 1.4 Analysis

#### 1.4.1 Training is healthy and has converged

- **Val mAP** climbs quickly (0.25 → 0.51 by epoch 6), then flattens. From epoch 15 onwards it stays within ±0.004 of 0.578, so 20 epochs is enough for this recipe.
- **Overfitting:** none visible. Training loss keeps falling slowly and val mAP does not drop. Part of the late plateau is simply the cosine schedule taking the learning rate toward 1 % of its peak.
- **Val vs eval gap:** val mAP 0.581 against eval clean mAP 0.557, a gap of 0.024. That size is normal for FSD50K. Val comes from the same pool of dev uploads as train, whereas eval is a separately collected set. Model selection on val adds a little optimism as well.

#### 1.4.2 The clean accuracy is competitive

These are the published numbers on the FSD50K eval set:

- FSD50K paper, VGG-like CNN baseline: **0.434**.
- PSLA (Gong et al. 2021), a single EfficientNet-B2 with attention pooling, ImageNet init and 16 kHz input: about **0.558**.

Our B0, which is smaller, reaches **0.5565** with the same recipe family, even with the robustness augmentations switched on. So the pipeline (features, balanced sampling, mixup, attention pooling) works as intended, and clean accuracy is **not** the bottleneck of this project.

#### 1.4.3 Finding: `brown_noise` is almost a no-op, so the protocol needs a fix

Even at 0 dB SNR, brown noise costs only 3 % relative mAP. The reason is the noise spectrum, not the model:

- Brown noise has power ∝ 1/f². For a 10 s, 16 kHz clip, **99.88 % of its power lies below 50 Hz**, and 50 Hz is `f_min` of the mel filterbank. The fraction that falls inside the 50–8000 Hz analysis band is 0.12 %, which is −29.2 dB.
- The SNR is set with broadband RMS. So "0 dB SNR" is really about **+29 dB inside the band the model sees**. That is milder than white noise at severity 1.

**Consequence:** the headline unseen score (0.466 / 0.837) is inflated by a condition that barely corrupts the input. Without it, unseen mAP is 0.427 (0.767 relative). This protocol issue has to be fixed before comparing models (see §1.5, step 2).

#### 1.4.4 Weak spot 1: band-limiting (`telephone`)

- **The drop:** relative mAP falls 0.89 → 0.69 → **0.42** as the pass band narrows (100–5000, then 300–3400, then 500–2000 Hz). This is the largest failure in the benchmark.
- **Why the model is exposed:** many FSD50K classes rely on energy outside 500–2000 Hz. Low-frequency examples are engines, thunder and bass instruments; high-frequency examples are birds, hiss, cymbals and glass. The training augmentations never remove whole frequency regions coherently:
  - SpecAugment masks at most a few narrow bands.
  - Frequency-wise MixStyle only rescales per-band statistics.
  - White noise and reverb do not change the spectral envelope in this way.

#### 1.4.5 Weak spot 2: heavy distortion (`clipping` sev 3) and extrapolated severities

- **Clipping:** it is harmless at thresholds of 50 % and 20 % of the peak (0.98 and 0.91 relative). At 5 % it falls to 0.70, because the signal becomes nearly a square wave, with strong harmonics and the dynamics destroyed.
- **Seen corruptions at severity 3 are not really "seen":**
  - White noise at severity 3 is 0 dB SNR, but training used 5–30 dB. That explains the 0.79 relative.
  - Reverb at severity 3 has RT60 1.0 s, but training used 0.2–0.8 s.
  - The "seen" score therefore mixes interpolation (sev 1–2) and extrapolation (sev 3). Report the two separately.
- **Reverb at severity 1 already costs 8 %** (0.513 vs 0.557), even though RT60 0.3 s lies inside the training range. The synthetic RIR (exponentially decaying Gaussian noise with a unit direct path) smears onsets. This is the most "in-distribution" loss in the table and worth watching once baselines exist.

#### 1.4.6 What this experiment cannot tell us yet

We have **only one model**. A relative score of 0.837 means nothing until we compare it with:

- `effnet_standard`: the same network and recipe, minus MixStyle and corruption augmentation. This isolates the effect of the robustness methods.
- `cnn_baseline`: a small CNN trained from scratch. This shows the effect of architecture and pretraining.

In addition, all numbers come from one seed. Differences below about 0.005 mAP should not be interpreted without repeated seeds.

### 1.5 Proposed next steps (in priority order)

#### Step 1: Run the two baselines (required; about 3.5 h of GPU in total)

Use the same commit, seed and batch size, so the evaluation noise is identical. The eval noise depends on `batch_size` (limitation M-1), so keep it at 32. (Resolved in v2: corruptions now use per-clip seeds, so batch size no longer affects evaluation.)

```bash
!python scripts/train.py    --experiment effnet_standard --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
!python scripts/evaluate.py --experiment effnet_standard --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
!python scripts/train.py    --experiment cnn_baseline    --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
!python scripts/evaluate.py --experiment cnn_baseline    --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
!python scripts/compare.py  --out /kaggle/working/summary.csv
```

- Each EfficientNet run takes about 1 h 45 min of training plus the evaluation. If one Kaggle session is not enough, split the runs across sessions.
- Save the output of each session as a Kaggle dataset/notebook output and attach it as input. `compare.py` also reads `/kaggle/input/**/robustness.csv`.
- **Question answered:** does robustness training improve the relative unseen score, and at what clean-accuracy cost? Expect `effnet_standard` to have a similar or slightly higher clean mAP but lower seen and unseen relative scores.

#### Step 2: Fix the evaluation protocol before drawing conclusions (evaluation only, no retraining)

Evaluation is cheap compared with training (the checkpoints stay the same), so apply these changes before or together with step 1:

1. **Make brown noise meaningful.** Either set the SNR over the 50–8000 Hz analysis band (band-pass the noise before measuring its RMS), or replace it with **pink noise** (1/f). Pink noise keeps about 14 % of its power in the band. The recommended fix is band-limited SNR for all additive noise, so white and brown noise are defined the same way.
2. **Report the severity split:** add columns for "seen, in training range" (sev 1–2) and "seen, extrapolated" (sev 3).
3. **Add at least one more genuinely unseen condition** so the unseen score does not hinge on three corruptions. Candidates:
   - Real background noise from a different corpus, if one is available on Kaggle (for example ESC-50 or MUSAN).
   - MP3/low-bit-rate codec compression.
   - Pitch/speed perturbation.

   Keep all of them test-only.
4. Re-evaluate `effnet_robust` with the new protocol and update §1.3, so all models are compared under the same rules.

#### Step 3: Target the telephone weakness with a *generic* spectral augmentation (`effnet_robust_v2`)

The rule of the study is that unseen corruptions must never be trained on. So we do **not** add telephone band-passing. Instead we add a generic family that randomises the spectral envelope:

- **FilterAugment** (Nam et al., ICASSP 2022): multiply the log-mel by a random piecewise-linear (or step) frequency weighting of ±6–12 dB over 2–5 random bands. It is cheap, runs on the spectrogram, and is known to help with microphone and channel shifts.
- **Random EQ / random low-pass or high-pass on the waveform**, with cut-offs drawn from wide ranges (for example low-pass 2–8 kHz, high-pass 20–400 Hz).

  Write down clearly that this family *overlaps* with telephone band-limiting. For the unseen claim to stay honest, report telephone results for both models, and state the overlap in the report.
- Widen the seen corruption ranges so severity 3 is covered: SNR 0–30 dB, RT60 0.2–1.0 s. This turns sev 3 of the seen corruptions into interpolation and should raise the seen score.

**Expected effect:** telephone sev 2–3 relative rises from 0.69 / 0.42 toward 0.8 / 0.6, with a clean-mAP change of at most about 0.005.

#### Step 4: Ablation of the robustness components (after steps 1–3)

Train with one component at a time, so each gain can be attributed:

| Run | MixStyle | Corruption aug | FilterAugment |
|---|:-:|:-:|:-:|
| effnet_standard | – | – | – |
| + MixStyle only | ✓ | – | – |
| + corruption only | – | ✓ | – |
| effnet_robust | ✓ | ✓ | – |
| effnet_robust_v2 | ✓ | ✓ | ✓ |

This needs new presets in `esr/config.py` (`EXPERIMENTS`). The CLI already accepts any preset name listed there.

#### Step 5 (optional, only if GPU budget remains): capacity and variance

- **Capacity:**
  - EfficientNet-B2 (PSLA's choice) for about +0.01 clean mAP. It costs about 1.6× the time per epoch.
  - Use both T4s (DDP) to halve the wall time; the code is currently single-GPU.
- **Variance:** repeat the best configuration with 2–3 seeds, and report mean ± std for clean, seen and unseen.

### 1.6 Takeaways

1. The pipeline is sound. EfficientNet-B0 reaches **0.557 clean mAP**, on par with the published PSLA single-model result, and the training has converged within 20 epochs.
2. Robustness varies strongly by condition. Additive noise and mild distortion are handled well. **Band-limiting is the main failure (0.42 relative at sev 3)**, followed by heavy clipping and 0 dB white noise.
3. The current `brown_noise` condition is not a real test (−29 dB in-band). Fix the protocol before comparing models.
4. The robustness claim is not supported yet. It needs the `effnet_standard` and `cnn_baseline` runs.

---

## Experiment 2 (planned): baselines + protocol v2 + `effnet_robust_v2`

Implementation plan: `docs/superpowers/plans/2026-09-27-esr-v2-next-steps.md`. Run order and decision gates: see the plan's "Kaggle Run Plan".
Protocol v2 = band-limited SNR (50–8000 Hz), per-clip eval seeds, 22 conditions (UNSEEN adds `speed`, `quantize`).

---

### Experiment 2a: `effnet_robust` under protocol v2

**Date:** 2026-09-28 · **Hardware:** Kaggle GPU (single T4) · **Protocol:** 2 (22 conditions)

#### 2a.1 Results

Summary (`compare.py`):

| Experiment | Clean | Seen | Unseen | Seen rel. | Unseen rel. |
|---|---:|---:|---:|---:|---:|
| effnet_robust | 0.5580 | 0.5001 | 0.4731 | 0.896 | 0.848 |

Relative mAP (corrupted / clean) per condition (`summary_by_condition.csv`). Protocol-v1 values from §1.3 are shown for comparison.

| Condition | Group | Sev 1 | Sev 2 | Sev 3 | v1 (sev 1 / 2 / 3) |
|---|---|---:|---:|---:|---|
| white_noise | seen | 0.979 | 0.931 | 0.790 | 0.967 / 0.922 / 0.787 |
| reverb | seen | 0.927 | 0.904 | 0.848 | 0.922 / 0.900 / 0.844 |
| brown_noise | unseen | 0.976 | 0.927 | 0.789 | 0.988 / 0.980 / 0.968 |
| speed | unseen (new) | 0.975 | 0.959 | 0.909 | – |
| clipping | unseen | 0.988 | 0.918 | 0.710 | 0.982 / 0.908 / 0.702 |
| quantize | unseen (new) | 0.983 | 0.915 | 0.694 | – |
| telephone | unseen | 0.888 | 0.691 | **0.398** | 0.891 / 0.695 / 0.422 |

The summary scores are **not comparable** with Experiment 1: unseen now averages 5 conditions instead of 3, and the noise definition changed. `compare` refuses to mix the two protocols.

#### 2a.2 Analysis

1. **The protocol fix works.**
   - At equal in-band SNR, brown noise now costs as much as white noise (0.976 / 0.927 / 0.789 vs 0.979 / 0.931 / 0.790).
   - In v1 it was almost a no-op (0.968 at 0 dB).
   - The unseen score is now carried by five meaningful conditions.
2. **Brown ≈ white is a promising sign of transfer, but not yet evidence.**
   - The model is as robust to a noise colour it never trained on as to the one it did.
   - Whether that comes from the white-noise augmentation or is simply how EfficientNet behaves can only be decided against `effnet_standard`:
     - If the standard model shows a larger brown–white gap, the augmentation transfers.
     - If not, it is architecture.
3. **Every condition breaks at severity 3.**
   - At severity 1 every condition keeps ≥ 0.97 of clean mAP, except telephone (0.89) and reverb (0.93).
   - At severity 3 most conditions drop to 0.69–0.79; speed (0.91) and reverb (0.85) hold up best.
4. **The unseen conditions rank:** telephone ≫ quantize ≈ clipping ≈ brown_noise > speed.
   - **Speed:** a 30 % speed-up (≈ +4.5 semitones) barely matters (0.91), so the model does not rely on exact pitch.
   - **Quantize:** 4-bit quantisation (0.69) hurts as much as clipping at 5 % of peak (0.71). At 4 bits, quiet events fall below one quantisation step and disappear.
5. **Telephone band-limiting is still the main failure** (0.40 at 500–2000 Hz). Nothing in the current recipe addresses it; the `effnet_robust_v2` random-EQ augmentation targets it.
6. **Reverb costs 7 % even at its mildest setting** (RT60 0.3 s), which lies inside the training range.

#### 2a.3 Run-to-run variation

Clean, telephone and clipping are deterministic, so re-evaluating the same checkpoint would reproduce §1.3 exactly. They moved slightly:

| Condition | §1.3 (v1) | 2a (v2) |
|---|---:|---:|
| clean | 0.5565 | 0.5580 |
| telephone sev 3 (absolute) | 0.2351 | 0.2221 |
| clipping sev 3 (absolute) | 0.3906 | 0.3962 |

So this model is a retraining of `effnet_robust`, not the Experiment 1 checkpoint. The gap gives a first estimate of seed-to-seed noise: **about ±0.002 on clean mAP and up to about ±0.015 at severity 3.** Treat model differences smaller than that as noise unless they are confirmed with a second seed.

(If this run did use `--checkpoint` on the Experiment 1 `best.pt`, then evaluation is not deterministic and must be investigated before comparing models.)

#### 2a.4 Next steps

1. **Session 1 of the Run Plan:** train and evaluate `effnet_standard` and `cnn_baseline` under protocol 2. Every robustness claim depends on this comparison, including the brown ≈ white transfer question.
2. **Session 2:** train and evaluate `effnet_robust_v2` and `effnet_eq`.
   - Success criterion: telephone sev 2–3 rises clearly above 0.69 / 0.40, while clean mAP drops by at most 0.005.
3. **Seed check:** if `effnet_robust_v2` and `effnet_robust` differ by less than about 0.015 at severity 3, run a second seed of both before concluding.
