# Open Stethoscope — Complete Experimental Report

**Heart murmur detection on the CirCor DigiScope dataset (PhysioNet Challenge 2022).**
Compiled from EXPERIMENTS.md (v3), EXPERIMENTS_v4.md (v4 iteration log) and
EXPERIMENTS_2026-08-20.md (independent re-validation, CV, ablations, ensembles).
All numbers in this report are reproducible from the committed artifacts — see
[README → Reproducibility](README.md).

---

## 1. Problem & Setup

**Task.** Classify each patient as murmur *Absent / Unknown / Present* from 1–4
phonocardiogram (PCG) recordings at auscultation positions AV, PV, TV, MV.
Official metric: weighted multi-class accuracy `s_murmur` with weights
Absent 1 / Unknown 3 / Present 5 (the George B. Moody PhysioNet Challenge 2022).

**Data.** CirCor DigiScope v1.0.3 public cohort: **942 patients / 3,158 recordings**
(A 695 / U 68 / P 179 at patient level). Fixed patient-level 70/15/15 split
(`v3_split_seed42.json`, 142 held-out test patients), no patient leakage.

**⚠️ Label-corruption note.** The popular HF parquet mirror of this dataset
mislabels 119 recordings (~19 Present patients) as Absent. All experiments here
use labels rebuilt from the **official `training_data.csv`** and verified against
official patient-level counts (exact match). Anyone reusing HF mirrors should
re-validate labels.

**Model — FusionNet (v4).** ~404K parameters, all positions share a 2D-CNN
encoder over log-mel spectrograms, fused by **masked learned attention**
(missing positions masked out — mirrors a clinician listening to every available
position). Class imbalance handled by sqrt-inverse-frequency patient sampling +
class-weighted CE + auxiliary binary murmur head (λ=0.3). Augmentation: random
8 s window, ±10% time stretch, SpecAugment-lite, Gaussian noise.
Trains in ~1 min on a Tesla P4; real-time CPU inference.

## 2. Experimental Timeline

| stage | what | test s_murmur |
|---|---|---|
| v2 → v3 | multi-position fusion, ablations, official metric | 0.7000 |
| v4 | longer training (30 ep, patience 7) + val-tuned thresholds | **0.7815** |
| 2026-08-20 | bit-for-bit reproduction of all 4 original seeds | 0.7593/0.7778/0.6556/0.7704 ✓ |
| 2026-08-20 | 5-fold patient-stratified CV (honest generalization) | **0.7036 ± 0.0604** |
| 2026-08-20 | 9-seed family + val-selected top-4 ensemble, tuned | **0.7926** (new best) |

## 3. Headline Results

**Best (test, 142 held-out patients):**

| config | s_murmur | recall [A, U, P] |
|---|---|---|
| **val-selected top-4 ensemble (9-seed family) + tuned offsets** | **0.7926** | — |
| best single (s43 + val-tuned dP +0.40 / dU +0.20) | 0.7815 | [0.886, 0.60, 0.741] |
| 4-seed probavg ensemble (s42–45) tuned | 0.7852 | [0.838, 0.80, 0.741] |
| 2022 Challenge champion (HearHeart, hidden test) | 0.780 | — |

**Honest generalization (5-fold CV, 942 patients):** 0.7036 ± 0.0604
(per fold: 0.759 / 0.665 / 0.754 / 0.737 / 0.603). OOF ensemble (every patient
scored by a model that never saw them): **0.7040**, recall [0.859, 0.544, 0.620].

**Interpretation.** The single-split 0.7815 sits at the favorable end of a
±0.06 seed/CV spread; true generalization is ≈ 0.70. Rare classes make
selection noisy (10 Unknown / 27 Present in val). The bottleneck is Present
recall 0.62 (weight 5× in the official metric).

## 4. Ablations & Interventions (fixed split, seed 43)

| variant | params | s_murmur | verdict |
|---|---|---|---|
| v4 FusionNet (baseline) | 404K | 0.7778 | best |
| v5 scaled encoder (4 conv, 384 ch) | 2.26M | 0.7296 | scale hurts (overfit 659-patient train) |
| focal loss γ=2 | 404K | 0.6111 | hurts (Present recall collapses) |
| frozen wav2vec2-base + MLP/attn head | 94.4M frozen | 0.6778 | hurts (both heads) |

v3 ablations (same split): class-balance sampling is the biggest lever;
learned attention ≈ mean pooling — the winning ingredient is **multi-position
fusion itself**, not the attention weights.

**Reading.** On this small dataset, a 404K from-scratch fusion model beats a
5.6× larger version of itself and frozen 94M self-supervised features.
Bigger is not better when the training set is 659 patients.

## 5. What Didn't Work (negative results, seed 43)

- **4-seed ensembles without selection** — no better than best single (val too
  small to rank reliably). *Resolved* by the 9-seed family + val selection → 0.7926.
- **Coarse threshold tuning on individual seeds** — noisy, often hurts.
- **Focal loss, larger encoder, frozen wav2vec2** — all worse (see above).
- **Naive ensembling (v3)** and **per-location voting** — worse than fusion.

## 6. Methodological Lessons

1. **Label integrity first.** Third-party mirrors can silently corrupt labels;
   always cross-check patient-level distributions against the official source.
2. **Honest generalization ≠ single split.** Report CV; the ±0.06 spread here
   would have been invisible in one seed.
3. **Selection needs seeds.** "Ensembling doesn't help" was an artefact of
   n=4 seeds with no selection; n=9 + val-selected top-k flipped the conclusion.
4. **404K beats 94M when data is small** — parameter count is not the bottleneck.

## 7. Artifacts

- Weights: `experiments/models/` (9-seed v4 family, kfold×5, v5, focal, QC)
- Probabilities: `experiments/probs/` (val/test per seed, ensemble inputs)
- Training logs: `experiments/logs/` (run_s42–s52_30ep.log, kfold, focal, v5, w2v)
- Results: `experiments/results/` (exp_results*.json, kfold_results.json,
  score_library.json, v3_split_seed42.json)

Reproduce in seconds on CPU:

```bash
python3 ensemble_topk.py            # → top4 tuned = 0.7926
python3 tune_ensemble.py --mode probavg s43_30ep   # → 0.7815
```

---

*Companion docs: [EXPERIMENTS.md](EXPERIMENTS.md) · [EXPERIMENTS_v4.md](EXPERIMENTS_v4.md) ·
[EXPERIMENTS_2026-08-20.md](EXPERIMENTS_2026-08-20.md)*
