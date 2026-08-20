# EXPERIMENTS — 2026-08-20 (New GPU server, independent re-validation)

Host: DeepLn Tesla P4 (8 GB) · torch 2.6.0+cu124 · Python 3.12
Dataset: CirCor public training set rebuilt from HF parquet (`miguellmartins/circor-digiscope-physionet22`),
**re-labeled from the official `training_data.csv`** (the HF parquet mislabels 119 recordings of ~19
Present patients as Absent; verified against official patient-level counts Absent 695 / Present 179 /
Unknown 68 — exact match after fix).
Final set: **942 patients / 3,158 AV-PV-TV-MV recordings** (official RECORDS list; 1 recording
`50782_MV_1` absent from the parquet). Same fixed patient-level 70/15/15 split (`v3_split_seed42.json`).

## 1. Exact reproduction of the v4 seed family (fixed split, test = 142 held-out patients)

| run | config | test s_murmur | original 2026-08-15 | recall [A,U,P] |
|---|---|---|---|---|
| s42_30ep | 30ep/pat7/macroF1 | 0.7593 | 0.7593 ✓ | [0.895, 0.7, 0.667] |
| s43_30ep | 30ep/pat7/macroF1 | 0.7778 | 0.7778 ✓ | [0.933, 0.4, 0.741] |
| s44_30ep | 30ep/pat7/macroF1 | 0.6556 | 0.6556 ✓ | [0.886, 0.8, 0.444] |
| s45_30ep | 30ep/pat7/macroF1 | 0.7704 | 0.7704 ✓ | [0.895, 0.8, 0.667] |
| **s43_30ep + val-tuned dP=+0.40 dU=+0.20** | — | **0.7815** | 0.7815 ✓ | [0.886, 0.6, 0.741] |

→ Bit-for-bit reproduction of the published baseline. Pipeline validated.

## 2. 5-fold patient-stratified CV (v4 config) — honest generalization

Each fold: 4/5 patients train (with 15% held out for early stopping), 1/5 test (never seen).
Official s_murmur per fold:

| fold | test patients [A,U,P] | test s_murmur | recall [A,U,P] |
|---|---|---|---|
| 1 | 189 [139,14,36] | 0.7590 | [0.878, 0.643, 0.694] |
| 2 | 189 | 0.6648 | [0.82, 0.5, 0.583] |
| 3 | 189 | 0.7535 | [0.914, 0.357, 0.722] |
| 4 | 189 | 0.7374 | [0.827, 0.615, 0.694] |
| 5 | 189 | 0.6034 | [0.856, 0.615, 0.4] |
| **mean ± std** | — | **0.7036 ± 0.0604** | — |

**Out-of-fold ensemble** (each patient voted by a model that never saw them; 942 patients):
acc 0.7909 · macro-F1 0.6507 · **s_murmur 0.7040** · recall [0.859, 0.544, 0.620].

→ The honest generalization estimate is ~0.70 ± 0.06. The single-split 0.7815 sits at the
favorable end of this variance (rare classes: 10 Unknown / 27 Present in val → noisy selection).
Bottleneck quantified: Present recall 0.62 (OOF) is the biggest lever (weight 5× in s_murmur).

## 3. Model interventions (fixed split, seed 43 unless noted)

| variant | params | test s_murmur | recall [A,U,P] | verdict |
|---|---|---|---|---|
| v4 FusionNet (baseline) | 404K | 0.7778 | [0.933, 0.4, 0.741] | best |
| v5 scaled encoder (4 conv, 384ch) | 2.26M | 0.7296 | [0.933, 0.3, 0.667] | scale hurts (overfit) |
| focal loss γ=2 (α=sqrt-inv-freq) | 404K | 0.6111 | [0.771, 0.8, 0.444] | hurts (P recall ↓) |
| frozen wav2vec2-base + MLP | 94.4M frozen + 0.13M | 0.6778 | [0.857, 0.6, 0.556] | hurts |
| frozen wav2vec2-base + attn-fusion | 94.4M frozen + 0.13M | 0.6778 | [0.895, 0.3, 0.593] | hurts |

→ On this small dataset, a 404K-param from-scratch multi-position fusion model beats
(a) a 5.6× larger version of itself, (b) focal reweighting, and (c) frozen 94M-param
self-supervised features. The winning ingredients remain multi-position fusion +
patient-level class-balance sampling + long-enough training (30ep).

## 4. Threshold tuning (coarse grid, val-tuned) — noisy, mostly hurts

| model | val-tuned dP,dU | test s_murmur |
|---|---|---|
| s43_30ep | +0.20, 0.00 (coarse) | 0.7593 (tuning noise) |
| s42_30ep | +0.60, 0.00 | 0.7037 |
| s44_30ep | +0.60, +0.20 | 0.6481 |
| s45_30ep | +0.40, 0.00 | 0.7630 |
| ENS4 (probavg s42-45) | argmax | 0.7667 |

→ Consistent with original finding: ensembles/val-tuning don't beat the best single
model because val is too small to rank reliably (10 U / 27 P).

## Files
- `train_v4.py` — unchanged baseline (v4 config)
- `train_kfold.py` — 5-fold CV + OOF ensemble
- `train_focal.py` — focal-loss variant
- `train_v5.py` — scaled-up FusionNetV5
- `extract_w2v.py` / `train_w2v_head.py` — frozen wav2vec2 feature extraction + heads
- `tune_offsets.py` — coarse threshold tuning on saved probs
- Remote artifacts: `/root/heart-train/` (checkpoints, logs, w2v_features.npz, kfold_results.json)
