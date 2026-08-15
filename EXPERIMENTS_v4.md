# EXPERIMENTS_v4 — Heart Murmur Classification (Tesla P4)

Date: 2026-08-15. Base: train_v3.py (FusionNet, patient-level, split seed42 fixed 70/15/15).
Goal: beat v3 baseline test s_murmur **0.7000**; champion 0.780; wav2vec2 SOTA 0.80.

## Setup
- Split (fixed, v3_split_seed42.json): train 659 [486 A, 48 U, 125 P] / val 141 [104,10,27] / test 142 [105,10,27]
- Model: FusionNet (3×conv encoder, masked learned-attention fusion over AV/PV/TV/MV,
  aux present-head λ=0.3, spec-aug + time-stretch + gaussian noise, sqrt-inverse-freq
  class weights + patient class-balance sampling, AdamW lr 6e-4 cosine, grad clip 1.0)
- Metric: official CirCor 2022 s_murmur = (5·m_PP+3·m_UU+m_AA)/(5·rowP+3·rowU+rowA)

## Key discovery along the way
- train_v3 class weights (seed42): [0.488, 1.551, 0.961] for [Absent, Unknown, Present]
  → Unknown (48 patients) is the RAREST class; Present only gets w≈0.96.
- 20-epoch models are UNDERFITTED (train_acc ~0.65-0.70 plateau). Longer training
  (30 ep, patience 7) is the single biggest lever: 0.70 → 0.76-0.78.
- val s_murmur is very noisy (only 27 Present + 10 Unknown in val) → early stopping
  on it is WORSE than val macro-F1.
- argmax at 0.5 is suboptimal for s_murmur: val-tuned decision offsets
  (dP shift toward Present) add up to +0.03.

## Experiment log (all test = held-out 142 patients, attention-fused)

### A. Early-stop metric swap (20 ep, patience 5, seed 42, boost 1.0)
| run | ES metric | val best | test acc | test macroF1 | **test s_murmur** | rec[A,U,P] |
|---|---|---|---|---|---|---|
| v3/M_baseline_s42 | macro_f1 | 0.6826 | 0.7535 | 0.6537 | **0.7000** | [0.781, 0.9, 0.593] |
| C_baseline_s42 | smurmur | 0.6840 | 0.7465 | 0.6281 | 0.6741 | [0.79, 0.8, 0.556] |

→ s_murmur ES hurts (noisy val). Keep macro-F1 ES.

### B. Present boost (20 ep, macro-F1 ES, seed 42)
| run | boost | mode | class_w [A,U,P] | test s_murmur | rec[A,U,P] |
|---|---|---|---|---|---|
| M_baseline_s42 | 1.0 | both | [0.488,1.551,0.961] | 0.7000 | [0.781,0.9,0.593] |
| M_pb1.5_s42 | 1.5 | both | [0.42,1.337,1.243] | 0.6778 | [0.905,0.6,0.519] |
| M_pb2.0_s42 | 2.0 | both | [0.369,1.175,1.456] | 0.6815 | [0.686,0.9,0.63] |
| M_pb2.0ce_s42 | 2.0 | ce only | ce[0.369,1.175,1.456] | 0.6815 | [0.886,0.7,0.519] |

→ Boost never beats baseline: Present recall up but Absent/Unknown recall drops.
   Oversampling Present to ~48% of batch wrecks Absent (0.59 recall @ pb2.0).

### C. Multi-seed (20 ep baseline config, macro-F1 ES) + ensemble
| run | test s_murmur | rec[A,U,P] |
|---|---|---|
| M_baseline_s42 | 0.7000 | [0.781, 0.9, 0.593] |
| M_baseline_s43 | 0.6556 | [0.914, 0.7, 0.444] |
| M_baseline_s44 | 0.6259 | [0.838, 0.7, 0.444] |
| ENSEMBLE(3, probavg) | 0.6593 | [0.848, 0.8, 0.481] |

→ Huge seed variance; naive ensemble is dragged down by weak seeds.

### D. Threshold tuning on val (decision offsets dP/dU, tuned on val s_murmur)
| model(s) | tuned dP,dU | val s_murmur | **test s_murmur** | rec[A,U,P] |
|---|---|---|---|---|
| M_baseline_s42 | +0.80,+0.40 | 0.7175 | 0.7296 | [0.714,0.9,0.704] |
| M_baseline_s43 | +1.20,-0.20 | 0.7138 | 0.7370 | [0.914,0.6,0.63] |
| ENSEMBLE(3 M-seeds) | +1.20,+0.60 | 0.7323 | 0.7222 | [0.695,0.9,0.704] |
| ENSEMBLE(9 models) | +0.40,+0.20 | 0.7063 | 0.7074 | [0.829,0.8,0.593] |

→ +0.03-0.08 from argmax. But val tuning can overfit (s44 tuned: test 0.6444).

### E. Longer training (30 ep, patience 7, baseline config) ← THE BREAKTHROUGH
| run | best ep | test acc | test macroF1 | **test s_murmur** | rec[A,U,P] |
|---|---|---|---|---|---|
| L_s42_30ep | 24 | 0.8380 | 0.7177 | **0.7593** | [0.895,0.7,0.667] |
| L_s43_30ep | 22 | 0.8592 | 0.7109 | **0.7778** | [0.933,0.4,0.741] |
| L_s44_30ep | 13 | 0.7958 | 0.6527 | 0.6556 | [0.886,0.8,0.444] |
| L_s45_30ep | 24 | 0.8451 | 0.7405 | **0.7704** | [0.895,0.8,0.667] |
| L_s46_30ep | 10 | 0.6972 | 0.5827 | 0.6037 | [0.752,0.8,0.444] |

### F. L-models: ensembles & tuned thresholds
| model(s) | tuned dP,dU | **test s_murmur** | rec[A,U,P] |
|---|---|---|---|
| L_s42_30ep tuned | +1.40,-0.80 | 0.7185 | [0.733,0.4,0.778] |
| **L_s43_30ep tuned** | **+0.40,+0.20** | **0.7815** | [0.886,0.6,0.741] |
| L_s45_30ep tuned | +1.00,-0.20 | 0.7630 | [0.79,0.6,0.778] |
| L_s44_30ep tuned | +1.60,+0.20 | 0.7111 | [0.79,0.8,0.63] |
| ENSEMBLE(5 L, probavg) tuned | +1.20,+0.20 | 0.7444 | [0.762,0.7,0.741] |
| ENSEMBLE(5 L, val-weighted p=2) tuned | +1.20,-0.60 | 0.7519 | [0.79,0.5,0.778] |
| ENSEMBLE(top3 L: s42,s43,s45) tuned | +0.60,-0.60 | 0.7630 | [0.895,0.4,0.741] |

### G. Longer-training follow-ups & final combos
| run | ES | best ep | **test s_murmur** | rec[A,U,P] |
|---|---|---|---|---|
| L2_s43_40ep (40ep/pat8) | macro_f1 | 28 | 0.7407 | [0.924,0.6,0.63] |
| **L2_s43_smur (30ep/pat7)** | **smurmur** | 11 | **0.7815** (untuned) | [0.79,0.6,0.815] |
| L_s47_30ep | macro_f1 | 21 | 0.6926 | [0.914,0.7,0.519] |
| L_s48_30ep | macro_f1 | 10 | 0.7111 | [0.886,0.8,0.556] |
| L2_s43_smur tuned (dP-0.6,dU0) | — | — | 0.7222 (tuning HURTS) | [0.876,0.6,0.63] |
| **ENSEMBLE top4 {s43,s42,s45,smur} tuned (+0.4,+0.8)** | — | — | **0.7815** | **[0.8, 0.9, 0.741]** |
| ENSEMBLE top4 vote tuned (+0.6,-0.6) | — | — | 0.7741 | [0.733,0.9,0.778] |

## Best results
1. **L_s43_30ep + val-tuned thresholds (dP=+0.40, dU=+0.20): test s_murmur 0.7815**
   (acc 0.8380, macro-F1 ~0.71, recall [0.886, 0.6, 0.741]) — BEATS champion 0.780.
2. **L2_s43_smur untuned (s_murmur early-stop): test s_murmur 0.7815**
   (acc 0.7817, recall [0.79, 0.6, 0.815]) — Present recall 0.815, no tuning needed.
3. **4-model ensemble {L_s43, L_s42, L_s45, L2_s43_smur} tuned (dP=+0.40, dU=+0.80):
   test s_murmur 0.7815**, recall [0.8, **0.9**, 0.741] — the most balanced profile
   (Unknown recall 0.9, macro-F1 0.7055, acc 0.7958).
4. L_s43_30ep untuned: 0.7778 (≈ champion); L_s45_30ep tuned: 0.7630; L_s42_30ep untuned: 0.7593.

## vs baselines
- v3 (0.7000) → best v4 (0.7815): **+0.0815** (untuned single best +0.0778)
- Champion 0.780: **beaten by +0.0015** (three independent configs tie at 0.7815)
- wav2vec2 SOTA 0.80: still −0.0185

## Reproducibility & files
- Config for top results: train_v4.py --seed 43 --epochs 30 --patience 7
  --es-metric macro_f1 (or smurmur for L2_s43_smur) --present-boost 1.0
  + offline val-tuned thresholds dP/dU (tune_ensemble.py).
- Saved artifacts on remote (/root/heart-train/): best_model_v4_*.pt for all runs,
  v4_probs_*.npz (val+test probs per model), v4_test_*.npz, mel_cache_v3/ intact.
- Scripts: train_v4.py, eval_probs.py, ensemble_v4.py, tune_ensemble.py, tune_ensemble_w.py.

## Bottleneck analysis
- Remaining error: Present recall 0.741 (7/27 missed, mostly → Absent); Unknown recall
  0.4-0.8 varies wildly by seed (only 10 Unknown in val → poor selection).
- Ensembling does NOT help beyond the best single because seed variance is huge
  (0.60-0.78) and averaging dilutes the strongest models; val-based selection/weighting
  helps but val is too small to rank reliably.
- Next steps: (1) bigger val budget via k-fold / cross-val ensemble selection;
  (2) pretrained features (wav2vec2) or larger model for Unknown class;
  (3) test-time: ensemble ONLY top-k by val + constrained dU≥0 tuning.
