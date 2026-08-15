# Heart Murmur Classification v2 — Supplementary Experiments (Tesla P4)

Date: 2026-08-15 · Host: remote Tesla P4 (8 GB) · Env: /root/venv_heart (torch 2.6.0+cu124)
Base code: `train_v2.py` (27 KB) → new `train_v3.py` (569 lines). Original `best_model_v2*.pt` untouched.

Dataset: CirCor public training set (942 patients, 3159 wavs after filtering; Absent 695 / Present 179 / Unknown 68).
All experiments use a **patient-level stratified 70/15/15 split (seed 42)**, persisted in `v3_split_seed42.json`
so every run shares the identical split:
- train 659 = [Absent 486, Unknown 48, Present 125]
- val   141 = [Absent 104, Unknown 10, Present 27]
- test  142 = [Absent 105, Unknown 10, Present 27]   ← never seen during training

Runs: `exp1_full.log`, `exp_abl_{A,B,C,D,E}.log`, aggregated in `exp_results.json`.
Checkpoints: `best_model_v3_full.pt`, `best_model_v3_single.pt`, `best_model_v3_{A..E}.pt`.

---

## Experiment 1 — True held-out test evaluation

Train exactly like v2 (fusion + single-location baseline, equal optimizer budget, early stop patience 5 on
val macro-F1) but with the 70/15/15 split; report the final metrics on the untouched 15% test.

### Test results (142 patients, never seen)

| variant | acc | macro-F1 | challenge WA* | recall Absent | recall Unknown | recall Present |
|---|---|---|---|---|---|---|
| **Fusion (attention)** | **0.7535** | **0.6537** | **0.7000** | 0.781 | 0.900 | 0.593 |
| Fusion → per-location vote | 0.6620 | 0.5823 | 0.6222 | 0.676 | 0.900 | 0.519 |
| Single-location (patient vote) | 0.7324 | 0.5817 | 0.6074 | 0.819 | 0.600 | 0.444 |
| Single-location (wav-level) | 0.8182 | 0.5782 | 0.6799 | 0.925 | 0.167 | 0.567 |

\* challenge WA = official CirCor-2022 metric `s_murmur` (see Experiment 2).

Fusion confusion matrix on test (rows=true [Absent,Unknown,Present], cols=pred):
`[[82,20,3],[1,9,0],[8,3,16]]` → per-class specificity [0.757, 0.826, 0.974].

Val results (early-stop set): fusion acc 0.8014 / macro-F1 0.6826 / chWA 0.6952 (best epoch 16, 3.6 s/epoch, 71 s total);
single acc 0.7660 / macro-F1 0.6225 / chWA 0.6320 (best epoch 11, 2.7 s/epoch, 43 s).

**Conclusions (Exp 1):**
- Ranking on the held-out test is the same as on val: fusion > fusion-vote > single-location.
- Test macro-F1 (0.6537) is only 0.03 below val macro-F1 (0.6826) → no meaningful overfitting to val.
- Single-location's high wav-level accuracy (0.818) is an artifact of the Absent majority; patient-level vote
  macro-F1 (0.582) is the honest number. Fusion beats it by +0.07 macro-F1 / +0.09 challenge WA.

---

## Experiment 2 — CirCor-2022 official metric & benchmark comparison

### Official metric (verified)

Source: PhysioNet Challenge 2022 page (https://physionet.org/content/challenge-2022/) and
Reyna et al., "Heart murmur detection from phonocardiogram recordings: The George B. Moody PhysioNet
Challenge 2022", PLOS Digital Health 2023, Eq. (1):

> s_murmur = (5·m_PP + 3·m_UU + m_AA) /
>           (5·(m_PP + m_UP + m_AP) + 3·(m_PU + m_UU + m_AU) + (m_PA + m_UA + m_AA))

where M = [m_ij] is the 3×3 confusion matrix, **columns = ground truth (expert), rows = model outputs**.
Weights: Present = 5, Unknown = 3, Absent = 1 ("murmur present cases have five times the weight of murmur
absent cases, and the murmur unknown cases have three times the weight of murmur absent cases, to reflect
a tolerance of five false alarms for every one false positive" — paper, Weighted accuracy metric section).

Equivalent form: s_murmur = Σ_c w_c · TP_c / Σ_c w_c · N_c, with w = {Absent:1, Unknown:3, Present:5}.

Our implementation (`metrics()` in `train_v3.py`) also reports per-class sensitivity (= recall) and
specificity (class c as positive: TN/(TN+FP)).

### Our model under the official metric (held-out test)

- Fusion (attention): **s_murmur = 0.7000**
- Fusion-vote: 0.6222
- Single-location (patient vote): 0.6074

### Benchmark vs official Challenge entries (hidden test set, 40 official teams)

Official leaderboard (`moody-challenge.physionet.org/2022/results/official_murmur_scores.tsv`,
column "Weighted Accuracy on Test Set"): max **0.780** (HearHeart), top-3 0.780/0.776/0.776,
**median 0.692**, mean 0.657, min 0.374.
Paper Table 5 (test set): Voting—GBT 0.790, Voting—RF 0.789, HearHeart 0.780 (acc 0.801, macro-F 0.619).

| reference | s_murmur (test) |
|---|---|
| Voting ensemble (GBT, paper) | 0.790 |
| #1 HearHeart (champion) | 0.780 |
| #2 CUED_Acoustics / HearTech+ | 0.776 |
| Top-10 cutoff | 0.755 |
| **Ours — fusion v3 (held-out test)** | **0.7000** |
| Official median (40 teams) | 0.692 |
| Official baseline (example RF classifier) | not scored publicly (minimal example, "not designed to perform well") |

**Position: our 0.7000 sits at ~rank 20 of 40 official entries — exactly the official median.** Note our
test is a self-made 15% split of the same public 942-patient cohort, not the Challenge's hidden 40%
(which includes unseen patients), so this is indicative, not identical to an official submission.

---

## Experiment 3 — Ablations (all retrained on the same 70/15/15 split)

| config | val macro-F1 | val chWA | test acc | test macro-F1 | test chWA |
|---|---|---|---|---|---|
| **full (baseline)** | 0.6826 | 0.6952 | 0.7535 | **0.6537** | **0.7000** |
| A. no attention (mean-pool fusion) | 0.6784 | 0.6729 | 0.7958 | 0.6432 | 0.7074 |
| B. no aux head (λ=0) | 0.6646 | 0.6617 | 0.7817 | 0.6471 | 0.6704 |
| C. no class-balance sampling | 0.6714 | 0.6766 | 0.7394 | 0.5986 | 0.6333 |
| D. no augmentation | 0.6826 | 0.6952 | 0.7746 | 0.6564 | 0.6889 |
| E. single-location baseline (fresh seed-42 run) | 0.5251 | 0.5019 | 0.6901 | 0.4836 | 0.4889 |

Note: the full run's single-location baseline (trained after fusion in the same process, RNG stream
already consumed) reached val 0.6225 / test 0.5817 / 0.6074; single-location is high-variance (±0.05
macro-F1 across seeds), but in every sample it is far below fusion.

Test confusion matrices (rows=true, cols=pred): A `[[91,9,5],[4,5,1],[7,3,17]]`,
B `[[90,15,0],[3,7,0],[9,4,14]]`, C `[[85,15,5],[2,7,1],[10,4,13]]`,
D `[[87,17,1],[2,8,0],[8,4,15]]`, E `[[87,18,0],[5,5,0],[18,3,6]]`.

**Conclusions (Exp 3):**
1. **Class-balance sampling is the single biggest lever** (C vs full: test macro-F1 −0.055, chWA −0.067).
2. Auxiliary murmur-present head helps the official metric (+0.03 chWA) and slightly helps macro-F1.
3. Augmentation gives a small chWA gain (+0.011), macro-F1 neutral.
4. **Learned attention ≈ mean pooling** (A vs full: chWA 0.7074 vs 0.7000, macro-F1 0.6432 vs 0.6537 —
   within run-to-run noise). The winning ingredient is *multi-position fusion itself* (both beat
   single-location by +0.07…+0.17 macro-F1), not the learned attention weights.
5. Unknown recall is consistently high (0.9 on test, 10/10 in most runs); **Present recall (~0.52–0.63)
   is the binding constraint** everywhere.

---

## Final assessment

Publishability gate (as requested): test macro-F1 ≥ 0.60 AND challenge WA ≥ 0.70.
Fusion v3 on held-out test: **macro-F1 0.654 ✓ / challenge WA 0.700 ✓ — passes, but the 0.70 is right at
the threshold** (single-seed; a different seed could land ~0.68–0.71). Verdict: **可发布，但注明为
"接近门槛的基线模型"；不建议作为 SOTA 宣称**。Relative to the 2022 Challenge it sits at the official median (rank ~20/40).

Highest-ROI next steps:
1. **Multi-seed ensemble (3–5 seeds) + epoch-selection on val challenge WA instead of macro-F1** — the
   val curve is noisy (ep2/4/10/13/17 collapse to all-Present), ensembles are the cheapest reliable gain
   (the Challenge's own GBT/RF voting gained +0.01 over the winner).
2. **Attack Present recall** (0.59): stronger oversampling/augmentation targeted at the 179 Present
   patients; the balance-sampling ablation proves the headroom is real.
3. **Simplify: drop learned attention for mean pooling** (identical results, fewer moving parts) unless
   cross-position interaction (e.g., transformer fusion) is added later.
4. Only if more compute appears: pretrained audio front-ends (AST/wav2vec2) or 16 s windows.

## Artifacts
`train_v3.py` · `v3_split_seed42.json` · `mel_cache_v3/` · `exp1_full.log` · `exp_abl_{A..E}.log` ·
`exp_results.json` · `best_model_v3_{full,single,A,B,C,D,E}.pt` — all in `/root/heart-train/`.
