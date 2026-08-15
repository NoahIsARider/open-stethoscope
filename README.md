# Open Stethoscope 🩺

**Open-source AI heart murmur detection for primary care.**

Cardiovascular disease is the leading cause of death in China. In county and township health centers, general practitioners often lack the training to interpret heart sounds. Open Stethoscope brings cardiology-grade auscultation to every village doctor — for free, open source, and on-device.

## Why

- CVD is the #1 cause of death in China and worldwide
- ~1.2 million heart sound recordings are needed at grassroots level each year, but there aren't enough cardiologists to read them
- Digital stethoscopes + AI can screen for heart murmurs (valvular disease) with accuracy approaching specialist auscultation
- Existing solutions are closed-source and enterprise-priced — this project is open and runs on edge hardware

## What

A deep learning system for heart murmur detection from phonocardiogram (PCG) recordings, built on the [CirCor DigiScope dataset](https://physionet.org/content/circor-heart-sound/) (PhysioNet Challenge 2022):

- **5,272 recordings** from **1,568 patients**, expert-annotated for murmurs
- **~400K parameters**, <1 GB VRAM, real-time CPU inference — designed for low-cost edge devices (Tesla P4 / CPU)
- Reproducible baseline with strict patient-disjoint evaluation and the official Challenge metric

## Approach

- **Multi-location attention fusion** — one patient = one sample. All available auscultation positions (AV/PV/TV/MV) are encoded by a shared 2D-CNN over log-mel spectrograms, then fused with a **masked learned attention** (missing positions masked out). Mirrors how a clinician listens to all positions before judging.
- **Class-imbalance handling** — sqrt-inverse-frequency patient sampling + class-weighted CE + auxiliary "murmur present?" binary head (λ=0.3).
- **Mel-domain augmentation** — random 8 s window, ±10% time stretch, SpecAugment-lite, Gaussian noise.
- **Patient-stratified evaluation** — strict patient-disjoint 70/15/15 train/val/test split (no leakage).

## Results

Held-out test set (142 patients, never seen during training), official CirCor-2022 challenge metric `s_murmur` (weights Absent:1 / Unknown:3 / Present:5):

| Model | Test acc | Test macro-F1 | Test s_murmur | recall A / U / P |
|-------|---------|---------------|---------------|------------------|
| **Fusion (masked attention)** | **0.7535** | **0.6537** | **0.7000** | 0.781 / 0.900 / 0.593 |
| Fusion → per-location vote | 0.6620 | 0.5823 | 0.6222 | 0.676 / 0.900 / 0.519 |
| Single-location (patient vote) | 0.7324 | 0.5817 | 0.6074 | 0.819 / 0.600 / 0.444 |

- Benchmark vs the official 2022 Challenge (40 teams, hidden test): champion HearHeart **0.780**, top-10 cutoff 0.755, **median 0.692** → our 0.7000 sits at ≈ rank 20/40 (official median). **Not a SOTA claim** — this is a reproducible, deployable baseline.
- Ablations (same split): class-balance sampling is the biggest lever (−0.055 macro-F1 / −0.067 s_murmur without it); auxiliary head +0.03 s_murmur; **learned attention ≈ mean pooling** — the winning ingredient is multi-position fusion itself (+0.07…+0.17 macro-F1 over single-location), not the attention weights.
- Present recall (~0.59) is the binding constraint; Unknown recall is consistently high (0.9).

Full details (official metric derivation, confusion matrices, per-ablation numbers): [EXPERIMENTS.md](EXPERIMENTS.md).

## Roadmap

- [x] Reproduce PhysioNet 2022 Challenge baseline
- [x] Train lightweight murmur classifier (multi-position fusion, s_murmur 0.70)
- [ ] Multi-seed ensemble + official-metric early stopping
- [ ] On-device inference demo (edge deployment)
- [ ] Chinese primary-care deployment guide

## Dataset

CirCor DigiScope v1.0.3 — open access, no application required:

```bash
wget https://physionet.org/static/published-projects/circor-heart-sound/circor-heart-sound-1.0.3.zip
```

## Quickstart

```bash
# 1. Download CirCor dataset (open access, ~560 MB)
wget https://physionet.org/static/published-projects/circor-heart-sound/circor-heart-sound-1.0.3.zip
unzip circor-heart-sound-1.0.3.zip

# 2. Train (fusion + single-location baseline, patient-level 70/15/15 split)
python train_v3.py --data-csv ./training_data.csv --data-dir ./training_data --workdir ./out

# 3. Ablations (same split, retrained)
python train_v3.py --ablation A   # no attention (mean pooling)
python train_v3.py --ablation B   # no auxiliary head
python train_v3.py --ablation C   # no class-balance sampling
python train_v3.py --ablation D   # no augmentation
```

Requirements: Python 3.10+, PyTorch (tested 2.6.0+cu124), librosa, soundfile, pandas, numpy. Full run ≈ 1 min on Tesla P4 / ~2 min on CPU.

## Repository layout

```
├── train_v2.py          # v2 training: multi-location masked-attention fusion
├── train_v3.py          # v3: 70/15/15 held-out split + ablation switches + official challenge metric
├── EXPERIMENTS.md       # full experimental report (test eval, benchmark, ablations)
├── exp_results.json     # aggregated metrics for all runs
├── v3_split_seed42.json # persisted patient-level split (reproducibility)
├── data/                # dataset (gitignored)
├── models/              # checkpoints (gitignored)
├── notebooks/           # exploratory notebooks
├── scripts/             # utility scripts
└── README.md
```

## Limitations

- Single-seed results; s_murmur 0.700 is right at the publishability threshold (different seeds land ~0.68–0.71). A multi-seed ensemble is the highest-ROI next step.
- Test split is a self-made 15% of the public cohort, not the Challenge's hidden 40% (includes unseen patients) — numbers are indicative, not an official submission.

## References

- Reyna et al., *Heart murmur detection from phonocardiogram recordings: The George B. Moody PhysioNet Challenge 2022*, PLOS Digital Health, 2023.
- Oliveira et al., *The CirCor DigiScope dataset: from murmur detection to heart sound classification*, 2021.

## License

MIT (code). Dataset: Open Data Commons Attribution License v1.0 (see [PhysioNet](https://physionet.org/content/circor-heart-sound/)).
