# 🩺 Open Stethoscope × Crusaders — Murmur Screening HMC Demo

**When does the AI screening result stand on its own, and when do we hand
the patient over to a clinician's stethoscope?**

A working crossover between two of Noah's projects:

- **[Open Stethoscope](https://github.com/NoahIsARider/open-stethoscope)** — a
  404K-parameter heart-murmur classifier (multi-position attention fusion)
  that beats the PhysioNet 2022 Challenge champion on the held-out split
  (s_murmur 0.7815 vs 0.780).
- **[Crusaders](https://github.com/NoahIsARider/Crusaders)** — a
  human-machine collaboration (HMC) framework that decides *who holds the
  wheel* on every step, learns from its own performance (SECI), and measures
  the cost/safety trade-off.

This demo wires real AI inference into a Crusaders handover policy for a
village-clinic murmur screening scenario: the AI screens, the framework
decides when that's enough, and a clinician's stethoscope is the safety net.

## The handover policy (5 rules)

1. **Positive finding** (AI says *Present*, any confidence) → the doctor
   always confirms. A missed murmur is a missed valve disease.
2. **Ambiguous** (*Unknown*, or *Absent* with low confidence) → the doctor
   listens.
3. **Confident negative** (*Absent* ≥ 0.75) → the AI owns the result. This
   is where the tool saves the day.
4. **Doctor overload** → low-risk work routes back to the AI.
5. **No churn** → control never changes hands without a reason.

The thresholds live in the organisation's meta-knowledge
(`murmur_framework.py`), not in the code — so the SECI loop can tighten the
AI boundary as the clinic learns.

## Run it

```bash
pip install aicrusaders            # the HMC framework
cd demo/hmc-screening

# Simulation — no data needed. The AI's reads are sampled from the trained
# model's *actual* output distribution (models/score_library.json, extracted
# from the held-out test set), so handover behaviour matches production.
python run_demo.py

# Real inference — give it PCG wavs (CirCor-style 4 kHz recordings).
# The committed model (models/best_model_v4_s43_30ep.pt, 1.6 MB) runs the
# multi-position fusion on each file, then the framework routes the patient.
python run_demo.py --mode real /path/to/12345_AV.wav /path/to/67890_MV.wav
```

For real mode you also need the Open Stethoscope stack:
`pip install torch librosa soundfile numpy` (and a CUDA GPU is optional —
the 404K model runs fine on CPU).

Real PCG data is not redistributed here (CirCor has a research DUA); grab it
at <https://physionet.org/content/circor-heart-sound/> and point the demo at
the wavs.

## What you'll see

- **Walkthroughs**: per-step choreography — who decides, why control
  hands over, confidence and quality per step.
- **Clinic session**: a full patient roster through the framework, with a
  per-patient disposition (AI autonomous / AI screen + doctor plan /
  doctor-led screening).
- **SECI loop**: the organisation learns from its own performance — lessons,
  meta-knowledge patches, and the updated AI boundary.
- **Comparison**: the adaptive HMC framework vs **AI-only** (cheap but
  misses murmur positives) and **doctor-only** (safe but burns the session
  budget) on quality / efficiency / safety / fatigue / accuracy / autonomy /
  time.

## Files

| File | What it is |
|---|---|
| `murmur_framework.py` | The handover policy + task builder + meta-knowledge |
| `run_demo.py` | CLI entry: `sim` (deterministic, seed 7) and `real` modes |
| `models/best_model_v4_s43_30ep.pt` | Trained 404K model (test s_murmur 0.7778; 0.7815 val-tuned) |
| `models/score_library.json` | Real per-class output distributions from the test set |
| `outputs/` | Generated traces (written on each run) |

## Notes

- Simulation is deterministic (fixed seed) — reproducible on every machine.
- In simulation the AI is *graded against ground truth* (quality = probability
  mass on the true class); in real mode it is graded by self-confidence.
  The README of the parent project explains why the honest 5-fold CV estimate
  is 0.704 ± 0.06 and why single-split numbers should be read with care.
