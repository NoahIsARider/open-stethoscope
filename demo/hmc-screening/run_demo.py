"""Open Stethoscope x Crusaders - murmur screening HMC demo.

Two modes:

    python run_demo.py                        # simulation (no data needed)
    python run_demo.py --mode real a.wav b.wav   # real AI inference on PCG wavs

The simulation is fully deterministic (fixed seed): the AI's reads are drawn
from the *actual output distribution* of the trained model (see
``models/score_library.json``, extracted from the held-out test set), so the
handover behaviour you see is the behaviour you get in production.

The real mode loads the committed 404K-parameter model and runs genuine
inference: per-position log-mel spectrograms -> multi-position attention
fusion -> 3-class probabilities, then routes the patient through the same
Crusaders handover framework.

Run with:  pip install aicrusaders
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from crusaders import (  # noqa: E402
    AlwaysAI,
    AlwaysExpert,
    HMCFramework,
    Role,
    SECIEngine,
    SimulationRunner,
)
from crusaders.adapters import AgentDecision  # noqa: E402
from crusaders.mediators import (  # noqa: E402
    CognitiveLoadMediator,
    DecisionTimeMediator,
    FatigueMediator,
    HandoverAccuracyMediator,
    MediatorBase,
    MetricResult,
)

import murmur_framework as MF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
OUTPUT_DIR = os.path.join(HERE, "outputs")
SCORE_LIB = json.load(open(os.path.join(MODELS, "score_library.json")))


# --------------------------------------------------------------------------- #
# Adapters: the real AI + a simulated clinician
# --------------------------------------------------------------------------- #

class MissedPositiveMediator(MediatorBase):
    """How many murmur-positive patients were handled by the AI alone?

    In simulation the ground truth is known, so this mediator counts the
    exact number of missed positives - the number a clinic must watch.
    Lower is better; zero is the only acceptable number.
    """

    key = "missed_positives"
    label = "Missed murmur positives (AI-owned Present patients)"
    higher_is_better = False

    def compute(self, outcome) -> MetricResult:
        meta = getattr(outcome.task, "metadata", {}) or {}
        tc = meta.get("true_class")
        present_total = 0
        ai_owned = 0
        for o in outcome.step_outcomes:
            if o.step.id.endswith("-screening"):
                if tc == 2:
                    present_total += 1
                    if o.controller is Role.AI:
                        ai_owned += 1
        return MetricResult(
            key=self.key,
            value=ai_owned,
            label=self.label,
            higher_is_better=self.higher_is_better,
            detail={"present_total": present_total, "ai_owned": ai_owned},
        )


class MurmurAIAdapter:
    """Runs the Open Stethoscope model (or its simulated score distribution).

    ``quality_estimate`` grades the read: in simulation we know ground truth
    and use the probability mass on the true class (honest grading); in real
    mode we fall back to the model's self-confidence.
    """

    def __init__(self, probs_by_patient: dict, true_by_patient: dict | None = None,
                 attention_by_patient: dict | None = None):
        self.probs = probs_by_patient
        self.true = true_by_patient or {}
        self.attention = attention_by_patient or {}

    def act(self, step, session) -> AgentDecision:
        pid = step.id.split("-")[0]
        probs = self.probs[pid]
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        if pid in self.true:
            quality = float(probs[self.true[pid]])
            graded = True
        else:
            quality = conf
            graded = False
        meta = {
            "probs": [round(float(x), 4) for x in probs],
            "verdict": MF.CLASS_NAMES[cls],
            "graded_against_truth": graded,
        }
        if pid in self.attention:
            meta["attention"] = {k: round(float(v), 3) for k, v in self.attention[pid].items()}
        return AgentDecision(
            role=Role.AI,
            step_id=step.id,
            action="ai_screening",
            content=f"AI: {MF.CLASS_NAMES[cls]} (p={conf:.2f})",
            confidence=conf,
            quality_estimate=quality,
            latency=2.5,
            metadata=meta,
        )


class MurmurExpertAdapter:
    """A competent village doctor with a stethoscope.

    Good but not perfect at auscultation (0.92), slow (listens to all four
    positions), and gets tired.
    """

    def __init__(self, accuracy: float = 0.92, latency: float = 12.0,
                 fatigue_growth: float = 0.06):
        self.accuracy = accuracy
        self.latency = latency
        self.fatigue_growth = fatigue_growth

    def act(self, step, session) -> AgentDecision:
        quality = max(0.05, self.accuracy - session.fatigue * 0.2)
        latency = self.latency * (0.7 + step.complexity * 0.6 + session.fatigue * 0.4)
        return AgentDecision(
            role=Role.EXPERT,
            step_id=step.id,
            action="auscultation_review",
            content="DOCTOR: auscultation review",
            confidence=quality,
            quality_estimate=quality,
            latency=latency,
            metadata={"expert": "village-doctor-sim"},
        )


# --------------------------------------------------------------------------- #
# Simulation roster
# --------------------------------------------------------------------------- #

def make_sim_roster(seed: int = 7) -> tuple[list, dict, dict]:
    """12 patients; AI reads sampled from the trained model's real score
    distribution (score_library) + small noise. Ground truth known."""
    rng = np.random.RandomState(seed)
    # class mix mirrors the held-out test set: 74% / 7% / 19%
    true_classes = [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 0]
    probs_by_patient, true_by_patient = {}, {}
    lib = {int(k): np.array(v, dtype=float) for k, v in SCORE_LIB["scores"].items()}
    for i, tc in enumerate(true_classes):
        pid = f"S{i+1:03d}"
        pool = lib[tc]
        base = pool[rng.randint(len(pool))]
        noise = rng.normal(0, 0.02, 3)
        probs = np.clip(base + noise, 0.01, 0.99)
        probs /= probs.sum()
        probs_by_patient[pid] = probs
        true_by_patient[pid] = tc
    tasks = [MF.build_task(pid, probs_by_patient[pid], true_by_patient[pid])
             for pid in sorted(probs_by_patient)]
    return tasks, probs_by_patient, true_by_patient


def make_runner(framework, seed: int = 7) -> SimulationRunner:
    return SimulationRunner(framework, mediators=[c(framework.metaknowledge) for c in DEFAULT_MEDIATORS], seed=seed)


DEFAULT_MEDIATORS = [
    FatigueMediator,
    CognitiveLoadMediator,
    DecisionTimeMediator,
    HandoverAccuracyMediator,
    MissedPositiveMediator,
]


def missed_positive_summary(report) -> None:
    """Print the safety line: how many Present patients slipped through AI-only screening."""
    missed = 0
    present_total = 0
    for run in report.runs:
        meta = getattr(run.outcome.task, "metadata", {}) or {}
        if meta.get("true_class") == 2:
            present_total += 1
            screening = next((o for o in run.outcome.step_outcomes
                              if o.step.id.endswith("-screening")), None)
            if screening is not None and screening.controller is Role.AI:
                missed += 1
    if present_total:
        print(f"  SAFETY: {present_total} murmur-positive patient(s) in roster, "
              f"{missed} handled by AI alone "
              f"({'OK' if missed == 0 else '⚠️  MISSED - the clinic must learn from this!'})")


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #

def _hr(char: str = "-", width: int = 74) -> str:
    return char * width


def walkthrough(framework, task, ai, expert) -> None:
    outcome = framework.run(task, ai=ai, expert=expert)
    print(f"\nPatient: {task.title}  ({task.id})")
    print(_hr())
    for o in outcome.step_outcomes:
        for event in o.handovers:
            print(f"  HANDOVER  {event.direction.value:<12} [{event.trigger.value}] {event.reason}")
        who = "AI" if o.controller is Role.AI else "DOCTOR"
        flag = "PASS" if o.passed else "FAIL"
        note = ""
        if o.step.id.endswith("-screening"):
            note = f"  ({o.decision.metadata.get('verdict', '')})"
        print(f"  {o.step.id:<20} {who:<6} conf={o.decision.confidence:.2f} "
              f"qual={o.decision.quality_estimate:.2f}  {flag}{note}")
    print(f"  -> {outcome.passed_steps}/{outcome.n_steps} steps passed, "
          f"{outcome.n_handovers} handover(s), {outcome.elapsed:.1f}s simulated")
    # safety flag: in simulation we know the truth - did the AI own a positive?
    meta = getattr(task, "metadata", {}) or {}
    if meta.get("true_class") == 2:
        screening = next((o for o in outcome.step_outcomes if o.step.id.endswith("-screening")), None)
        if screening is not None and screening.controller is Role.AI:
            print("  ⚠️  MISSED POSITIVE: AI owned a murmur-PRESENT patient - "
                  "this is the failure the clinic must learn from")


def _disposition(outcome) -> str:
    doctor_any = any(o.controller is Role.EXPERT for o in outcome.step_outcomes)
    screening = next((o for o in outcome.step_outcomes if o.step.id.endswith("-screening")), None)
    if screening is not None and screening.controller is Role.EXPERT:
        return "doctor-led screening"
    if doctor_any:
        return "AI screen + doctor plan"
    return "AI handles autonomously"


def session_table(framework, tasks, ai, expert) -> None:
    report = make_runner(framework).evaluate_tasks(tasks, ai=ai, expert=expert)
    print(f"\nClinic morning - {len(tasks)} patients through '{framework.name}'")
    print(_hr())
    print(f"{'patient':<9}{'title':<30}{'who handled it':<24}{'handovers':>9}  pass")
    for run in report.runs:
        print(f"{run.task_id:<9}{run.outcome.task.title:<30}"
              f"{_disposition(run.outcome):<24}{run.outcome.n_handovers:>9}  "
              f"{run.outcome.passed_steps}/{run.outcome.n_steps}")
    print()
    missed_positive_summary(report)
    return report


def seci_loop(framework, tasks, ai, expert, rounds: int = 3):
    print(f"\nSECI feedback loop - the clinic learns ({rounds} rounds)")
    print(_hr())
    meta = framework.metaknowledge
    for r in range(1, rounds + 1):
        report = make_runner(framework).evaluate_tasks(tasks, ai=ai, expert=expert)
        missed_positive_summary(report)
        update = SECIEngine(meta, learning_rate=0.2).run(report)
        print(f"\nRound {r}:")
        for lesson in update.lessons:
            print(f"  [{lesson.stage:<15}] {lesson.content}")
        if update.patch:
            print(f"  patch       {update.patch}")
        if update.recommendations:
            print(f"  recommend   {update.recommendations}")
        meta = update.apply(meta)
        # Domain learning: a missed positive is a hard signal - tighten the
        # AI boundary so ambiguous / positive-prone profiles reach the doctor.
        missed = sum(1 for run in report.runs
                     if (getattr(run.outcome.task, "metadata", {}) or {}).get("true_class") == 2
                     and any(o.step.id.endswith("-screening") and o.controller is Role.AI
                             for o in run.outcome.step_outcomes))
        if missed:
            old_floor = meta.ai_boundary.get("min_confidence", 0.75)
            meta.ai_boundary["min_confidence"] = min(0.95, old_floor + 0.10)
            old_cx = meta.ai_boundary.get("max_complexity", 0.6)
            meta.ai_boundary["max_complexity"] = max(0.3, old_cx - 0.1)
            print(f"  domain     missed positives detected ({missed}) -> AI boundary tightened "
                  f"(min_confidence {old_floor:.2f} -> {meta.ai_boundary['min_confidence']:.2f}, "
                  f"max_complexity {old_cx:.2f} -> {meta.ai_boundary['max_complexity']:.2f})")
        framework.metaknowledge = meta
    print("\nMeta-knowledge after learning:")
    print(json.dumps(meta.summary(), ensure_ascii=False, indent=2))
    return meta


def comparison(tasks, ai, expert, seed: int = 7) -> None:
    meta = MF.default_metaknowledge()

    def _baseline(name, policies):
        return HMCFramework(name=name, metaknowledge=meta, policies=policies,
                            step_evaluator=MF.murmur_step_evaluator)

    frameworks = {
        "adaptive HMC (this demo)": MF.MurmurScreeningFramework(
            name="murmur-hmc-adaptive", metaknowledge=meta),
        "doctor-only": _baseline("doctor-only", [AlwaysExpert()]),
        "ai-only": _baseline("ai-only", [AlwaysAI()]),
    }
    print(f"\nComparison - one clinic morning under 3 frameworks (seed={seed})")
    print(_hr())
    header = f"{'framework':<26}{'quality':>8}{'eff':>7}{'missed':>8}{'fatigue':>9}{'accuracy':>10}{'handover':>9}{'time(s)':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for name, fw in frameworks.items():
        report = make_runner(fw, seed=seed).evaluate_tasks(tasks, ai=ai, expert=expert)
        agg = report.aggregated()
        missed = sum(
            1 for run in report.runs
            if (getattr(run.outcome.task, "metadata", {}) or {}).get("true_class") == 2
            and any(o.step.id.endswith("-screening") and o.controller is Role.AI
                    for o in run.outcome.step_outcomes)
        )
        handovers = sum(run.outcome.n_handovers for run in report.runs) / max(1, len(report.runs))
        rows.append((name, report))
        print(f"{name:<26}{agg['quality']:>8.2f}{agg['efficiency']:>7.2f}{missed:>8}"
              f"{agg['fatigue']:>9.2f}{agg.get('handover_accuracy', 0):>10.2f}{handovers:>9.1f}"
              f"{agg['decision_time']:>9.2f}")
    return rows


# --------------------------------------------------------------------------- #
# Real inference
# --------------------------------------------------------------------------- #

def real_inference(wav_paths: list[str]):
    """Load the committed model and predict on real PCG wavs."""
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root for train_v4
    import train_v4 as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(MODELS, "best_model_v4_s43_30ep.pt"), map_location=device)
    model = T.FusionNet(n_class=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[model] loaded {ckpt['arch']} (seed {ckpt.get('seed')}, best ep {ckpt.get('epoch')}) "
          f"on {device}", flush=True)

    probs_by_patient, attention_by_patient = {}, {}
    for wav in wav_paths:
        pid = os.path.splitext(os.path.basename(wav))[0]
        x, sr = T.sf.read(wav, dtype="float32")
        if sr != T.SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=T.SR)
        m = _mel_from_audio(T, x)
        loc = "MV"  # default location for a bare wav
        m = m[:, :T.N_FRAMES] if m.shape[1] >= T.N_FRAMES else np.pad(
            m, ((0, 0), (0, T.N_FRAMES - m.shape[1])))
        xb = np.zeros((1, 4, 1, T.N_MELS, T.N_FRAMES), np.float32)
        xb[0, T.LOC2IDX[loc], 0] = m
        mask = np.zeros((1, 4), bool); mask[0, T.LOC2IDX[loc]] = True
        with torch.no_grad():
            logits, _, alpha = model(torch.from_numpy(xb).to(device),
                                     torch.from_numpy(mask).to(device))
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        probs_by_patient[pid] = probs
        attention_by_patient[pid] = dict(zip(T.LOCS, alpha[0].cpu().numpy()))
        attn_str = {k: round(float(v), 3) for k, v in attention_by_patient[pid].items()}
        print(f"[infer] {wav} -> {MF.CLASS_NAMES[int(np.argmax(probs))]} "
              f"p={np.round(probs, 3).tolist()} attn={attn_str}", flush=True)
    return probs_by_patient, attention_by_patient


def _mel_from_audio(T, x):
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=T.SR, n_fft=T.N_FFT, hop_length=T.HOP,
                                       n_mels=T.N_MELS, fmin=T.FMIN, fmax=T.FMAX)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Open Stethoscope x Crusaders HMC demo")
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("wavs", nargs="*", help="PCG wav files for --mode real")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 74)
    print("Open Stethoscope x Crusaders")
    print("Human-machine collaboration for heart-murmur screening")
    print("=" * 74)

    meta = MF.default_metaknowledge()
    framework = MF.MurmurScreeningFramework(name="murmur-hmc-adaptive", metaknowledge=meta)

    if args.mode == "sim":
        tasks, probs, true = make_sim_roster()
        ai = MurmurAIAdapter(probs, true)
        expert = MurmurExpertAdapter()

        print("\n[1] Walkthroughs (3 representative patients)")
        for pid in ("S002", "S008", "S011"):  # a confident negative, an Unknown, a Present
            task = next(t for t in tasks if t.id == pid)
            walkthrough(framework, task, ai, expert)

        print("\n[2] Full clinic morning")
        session_table(framework, tasks, ai, expert)

        print("\n[3] SECI feedback loop")
        seci_loop(framework, tasks, ai, expert)

        print("\n[4] Framework comparison")
        rows = comparison(tasks, ai, expert)
    else:
        if not args.wavs:
            print("usage: python run_demo.py --mode real <wav> [<wav> ...]")
            return 1
        probs, attn = real_inference(args.wavs)
        ai = MurmurAIAdapter(probs, attention_by_patient=attn)
        expert = MurmurExpertAdapter()
        tasks = [MF.build_task(pid, probs[pid]) for pid in sorted(probs)]
        print("\n[1] Screening walkthroughs (real inference)")
        for task in tasks:
            walkthrough(framework, task, ai, expert)
        print("\n[2] Clinic session")
        session_table(framework, tasks, ai, expert)
        rows = None

    print("\nAll done. Reports would be written to ./outputs/ (see smart_clinic for the "
          "full report-writing pattern).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
