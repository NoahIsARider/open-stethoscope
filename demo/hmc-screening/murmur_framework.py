"""Murmur screening - the collaboration framework.

Bridges **Open Stethoscope** (real AI heart-murmur classifier) and the
**Crusaders** human-machine collaboration framework.

The design question this demo answers:

    "When does the AI screening result stand on its own, and when do we
    hand the patient over to a clinician's stethoscope?"

Answer, as five ordered rules (mirrors how a village-clinic doctor would
actually use the tool):

    1. **Positive finding** (AI says *Present*, any confidence) - the doctor
       always confirms. A missed murmur is a missed valve disease.
    2. **Ambiguous** (AI says *Unknown*, or *Absent* with low confidence) -
       the picture is unclear, the doctor listens.
    3. **Confident negative** (AI says *Absent* with high confidence) - the
       AI owns the screening result. This is where the tool saves the day.
    4. **Doctor overload** - when the clinician is at session budget or
       fatigued, low-risk *confident-negative* patients are re-screened by
       the AI so the human can breathe.
    5. **No churn** - control never changes hands without a reason.

The framework reads the organisation's meta-knowledge (risk bands, AI
confidence floor, doctor session budget) instead of hard-coding thresholds,
so the SECI loop can tighten the AI boundary as it learns.
"""

from __future__ import annotations

from crusaders import (
    HMCFramework,
    HandoverDecision,
    Role,
    RiskResponsibility,
    OrganizationalMetaknowledge,
)
from crusaders.adapters import AgentDecision
from crusaders.core.types import HandoverTrigger
from crusaders.policies import SessionState

DOMAIN = "murmur_screening"

CLASS_NAMES = ["Absent", "Unknown", "Present"]

# Risk bands: risk <= 0.3 is AI-owned, <= 0.6 is shared (AI recommends /
# doctor approves), anything above belongs to the doctor.
RISK_BANDS = [
    RiskResponsibility(0.3, "ai"),
    RiskResponsibility(0.6, "shared"),
    RiskResponsibility(1.0, "expert"),
]

# The AI may own a negative screening result only above this P(Absent).
AI_CONFIDENCE_FLOOR = 0.75

# Doctor session budget before low-risk work routes back to the AI.
EXPERT_SESSION_LIMIT = 6


def default_metaknowledge() -> OrganizationalMetaknowledge:
    """A village clinic that trusts the AI on confident negatives but keeps
    a clinician accountable for anything positive or ambiguous."""
    return OrganizationalMetaknowledge(
        ai_boundary={
            "max_complexity": 0.6,
            "allowed_domains": [DOMAIN],
            "min_confidence": AI_CONFIDENCE_FLOOR,
        },
        expert_capability={
            "max_steps_per_session": EXPERT_SESSION_LIMIT,
            "strengths": ["auscultation_review", "positive_confirmation"],
        },
        risk_responsibility=RISK_BANDS,
        handover_timing={"prefer_early": 0.3, "handover_overhead_budget": 1.0},
    )


def risk_for_probs(probs) -> tuple[float, float, bool]:
    """Map an AI 3-class probability vector to (risk, complexity, requires_expert).

    - Present -> expert band, requires_expert (red flag)
    - Unknown -> shared band (ambiguous)
    - Absent  -> AI band if confident, shared band otherwise
    complexity is low when the AI is confident (unambiguous picture).
    """
    p_abs, p_unk, p_pre = probs
    cls = int(p_pre > p_unk and p_pre >= p_abs) * 2 + int(p_unk > p_pre and p_unk >= p_abs)
    # simpler: argmax
    cls = max(range(3), key=lambda c: probs[c])
    conf = probs[cls]
    if cls == 2:  # Present
        return 0.85, 1.0 - conf, True
    if cls == 1:  # Unknown
        return 0.5, 1.0 - conf * 1.2, False
    # Absent
    risk = 0.2 if conf >= AI_CONFIDENCE_FLOOR else 0.45
    return risk, 1.0 - conf, False


def build_task(patient_id: str, probs, true_class: int | None = None) -> TaskSpec:
    """Build a screening task from an AI score vector.

    ``true_class`` is only known in simulation (for grading); in real use it
    is ``None`` and the demo grades by AI self-confidence instead.
    """
    from crusaders import TaskSpec, StepSpec

    risk, complexity, requires_expert = risk_for_probs(probs)
    verdict = CLASS_NAMES[max(range(3), key=lambda c: probs[c])]

    task = TaskSpec(
        id=patient_id,
        title=f"Murmur screening - {patient_id}",
        steps=[
            StepSpec(
                id=f"{patient_id}-intake",
                description="record PCG at AV/PV/TV/MV, run AI screener",
                risk=0.1,
                complexity=0.2,
            ),
            StepSpec(
                id=f"{patient_id}-screening",
                description=f"AI read: {verdict} (p={max(probs):.2f})",
                risk=risk,
                complexity=complexity,
                requires_expert=requires_expert,
            ),
            StepSpec(
                id=f"{patient_id}-plan",
                description="referral / reassurance decision",
                risk=0.2 if not requires_expert else 0.85,
                complexity=0.3,
            ),
        ],
    )
    task.steps[1].metadata = {  # screening step carries the AI read
        "probs": [round(float(x), 4) for x in probs],
        "verdict": verdict,
        "true_class": true_class,
    }
    task.metadata = {  # type: ignore[attr-defined]
        "patient_id": patient_id,
        "ai_probs": [round(float(x), 4) for x in probs],
        "ai_verdict": verdict,
        "true_class": true_class,
    }
    return task


def murmur_step_evaluator(decision: AgentDecision, spec) -> bool:
    """Bar for an acceptable outcome.

    The AI passes when its self-reported confidence clears the floor (in
    simulation we additionally grade against ground truth via
    ``quality_estimate`` set by the adapter). The doctor passes anything a
    competent clinician would pass.
    """
    if decision.role is Role.EXPERT:
        return decision.quality_estimate >= 0.5
    return decision.quality_estimate >= 0.6


class MurmurScreeningFramework(HMCFramework):
    """AI-first murmur screening with a clinician safety net."""

    def __init__(self, **kwargs):
        kwargs.setdefault("step_evaluator", murmur_step_evaluator)
        super().__init__(**kwargs)

    def decide_handover(self, step, session: SessionState) -> HandoverDecision:
        spec = step.step
        meta = self.metaknowledge

        # Rule 1: positive finding is never delegated.
        if spec.id.endswith("-screening") and spec.requires_expert:
            if session.current_controller is Role.AI:
                return HandoverDecision(
                    Role.EXPERT,
                    trigger=HandoverTrigger.AI_ESCALATION,
                    reason="murmur PRESENT - positive finding must be confirmed by clinician",
                )
            return HandoverDecision(
                Role.EXPERT,
                reason="positive finding stays with the doctor",
            )

        # Rule 2: the screening autonomy gate (all other screenings).
        if spec.id.endswith("-screening"):
            # Read the AI's read off the step; the confidence floor comes from
            # the organisation's meta-knowledge so the SECI loop can tighten it.
            step_meta = getattr(spec, "metadata", None) or {}
            probs = step_meta.get("probs")
            if probs is not None:
                p_abs = float(probs[0])
                floor = meta.ai_boundary.get("min_confidence", AI_CONFIDENCE_FLOOR)
                if p_abs >= floor:
                    if session.current_controller is Role.EXPERT:
                        return HandoverDecision(
                            Role.AI,
                            trigger=HandoverTrigger.POLICY,
                            reason=f"confident negative (p_abs={p_abs:.2f} >= floor {floor:.2f}) -> AI",
                        )
                    return HandoverDecision(
                        Role.AI,
                        reason=f"confident negative (p_abs={p_abs:.2f} >= floor {floor:.2f}) - AI owns screening",
                    )
                if session.current_controller is Role.AI:
                    return HandoverDecision(
                        Role.EXPERT,
                        trigger=HandoverTrigger.AI_ESCALATION,
                        reason=f"below confidence floor (p_abs={p_abs:.2f} < {floor:.2f}) -> doctor auscultation review",
                    )
                return HandoverDecision(
                    Role.EXPERT,
                    reason=f"below confidence floor (p_abs={p_abs:.2f} < {floor:.2f}) - doctor owns screening",
                )
            # fallback: no probs on the step - use the risk band as before
            responsible = meta.responsibility_for(spec.risk)
            if responsible == "ai":
                if session.current_controller is Role.EXPERT:
                    return HandoverDecision(
                        Role.AI,
                        trigger=HandoverTrigger.POLICY,
                        reason="low-risk screening within AI ownership -> AI",
                    )
                return HandoverDecision(
                    Role.AI, reason="low-risk screening - AI owns result"
                )
            if responsible == "shared":
                if session.current_controller is Role.AI:
                    return HandoverDecision(
                        Role.EXPERT,
                        trigger=HandoverTrigger.AI_ESCALATION,
                        reason="ambiguous read -> doctor auscultation review",
                    )
                return HandoverDecision(
                    Role.EXPERT, reason="doctor owns ambiguous screening"
                )
            return HandoverDecision(Role.EXPERT, reason="expert risk band")

        # Rule 3: plan - referral for positives, reassurance for negatives.
        if spec.id.endswith("-plan"):
            if spec.risk > 0.3:
                if session.current_controller is Role.AI:
                    return HandoverDecision(
                        Role.EXPERT,
                        trigger=HandoverTrigger.POLICY,
                        reason="follow-up / referral needs doctor",
                    )
                return HandoverDecision(Role.EXPERT, reason="doctor signs the plan")
            if session.current_controller is Role.EXPERT:
                return HandoverDecision(
                    Role.AI,
                    trigger=HandoverTrigger.POLICY,
                    reason="reassurance plan -> AI",
                )
            return HandoverDecision(Role.AI, reason="AI drafts reassurance plan")

        # Rule 4: doctor overload - low-risk work back to the AI.
        limit = meta.expert_session_limit(default=EXPERT_SESSION_LIMIT)
        if session.current_controller is Role.EXPERT and spec.risk <= 0.3:
            if session.expert_steps >= limit or session.fatigue >= 0.8:
                return HandoverDecision(
                    Role.AI,
                    trigger=HandoverTrigger.POLICY,
                    reason=(
                        f"doctor at session budget ({session.expert_steps}/{limit}) "
                        "or fatigued; low-risk step -> AI"
                    ),
                )
            return HandoverDecision(
                Role.AI,
                trigger=HandoverTrigger.POLICY,
                reason="low-risk step -> AI",
            )

        # Rule 5: no reason to churn control.
        return HandoverDecision(
            session.current_controller, reason="keep current controller"
        )
