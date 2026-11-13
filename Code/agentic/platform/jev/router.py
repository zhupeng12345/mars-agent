"""Risk-aware diagnosis routing with a deterministic gate and optional Laya backend."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from agentic.platform.contracts import InspectionFeatures, JevRouteDecision

from .policy import JevRuntimePolicy
from .rules import safety_route


class JevRouter:
    """Route a diagnosis to one of four controlled loops."""

    def __init__(
        self,
        *,
        gate_version: str = "alu-jev-safety-0.1",
        model_profile: str = "deterministic_alu_router_v1",
        adapter: Any = None,
        policy: Optional[JevRuntimePolicy] = None,
        policy_version: str = "base",
        policy_hash: str = "builtin",
    ):
        self.gate_version = gate_version
        self.model_profile = model_profile
        self.adapter = adapter
        self.policy = policy or JevRuntimePolicy()
        self.policy_version = policy_version
        self.policy_hash = policy_hash

    def route(self, features: InspectionFeatures | Mapping[str, Any]) -> JevRouteDecision:
        if not isinstance(features, InspectionFeatures):
            features = InspectionFeatures.from_dict(dict(features))
        fallback_reasons: list[str] = []
        route, reasons = safety_route(features)
        confidence = float(features.visual_confidence or 0.0)
        router_backend = "deterministic_rule"
        if self.adapter is not None and route == "full_loop":
            try:
                proposal = self.adapter.predict(features)
                if proposal and proposal.get("route") in {"simple_loop", "full_loop", "retake", "human"} and float(proposal.get("confidence", 0.0)) >= self.policy.minimum_laya_confidence:
                    proposal_route = str(proposal["route"])
                    # A learned/remote router is an optional optimization.  It
                    # may select a safer route, but it must not downgrade a
                    # hard safety condition to the cheap loop.
                    hard_gates = self._hard_safety_gates(features)
                    if proposal_route == "simple_loop" and hard_gates:
                        reasons.extend(["adapter_rejected_hard_gate", *hard_gates])
                    else:
                        route = proposal_route
                        router_backend = "laya_adapter"
                        reasons.extend(["adapter_proposal"])
                        if proposal.get("reason"):
                            reasons.append(f"adapter_reason:{str(proposal['reason'])[:120]}")
                        confidence = float(proposal.get("confidence", confidence))
                else:
                    router_backend = "laya_adapter_fallback"
                    reasons.append("adapter_abstain_or_low_confidence")
            except Exception:
                fallback_reasons.append("adapter_error")
                route = "human"
                router_backend = "safety_fallback"
        if fallback_reasons:
            reasons.extend(fallback_reasons)
        decision = JevRouteDecision(
            route=route,
            difficulty_score=min(1.0, max(float(features.novelty_score or 0.0), float(features.evidence_conflict or 0.0), 1.0 - confidence)),
            novelty_score=float(features.novelty_score or 0.0),
            confidence=min(1.0, max(0.0, confidence)),
            risk_level="high" if float(features.risk_score or 0.0) >= 0.75 else ("medium" if float(features.risk_score or 0.0) >= 0.45 else "low"),
            similar_case_ids=self._similar_case_ids(features),
            evidence_gaps=["knowledge" ] if float(features.knowledge_coverage or 0.0) < 0.34 else [],
            reasons=reasons,
            process_context_used=bool(features.process_snapshot and features.process_snapshot.alignment_quality in {"exact", "interval", "segment"}),
            alignment_quality=features.process_snapshot.alignment_quality if features.process_snapshot else "unknown",
            router_backend=router_backend,
            model_profile=self.model_profile,
            gate_version=self.gate_version,
            policy_version=self.policy_version,
            policy_hash=self.policy_hash,
            run_id=features.run_id,
            thread_id=features.thread_id,
        )
        decision.validate()
        return decision

    decide = route

    @staticmethod
    def _hard_safety_gates(features: InspectionFeatures) -> list[str]:
        """Return conditions that prohibit a model downgrade to simple_loop.

        These gates intentionally duplicate the policy thresholds in
        ``jev.rules`` at the adapter boundary.  Keeping the check here means
        a remote or experimental adapter cannot bypass deterministic safety
        behavior when it proposes a cheap route.
        """
        values = features.features or {}
        quality = str(values.get("quality_status", "pass")).lower()
        gates: list[str] = []
        if quality in {"retake", "human"}:
            gates.append(f"quality_status:{quality}")
        if values.get("target_present") is False:
            gates.append("target_missing")
        if float(features.novelty_score or 0.0) >= 0.85:
            gates.append("novelty_high")
        if float(features.visual_confidence or 0.0) < 0.45:
            gates.append("visual_confidence_low")
        if float(features.risk_score or 0.0) >= 0.75:
            gates.append("risk_high")
        if float(features.evidence_conflict or 0.0) >= 0.45:
            gates.append("evidence_conflict_high")
        if float(features.knowledge_coverage or 0.0) < 0.34:
            gates.append("knowledge_coverage_low")
        return gates

    @staticmethod
    def _similar_case_ids(features: InspectionFeatures) -> list[str]:
        """Extract stable case IDs supplied by retrieval without trusting text."""
        values = features.features or {}
        raw = values.get("similar_case_ids", [])
        if not raw:
            raw = [item.get("case_id") for item in values.get("case_matches", []) if isinstance(item, Mapping)]
        result: list[str] = []
        for item in raw if isinstance(raw, (list, tuple, set)) else []:
            text = str(item).strip()
            if text and text not in result and len(text) <= 256:
                result.append(text)
        return result[:20]


class LayaJevRouter(JevRouter):
    """Laya typed-decisions proposals guarded by the aluminium process policy."""

    pass


def create_jev_router(
    backend: str = "deterministic_rule",
    *,
    laya_model: str = "typed-decisions",
    policy: Optional[JevRuntimePolicy] = None,
    policy_version: str = "base",
    policy_hash: str = "builtin",
) -> JevRouter:
    """Create a configured router without loading Laya weights until inference."""
    if backend == "deterministic_rule":
        return JevRouter(policy=policy, policy_version=policy_version, policy_hash=policy_hash)
    if backend == "laya_adapter":
        from .laya_adapter import LayaAdapter

        return LayaJevRouter(
            gate_version="alu-jev-safety-0.1",
            model_profile=f"laya_{laya_model.replace('-', '_')}_alu_adapter_v1",
            adapter=LayaAdapter(model=laya_model),
            policy=policy,
            policy_version=policy_version,
            policy_hash=policy_hash,
        )
    raise ValueError(f"unsupported JEV router backend: {backend}")
