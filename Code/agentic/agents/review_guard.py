"""Deterministic safety and evidence checks for diagnosis drafts."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from agentic.platform.contracts import DiagnosisDraft


SUPERHEAT_PATTERN = re.compile(
    r"(?:"
    r"(?<![\w.])[-+]?\d+(?:\.\d+)?\s*(?:°\s*[CFK]|℃|℉|摄氏度|华氏度|[CFK]\b|"
    r"degrees?\s*(?:C|F|K)|celsius\b|fahrenheit\b|kelvin\b)"
    r"|(?:Δ\s*T|delta\s*T)\s*(?:(?:=|为|:|is)\s*)?[-+]?\d+(?:\.\d+)?"
    r"|(?:过热度|superheat)\s*(?:(?:=|为|:|is|of)\s*)?[-+]?\d+(?:\.\d+)?"
    r")",
    re.IGNORECASE,
)
CONTROL_PATTERN = re.compile(
    r"(?:直接|立即|执行|下发|指令|调压|升压|降压|加料|减料|加氟|投料|改变槽电压|修改工艺参数)"
)


CONTROL_ACTION_PATTERN = re.compile(
    r"(?:停止(?:操作|生产|运行)|暂停(?:操作|生产|运行)|"
    r"stop\s+(?:operation|production)|halt\s+(?:operation|production)|"
    r"(?:adjust|increase|raise|decrease|lower)\s+(?:the\s+)?(?:cell\s+)?voltage|"
    r"(?:add|increase|decrease|reduce)\s+(?:the\s+)?(?:feed|material)|"
    r"change\s+(?:the\s+)?process\s+parameters)",
    re.IGNORECASE,
)


def _is_finite_measurement(value: Any) -> bool:
    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, bool) or value in (None, ""):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return number == number and abs(number) != float("inf")


def _text_fields(draft: DiagnosisDraft) -> list[str]:
    values = [draft.diagnosis, draft.summary, *draft.findings, *draft.recommendations, *draft.inspection_items, *draft.open_questions]
    values.extend(str(value) for value in draft.metadata.values())
    return [value for value in values if value]


def review_draft(
    draft: DiagnosisDraft | Mapping[str, Any],
    known_source_ids: Iterable[str] = (),
    measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a review action without silently approving unsupported claims.

    ``measurements`` must contain explicit measured temperature, composition and
    initial crystallization temperature before numerical superheat language can
    pass. Visual C/N/H labels remain evidence, never a measured temperature.
    """

    if not isinstance(draft, DiagnosisDraft):
        draft = DiagnosisDraft.from_dict(dict(draft))
    known = set(known_source_ids)
    issues: list[str] = []
    cited = set(draft.source_ids)
    cited.update(ref.source_id for ref in draft.evidence)
    if cited and not known:
        issues.append("证据来源注册表为空，无法验证引用来源")
    elif known:
        missing = sorted(source_id for source_id in cited if source_id and source_id not in known)
        if missing:
            issues.append("引用了不存在的证据来源: " + ", ".join(missing))
    measurements = dict(measurements or {})
    has_superheat_inputs = all(
        _is_finite_measurement(measurements.get(name))
        for name in ("temperature", "composition", "initial_crystallization_temperature")
    )
    text = "\n".join(_text_fields(draft))
    if SUPERHEAT_PATTERN.search(text) and not has_superheat_inputs:
        issues.append("数值过热度表达缺少温度、成分和初晶温度实测依据")
    if CONTROL_PATTERN.search(text) or CONTROL_ACTION_PATTERN.search(text):
        issues.append("文本包含需要人工授权的工艺控制指令")
    if str(draft.metadata.get("llm_status", "")).startswith("fallback"):
        issues.append("诊断来自降级输出，必须人工复核")
    if not draft.findings and not draft.evidence:
        issues.append("诊断缺少可复核发现或证据")
    if (draft.diagnosis or draft.summary or draft.findings or draft.recommendations) and not cited:
        issues.append("missing_traceable_source_citation")
    if issues:
        action = "human_review" if any("降级" in item or "控制指令" in item for item in issues) else "request_more_evidence"
    else:
        action = "pass"
    if issues and (CONTROL_ACTION_PATTERN.search(text) or CONTROL_PATTERN.search(text) or str(draft.metadata.get("llm_status", "")).startswith("fallback")):
        action = "human_review"
    return {
        "action": action,
        "approved": action == "pass" and not draft.requires_human_review,
        "issues": issues,
        "evidence_ids": sorted(cited),
        "measurements_used": has_superheat_inputs,
        "implementation": "deterministic_review_guard_v1",
    }
