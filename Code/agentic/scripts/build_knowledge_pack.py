"""Build the first versioned aluminum-electrolysis knowledge pack."""

from __future__ import annotations

import json
from pathlib import Path


RULES = [
    {"rule_id": "superheat-definition-v1", "title": "过热度定义", "source": "aluminum-electrolysis-manual", "source_version": "v1", "effective_date": "2026-01-01", "preconditions": ["temperature_measured", "initial_crystallization_temperature_measured"], "applicability": ["process", "superheat"], "risk_level": "high", "allowed_output": ["measurement_interpretation", "human_review"], "text": "过热度需要温度和初晶温度的实测依据；火眼视觉标签用于状态证据，不能单独替代实测。"},
    {"rule_id": "fire-eye-visual-state-v1", "title": "火眼视觉状态", "source": "fire-eye-inspection-handbook", "source_version": "v1", "effective_date": "2026-01-01", "preconditions": ["quality_status_pass"], "applicability": ["visual", "fire_eye"], "risk_level": "medium", "allowed_output": ["visual_observation", "inspection_item"], "text": "视觉 Agent 记录亮度、纹理、火焰形态和跨帧变化，并将证据绑定到关键帧。"},
    {"rule_id": "evidence-boundary-v1", "title": "证据与建议边界", "source": "inspection-governance", "source_version": "v1", "effective_date": "2026-01-01", "preconditions": ["evidence_reference"], "applicability": ["report", "review"], "risk_level": "high", "allowed_output": ["inspection_suggestion", "human_review"], "text": "报告建议现场复核和持续观察事项，生产控制动作必须经人工授权。"},
]


def main() -> int:
    output = Path(__file__).resolve().parents[1] / "knowledge" / "rules_v1.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in RULES), encoding="utf-8")
    print(json.dumps({"output": str(output), "rules": len(RULES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
