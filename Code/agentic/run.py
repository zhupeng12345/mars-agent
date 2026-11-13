from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agentic.workflow import run_diagnosis
from agentic.scripts.demo_run import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local fire-eye multi-agent diagnosis system.")
    parser.add_argument("--input", required=True, help="Image or video path.")
    parser.add_argument("--mode", choices=("modern", "legacy"), default="modern", help="modern multi-frame/JEV/HITL flow or the legacy single-frame flow")
    parser.add_argument("--model", default=os.getenv("MARS_AGENT_MODEL", "qwen3-vl:latest"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--vision-device", default=os.getenv("MARS_VISION_DEVICE", "cpu"))
    parser.add_argument("--knowledge-dir", type=Path, default=Path(__file__).resolve().parent / "knowledge")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON result.")
    parser.add_argument("--run-id", help="Stable run ID for replay and human review.")
    parser.add_argument("--tenant-id", default="tenant_default")
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--operator-id", default="local_operator")
    parser.add_argument("--runtime-dir", type=Path, help="Directory for JSONL traces and candidate cases.")
    parser.add_argument("--no-model", action="store_true", help="Use the model-free visual smoke path in modern mode.")
    parser.add_argument("--weights", type=Path, help="Optional F-GMVAE-T checkpoint in modern mode.")
    parser.add_argument("--llm-profile", choices=("fake_json", "ollama_local", "qwen_api", "fallback_human"), default=None, help="Diagnosis aggregation provider profile in modern mode.")
    parser.add_argument("--jev-backend", choices=("deterministic_rule", "laya_adapter"), default=None, help="Risk-aware diagnostic route backend in modern mode.")
    parser.add_argument("--laya-model", choices=("typed-decisions",), default=None, help="Explicit Laya checkpoint used by laya_adapter.")
    parser.add_argument("--jev-policy-registry", type=Path, help="Explicit approved RSI registry root for the bounded JEV runtime policy.")
    parser.add_argument("--review-action", choices=("approve", "amend", "reject", "request_retake"), help="Complete the pending modern HITL review.")
    args = parser.parse_args()

    try:
        if args.mode == "modern":
            root = Path(__file__).resolve().parents[1]
            output = args.output or (root / "runtime_data" / f"run_{Path(args.input).stem}.json")
            result = run_demo(root, Path(args.input), output=output, no_model=args.no_model, weights=args.weights, review_action=args.review_action, run_id=args.run_id, llm_profile=args.llm_profile, jev_backend=args.jev_backend, laya_model=args.laya_model, runtime_dir=args.runtime_dir, jev_policy_registry_root=args.jev_policy_registry)
        else:
            result = run_diagnosis(
                input_path=args.input,
                model=args.model,
                base_url=args.ollama_url,
                vision_device=args.vision_device,
                knowledge_dir=args.knowledge_dir,
                run_id=args.run_id,
                tenant_id=args.tenant_id,
                session_id=args.session_id,
                operator_id=args.operator_id,
                runtime_dir=args.runtime_dir,
            )
    except Exception as exc:
        print(f"Agent 流程失败: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output and args.mode == "legacy":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
