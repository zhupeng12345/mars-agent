"""Generate presentation-ready structure validation visuals and sidecars.

The visuals are deliberately labelled ``flow_validation``. They communicate the
implemented interfaces and runtime roles without presenting synthetic numbers as
model accuracy. Each page gets one deterministic 16:9 PNG and a JSON sidecar.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PAGE_SPECS = (
    (1, "项目目标", "视觉证据驱动的铝电解多智能体诊断", ["视频/图片输入", "JEV 诊断分流", "Agent 协作 + 人工确认", "案例与 RSI 闭环"], "agentic/orchestration/diagnosis_graph.py"),
    (2, "巡检输入", "视频、图片与任务说明进入统一协议", ["MediaManifest", "sha256 / fps / duration", "采样窗口", "run_id / thread_id"], "agentic/media/normalize.py"),
    (3, "火眼视觉证据", "从多帧输入形成可回放视觉证据", ["ROI / bbox", "关键帧", "置信度", "VisualEvidence"], "agentic/agents/visual_evidence.py"),
    (4, "多帧窗口", "逐帧取证、跟踪和缺失帧掩码", ["frame sequence", "ROI tracking", "missing mask", "contact sheet"], "agentic/models/roi_sequence.py"),
    (5, "质量门控", "先检查图像质量，再选择诊断路径", ["blur / brightness", "overexposure", "target present", "pass / retake / human"], "agentic/agents/image_quality.py"),
    (6, "标注与数据协议", "所有 Agent 通过统一结构化协议交接", ["InspectionTask", "AgentResult", "EvidenceRef", "TraceEvent"], "agentic/platform/contracts.py"),
    (7, "总体架构", "视觉、过程、知识、案例和人工闭环", ["视觉取证", "Process State", "知识 RAG", "Laya-JEV", "HITL"], "agentic/orchestration/diagnosis_graph.py"),
    (8, "任务状态机", "任务状态随证据和审核动作推进", ["received", "evidence_check", "diagnosis", "human_review", "archived"], "agentic/platform/tickets.py"),
    (9, "Process State Agent", "14 维过程变量形成独立工况先验", ["电气状态", "波动动态", "加料操作", "设备状态"], "agentic/agents/process_state.py"),
    (10, "14维变量分组", "过程变量按工程语义分组展示", ["potVolt / potCurrent", "fluctDelta / slopeData", "feedingState", "anodeMoveState"], "data/v1/step2_ew_202111"),
    (11, "非配对数据治理", "过程数据作为独立先验，不伪造视频对齐", ["process_missing", "alignment_quality", "independent prior", "source trace"], "agentic/agents/process_state.py"),
    (12, "F-GMVAE-T", "静态、动态和模糊先验进入时序分类器", ["spatial encoder", "temporal encoder", "fuzzy guide", "mixture prior"], "agentic/models/fgmvae_temporal.py"),
    (13, "时序编码器", "从池化基线逐步扩展到时序建模", ["mean/max pooling", "GRU", "TCN", "Temporal Transformer"], "agentic/models/temporal.py"),
    (14, "JEV 路由", "Laya-JEV 只决定诊断循环", ["simple_loop", "full_loop", "retake", "human"], "agentic/platform/jev/router.py"),
    (15, "simple_loop", "证据稳定时使用轻量诊断路径", ["case/rule match", "light diagnosis", "deterministic review", "upgrade full_loop"], "agentic/orchestration/diagnosis_graph.py"),
    (16, "full_loop 并行协作", "五类 Agent fan-out，再聚合和复核", ["visual review", "quality review", "knowledge RAG", "case compare", "process state"], "agentic/orchestration/diagnosis_graph.py"),
    (17, "知识 RAG", "工艺规则带来源、版本和适用条件", ["query rewrite", "rule retrieval", "applicability", "citation"], "agentic/platform/knowledge.py"),
    (18, "特殊案例库", "人工审核结果沉淀为可治理记忆", ["candidate", "eval_only", "approved", "active / deprecated"], "agentic/platform/case_memory.py"),
    (19, "证据复核", "诊断聚合后检查引用、冲突和缺口", ["DiagnosisDraft", "source validation", "conflict", "review decision"], "agentic/review.py"),
    (20, "LLM Gateway", "多模型 profile 统一接入和降级", ["fake_json", "ollama_local", "qwen_api", "fallback_human"], "agentic/platform/llm_gateway.py"),
    (21, "人工 interrupt/resume", "人工拥有最终授权并可恢复任务", ["interrupt", "thread_id", "approve / amend", "reject / retake"], "agentic/orchestration/diagnosis_graph.py"),
    (22, "审计与报告", "运行轨迹、报告和人工决定可回放", ["run_id", "node trace", "JSON / Markdown / HTML", "audit"], "agentic/platform/hitl.py"),
    (23, "RSI 候选演化", "运行反馈驱动离线候选版本治理", ["trajectory", "critique", "candidate diff", "gate / publish / rollback"], "agentic/rsi/"),
    (24, "冻结评测与消融", "用结构化卡片呈现质量、安全和效率维度", ["visual", "process", "route", "agent", "system", "ablation"], "agentic/evaluation/run_all.py"),
    (25, "端到端演示", "输入、分流、协作、审核和归档完整闭环", ["真实视频", "多帧 ROI", "JEV route", "Agent trace", "HITL", "report"], "agentic/scripts/demo_run.py"),
)


COLORS = {
    "navy": (13, 32, 74),
    "blue": (39, 104, 196),
    "cyan": (59, 180, 202),
    "green": (69, 157, 104),
    "orange": (224, 139, 46),
    "purple": (127, 89, 180),
    "red": (194, 79, 79),
    "ink": (31, 43, 61),
    "muted": (94, 110, 132),
    "light": (244, 247, 252),
    "white": (255, 255, 255),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    result: list[str] = []
    current = ""
    for char in str(text):
        trial = current + char
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width or not current:
            current = trial
        else:
            result.append(current)
            current = char
    if current:
        result.append(current)
    return result


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line([start, end], fill=color, width=5)
    x, y = end
    draw.polygon([(x, y), (x - 16, y - 9), (x - 16, y + 9)], fill=color)


def render_page(page: int, title: str, subtitle: str, blocks: Iterable[str], source: str, out: Path, run_id: str) -> dict:
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), COLORS["light"])
    draw = ImageDraw.Draw(image)
    title_font = _font(46, bold=True)
    subtitle_font = _font(24)
    block_font = _font(25, bold=True)
    small_font = _font(20)
    tiny_font = _font(18)

    # Header
    draw.rectangle((0, 0, width, 126), fill=COLORS["navy"])
    draw.rectangle((0, 118, width, 126), fill=COLORS["cyan"])
    draw.text((64, 28), f"P{page:02d}  {title}", fill=COLORS["white"], font=title_font)
    draw.text((68, 82), subtitle, fill=(210, 226, 250), font=subtitle_font)

    # Main visual cards
    block_list = list(blocks)
    n = len(block_list)
    cols = 3 if n >= 5 else 2
    rows = (n + cols - 1) // cols
    card_w = 420 if cols == 3 else 560
    card_h = 150
    gap_x = 48
    gap_y = 42
    total_w = cols * card_w + (cols - 1) * gap_x
    start_x = (width - total_w) // 2
    start_y = 220
    accent = (COLORS["orange"], COLORS["blue"], COLORS["green"], COLORS["purple"], COLORS["red"], COLORS["cyan"])
    centers: list[tuple[int, int]] = []
    for idx, block in enumerate(block_list):
        row, col = divmod(idx, cols)
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y)
        color = accent[idx % len(accent)]
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=18, fill=COLORS["white"], outline=color, width=5)
        draw.rectangle((x, y, x + 12, y + card_h), fill=color)
        lines = _wrap(draw, block, block_font, card_w - 52)
        text_y = y + (card_h - len(lines) * 34) // 2
        for line in lines:
            draw.text((x + 34, text_y), line, fill=COLORS["ink"], font=block_font)
            text_y += 34
        centers.append((x + card_w // 2, y + card_h // 2))
    for idx in range(len(centers) - 1):
        # Connect adjacent cards in reading order; this communicates a pipeline
        # without claiming the underlying implementation is linear.
        if idx % cols != cols - 1:
            _draw_arrow(draw, (centers[idx][0] + card_w // 2 - 14, centers[idx][1]), (centers[idx + 1][0] - card_w // 2 + 14, centers[idx + 1][1]), COLORS["muted"])

    # Footer provenance and status
    footer_y = height - 112
    draw.line((64, footer_y, width - 64, footer_y), fill=(205, 214, 229), width=2)
    draw.text((68, footer_y + 22), "结构验证素材 · 用于 PPT 演示，不代表模型精度或生产控制结果", fill=COLORS["muted"], font=small_font)
    draw.text((68, footer_y + 58), f"source: {source}  |  run_id: {run_id}", fill=COLORS["muted"], font=tiny_font)
    badge = "FLOW VALIDATION"
    bbox = draw.textbbox((0, 0), badge, font=small_font)
    badge_x = width - 68 - (bbox[2] - bbox[0]) - 28
    draw.rounded_rectangle((badge_x, footer_y + 18, width - 68, footer_y + 62), radius=20, fill=(230, 242, 255), outline=COLORS["blue"], width=2)
    draw.text((badge_x + 14, footer_y + 27), badge, fill=COLORS["blue"], font=small_font)

    image.save(out, format="PNG", optimize=True)
    digest = hashlib.sha256(image.tobytes()).hexdigest()[:16]
    metadata = {
        "asset_id": out.stem,
        "page": page,
        "title": title,
        "source_files": [source],
        "run_id": run_id,
        "model_profile": "presentation_structure_validation_v1",
        "knowledge_version": "knowledge-pack-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "width": width,
        "height": height,
        "status": "flow_validation",
        "script": "agentic.scripts.build_ppt_visual_assets",
        "script_version": "ppt-assets-1",
        "content_sha256": digest,
        "claim_policy": "structure validation only; no visual accuracy or production control claim",
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_assets(root: Path, output_dir: Path, run_id: str = "ppt-demo-structure-001") -> dict:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for page, title, subtitle, blocks, source in PAGE_SPECS:
        out = output_dir / f"asset-page-{page:02d}.png"
        rows.append(render_page(page, title, subtitle, blocks, source, out, run_id))
    return {"schema_version": "ppt-structure-assets-1", "run_id": run_id, "status": "flow_validation", "asset_count": len(rows), "assets": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="ppt-demo-structure-001")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = args.output_dir or root / "report_assets" / "multi_agent_v2"
    payload = build_assets(root, output_dir, args.run_id)
    index = output_dir / "ppt_structure_assets.json"
    index.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(index), "assets": payload["asset_count"], "status": payload["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
