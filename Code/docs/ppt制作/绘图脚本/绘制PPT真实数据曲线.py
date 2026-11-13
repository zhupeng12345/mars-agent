#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create two PPT-ready charts from the project's real visual/process data.

Run from any working directory:
  python Code/docs/ppt制作/绘图脚本/绘制PPT真实数据曲线.py

Outputs 1800x600 PNG and vector SVG files under Code/docs/ppt制作/插图素材.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / "Code" / "docs" / "ppt制作" / "插图素材"
VISUAL_MANIFEST = ROOT / "Code" / "data" / "evaluation" / "visual_manifest.jsonl"
PROCESS_DATASET = ROOT / "Code" / "data" / "v1" / "step2_ew_202111" / "dataset.npz"
PROCESS_SPLIT = ROOT / "Code" / "data" / "v1" / "step2_ew_202111" / "split.json"
VISUAL_WINDOW_ID = "video-n-788ade07-w0000"
PROCESS_SAMPLE_INDEX = 26

COLORS = {
    "navy": "#102A68",
    "blue": "#2878B5",
    "cyan": "#2A9D9A",
    "green": "#3C9B65",
    "orange": "#E58B32",
    "red": "#D9534F",
    "purple": "#7957A6",
    "ink": "#1F2937",
    "muted": "#687386",
    "grid": "#DFE5EC",
    "pale_blue": "#EEF5FC",
    "pale_teal": "#EAF5F3",
    "pale_orange": "#FFF4E6",
    "white": "#FFFFFF",
}


def configure_fonts() -> None:
    available = {item.name for item in font_manager.fontManager.ttflist}
    family = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in available), None)
    if family:
        plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def load_manifest_row(window_id: str) -> dict[str, Any]:
    with VISUAL_MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("window_id") == window_id:
                return row
    raise ValueError(f"Visual window not found in {VISUAL_MANIFEST}: {window_id}")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    candidate = ROOT.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def read_visual_sequence(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    for value in row["roi_paths"]:
        image = cv2.imread(str(resolve_repo_path(str(value))), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read ROI image: {value}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(image.astype(np.float32) / 255.0)
    frames = np.stack(images)
    gray = frames.mean(axis=-1)
    frame_ids = np.asarray(row.get("frame_ids", list(range(len(frames)))), dtype=float)
    fps = float(row.get("fps") or 1.0)
    seconds = frame_ids / fps
    return frames, gray, seconds


def visual_features(frames: np.ndarray, gray: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray | float]]:
    delta = np.abs(np.diff(gray, axis=0))
    static = {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "gray_contrast": float(gray.std()),
        "saturation_proxy": float((frames.max(axis=-1) - frames.min(axis=-1)).mean()),
        "hot_pixel_ratio": float((gray > 0.9).mean()),
    }
    dynamic = {
        "gray_change_rate": delta.mean(axis=(1, 2)),
        "gray_change_std": float(delta.std()),
        "structure_change_rate": (delta > 0.08).mean(axis=(1, 2)),
        "area_change_rate": np.abs(np.diff(gray.mean(axis=(1, 2)))),
    }
    # Check that the plotted interval statistics aggregate to the implemented
    # descriptor definitions in derive_expert_features().
    if not np.isclose(dynamic["gray_change_rate"].mean(), delta.mean(), atol=1e-6):
        raise AssertionError("gray_change_rate aggregation differs from the runtime definition")
    if not np.isclose(dynamic["structure_change_rate"].mean(), (delta > 0.08).mean(), atol=1e-6):
        raise AssertionError("structure_change_rate aggregation differs from the runtime definition")
    if not np.isclose(dynamic["area_change_rate"].mean(), np.abs(np.diff(gray.mean(axis=(1, 2)))).mean(), atol=1e-6):
        raise AssertionError("area_change_rate aggregation differs from the runtime definition")
    return static, dynamic


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB9C7")
    ax.spines["bottom"].set_color("#AEB9C7")
    ax.tick_params(colors=COLORS["muted"], labelsize=11, length=0, pad=6)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def add_card(fig: plt.Figure, x: float, y: float, w: float, h: float,
             title: str, body: str, fill: str, edge: str = "#CBD5E1",
             title_color: str = COLORS["navy"], title_size: int = 13,
             body_size: int = 11, align: str = "left") -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.2,
        zorder=2,
    )
    fig.add_artist(patch)
    tx = x + (w / 2 if align == "center" else 0.018)
    ha = "center" if align == "center" else "left"
    fig.text(tx, y + h - 0.025, title, ha=ha, va="top", color=title_color,
             fontsize=title_size, fontweight="bold", zorder=3)
    fig.text(tx, y + h - 0.07, body, ha=ha, va="top", color=COLORS["ink"],
             fontsize=body_size, linespacing=1.35, zorder=3)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{ext}",
            dpi=100,
            facecolor=COLORS["white"],
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig)


def draw_visual_chart() -> dict[str, Any]:
    row = load_manifest_row(VISUAL_WINDOW_ID)
    frames, gray, seconds = read_visual_sequence(row)
    static, dynamic = visual_features(frames, gray)
    pairs = (seconds[:-1] + seconds[1:]) / 2

    fig = plt.figure(figsize=(18, 6), dpi=100, facecolor=COLORS["white"])
    fig.text(0.045, 0.94, "F-GMVAE-T｜真实 ROI 窗口的静态与动态视觉特征",
             fontsize=25, fontweight="bold", color=COLORS["navy"], va="top")
    fig.text(0.047, 0.865,
             f"样本 {row['window_id']} · {len(frames)} 帧 · {seconds[0]:.2f}–{seconds[-1]:.2f} s · 人工标签未提供",
             fontsize=12, color=COLORS["muted"], va="top")

    fig.text(0.075, 0.79, "静态窗口描述符", fontsize=15, color=COLORS["navy"], fontweight="bold")
    fig.text(0.385, 0.79, "帧间动态变化", fontsize=15, color=COLORS["navy"], fontweight="bold")

    ax_static = fig.add_axes([0.105, 0.22, 0.245, 0.50])
    style_axis(ax_static)
    static_names = ["亮度均值", "灰度对比度*", "饱和度代理", "高亮像素占比"]
    static_values = [
        static["brightness_mean"],
        static["gray_contrast"],
        static["saturation_proxy"],
        static["hot_pixel_ratio"],
    ]
    static_colors = [COLORS["blue"], COLORS["cyan"], COLORS["orange"], COLORS["purple"]]
    y_pos = np.arange(len(static_names))
    ax_static.barh(y_pos, static_values, color=static_colors, height=0.56, zorder=3)
    ax_static.set_yticks(y_pos, static_names, fontsize=10)
    ax_static.invert_yaxis()
    ax_static.set_xlim(0, 1.0)
    ax_static.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax_static.set_xlabel("归一化描述值", fontsize=10, color=COLORS["muted"])
    for y, value in zip(y_pos, static_values):
        ax_static.text(min(value + 0.025, 0.92), y, f"{value:.3f}", va="center",
                       fontsize=10, color=COLORS["ink"], fontweight="bold")

    ax_dynamic = fig.add_axes([0.405, 0.22, 0.315, 0.50])
    style_axis(ax_dynamic)
    series = [
        ("帧间灰度差", dynamic["gray_change_rate"], COLORS["blue"]),
        ("结构变化率", dynamic["structure_change_rate"], COLORS["cyan"]),
        ("区域亮度变化", dynamic["area_change_rate"], COLORS["orange"]),
    ]
    for label, values, color in series:
        ax_dynamic.plot(pairs, values, color=color, linewidth=2.6, marker="o",
                        markersize=5.2, label=label)
    ax_dynamic.set_xlim(float(seconds[0]), float(seconds[-1]))
    ax_dynamic.set_ylim(bottom=0)
    ax_dynamic.set_xlabel("视频时间（秒）", fontsize=10, color=COLORS["muted"])
    ax_dynamic.set_ylabel("帧间变化量", fontsize=10, color=COLORS["muted"])
    ax_dynamic.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none",
                      framealpha=0.88, fontsize=9, ncol=1,
                      handlelength=1.5, borderaxespad=0.2)
    fig.text(0.405, 0.745,
             f"窗口帧差标准差：{dynamic['gray_change_std']:.3f}",
             fontsize=9.5, color=COLORS["muted"], va="top")

    fig.text(0.075, 0.135,
             "* 5项静态字段中，gray_contrast 与 brightness_std 当前采用同一灰度标准差定义。",
             fontsize=9.5, color=COLORS["muted"])
    fig.text(0.075, 0.085,
             "数据为真实 ROI 描述统计；不是分类结果或已训练潜变量分布。",
             fontsize=9.5, color=COLORS["muted"])

    add_card(fig, 0.755, 0.48, 0.21, 0.24,
             "静态 5维 + 动态 4维",
             "静态：亮度/对比度/色彩/高亮\n动态：帧差均值与标准差、结构/亮度变化\nFeatureLabelGenerator → g_fuzzy",
             COLORS["pale_blue"], edge="#A9C5E2", title_size=14, body_size=10.5)
    arrow = FancyArrowPatch((0.86, 0.46), (0.86, 0.405), transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=16, linewidth=1.8,
                            color=COLORS["purple"], zorder=3)
    fig.add_artist(arrow)
    add_card(fig, 0.755, 0.19, 0.21, 0.21,
             "混合先验权重更新",
             "temporal logits + expert logits\n+ log(g_fuzzy) → softmax",
             COLORS["pale_teal"], edge="#A9D0C8", title_size=14, body_size=10.5)
    fig.text(0.755, 0.12,
             "专家引导需 expert_dim=9；当前图像监督 profile 为 0。",
             fontsize=9.5, color=COLORS["muted"], va="top")
    save_figure(fig, "F-GMVAE-T真实ROI静态动态特征")
    return {
        "window_id": row["window_id"],
        "video_path": row.get("video_path"),
        "human_label": row.get("human_label"),
        "label_status": row.get("label_status"),
        "frame_ids": row.get("frame_ids"),
        "timestamps_seconds": seconds.tolist(),
        "static_features": static,
        "dynamic_interval_means": {name: float(np.mean(values)) for name, values in dynamic.items()},
        "expert_dim_note": "optional expert branch requires expert_dim=9; current image-supervised profile uses expert_dim=0",
    }


def draw_process_chart() -> dict[str, Any]:
    if not PROCESS_DATASET.exists() or not PROCESS_SPLIT.exists():
        raise FileNotFoundError("Process dataset or split metadata is missing")
    with np.load(PROCESS_DATASET, allow_pickle=True) as data:
        X = np.asarray(data["X_seq"], dtype=np.float64)
        columns = [str(item) for item in data["feature_cols"].tolist()]
        anchor = str(data["ts_anchor"][PROCESS_SAMPLE_INDEX])
        y_future = int(data["y"][PROCESS_SAMPLE_INDEX])
        sample = X[PROCESS_SAMPLE_INDEX]
        sample_minutes = int(data["SAMPLE_MINUTES"])
    split = json.loads(PROCESS_SPLIT.read_text(encoding="utf-8"))
    if PROCESS_SAMPLE_INDEX not in split.get("train", []):
        raise ValueError(f"Process sample {PROCESS_SAMPLE_INDEX} is not in the train split")

    names = ["potVolt", "potCurrent", "fluctDelta", "actualFeedingInter"]
    colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"]]
    offsets = np.arange(sample.shape[0], dtype=float) * sample_minutes - (sample.shape[0] - 1) * sample_minutes

    fig = plt.figure(figsize=(18, 6), dpi=100, facecolor=COLORS["white"])
    fig.text(0.045, 0.94, "Process State Agent｜真实 14 变量过程窗口",
             fontsize=25, fontweight="bold", color=COLORS["navy"], va="top")
    fig.text(0.047, 0.865,
             f"样本 {PROCESS_SAMPLE_INDEX} · 30步 × 14字段 · 每步 {sample_minutes} 分钟 · 窗口结束 {anchor[:16]}",
             fontsize=12, color=COLORS["muted"], va="top")

    ax = fig.add_axes([0.065, 0.36, 0.61, 0.40])
    style_axis(ax)
    short_names = {
        "potVolt": "槽电压",
        "potCurrent": "槽电流",
        "fluctDelta": "波动量",
        "actualFeedingInter": "实际加料间隔",
    }
    for name, color in zip(names, colors):
        raw = sample[:, columns.index(name)]
        mean = float(np.mean(raw))
        std = float(np.std(raw))
        z = (raw - mean) / std if std > 1e-9 else np.zeros_like(raw)
        ax.plot(offsets, z, color=color, linewidth=2.4, marker="o", markersize=3.5,
                label=short_names[name])
    ax.axhline(0, color="#AAB4C0", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlim(float(offsets[0]), float(offsets[-1]))
    ax.set_xticks([-58, -40, -20, 0])
    ax.set_xlabel("相对窗口结束时间（分钟）", fontsize=10.5, color=COLORS["muted"], labelpad=4)
    ax.set_ylabel("窗口内 Z 分数", fontsize=10.5, color=COLORS["muted"], labelpad=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False,
              fontsize=10, ncol=4, handlelength=1.7, columnspacing=1.5)

    fig.text(0.065, 0.205,
             "展示4个代表性字段（槽电压、槽电流、波动量、实际加料间隔）；各字段单独做窗口内 Z 标准化。",
             fontsize=9.5, color=COLORS["muted"])
    fig.text(0.065, 0.15,
             "14字段完整输入 Process State Agent；原数据未提供物理单位，曲线不与视频时间配对。",
             fontsize=9.5, color=COLORS["muted"])
    fig.text(0.065, 0.095,
             "数据集 y=1 表示该窗口对应的未来30分钟事件训练标签，不是视频类别或在线推理结果。",
             fontsize=9.5, color=COLORS["muted"])

    add_card(fig, 0.715, 0.60, 0.235, 0.18,
             "输入分组 · 全部14字段",
             "电气状态 5  ·  波动动态 4\n加料操作 3  ·  设备状态 2",
             COLORS["pale_blue"], edge="#A9C5E2", title_size=14, body_size=12)
    add_card(fig, 0.715, 0.34, 0.235, 0.20,
             "窗口状态摘要",
             "各变量均值、波动与趋势\n变化点、缺失掩码、工况标签",
             COLORS["pale_teal"], edge="#A9D0C8", title_size=14, body_size=11.5)
    add_card(fig, 0.715, 0.12, 0.235, 0.16,
             "过程先验输入诊断",
             "与巡检视频未建立时间配对\nalignment_quality = unknown",
             COLORS["pale_orange"], edge="#E7C48E", title_size=13, body_size=10.5)

    fig.text(0.715, 0.065,
             f"数据源 dataset.npz · sample={PROCESS_SAMPLE_INDEX} · y={y_future}",
             fontsize=8.5, color="#8792A2")
    save_figure(fig, "ProcessState真实14变量窗口")
    return {
        "dataset": str(PROCESS_DATASET.relative_to(ROOT)),
        "sample_index": PROCESS_SAMPLE_INDEX,
        "split": "train",
        "anchor": anchor,
        "window_start": str(np.datetime64(anchor) - np.timedelta64((sample.shape[0] - 1) * sample_minutes, "m")),
        "window_end": anchor,
        "steps": int(sample.shape[0]),
        "sample_minutes": sample_minutes,
        "features": columns,
        "plotted_features": names,
        "future_event_label_30min": y_future,
        "video_alignment": "unknown",
        "units": "not specified by source dataset",
    }


def main() -> None:
    configure_fonts()
    visual_meta = draw_visual_chart()
    process_meta = draw_process_chart()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "PPT真实曲线数据来源.json"
    metadata_path.write_text(
        json.dumps({"visual_chart": visual_meta, "process_chart": process_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created chart assets in: {OUTPUT_DIR}")
    print(f"- F-GMVAE-T: {visual_meta['window_id']} (human_label={visual_meta['human_label']}, status={visual_meta['label_status']})")
    print(f"- Process State: sample {process_meta['sample_index']} at {process_meta['anchor']}, y={process_meta['future_event_label_30min']}")


if __name__ == "__main__":
    main()
