#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render two publication-style illustrative figures for the project slides.

All curves in these figures are deterministic synthetic examples. They are
visual explanations, not measured data, model outputs, or evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / "Code" / "docs" / "ppt制作" / "插图素材"
SEED = 20261110

COLORS = {
    "navy": "#132B52",
    "blue": "#3478B8",
    "teal": "#2A9D8F",
    "orange": "#E58A35",
    "red": "#D65F5F",
    "purple": "#7856A6",
    "green": "#4B9667",
    "ink": "#273449",
    "muted": "#69778C",
    "grid": "#E1E7EF",
    "pale_blue": "#EDF4FB",
    "pale_teal": "#EAF5F2",
    "pale_orange": "#FBF1E5",
    "pale_purple": "#F1ECF8",
    "white": "#FFFFFF",
}


def configure_fonts() -> None:
    available = {item.name for item in font_manager.fontManager.ttflist}
    family = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in available), None)
    if family:
        plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB9C7")
    ax.spines["bottom"].set_color("#AEB9C7")
    ax.tick_params(colors=COLORS["muted"], labelsize=9.5, length=0, pad=5)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.75)
    ax.set_axisbelow(True)


def gaussian(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def add_flow_card(fig: plt.Figure, x: float, y: float, w: float, h: float,
                  title: str, body: str, face: str, edge: str) -> None:
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.25,
    )
    fig.add_artist(card)
    fig.text(x + 0.015, y + h - 0.022, title, color=COLORS["navy"],
             fontsize=11.5, fontweight="bold", va="top")
    fig.text(x + 0.015, y + h - 0.057, body, color=COLORS["ink"],
             fontsize=9.4, linespacing=1.35, va="top")


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"{name}.{suffix}",
            dpi=100,
            facecolor=COLORS["white"],
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig)


def draw_fgmvae_figure() -> dict[str, object]:
    x = np.linspace(-3.4, 3.4, 600)
    static = gaussian(x, -0.65, 0.72)
    temporal = gaussian(x, 0.55, 0.82)
    expert = 0.56 * gaussian(x, -0.8, 0.52) + 0.44 * gaussian(x, 0.9, 0.58)
    fused = 0.28 * gaussian(x, -1.0, 0.58) + 0.46 * gaussian(x, 0.05, 0.68) + 0.26 * gaussian(x, 1.15, 0.55)
    fused /= np.trapezoid(fused, x)

    fig = plt.figure(figsize=(18, 6), dpi=100, facecolor=COLORS["white"])
    fig.text(0.045, 0.94, "F-GMVAE-T｜静态—动态特征的模糊先验融合",
             fontsize=24, fontweight="bold", color=COLORS["navy"], va="top")
    fig.text(0.047, 0.865, "机制示意 · 多帧 ROI 表征与专家描述符共同调节潜空间混合先验",
             fontsize=11.5, color=COLORS["muted"], va="top")

    ax = fig.add_axes([0.075, 0.22, 0.50, 0.53])
    style_axis(ax)
    ax.fill_between(x, static, color=COLORS["blue"], alpha=0.12)
    ax.plot(x, static, color=COLORS["blue"], linewidth=2.2, label="静态视觉表征")
    ax.fill_between(x, temporal, color=COLORS["teal"], alpha=0.12)
    ax.plot(x, temporal, color=COLORS["teal"], linewidth=2.2, label="动态时序表征")
    ax.plot(x, expert, color=COLORS["orange"], linewidth=2.0, linestyle=(0, (5, 3)),
            label="专家模糊引导")
    ax.fill_between(x, fused, color=COLORS["purple"], alpha=0.15)
    ax.plot(x, fused, color=COLORS["purple"], linewidth=3.0, label="融合混合先验")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("潜变量坐标（标准化）", fontsize=10, color=COLORS["muted"], labelpad=7)
    ax.set_ylabel("密度（示意）", fontsize=10, color=COLORS["muted"], labelpad=7)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9.2,
              columnspacing=1.4, handlelength=2.0)
    ax.text(0.99, 0.94, "Gaussian mixture prior", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=COLORS["muted"])

    fig.text(0.605, 0.78, "融合路径", fontsize=14, color=COLORS["navy"], fontweight="bold")
    add_flow_card(fig, 0.605, 0.60, 0.34, 0.15,
                  "双路编码",
                  "ROI 帧序列 → 空间 / 时序编码器\n专家描述符 → FeatureLabelGenerator",
                  COLORS["pale_blue"], "#A9C5E2")
    add_flow_card(fig, 0.605, 0.385, 0.34, 0.17,
                  "模糊混合引导",
                  "g_fuzzy = softmax(MLP(f_static, f_dynamic))\nπ = softmax(ℓ_temporal + ℓ_expert + log g_fuzzy)",
                  COLORS["pale_teal"], "#A9D0C8")
    add_flow_card(fig, 0.605, 0.17, 0.34, 0.17,
                  "诊断表征",
                  "混合先验约束潜变量 z\n分类头输出类别概率与不确定性",
                  COLORS["pale_purple"], "#C7B5DF")
    for y1, y2 in ((0.59, 0.56), (0.375, 0.345)):
        fig.add_artist(FancyArrowPatch((0.775, y1), (0.775, y2), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
                                       color=COLORS["purple"]))

    fig.text(0.075, 0.105,
             "曲线为合成分布，用于说明特征对齐与混合先验更新；不表示训练集统计或模型实测输出。",
             fontsize=9.3, color=COLORS["muted"])
    fig.text(0.605, 0.105,
             "颜色编码：静态视觉 · 动态时序 · 专家引导 · 融合先验",
             fontsize=9.2, color=COLORS["muted"])
    save_figure(fig, "F-GMVAE-T融合分布_示意")
    return {
        "kind": "synthetic illustrative distribution",
        "seed": SEED,
        "components": 3,
        "curves": ["static visual representation", "dynamic temporal representation", "fuzzy expert guide", "fused mixture prior"],
        "is_measured_data": False,
        "is_model_output": False,
    }


PROCESS_FIELDS = [
    ("槽电压", "electrical"),
    ("槽电流", "electrical"),
    ("设定电压", "electrical"),
    ("电压上限", "electrical"),
    ("电压下限", "electrical"),
    ("效应等待", "equipment"),
    ("波动量", "fluctuation"),
    ("波动差", "fluctuation"),
    ("电压斜率", "fluctuation"),
    ("累计斜率", "fluctuation"),
    ("基准加料间隔", "feeding"),
    ("实际加料间隔", "feeding"),
    ("加料状态", "feeding"),
    ("阳极移动状态", "equipment"),
]

GROUP_COLORS = {
    "electrical": COLORS["blue"],
    "fluctuation": COLORS["teal"],
    "feeding": COLORS["orange"],
    "equipment": COLORS["purple"],
}


def make_synthetic_process_window() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    time = np.linspace(0, 1, 60)
    rows = []
    for index, (name, group) in enumerate(PROCESS_FIELDS):
        noise = np.zeros_like(time)
        innovations = rng.normal(0, 0.13 + (index % 4) * 0.012, size=time.size)
        for step in range(1, time.size):
            noise[step] = 0.86 * noise[step - 1] + innovations[step]
        common_shift = 0.7 / (1 + np.exp(-(time - 0.57) * 35))
        pulse = np.exp(-0.5 * ((time - (0.62 + (index % 3) * 0.025)) / 0.055) ** 2)
        trend = 0.35 * np.sin(2 * np.pi * (time * (0.7 + index * 0.035) + index * 0.07))

        if group == "electrical":
            values = 0.45 * trend + (0.25 + index * 0.035) * common_shift + noise
        elif group == "fluctuation":
            values = 0.45 * trend + (0.5 + (index % 2) * 0.15) * pulse + 0.8 * noise
        elif group == "feeding":
            values = 0.2 * trend + 0.65 * np.where(time > 0.55, 1.0, -0.45) + 0.5 * noise
        else:
            values = 0.22 * trend + 0.7 * np.where(time > 0.62, 0.8, -0.35) + 0.45 * noise

        if name == "加料状态":
            values = np.where((time > 0.54) & (time < 0.72), 1.0, 0.0) + rng.normal(0, 0.035, size=time.size)
        elif name == "阳极移动状态":
            values = np.where((time > 0.62) & (time < 0.69), 1.0, 0.0) + rng.normal(0, 0.025, size=time.size)
        kernel = np.asarray([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        smooth_values = np.convolve(np.pad(values, (2, 2), mode="edge"), kernel, mode="valid")
        rows.append(smooth_values)

    matrix = np.stack(rows)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return np.clip((matrix - mean) / np.maximum(std, 1e-6), -2.8, 2.8)


def draw_process_figure() -> dict[str, object]:
    z = make_synthetic_process_window()
    time = np.arange(z.shape[1])
    selected = [1, 2, 6, 8, 11]
    line_colors = [COLORS["blue"], COLORS["red"], COLORS["teal"], COLORS["green"], COLORS["orange"]]

    fig = plt.figure(figsize=(18, 6), dpi=100, facecolor=COLORS["white"])
    fig.text(0.045, 0.94, "Process State Agent｜多变量窗口与状态摘要",
             fontsize=24, fontweight="bold", color=COLORS["navy"], va="top")
    fig.text(0.047, 0.865, "机制示意 · 14 变量 × 60 个采样步 · 合成时序 · 逐变量标准化",
             fontsize=11.5, color=COLORS["muted"], va="top")

    ax_line = fig.add_axes([0.07, 0.29, 0.48, 0.43])
    style_axis(ax_line)
    for idx, color in zip(selected, line_colors):
        ax_line.plot(time, z[idx], color=color, linewidth=1.9, label=PROCESS_FIELDS[idx][0])
    ax_line.axvline(37, color=COLORS["red"], linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.8)
    ax_line.text(38, 2.55, "变化点", fontsize=8.5, color=COLORS["red"], va="top")
    ax_line.axhline(0, color="#AEB9C7", linewidth=0.8, linestyle="--")
    ax_line.set_xlim(0, 59)
    ax_line.set_ylim(-2.9, 2.9)
    ax_line.set_xticks([0, 15, 30, 45, 59], ["t−59", "t−44", "t−29", "t−14", "t"])
    ax_line.set_xlabel("窗口相对采样步", fontsize=9.5, color=COLORS["muted"], labelpad=6)
    ax_line.set_ylabel("窗口内 Z 分数", fontsize=9.5, color=COLORS["muted"], labelpad=6)
    ax_line.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False,
                   ncol=3, fontsize=8.7, columnspacing=1.25, handlelength=1.7)

    ax_heat = fig.add_axes([0.64, 0.25, 0.245, 0.50])
    image = ax_heat.imshow(z, cmap="RdYlBu_r", aspect="auto", interpolation="nearest", vmin=-2.8, vmax=2.8)
    ax_heat.set_yticks(np.arange(len(PROCESS_FIELDS)), [name for name, _ in PROCESS_FIELDS], fontsize=8.5)
    for tick, (_, group) in zip(ax_heat.get_yticklabels(), PROCESS_FIELDS):
        tick.set_color(GROUP_COLORS[group])
    ax_heat.set_xticks([0, 15, 30, 45, 59], ["0", "15", "30", "45", "59"], fontsize=8.5)
    ax_heat.set_xlabel("时间步", fontsize=9, color=COLORS["muted"], labelpad=5)
    ax_heat.tick_params(length=0, pad=4)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.06, pad=0.035)
    colorbar.set_label("Z", fontsize=8.5, color=COLORS["muted"])
    colorbar.ax.tick_params(labelsize=8, colors=COLORS["muted"], length=0)

    fig.text(0.64, 0.785, "14变量状态矩阵", fontsize=13, color=COLORS["navy"], fontweight="bold")
    fig.text(0.905, 0.70, "变量组", fontsize=10.5, color=COLORS["navy"], fontweight="bold")
    group_items = [
        ("电气状态 · 5", "electrical"),
        ("波动动态 · 4", "fluctuation"),
        ("加料操作 · 3", "feeding"),
        ("设备状态 · 2", "equipment"),
    ]
    for i, (label, group) in enumerate(group_items):
        y = 0.65 - i * 0.055
        fig.add_artist(plt.Line2D([0.906, 0.922], [y, y], transform=fig.transFigure,
                                  color=GROUP_COLORS[group], linewidth=3.2, solid_capstyle="round"))
        fig.text(0.93, y, label, fontsize=8.8, color=COLORS["ink"], va="center")

    fig.text(0.07, 0.17, "代表变量趋势", fontsize=11, color=COLORS["navy"], fontweight="bold")
    fig.text(0.07, 0.105,
             "合成数据用于说明：多变量窗口 → 趋势/波动/变化点 → 工况状态摘要。",
             fontsize=9.3, color=COLORS["muted"])
    fig.text(0.64, 0.105,
             "变量值为标准化模拟量；形状仅用于展示状态特征提取。",
             fontsize=9.3, color=COLORS["muted"])
    save_figure(fig, "ProcessStateAgent14变量趋势热图_示意")
    return {
        "kind": "synthetic illustrative multivariate time series",
        "seed": SEED,
        "steps": int(z.shape[1]),
        "variables": [name for name, _ in PROCESS_FIELDS],
        "groups": {key: sum(group == key for _, group in PROCESS_FIELDS) for key in GROUP_COLORS},
        "is_measured_data": False,
        "is_model_output": False,
    }


def main() -> None:
    configure_fonts()
    metadata = {
        "notice": "All plotted values are synthetic illustrative data; do not cite as measurements, training results, or evaluation results.",
        "seed": SEED,
        "figures": [draw_fgmvae_figure(), draw_process_figure()],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "PPT模拟示意图说明.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created illustrative assets in: {OUTPUT_DIR}")
    print("- F-GMVAE-T: synthetic distribution/fusion mechanism")
    print("- Process State Agent: synthetic 14-variable trajectories and heatmap")


if __name__ == "__main__":
    main()
