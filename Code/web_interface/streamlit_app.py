# streamlit_app.py
# -*- coding: utf-8 -*-
"""
MARS-Agent 火眼诊断系统 —— Web 端（科技风驾驶舱）

v6 相对 v5 的变化：
  * 导航从左侧栏移到**顶部**，五个入口：性能监控 / 过程诊断 / 加药分析 / 历史管理 / 系统设置
  * 顶部标题改为「基于YOLO火眼识别与多模态大模型铝电解诊断系统」，放大并置于左上角，删除 MARS-AGENT 字样
  * 整站改为科技风主题：深空底色 + 网格 + 霓虹青描边 + 等宽数字 + 面板四角标记
  * 新增「性能监控」页（系统/推理性能）与「加药分析」页（AlF3 规则推算）
"""

import base64
import hashlib
import html
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo_web_bridge import (  # noqa: E402
    CLASS_CN, FireEyeAnalyzer, get_device, load_bgr_from_bytes,
    load_frames_from_bytes, models_ready, summarize,
)
import dosing_rules  # noqa: E402
import history_store  # noqa: E402

# 注意：except 里的 `as _e` 在 except 块结束后会被 Python 删除，
# 不能在 lambda 里闭包引用它（否则调用时 NameError）。这里先存成模块级变量。
_LLM_IMPORT_ERROR = None
try:
    from llm_web import backend_summary as llm_backend_summary, llava_analyze_frame
except Exception as _exc:  # pragma: no cover
    _LLM_IMPORT_ERROR = _exc
    llm_backend_summary = lambda: f"未加载（{_LLM_IMPORT_ERROR}）"   # noqa: E731
    llava_analyze_frame = None

# 多智能体工作流是可选模块，保持旧版网页在未安装 LangGraph 时仍能打开。
_AGENT_IMPORT_ERROR = None
try:
    from agentic.workflow import run_diagnosis_trace, workflow_spec
except Exception as _agent_exc:  # pragma: no cover
    _AGENT_IMPORT_ERROR = _agent_exc
    run_diagnosis_trace = None
    workflow_spec = None

_MODERN_IMPORT_ERROR = None
try:
    from agentic.platform.hitl import HumanReviewService
    from agentic.platform.presentation_review import PresentationReviewStore
    from agentic.scripts.demo_run import (
        load_demo_run as load_persisted_modern_demo,
        resume_demo_review,
        run_demo as run_modern_demo,
    )
except Exception as _modern_exc:  # pragma: no cover
    _MODERN_IMPORT_ERROR = _modern_exc
    HumanReviewService = None
    PresentationReviewStore = None
    run_modern_demo = None

DEFAULT_SAMPLE_FRAMES = 24
PAGES = [("perf", "性能监控"), ("diagnose", "过程诊断"), ("dosing", "加药分析"),
         ("agent", "智能体协作"), ("history", "历史管理"), ("settings", "系统设置")]

STATE_COLOR = {"C": "#4d8dff", "H": "#c03030", "N": "#3ddc84", "NoTarget": "#8b97c9"}
STATE_TEXT = {"C": "温度偏低 COLD", "H": "严重过热 HOT", "N": "工况正常 NORMAL", "NoTarget": "等待目标 …"}
CYAN, DIM, MONO = "#5a7bff", "#8b97c9", "var(--mono)"

st.set_page_config(page_title="基于YOLO火眼识别与多模态大模型铝电解诊断系统",
                   page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# 科技风主题
# ============================================================
st.markdown("""
<style>
:root{
    /* 参考图配色：深藏青底 + 靛蓝块 + 白字 + 红色点缀 */
    --bg0:#000410; --bg1:#000830; --bg2:#001038;
    --panel-a:#0c1140; --panel-b:#070b28;
    --line:#1b2a6b; --line-soft:#152057;
    --accent:#5a7bff; --accent-hi:#8ba4ff; --accent-deep:#2a37c9;
    --indigo:#101060;
    --text:#eef2ff; --muted:#8b97c9;
    --ok:#3ddc84; --warn:#ffb020; --bad:#c03030;
    --mono:"JetBrains Mono","Cascadia Mono",Consolas,"DejaVu Sans Mono",ui-monospace,monospace;
    --cn:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","Source Han Sans SC",sans-serif;
}
header[data-testid="stHeader"]{display:none;}
#MainMenu,footer{visibility:hidden;}

html,body{height:100vh;overflow:hidden;}
[data-testid="stAppViewContainer"]{
    height:100vh;overflow:hidden;
    background:
        radial-gradient(1200px 620px at 8% -14%, rgba(28,40,150,.38), transparent 62%),
        radial-gradient(900px 520px at 102% -6%, rgba(16,16,96,.50), transparent 58%),
        linear-gradient(180deg,#000830 0%,#000410 100%);
    font-family:var(--cn); color:var(--text);
}
[data-testid="stAppViewContainer"]::before{
    content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.45;
    background-image:
        linear-gradient(rgba(90,123,255,.055) 1px,transparent 1px),
        linear-gradient(90deg,rgba(90,123,255,.055) 1px,transparent 1px);
    background-size:46px 46px;
    -webkit-mask-image:radial-gradient(circle at 50% 0%,#000 26%,transparent 84%);
    mask-image:radial-gradient(circle at 50% 0%,#000 26%,transparent 84%);
}
[data-testid="stMain"],[data-testid="stMainBlockContainer"],.block-container{
    height:100vh!important;overflow:hidden;position:relative;z-index:1;
}
.block-container{max-width:100%!important;padding:.45rem .95rem .35rem .95rem!important;}
[data-testid="stVerticalBlock"]{gap:.32rem;}
[data-testid="stHorizontalBlock"]{gap:.5rem;}

/* ---------- 标题：左上大字 + 左侧竖向色条 ---------- */
.app-title-wrap{display:flex;align-items:center;gap:15px;padding:2px 0 2px 2px;}
.app-title-bar{
    width:6px;height:48px;border-radius:3px;flex:0 0 auto;
    background:linear-gradient(180deg,var(--accent-hi) 0%,var(--accent-deep) 65%,var(--indigo) 100%);
    box-shadow:0 0 20px rgba(90,123,255,.80),0 0 44px rgba(42,55,201,.45);
}
.app-title{
    font-family:var(--cn);font-size:34px;font-weight:900;letter-spacing:2px;line-height:1.16;
    color:#f4f7ff;text-shadow:0 0 18px rgba(90,123,255,.45);
}
.app-title .accent{color:var(--accent-hi);}
.hdr-right{
    text-align:right;font-family:var(--mono);font-size:21px;font-weight:700;
    color:#b9c6ff;letter-spacing:1.2px;line-height:1.25;padding-top:10px;
    text-shadow:0 0 14px rgba(90,123,255,.45);
}
.hr-tech{display:none;}

/* ---------- 顶部导航：下划线标签（tab 风格） ---------- */
.st-key-tabbar{border-bottom:1px solid var(--line-soft);margin:6px 0 7px 0;}
.st-key-tabbar [data-testid="stHorizontalBlock"]{gap:0;}
.st-key-tabbar div.stButton > button{
    width:100%;background:transparent;border:none;border-radius:0;
    border-bottom:2px solid transparent;
    color:var(--muted);font-size:15px;font-weight:600;letter-spacing:2px;
    padding:7px 4px 9px 4px;box-shadow:none;transition:color .15s ease,border-color .15s ease;
}
.st-key-tabbar div.stButton > button:hover{
    color:var(--text);background:rgba(90,123,255,.10);
    border-bottom-color:rgba(90,123,255,.50);box-shadow:none;
}
.st-key-tabbar div.stButton > button[kind="primary"]{
    color:#ffffff;background:transparent;border-bottom:2px solid var(--accent);
    box-shadow:0 10px 20px -14px var(--accent);
    text-shadow:0 0 14px rgba(90,123,255,.70);
}

/* ---------- 整页外框 ---------- */
.st-key-pageframe,.st-key-pageframe-sm{
    border:1px solid var(--line);
    border-radius:5px;
    background:linear-gradient(180deg,rgba(12,17,64,.62),rgba(5,8,30,.62));
    box-shadow:0 0 30px rgba(28,40,150,.20),inset 0 0 56px rgba(28,40,150,.10);
    padding:9px 11px 8px 11px;
    margin-top:2px;
    overflow-y:auto;overflow-x:hidden;
}
.st-key-pageframe{max-height:calc(100vh - 210px);}
.st-key-pageframe-sm{max-height:calc(100vh - 180px);}
.st-key-pageframe::-webkit-scrollbar,.st-key-pageframe-sm::-webkit-scrollbar{width:7px;}
.st-key-pageframe::-webkit-scrollbar-thumb,.st-key-pageframe-sm::-webkit-scrollbar-thumb{
    background:#25357f;border-radius:3px;}
.st-key-pageframe::-webkit-scrollbar-track,.st-key-pageframe-sm::-webkit-scrollbar-track{
    background:transparent;}

/* ---------- 面板 ---------- */
.panel{
    position:relative;border:1px solid var(--line);border-radius:4px;
    background:linear-gradient(180deg,var(--panel-a),var(--panel-b));
    padding:10px 12px;margin-bottom:8px;
    box-shadow:inset 0 0 30px rgba(28,40,150,.12);
}
.panel h4,.panel h4.ptitle{
    display:flex;align-items:center;gap:8px;
    margin:0 0 8px 0;font-size:13px;font-weight:700;letter-spacing:1.6px;
    color:#dbe4ff;
}
/* 标题前的小竖条：统一版式节奏 */
.panel h4::before,.panel h4.ptitle::before{
    content:"";flex:0 0 auto;width:3px;height:13px;border-radius:2px;
    background:linear-gradient(180deg,#8ba4ff,#2a37c9);
    box-shadow:0 0 9px rgba(90,123,255,.75);
}
/* 大标题变体（3D 剖视面板用） */
h4.ptitle.big{font-size:17px;letter-spacing:2px;margin-bottom:9px;}
h4.ptitle.big::before{width:4px;height:17px;}
.panel.accent-hot{border-color:#5c1b1b;background:linear-gradient(180deg,rgba(60,10,10,.85),rgba(30,6,6,.85));}
[data-testid="stVerticalBlockBorderWrapper"]{
    border:1px solid var(--line)!important;
    border-radius:4px;
    background:linear-gradient(180deg,var(--panel-a),var(--panel-b));
    padding:10px 12px;margin-bottom:8px;
    box-shadow:inset 0 0 30px rgba(28,40,150,.12);
}
h4.ptitle{
    display:flex;align-items:center;gap:8px;
    margin:0 0 8px 0;font-size:13px;font-weight:700;letter-spacing:1.6px;
    color:#dbe4ff;
}
h4.ptitle::before{
    content:"";flex:0 0 auto;width:3px;height:13px;border-radius:2px;
    background:linear-gradient(180deg,#8ba4ff,#2a37c9);
    box-shadow:0 0 9px rgba(90,123,255,.75);
}

/* ---------- 文本工具 ---------- */
.muted{color:var(--muted);font-size:12px;}
.num{font-family:var(--mono);letter-spacing:.6px;color:var(--text);}
.kv{display:flex;justify-content:space-between;align-items:baseline;
    padding:3.5px 0;border-bottom:1px dashed rgba(90,123,255,.20);font-size:13px;}
.kv:last-child{border-bottom:0;}
.kv .k{color:#93a0d0;}
.kv .v{font-family:var(--mono);color:var(--text);font-weight:600;}

/* ---------- 状态大字 ---------- */
.state-card{text-align:center;padding:11px 8px;border-radius:4px;
    background:rgba(8,12,48,.92);border:1px solid var(--line);}
.state-title{font-size:11px;color:var(--muted);letter-spacing:3px;}
.state-value{font-size:27px;font-weight:900;letter-spacing:2px;margin:5px 0 3px 0;}
.state-conf{font-size:12px;color:#c3cdf0;}

/* ---------- 概率条 ---------- */
.probrow{display:flex;align-items:center;gap:7px;margin:4.5px 0;font-size:12px;}
.probrow .nm{width:52px;color:#a9b6e0;}
.probrow .bar{flex:1;height:8px;background:#0d1440;border-radius:4px;overflow:hidden;
              box-shadow:inset 0 0 6px rgba(0,0,0,.6);}
.probrow .bar>i{display:block;height:100%;border-radius:4px;box-shadow:0 0 8px currentColor;}
.probrow .pc{width:48px;text-align:right;font-family:var(--mono);color:var(--text);}

.badge{display:inline-block;padding:1px 8px;border-radius:2px;font-size:11px;font-weight:700;letter-spacing:.5px;}
.badge-ok{background:rgba(0,110,60,.28);color:#5ee79a;border:1px solid #17663c;}
.badge-warn{background:rgba(120,20,20,.34);color:#ff9a90;border:1px solid #7f1d1d;}
.badge-info{background:rgba(28,40,150,.38);color:#a8b8ff;border:1px solid #25357f;}

/* ---------- 指标卡 ---------- */
.statcard{border:1px solid var(--line);border-radius:4px;padding:9px 10px;text-align:center;
    background:linear-gradient(180deg,var(--panel-a),var(--panel-b));}
.statcard .lb{font-size:11px;color:var(--muted);letter-spacing:1.4px;}
.statcard .vl{font-family:var(--mono);font-size:23px;font-weight:800;margin-top:3px;}
.history-mobile-list{display:none;gap:7px;margin:6px 0 10px 0;}
.history-record-card{border:1px solid var(--line);border-radius:4px;padding:9px 10px;
    background:linear-gradient(180deg,var(--panel-a),var(--panel-b));min-width:0;}
.history-record-head{display:flex;justify-content:space-between;gap:8px;align-items:flex-start;
    padding-bottom:6px;margin-bottom:5px;border-bottom:1px solid var(--line-soft);}
.history-record-head strong{font-size:13px;overflow-wrap:anywhere;}
.history-record-status{color:var(--accent-hi);font-weight:700;text-align:right;overflow-wrap:anywhere;}
.history-record-fields{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-size:12px;}
.history-record-fields span{color:var(--muted);}
.history-record-fields b{color:var(--text);font-weight:600;overflow-wrap:anywhere;}

/* ---------- 检测结果 / 汇总：紧凑网格 ---------- */
.resgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:3px;}
.rescell{border:1px solid var(--line);border-radius:3px;padding:3px 5px;text-align:center;
    background:linear-gradient(180deg,rgba(14,20,70,.85),rgba(7,11,38,.85));}
.rescell .rk{font-size:9px;color:var(--muted);letter-spacing:.6px;}
.rescell .rv{font-family:var(--mono);font-size:14px;font-weight:800;color:var(--text);margin-top:1px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.rescell .rv.sm{font-size:10px;font-weight:600;color:#a8b8ff;font-family:var(--mono);}
.probwrap{border-top:1px dashed rgba(90,123,255,.20);padding-top:3px;margin-top:2px;}

/* ---------- 视频区：高度随画面比例自适应，等比完整显示 ---------- */
.st-key-videobox{
    width:100%;height:auto;min-height:200px;
    background:radial-gradient(circle at 50% 40%,#050a2a 0%,#00030f 88%);
    border:1px solid var(--line);border-radius:4px;
    display:flex;align-items:center;justify-content:center;
    overflow:hidden;padding:0;
}
/* 有画面时：框体贴合画面本身并居中，消除两侧大片深蓝底色。
   竖屏视频（如 720x1280）在宽框里只能占 26%，剩下 74% 都是这个蓝底 ——
   这就是"视频显示是蓝色"的原因。 */
.st-key-videobox:has(.videoinner){
    width:fit-content;max-width:100%;margin:0 auto;
    background:transparent;
}
.videoinner{display:flex;align-items:center;justify-content:center;}
.videoinner img{
    width:100%;height:100%;max-height:100%;
    object-fit:contain;object-position:center;
    display:block;border-radius:3px;
}
.st-key-videobox video{width:100%;height:42vh;object-fit:contain;display:block;}
.video-empty{display:flex;align-items:center;justify-content:center;height:200px;width:100%;
    color:#5b6b9a;font-size:14px;letter-spacing:1px;text-align:center;padding:10px;}

/* ---------- 上传：小按钮（popover） ---------- */
[data-testid="stPopover"] > button{
    background:rgba(28,40,150,.42);color:#b9c6ff;border:1px solid var(--line);
    border-radius:4px;padding:6px 13px;font-size:14px;font-weight:600;
    letter-spacing:.6px;font-family:var(--cn);width:auto!important;
}
[data-testid="stPopover"] > button:hover{
    background:rgba(90,123,255,.32);color:#ffffff;border-color:var(--accent);
    box-shadow:0 0 16px rgba(90,123,255,.45);
}
[data-testid="stPopoverBody"]{
    background:#060a26;border:1px solid var(--line);border-radius:5px;
    box-shadow:0 10px 30px rgba(0,0,0,.6);}
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"]{
    padding:6px 10px!important;min-height:0!important;
    background:rgba(9,13,48,.85);border:1px dashed var(--line);}
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] span{font-size:12px;}

/* ---------- 通用控件 ---------- */
div.stButton>button{
    width:100%;background:rgba(28,40,150,.34);color:#b9c6ff;
    border:1px solid var(--line);border-radius:4px;padding:7px 10px;font-size:15px;
    font-weight:600;letter-spacing:1.2px;font-family:var(--cn);transition:all .15s ease;
}
div.stButton>button:hover{background:rgba(90,123,255,.30);color:#ffffff;border-color:var(--accent);
    box-shadow:0 0 16px rgba(90,123,255,.40);}
div.stButton>button[kind="primary"]{
    background:linear-gradient(180deg,rgba(90,123,255,.42),rgba(42,55,201,.42));
    color:#ffffff;border-color:var(--accent);box-shadow:0 0 18px rgba(90,123,255,.45);}
[data-testid="stFileUploaderDropzone"]{padding:5px 10px!important;min-height:0!important;
    background:rgba(9,13,48,.8);border:1px dashed var(--line);}
[data-testid="stFileUploaderDropzone"] span{font-size:12px;color:#a8b8ff;}
[data-testid="stDataFrame"]{border:1px solid var(--line);}
.stSlider label,.stSelectbox label,.stTextInput label{font-size:12px!important;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:var(--mono);color:var(--accent-hi);}
@media (max-width: 700px){
    .st-key-historypage [data-testid="stHorizontalBlock"]{
        flex-wrap:wrap!important;align-items:stretch!important;
    }
    .st-key-historypage [data-testid="stHorizontalBlock"] > [data-testid="column"]{
        flex:1 1 100%!important;width:100%!important;min-width:0!important;max-width:100%!important;
    }
    .st-key-historypage .st-key-historykpis [data-testid="stHorizontalBlock"] > [data-testid="column"]{
        flex:1 1 calc(50% - .5rem)!important;width:calc(50% - .5rem)!important;
        min-width:0!important;max-width:calc(50% - .25rem)!important;
    }
    .st-key-historypage [data-testid="stDataFrame"],
    .st-key-historypage [data-testid="stPlotlyChart"],
    .st-key-historypage iframe{
        width:100%!important;max-width:100%!important;min-width:0!important;
    }
    .st-key-historytable{display:none!important;}
    .history-mobile-list{display:grid!important;}
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 会话状态
# ============================================================
def init_state():
    ss = st.session_state
    ss.setdefault("page", "perf")
    ss.setdefault("analyzer", FireEyeAnalyzer())
    ss.setdefault("upload_sig", None)
    # 已载入的源，用普通 session_state 保存（不是组件状态），
    # 这样切到「历史管理/系统设置」把上传组件清掉后，源和结果依然保留。
    ss.setdefault("src_sig", None)
    ss.setdefault("src_name", None)
    ss.setdefault("src_mime", "")
    ss.setdefault("src_bytes", None)
    ss.setdefault("results", None)
    ss.setdefault("summary", None)
    ss.setdefault("ai_report", "")
    ss.setdefault("display_bgr", None)
    ss.setdefault("source_name", "")
    ss.setdefault("sample_frames", DEFAULT_SAMPLE_FRAMES)
    ss.setdefault("agent_result", None)
    ss.setdefault("agent_error", None)
    ss.setdefault("agent_trace", [])
    ss.setdefault("agent_workflow", None)
    ss.setdefault("agent_job", None)
    ss.setdefault("agent_source_sig", None)
    ss.setdefault("modern_demo", None)
    ss.setdefault("modern_error", None)


init_state()


def reset_analysis():
    ss = st.session_state
    ss.results = ss.summary = ss.display_bgr = None
    ss.ai_report = ""
    ss.agent_result = None
    ss.agent_error = None
    ss.agent_trace = []
    ss.agent_workflow = None
    ss.agent_job = None
    ss.agent_source_sig = None
    ss.modern_demo = None
    ss.modern_error = None
    ss.analyzer.reset()


def cell_from_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return f"{stem}号槽" if stem else "未知来源"


def navigate(page: str):
    """切换页面。注意：不要叫 go —— 会遮蔽 `import plotly.graph_objects as go`。"""
    st.session_state.page = page
    st.rerun()


# ============================================================
# 图表（统一科技风）
# ============================================================
def _layout(fig, height=140, show_legend=False):
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=8, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Consolas, DejaVu Sans Mono, monospace", size=10, color="#8b97c9"),
        showlegend=show_legend,
        legend=dict(orientation="h", y=1.22, x=0, font=dict(size=10, color="#93a0d0")),
        xaxis=dict(showgrid=False, visible=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(23,80,125,.45)", zeroline=False),
    )
    return fig


def chart_series(results, key, color=CYAN, height=132, name=None):
    vals = [r.get(key, 0) for r in results]
    fig = go.Figure(go.Scatter(
        y=vals, mode="lines+markers", name=name or key,
        line=dict(color=color, width=2, shape="spline"),
        marker=dict(size=4, color=color),
        fill="tozeroy", fillcolor="rgba(0,234,255,.07)"))
    return _layout(fig, height)


def chart_multi(results, series, height=140):
    fig = go.Figure()
    for name, key, color, dash in series:
        fig.add_scatter(y=[r.get(key, 0) for r in results], mode="lines", name=name,
                        line=dict(color=color, width=2, dash=dash))
    return _layout(fig, height, show_legend=True)


def chart_donut(counts, height=190):
    keys = [k for k in ("H", "N", "C", "NoTarget") if counts.get(k, 0) > 0]
    values = [counts[k] for k in keys]
    labels = [CLASS_CN.get(k, k) for k in keys]
    colors = [STATE_COLOR[k] for k in keys]
    if not values:
        values, labels, colors = [1], ["暂无数据"], ["#12314c"]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.66,
                           marker=dict(colors=colors, line=dict(color="#04070f", width=2)),
                           textinfo="label+percent", textfont=dict(size=11, color="#d6e6f5")))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ============================================================
# 小组件
# ============================================================
def prob_bars(probs):
    rows = [("偏低 C", "#4d8dff", 0), ("偏高 H", "#c03030", 1), ("正常 N", "#3ddc84", 2)]
    html = []
    for cn, color, i in rows:
        p = float(probs[i]) if probs and i < len(probs) else 0.0
        html.append(
            f'<div class="probrow"><span class="nm">{cn}</span>'
            f'<span class="bar"><i style="width:{p*100:.1f}%;background:{color};color:{color}"></i></span>'
            f'<span class="pc">{p*100:.1f}%</span></div>')
    return "".join(html)


def kv_html(items):
    return "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>'
                   for k, v in items)


def stat_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        with col:
            st.markdown(f'<div class="statcard"><div class="lb">{label}</div>'
                        f'<div class="vl" style="color:{color}">{value}</div></div>',
                        unsafe_allow_html=True)


def no_data_hint(text="尚未分析。请上传火眼图片 / 视频并点「启动分析」。"):
    st.markdown(f'<div class="panel"><div class="muted">{text}</div></div>', unsafe_allow_html=True)


# ============================================================
# 铝电解槽 3D 结构展示（剖视）
# ============================================================
# 立方体三角面索引（8 顶点 → 12 三角形，顶点顺序与 _box_trace 的 x/y/z 一致）
_CUBE_I = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
_CUBE_J = [1, 2, 6, 7, 5, 4, 2, 6, 3, 7, 5, 6]
_CUBE_K = [2, 3, 5, 6, 1, 5, 6, 7, 7, 4, 6, 2]


def _box_trace(x0, y0, z0, x1, y1, z1, color, name, opacity=1.0):
    """一个长方体（plotly Mesh3d）。"""
    return go.Mesh3d(
        x=[x0, x1, x1, x0, x0, x1, x1, x0],
        y=[y0, y0, y1, y1, y0, y0, y1, y1],
        z=[z0, z0, z0, z0, z1, z1, z1, z1],
        i=_CUBE_I, j=_CUBE_J, k=_CUBE_K,
        color=color, opacity=opacity, name=name, showlegend=False, showscale=False,
        flatshading=True, hoverinfo="name",
        lighting=dict(ambient=0.74, diffuse=0.62, specular=0.16, roughness=0.88),
    )


def cell_3d(height=290, active=False):
    """
    铝电解槽本体 3D 剖视图（可拖拽旋转 / 滚轮缩放）。

    自下而上：阴极炭块 → 铝液层 → 电解质（冰晶石熔体 ~960℃）→ 壳面结壳；
    上方是浸入电解质的阳极炭块、阳极导杆与阳极母线；
    槽壳为半透明钢壳，正面剖开便于观察内部。
    """
    L, W = 5.0, 2.2          # 槽内腔半长 / 半宽
    wall = 0.34              # 槽壁厚
    traces = []

    # ---- 槽壳（钢壳）：底 + 后/左/右三面墙，正面剖开 ----
    traces.append(_box_trace(-L - wall, -W - wall, -0.34, L + wall, W + wall, 0.0,
                             "#54748f", "槽壳（钢壳）· 槽底", 0.95))
    traces.append(_box_trace(-L - wall, -W - wall, 0.0, -L, W + wall, 3.3,
                             "#54748f", "槽壳 · 侧壁", 0.40))
    traces.append(_box_trace(L, -W - wall, 0.0, L + wall, W + wall, 3.3,
                             "#54748f", "槽壳 · 侧壁", 0.40))
    traces.append(_box_trace(-L, W, 0.0, L, W + wall, 3.3,
                             "#54748f", "槽壳 · 端壁", 0.40))

    # ---- 槽内自下而上的各层 ----
    traces.append(_box_trace(-L, -W, 0.0, L, W, 0.5, "#2b2f38", "阴极炭块（槽底）", 1.0))
    traces.append(_box_trace(-L, -W, 0.5, L, W, 1.25, "#c9d3de", "铝液层", 1.0))
    traces.append(_box_trace(-L, -W, 1.25, L, W, 2.5,
                             "#ffb454" if active else "#ff9d3c",
                             "电解质（冰晶石熔体 ~960℃）", 0.95 if active else 0.88))

    # ---- 壳面结壳（中间留出火眼口）----
    cz0, cz1 = 2.5, 2.92
    traces.append(_box_trace(-L, -W, cz0, -1.0, W, cz1, "#6b5b4a", "壳面结壳", 1.0))
    traces.append(_box_trace(1.0, -W, cz0, L, W, cz1, "#6b5b4a", "壳面结壳", 1.0))
    traces.append(_box_trace(-1.0, -W, cz0, 1.0, -0.65, cz1, "#6b5b4a", "壳面结壳", 1.0))
    traces.append(_box_trace(-1.0, 0.65, cz0, 1.0, W, cz1, "#6b5b4a", "壳面结壳", 1.0))

    # ---- 阳极炭块 + 阳极导杆（一排 4 组，下端浸入电解质）----
    n = 4
    step = (2 * L) / n
    for k in range(n):
        cx = -L + step * (k + 0.5)
        traces.append(_box_trace(cx - step * 0.34, -W * 0.74, 1.95,
                                 cx + step * 0.34, W * 0.74, 3.75,
                                 "#1b1b22", "阳极炭块", 1.0))
        traces.append(_box_trace(cx - 0.09, -0.09, 3.75, cx + 0.09, 0.09, 4.75,
                                 "#b87333", "阳极导杆", 1.0))

    # ---- 阳极母线（直流供电）----
    traces.append(_box_trace(-L, -0.15, 4.75, L, 0.15, 5.1,
                             "#b87333", "阳极母线（直流供电）", 1.0))

    # ---- 有分析数据时，高亮火眼观察口 ----
    if active:
        traces.append(go.Scatter3d(
            x=[0], y=[0], z=[2.75], mode="markers+text",
            marker=dict(size=8, color="#5a7bff", symbol="diamond",
                        line=dict(color="#ffffff", width=1)),
            text=["火眼观察口"], textposition="top center",
            textfont=dict(size=10, color="#f4f7ff",
                          family="Microsoft YaHei, PingFang SC, sans-serif"),
            hoverinfo="skip", showlegend=False))

    # ---- 部件标注 ----
    ann = [
        ("槽壳",     L + wall + 0.15, W + 0.15, 2.5),
        ("壳面结壳", -L + 0.8,        W + 0.15, 3.1),
        ("阳极炭块", -L + step * 0.5, -W - 0.25, 3.7),
        ("阳极导杆", L - step * 0.5,  -W - 0.25, 4.5),
        ("阳极母线", 0.0,             -W - 0.25, 5.2),
        ("电解质",   0.0,             -W - 0.25, 2.0),
        ("铝液层",   0.0,             -W - 0.25, 0.85),
        ("阴极炭块", 0.0,             -W - 0.25, 0.2),
    ]
    traces.append(go.Scatter3d(
        x=[a[1] for a in ann], y=[a[2] for a in ann], z=[a[3] for a in ann],
        mode="text", text=[a[0] for a in ann],
        textfont=dict(size=10, color="#a8b8ff",
                      family="Microsoft YaHei, PingFang SC, sans-serif"),
        hoverinfo="skip", showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=0.95)),
            aspectmode="manual", aspectratio=dict(x=1.45, y=0.8, z=0.95),
        ),
    )
    return fig


def result_grid(cells):
    """检测结果紧凑网格：cells = [(标签, 值, 颜色, 是否小字)]，3 列自动换行。"""
    items = []
    for label, value, color, small in cells:
        cls = "rv sm" if small else "rv"
        style = "" if small else f' style="color:{color}"'
        items.append(f'<div class="rescell"><div class="rk">{label}</div>'
                     f'<div class="{cls}"{style}>{value}</div></div>')
    return f'<div class="resgrid">{"".join(items)}</div>'


def b64_jpeg(bgr, max_w=1280, quality=85):
    """
    把 BGR 帧编码成 data URI 的内联 JPEG。

    为什么不直接用 st.image：Streamlit 会给 <img> 加自己的内联样式与包裹结构，
    外部 CSS 经常抢不过它，导致画面显示得很小。自己拼 <img> 就能完全由 CSS 控制尺寸。
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return ""
    try:
        h, w = bgr.shape[:2]
        if w > max_w:                       # 限宽，避免 base64 过大拖慢页面
            bgr = cv2.resize(bgr, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def show_big_frame(bgr, cls="videoinner", scale=1.0):
    """
    按画面自身比例给容器**确定的高度**，然后铺满显示。

    关键点：`.videoinner` 必须自己带 `height:<n>vh`，不能写 `height:100%` ——
    Streamlit 在容器外面还套了 stElementContainer / stMarkdown 等层，
    这些中间层高度是 auto，`height:100%` 会解析成 auto，图片就退回原始尺寸，
    竖屏视频会垂直溢出并被 `overflow:hidden` 裁掉上下画面。
    竖屏给更高、宽屏给更矮，尽量把画面放大又不超出预算。
    """
    uri = b64_jpeg(bgr)
    if not uri:
        return False
    h, w = bgr.shape[:2]
    ratio = w / max(1, h)
    vh = (52 if ratio < 1 else (42 if ratio < 1.6 else 38)) * scale
    style = (f"height:{vh}vh;aspect-ratio:{w}/{h};width:auto;"
             f"max-width:100%;margin:0 auto;")
    st.markdown(f'<div class="{cls}" style="{style}"><img src="{uri}"/></div>',
                unsafe_allow_html=True)
    return True


def perf_charts(results):
    """四张性能指标曲线：耗时 / 吞吐 / 检出目标数 / 分类置信度。"""
    def prep():
        ms = [r.get("infer_ms", 0) for r in results]
        fps = [(1000.0 / m) if m else 0 for m in ms]
        objs = [len(r.get("boxes", {}).get("eye", [])) + len(r.get("boxes", {}).get("fire", []))
                for r in results]
        conf = [float(r.get("cls_conf") or 0) * 100 for r in results]
        return ms, fps, objs, conf

    specs = [
        ("单帧推理耗时 (ms)", "#5a7bff", 0),
        ("等效吞吐 (FPS)", "#3ddc84", 1),
        ("检出目标数 (个)", "#ffb020", 2),
        ("分类置信度 (%)", "#a58bff", 3),
    ]
    ms, fps, objs, conf = prep()
    data = [ms, fps, objs, conf]
    figs = []
    for i, (title, color, _) in enumerate(specs):
        y = data[i]
        fig = go.Figure(go.Scatter(
            y=y, mode="lines+markers",
            line=dict(color=color, width=2, shape="spline"),
            marker=dict(size=3.5, color=color),
            fill="tozeroy", fillcolor=f"rgba{tuple(int(color.lstrip('#')[j:j+2],16) for j in (0,2,4)) + (0.09,)}"))
        fig.update_layout(
            height=66, margin=dict(l=3, r=3, t=14, b=2),
            title=dict(text=title, font=dict(size=11, color="#93a0d0"), x=0.01, y=0.97),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(family="Consolas, DejaVu Sans Mono, monospace", size=9, color="#8b97c9"),
            xaxis=dict(showgrid=False, visible=False, zeroline=False),
            yaxis=dict(gridcolor="rgba(23,80,125,.42)", zeroline=False),
        )
        figs.append(fig)
    return specs, data, figs


# ============================================================
# 分析执行
# ============================================================
def run_analysis(data: bytes, name: str, mime: str = ""):
    ss = st.session_state
    is_image = (mime or "").startswith("image")
    analyzer = FireEyeAnalyzer()

    if is_image:
        bgr = load_bgr_from_bytes(data)
        if bgr is None:
            st.error("图片解码失败")
            return False
        res = analyzer.analyze_frame(bgr, immediate=True)
        results = [res]
        thumb = res["annotated_bgr"] if res.get("has_target") else bgr
    else:
        frames, _info = load_frames_from_bytes(data, max_frames=int(ss.sample_frames))
        if not frames:
            st.error("视频解码失败，或该视频没有可读帧")
            return False
        results = []
        bar = st.progress(0.0, text=f"正在逐帧推理 0/{len(frames)}（{get_device()}）")
        for i, f in enumerate(frames):
            results.append(analyzer.analyze_frame(f))
            bar.progress((i + 1) / len(frames),
                         text=f"正在逐帧推理 {i+1}/{len(frames)}（{get_device()}）")
        bar.empty()
        thumb = results[-1]["annotated_bgr"]

    summary = summarize(results)
    last = results[-1]
    ss.results, ss.summary = results, summary
    ss.display_bgr, ss.source_name, ss.ai_report = thumb, name, ""

    record = history_store.build_record(
        source=name, cell=cell_from_name(name), summary=summary, last=last, thumb_bgr=thumb)
    dosing = dosing_rules.analyze_dosing(results, summary)
    if dosing.get("ready"):
        record["dosing"] = {"action": dosing["action"], "delta": dosing["delta_text"],
                            "severity": dosing["severity"], "score": dosing["score"]}
    history_store.append_record(record)
    return True


def control_bar():
    """
    按页面自适应的控制条：

      性能监控：只有「载入视频/图片」——**载入即自动执行分析**，无需按钮
      过程诊断：载入（同样自动分析）+ 「🤖 AI 诊断」生成大模型报告
      加药分析：只有载入（自动分析），建议由分析结果推算
    """
    ss = st.session_state
    page = ss.page
    show_ai = (page == "diagnose")

    # 一行搞定：只有小按钮（+ 过程诊断页的 AI 诊断），不占整行、不带提示文字
    if show_ai:
        c1, c2 = st.columns([1.7, 1.6])
    else:
        c1, _ = st.columns([1.7, 8.3])

    with c1:
        with st.popover("📁 载入视频 / 图片"):
            up = st.file_uploader("源", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
                                  label_visibility="collapsed", key="main_upload")
            st.caption("支持 mp4 / avi / mov / jpg / png")

    ai_btn = False
    if show_ai:
        with c2:
            ai_btn = st.button("🤖 AI 诊断", type="primary", key="ai_btn")

    # ---- 只有"真的选了一个新文件"才清空重算 ----
    # 注意：切到「历史管理/系统设置」时本组件不渲染，Streamlit 会把它的状态清掉，
    # 回到分析页时 up 会变成 None。如果不加 `up is not None` 判断，就会误判成
    # "换了新文件"而调用 reset_analysis()，把已载入的视频和结果清空 ——
    # 表现就是"每进一个新页面都要重新载入"。
    if up is not None:
        data = up.getvalue()
        sig = f"{up.name}:{len(data)}"
        if sig != ss.get("src_sig"):
            ss.src_sig = sig
            ss.src_name = up.name
            ss.src_mime = up.type or ""
            ss.src_bytes = data
            reset_analysis()
            with st.spinner(f"正在推理…（{get_device()}）"):
                if run_analysis(data, up.name, up.type or ""):
                    st.rerun()

    # ---- AI 诊断（仅过程诊断页）----
    last = ss.results[-1] if ss.results else None
    if ai_btn:
        if last is None or ss.display_bgr is None:
            st.warning("请先载入火眼图片或视频")
        elif llava_analyze_frame is None:
            st.error("AI 模块未加载")
        else:
            with st.spinner(f"大模型推理中…（{llm_backend_summary()}）"):
                try:
                    frame = last["roi_bgr"] if last.get("roi_bgr") is not None else ss.display_bgr
                    ss.ai_report = llava_analyze_frame(frame, last, model="llava")
                except Exception as e:
                    ss.ai_report = f"AI 诊断失败：{e}"
    ss["_up"] = up          # 控制条现在由 main() 统一渲染，各页面从这里取
    return up


# ============================================================
# 1) 性能监控 —— 系统 / 推理性能
# ============================================================
def left_video_panel(scale=1.0, with_results=True):
    """
    左侧固定面板：视频 / 图像源 + 检测结果。

    三个分析页（性能监控 / 过程诊断 / 加药分析）共用同一个面板，
    源与分析结果都存在 session_state 里，**切页不会丢，不需要重新载入**。
    """
    ss = st.session_state
    src = ss.get("src_name")
    with st.container(key="videobox"):
        if ss.display_bgr is not None:
            show_big_frame(ss.display_bgr, scale=scale)
        elif ss.get("src_bytes") and not (ss.get("src_mime") or "").startswith("image"):
            st.video(ss.src_bytes, width="stretch")      # 已载入但还没分析时的预览
        else:
            st.markdown('<div class="video-empty">◈ 尚未载入画面 ◈<br>'
                        '点左上「📁 载入视频 / 图片」，载入后自动分析</div>',
                        unsafe_allow_html=True)

    if not with_results:
        return

    results = ss.results or []
    last = results[-1] if results else None
    code = (last or {}).get("cls_result", "NoTarget")
    conf = float((last or {}).get("cls_conf") or 0) * 100
    roi = last.get("roi_size") if last else None
    with_t = sum(1 for r in results if r.get("has_target")) if results else 0
    with st.container(border=True):
        st.markdown('<h4 class="ptitle">检测结果</h4>' + result_grid([
            ("工况判定", CLASS_CN.get(code, code), STATE_COLOR.get(code, DIM), False),
            ("置信度", f"{conf:.1f}%", "#f4f7ff", False),
            ("检出目标帧", f'{with_t} / {len(results)}' if results else "—", "#8ba4ff", False),
            ("采样帧数", str((ss.summary or {}).get("frames", 0)), "#8ba4ff", False),
            ("ROI 尺寸", f"{roi[0]}×{roi[1]}" if roi else "—", "#f4f7ff", False),
            ("来源", src or "未选择", "#a8b8ff", True),
        ]) + '<div class="probwrap">' + prob_bars(last["cls_probs"] if last else [0, 0, 0])
            + '</div>', unsafe_allow_html=True)


def agent_flow_chart(result, trace=None):
    """Render the actual local workflow as a compact, auditable graph."""
    trace = trace or []
    visual = (result or {}).get("visual", {})
    diagnosis = (result or {}).get("diagnosis", {})
    review = (result or {}).get("review", {})
    completed = {
        event.get("node"): event
        for event in trace
        if event.get("status") in {"completed", "error"}
    }
    node_status = {
        "perception_agent": completed.get("perception_agent", {}).get("status"),
        "knowledge_agent": completed.get("knowledge_agent", {}).get("status"),
        "diagnosis_agent": completed.get("diagnosis_agent", {}).get("status"),
        "review_agent": completed.get("review_agent", {}).get("status"),
        "finalize": completed.get("finalize", {}).get("status"),
    }
    nodes = [
        ("多帧视觉取证", "YOLO ROI + F-GMVAE-T", bool(visual.get("has_target")), "perception_agent"),
        ("知识检索", f"{len((result or {}).get('knowledge_sources', []))} 条来源",
         bool((result or {}).get("knowledge_sources")), "knowledge_agent"),
        ("诊断 Agent", "Qwen3-VL JSON", diagnosis.get("llm_status") == "qwen_vl_json", "diagnosis_agent"),
        ("证据复核", "通过" if review.get("approved") else "需复核", bool(review), "review_agent"),
        ("人工门控", "必须确认", False, "finalize"),
    ]
    fig = go.Figure()
    for index in range(len(nodes) - 1):
        fig.add_annotation(
            x=index + 0.5, y=0, ax=index + 0.15, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor="#5a7bff",
        )
    for index, (title, detail, passed, node_id) in enumerate(nodes):
        event = completed.get(node_id, {})
        status = node_status.get(node_id)
        if index == 4:
            color = "#ffb020"
        elif status == "error":
            color = "#c03030"
        elif status == "completed" and not passed:
            color = "#ffb020"
        elif passed:
            color = "#3ddc84"
        else:
            color = "#c03030" if result else "#5a7bff"
        if event.get("elapsed_ms") is not None:
            detail = f"{detail} · {event['elapsed_ms']:.1f} ms"
        fig.add_trace(go.Scatter(
            x=[index], y=[0], mode="markers+text",
            marker=dict(size=30, color=color, line=dict(color="#eef2ff", width=1.2)),
            text=[f"{index + 1}"], textposition="middle center",
            textfont=dict(size=12, color="#061027"),
            customdata=[[title, detail]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>状态: "
                f"{status or '未运行'}<br>%{{customdata[1]}}<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.add_annotation(
            x=index, y=-0.34, xref="x", yref="y", text=f"<b>{title}</b><br><sup>{detail}</sup>",
            showarrow=False, align="center", font=dict(size=11, color="#eef2ff"),
        )
    fig.update_layout(
        height=150, margin=dict(l=18, r=18, t=16, b=46),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-0.45, len(nodes) - 0.55], visible=False),
        yaxis=dict(range=[-0.62, 0.55], visible=False), showlegend=False,
    )
    return fig


def run_agent_diagnosis():
    """Run the optional graph against the currently loaded image/video."""
    ss = st.session_state
    if run_diagnosis_trace is None:
        st.error(f"多智能体模块未加载：{_AGENT_IMPORT_ERROR}")
        st.code("pip install -r Code\\agentic\\requirements.txt", language="powershell")
        return
    if not ss.get("src_bytes") or not ss.get("src_name"):
        st.warning("请先通过左上角载入视频或图片。")
        return

    suffix = os.path.splitext(ss.src_name)[1] or ".bin"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(ss.src_bytes)
            temp_path = handle.name
        node_labels = {
            "perception_agent": "视觉取证 Agent（YOLO ROI + F-GMVAE-T）",
            "knowledge_agent": "工艺知识 Agent（本地检索）",
            "diagnosis_agent": "多模态诊断 Agent（Qwen3-VL）",
            "review_agent": "证据复核 Agent（规则校验）",
            "finalize": "人工复核门控",
        }
        stage = st.status("多智能体协作中：准备运行…", expanded=True)

        def on_trace(event):
            node = node_labels.get(event.get("node"), event.get("node", "未知节点"))
            if event.get("status") == "running":
                stage.update(label=f"正在执行：{node}", state="running")
            elif event.get("status") == "completed":
                elapsed = event.get("elapsed_ms")
                suffix_text = f"（{elapsed:.0f} ms）" if elapsed is not None else ""
                stage.update(label=f"已完成：{node} {suffix_text}", state="running")
            elif event.get("status") == "error":
                stage.update(label=f"节点失败：{node}", state="error")

        trace_payload = None
        try:
            trace_payload = run_diagnosis_trace(
                temp_path,
                model=os.getenv("MARS_AGENT_MODEL", "qwen3-vl:latest"),
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                vision_device=os.getenv("MARS_VISION_DEVICE", "cpu"),
                trace_callback=on_trace,
            )
            if trace_payload.get("error"):
                stage.update(label="多智能体流程中断，请查看下方错误和运行轨迹", state="error")
            else:
                stage.update(label="多智能体流程完成", state="complete")
        except Exception:
            stage.update(label="多智能体流程失败", state="error")
            raise
        if trace_payload is not None:
            ss.agent_result = trace_payload["result"]
            ss.agent_error = trace_payload.get("error")
            ss.agent_trace = trace_payload.get("trace", [])
            ss.agent_workflow = trace_payload.get("workflow")
        ss.agent_source_sig = ss.src_sig
    except Exception as exc:
        ss.agent_result = None
        ss.agent_error = str(exc)
        ss.agent_trace = []
        ss.agent_workflow = None
        st.error(f"多智能体流程失败：{exc}")
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _modern_runtime_root() -> Path:
    configured = os.environ.get("MARS_AGENT_RUNTIME_DIR", "")
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parents[1] / "runtime_data"


def run_modern_agent_page(*, parent_run_id: Optional[str] = None, retake_of_run_id: Optional[str] = None):
    """Run the current JEV + multi-frame + Process State demo from the UI."""
    ss = st.session_state
    if run_modern_demo is None:
        ss.modern_error = f"modern agent backend unavailable: {_MODERN_IMPORT_ERROR}"
        return
    if not ss.get("src_bytes") or not ss.get("src_name"):
        ss.modern_error = "Please load a video or image first."
        return
    try:
        code_root = Path(__file__).resolve().parents[1]
        runtime_root = _modern_runtime_root()
        media_root = (
            runtime_root / "modern_demos" / "retakes" / parent_run_id
            if parent_run_id
            else runtime_root / "modern_demos" / "media"
        )
        source_name = Path(ss.src_name).name
        media_digest = hashlib.sha256(ss.src_bytes).hexdigest()
        media_path = media_root / media_digest / source_name
        media_path.parent.mkdir(parents=True, exist_ok=True)
        if not media_path.exists():
            media_path.write_bytes(ss.src_bytes)
        output = runtime_root / f"streamlit_{time.time_ns()}.json"
        profile = ss.get("modern_llm_profile") or os.environ.get("MARS_AGENT_LLM_PROFILE", "fake_json")
        jev_backend = ss.get("modern_jev_backend") or os.environ.get("MARS_AGENT_JEV_BACKEND", "deterministic_rule")
        ss.modern_demo = run_modern_demo(
            code_root,
            media_path,
            output=output,
            llm_profile=profile,
            jev_backend=jev_backend,
            runtime_dir=runtime_root,
            native_hitl=True,
            parent_run_id=parent_run_id,
            retake_of_run_id=retake_of_run_id,
            jev_policy_registry_root=os.environ.get("MARS_AGENT_RSI_REGISTRY"),
        )
        ss.modern_error = None
    except Exception as exc:  # keep the UI usable and show the exact failure
        ss.modern_demo = None
        ss.modern_error = str(exc)


def _run_modern_retake(parent_demo: Mapping[str, Any], uploaded_file: Any) -> None:
    """Persist replacement media and start a new run linked to its retake parent."""
    parent_run_id = str(parent_demo.get("run_id", ""))
    runtime_root = _modern_runtime_root()
    decision = (parent_demo.get("human_review") or {}).get("decision") or {}
    if not parent_run_id or decision.get("action") != "request_retake":
        st.session_state.modern_error = "A retake can only continue a run with a recorded request_retake decision."
        return
    try:
        persisted_parent = load_persisted_modern_demo(runtime_root, parent_run_id)
        persisted_decision = (persisted_parent.get("human_review") or {}).get("decision") or {}
        parent_audit = HumanReviewService(runtime_root).history(run_id=parent_run_id)
    except Exception as exc:
        st.session_state.modern_error = f"Unable to verify the parent retake audit: {exc}"
        return
    if (
        persisted_decision.get("action") != "request_retake"
        or not any(item.get("action") == "request_retake" for item in parent_audit)
    ):
        st.session_state.modern_error = "The persisted parent run has no request_retake audit; continuation was not started."
        return
    if uploaded_file is None:
        st.session_state.modern_error = "Upload a replacement image or video before starting the retake run."
        return
    try:
        data = uploaded_file.getvalue()
        filename = Path(str(uploaded_file.name).replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"}:
            raise ValueError("unsupported retake media type")
        st.session_state.src_bytes = data
        st.session_state.src_name = filename
        st.session_state.src_mime = getattr(uploaded_file, "type", "") or ""
        run_modern_agent_page(parent_run_id=parent_run_id, retake_of_run_id=parent_run_id)
        if st.session_state.get("modern_error"):
            st.session_state.modern_demo = parent_demo
    except Exception as exc:
        st.session_state.modern_error = f"Unable to start retake run: {exc}"


def _presentation_demo_catalog():
    """Return the local, reproducible demo traces used by the PPT walkthrough."""
    code_root = Path(__file__).resolve().parents[1]
    configured_dir = os.environ.get("MARS_AGENT_PRESENTATION_DEMO_DIR", "").strip()
    demo_dir = Path(configured_dir).expanduser() if configured_dir else next(
        (candidate for candidate in (
            code_root / "runtime_data" / "presentation_demo",
            code_root / "runtime_data" / "presentation_demos",
        ) if candidate.exists()),
        code_root / "runtime_data" / "presentation_demo",
    )
    if not demo_dir.exists():
        return {}
    catalog = {}
    for path in sorted(demo_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and path.name not in {"index.json"} and not path.name.endswith(".sidecar.json"):
            case_id = str(payload.get("presentation_case", path.stem))
            catalog[case_id] = {"path": path, "payload": payload}
    return catalog


def _load_latest_evaluation_report():
    """Load the latest generated aggregate evaluation, if one is available."""
    runtime_dir = _modern_runtime_root()
    candidates = [
        runtime_dir / "evaluation_report.json",
        *runtime_dir.glob("evaluation.e2_*.json"),
        *runtime_dir.glob("evaluation.e3_*.json"),
    ]
    candidates = [path for path in candidates if path.is_file()]
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == "evaluation-report-1":
            return path, payload
    return None, None


def _load_latest_delivery_snapshot():
    """Load the newest fail-closed delivery evidence audit for the UI."""
    runtime_dir = _modern_runtime_root()
    candidates = [
        *runtime_dir.glob("delivery_snapshot*.json"),
        *runtime_dir.glob("**/delivery_snapshot*.json"),
    ]
    seen: set[Path] = set()
    for path in sorted((item for item in candidates if item.is_file()), key=lambda item: item.stat().st_mtime, reverse=True):
        if path in seen:
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == "delivery-snapshot-1":
            return path, payload
    return None, None


def _normalize_presentation_payload(payload):
    """Adapt compact PPT fixtures to the normal modern-page payload shape."""
    if not isinstance(payload, dict) or payload.get("schema_version") != "presentation-demo-1":
        return payload
    normalized = dict(payload)
    route_decision = dict(payload.get("route_decision") or {})
    if route_decision.get("implementation") == "presentation_fixture":
        route_decision.update({
            "router_backend": "deterministic_rule",
            "router_version": "deterministic_alu_router_v1",
            "gate_version": "alu-jev-safety-0.1",
            "model_profile": "deterministic_alu_router_v1",
            "policy_version": "base",
            "policy_hash": "builtin",
        })
    input_manifest = dict(payload.get("input") or {})
    source = input_manifest.get("source", "")
    input_manifest.setdefault("path", source)
    input_manifest.setdefault("media_type", "video" if input_manifest.get("kind") == "video" else "fixture")
    input_manifest.setdefault("width", "—")
    input_manifest.setdefault("height", "—")
    normalized["input"] = input_manifest
    normalized["quality"] = {
        "quality_status": input_manifest.get("quality_status", "pass"),
        "issues": [] if input_manifest.get("quality_status") == "pass" else ["presentation_fixture_quality_gate"],
        "target_present": input_manifest.get("quality_status") not in {"retake", "human"},
    }
    normalized["visual"] = dict(payload.get("visual_evidence") or {})
    normalized["pre_route_decision"] = route_decision
    normalized.setdefault("router_backend", route_decision.get("router_backend"))
    normalized.setdefault("router_version", route_decision.get("router_version"))
    normalized.setdefault("knowledge", [])
    normalized.setdefault("cases", [])
    normalized.setdefault("knowledge_version", "knowledge-pack-v1")
    normalized.setdefault("prompt_version", "diagnosis-aggregator-v2-flow-validation")
    ticket = {
        "ticket_id": f"ticket-{payload.get('run_id', 'presentation')}",
        "run_id": payload.get("run_id", ""),
        "thread_id": payload.get("thread_id", ""),
        "stage": "human_review",
        "status": "waiting_review",
        "route": route_decision.get("route", "unknown"),
    }
    normalized["ticket"] = ticket
    agents = list(payload.get("agents") or [])
    trace = payload.get("graph_trace")
    if not isinstance(trace, list):
        trace = [{
            "run_id": payload.get("run_id", ""),
            "thread_id": payload.get("thread_id", ""),
            "node": f"full_loop_worker:{item.get('agent_id', 'agent')}",
            "status": item.get("status", "completed"),
            "elapsed_ms": item.get("duration_ms"),
            "agent_id": item.get("agent_id", "agent"),
            "attempt": item.get("attempt", 1),
        } for item in agents]
        trace.insert(0, {
            "run_id": payload.get("run_id", ""),
            "thread_id": payload.get("thread_id", ""),
            "node": "jev_router",
            "status": "completed",
            "elapsed_ms": None,
            "route": route_decision.get("route", "unknown"),
            "router_backend": route_decision.get("router_backend", "unknown"),
            "model_profile": route_decision.get("model_profile", "unknown"),
            "gate_version": route_decision.get("gate_version", "unknown"),
            "policy_version": route_decision.get("policy_version", "base"),
            "policy_hash": route_decision.get("policy_hash", "builtin"),
        })
    graph_result = payload.get("graph_result") if isinstance(payload.get("graph_result"), dict) else {}
    route_history = payload.get("route_history") or graph_result.get("route_history") or [route_decision.get("route", "unknown")]
    initial_route = payload.get("initial_route") or graph_result.get("initial_route") or route_history[0]
    upgrade_count = payload.get("upgrade_count", graph_result.get("upgrade_count", 0))
    final_route = str(route_history[-1]) if route_history else route_decision.get("route", "unknown")
    ticket = graph_result.get("ticket") if isinstance(graph_result.get("ticket"), dict) else ticket
    normalized["graph"] = {
        "route": final_route,
        "route_decision": route_decision,
        "status": "waiting_review",
        "trace": trace,
        "evidence_bundle": agents,
        "ticket": ticket,
        "result": {
            "diagnosis": graph_result.get("diagnosis", payload.get("diagnosis", {})),
            "review": graph_result.get("review", payload.get("review", payload.get("human_review", {}))),
            "evidence_bundle": agents,
            "ticket": ticket,
            "initial_route": initial_route,
            "route_history": route_history,
            "upgrade_count": upgrade_count,
        },
    }
    return normalized


def _load_presentation_demo(case_id):
    """Load a precomputed trace without invoking a model provider."""
    catalog = _presentation_demo_catalog()
    item = catalog.get(case_id)
    if not item:
        raise FileNotFoundError(f"presentation demo not found: {case_id}")
    payload = _normalize_presentation_payload(item["payload"])
    if PresentationReviewStore is not None:
        persisted = PresentationReviewStore(_modern_runtime_root()).load(str(payload.get("run_id", "")))
        if persisted is not None:
            payload = persisted
    code_root = Path(__file__).resolve().parents[1]
    manifest = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    source_value = manifest.get("path") or manifest.get("input_path") or payload.get("source_media")
    if source_value:
        source = Path(str(source_value))
        candidates = [source, code_root / source, code_root.parent / source]
        source = next((candidate for candidate in candidates if candidate.exists()), source)
        if source.exists() and source.is_file():
            st.session_state.src_bytes = source.read_bytes()
            st.session_state.src_name = source.name
            st.session_state.src_sig = f"presentation:{case_id}:{source.stat().st_mtime_ns}"
    st.session_state.modern_demo = payload
    st.session_state.modern_error = None
    st.session_state.presentation_case = case_id


def _apply_presentation_review(demo, action, amendment=""):
    """Apply an allowlisted review action to a presentation-only trace."""
    if PresentationReviewStore is None:
        raise RuntimeError("presentation review storage is unavailable")
    return PresentationReviewStore(_modern_runtime_root()).apply(demo, action, amendment)


def _presentation_report(demo):
    """Build a deterministic archive report for presentation-only traces."""
    review = dict(demo.get("human_review") or {})
    graph = dict(demo.get("graph") or {})
    final_diagnosis = dict((graph.get("result") or {}).get("diagnosis") or demo.get("diagnosis") or {})
    return {
        "schema_version": "presentation-report-1",
        "status": graph.get("status", review.get("status", "waiting_review")),
        "claim_status": "flow_validation",
        "run_id": demo.get("run_id"),
        "thread_id": demo.get("thread_id"),
        "route": graph.get("route", demo.get("route")),
        "initial_route": (graph.get("result") or {}).get("initial_route"),
        "route_history": (graph.get("result") or {}).get("route_history", []),
        "upgrade_count": (graph.get("result") or {}).get("upgrade_count", 0),
        "final_diagnosis": final_diagnosis,
        "router": demo.get("router", {}),
        "human_review": review,
        "trace": demo.get("trace", []),
        "agents": demo.get("agents", []),
        "case_candidate": demo.get("case_candidate"),
        "knowledge_version": demo.get("knowledge_version"),
        "prompt_version": demo.get("prompt_version"),
        "claim_policy": demo.get("claims", {
            "metrics": False,
            "real_superheat_measurement": False,
            "video_process_alignment": False,
        }),
    }


def _presentation_report_exports(demo, report):
    """Return report export payloads without depending on the durable service."""
    run_id = str(report.get("run_id") or "presentation-demo")
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    review = report.get("human_review") or {}
    agent_rows = report.get("agents") or []
    markdown = "\n".join([
        f"# 铝电解火眼多智能体演示报告",
        "",
        f"- run_id: `{run_id}`",
        f"- route: `{report.get('route', '—')}`",
        f"- status: `{report.get('status', '—')}`",
        f"- review: `{review.get('action', 'pending')}`",
        f"- claim status: `{report.get('claim_status', 'flow_validation')}`",
        "",
        "## Agent 结果",
        "",
        *[f"- `{row.get('agent_id', 'agent')}`: {row.get('status', 'unknown')} / attempt {row.get('attempt', 1)}" for row in agent_rows],
        "",
        "该报告用于流程演示和回放，不代表视觉精度、真实过热度测量或视频-过程数据对齐效果。",
        "",
    ])
    html = "<html><head><meta charset='utf-8'><title>铝电解火眼多智能体演示报告</title></head><body>"
    html += f"<h1>铝电解火眼多智能体演示报告</h1><p>run_id: <code>{run_id}</code></p>"
    html += f"<p>route: <code>{report.get('route', '—')}</code> · status: <code>{report.get('status', '—')}</code> · review: <code>{review.get('action', 'pending')}</code></p>"
    html += "<h2>Agent 结果</h2><ul>"
    html += "".join(f"<li><code>{row.get('agent_id', 'agent')}</code>: {row.get('status', 'unknown')} / attempt {row.get('attempt', 1)}</li>" for row in agent_rows)
    html += "</ul><p>流程验证素材，不代表模型精度、真实过热度测量或视频-过程数据对齐效果。</p></body></html>"
    png_bytes = None
    try:
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (1600, 900), "#08122d")
        draw = ImageDraw.Draw(image)
        lines = [
            "铝电解火眼多智能体演示报告",
            f"run_id: {run_id}",
            f"route: {report.get('route', '—')}    status: {report.get('status', '—')}",
            f"review: {review.get('action', 'pending')}    claim: flow_validation",
            "",
            "Agent 状态:",
        ]
        lines.extend(f"  {row.get('agent_id', 'agent')}: {row.get('status', 'unknown')}" for row in agent_rows)
        lines.extend(["", "流程验证素材，不代表模型精度或真实过热度测量。"])
        draw.multiline_text((80, 70), "\n".join(lines), fill="#eef2ff", spacing=18)
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
    except Exception:
        png_bytes = None
    return {
        "json": report_json,
        "markdown": markdown,
        "html": html,
        "png": png_bytes,
    }


def _render_modern_agent_page():
    """Render the implemented multi-agent path and durable HITL controls."""
    ss = st.session_state
    demo = ss.get("modern_demo")
    st.markdown('<h4 class="ptitle big">多智能体协作流程</h4>', unsafe_allow_html=True)
    st.caption("多帧视觉取证 → Process State → 风险感知 JEV 路由 → LangGraph 多 Agent → 人工确认 → 报告归档")
    c1, c2, c3 = st.columns([1.5, 1.5, 4.5])
    with c1:
        if st.button("▶ 运行多智能体", type="primary", key="modern_run"):
            run_modern_agent_page()
            st.rerun()
    with c2:
        if st.button("清除流程", key="modern_clear"):
            ss.modern_demo = None
            ss.modern_error = None
            st.rerun()
    with c3:
        st.caption(f"当前输入：{ss.get('src_name') or '未加载'}")
    runtime_root = _modern_runtime_root()
    try:
        pending_runs = [
            item for item in HumanReviewService(runtime_root).pending_reviews()
            if (runtime_root / "modern_demos" / f"{item.get('run_id', '')}.json").is_file()
        ] if HumanReviewService is not None else []
    except Exception:
        pending_runs = []
    if pending_runs:
        with st.expander("恢复待审任务", expanded=not bool(demo)):
            options = [str(item.get("run_id", "")) for item in pending_runs]
            selected_run = st.selectbox("待审 run_id", options, key="modern_pending_run_select")
            selected_checkpoint = next((item for item in pending_runs if item.get("run_id") == selected_run), {})
            st.caption(f"thread_id: {selected_checkpoint.get('thread_id', '')} · route: {selected_checkpoint.get('route', '')}")
            if st.button("加载待审任务", key="load_pending_modern_demo"):
                try:
                    restored = load_persisted_modern_demo(runtime_root, selected_run)
                    ss.modern_demo = restored
                    media_path = Path(str((restored.get("input") or {}).get("path", "")))
                    if media_path.is_file():
                        ss.src_bytes = media_path.read_bytes()
                        ss.src_name = media_path.name
                    ss.modern_error = None
                    st.rerun()
                except Exception as exc:
                    ss.modern_error = f"Unable to restore pending run: {exc}"
    presentation_catalog = _presentation_demo_catalog()
    if presentation_catalog:
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">汇报演示模式</h4>', unsafe_allow_html=True)
            st.caption("加载预生成的结构化 trace，现场无需等待模型调用；所有结果标记为 flow_validation。")
            demo_options = list(presentation_catalog)
            current_case = ss.get("presentation_case")
            selected_case = st.selectbox(
                "演示案例",
                demo_options,
                index=demo_options.index(current_case) if current_case in demo_options else 0,
                format_func=lambda value: str((presentation_catalog[value]["payload"] or {}).get("presentation_title", value)),
                key="presentation_case_select",
            )
            if st.button("加载演示 Trace", key="load_presentation_demo", use_container_width=True):
                try:
                    _load_presentation_demo(selected_case)
                    st.rerun()
                except Exception as exc:
                    ss.modern_error = f"演示 Trace 加载失败：{exc}"
    jev_options = ("deterministic_rule", "laya_adapter")
    configured_jev = ss.get("modern_jev_backend", os.environ.get("MARS_AGENT_JEV_BACKEND", "deterministic_rule"))
    ss.modern_jev_backend = st.selectbox(
        "JEV 路由后端",
        jev_options,
        index=jev_options.index(configured_jev) if configured_jev in jev_options else 0,
        format_func=lambda value: "Laya typed-decisions + 工艺安全门" if value == "laya_adapter" else "确定性规则基线",
        key="modern_jev_backend_select",
        help="Laya 后端仅在安全规则判定为复杂任务时参与路径建议；硬安全条件仍由工艺规则门控。首次调用可能需要下载 Laya 模型权重。",
    )
    ss.modern_llm_profile = st.selectbox(
        "诊断聚合模型",
        ("fake_json", "ollama_local", "qwen_api", "fallback_human"),
        index=("fake_json", "ollama_local", "qwen_api", "fallback_human").index(ss.get("modern_llm_profile", os.environ.get("MARS_AGENT_LLM_PROFILE", "fake_json"))) if ss.get("modern_llm_profile", os.environ.get("MARS_AGENT_LLM_PROFILE", "fake_json")) in {"fake_json", "ollama_local", "qwen_api", "fallback_human"} else 0,
        key="modern_llm_profile_select",
        help="JEV 是诊断路径路由层；Laya 后端为可选开源路由模型适配。安全门控负责硬约束，诊断聚合模型仅在 full_loop 并行取证后调用。",
    )
    if ss.get("modern_error"):
        st.error(ss.modern_error)
    if not demo:
        st.info("加载视频后运行，页面将显示 JEV 路由、并行 Agent 和人工审核状态。")
        return
    graph = demo.get("graph", {})
    visual = demo.get("visual", {})
    quality = demo.get("quality", {})
    process = demo.get("process_state", {})
    route = graph.get("route", "unknown")
    status = graph.get("status", graph.get("result", {}).get("status", "unknown"))
    status_label = {
        "waiting_review": "待人工确认",
        "waiting_for_retake": "待补拍",
        "demo_summary_waiting_review": "演示待人工确认",
        "retake_requested": "请求补拍",
        "request_more_evidence": "待补充证据",
        "completed": "已完成",
        "approved": "已批准",
        "rejected": "已驳回",
        "failed": "执行失败",
    }.get(str(status), str(status))
    process_status = {
        "demo_summary": "演示摘要",
    }.get(str(process.get("process_regime", "unknown")), str(process.get("process_regime", "unknown")))
    with st.container(border=True):
        st.markdown('<h4 class="ptitle">诊断状态</h4>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("JEV 诊断路径", route)
        m2.metric("质量门控", quality.get("quality_status", "unknown"))
        m3.metric("视觉置信度", f"{float(visual.get('confidence', 0.0)) * 100:.1f}%")
        m4.metric("过程状态", process_status)
        m5.metric("流程状态", status_label)
        flow_result = graph.get("result", {})
        route_history = flow_result.get("route_history", [])
        if isinstance(route_history, list) and len(route_history) > 1:
            st.caption(f"诊断循环：{' → '.join(str(item) for item in route_history)} · 升级 {flow_result.get('upgrade_count', 0)} 次")
        ticket = demo.get("ticket") or graph.get("ticket", {}) or {}
        route_info = demo.get("jev_router", {})
        policy_version = route_info.get("policy_version", (demo.get("pre_route_decision") or {}).get("policy_version", "base"))
        rsi_status = (demo.get("rsi_runtime") or {}).get("status", "unavailable")
        st.caption(f"run_id: {demo.get('run_id')} · thread_id: {demo.get('thread_id')} · ticket_id: {ticket.get('ticket_id', '—')} · stage: {ticket.get('stage', '—')} · backend: {route_info.get('decision_backend', 'unknown')} · profile: {route_info.get('model_profile', 'unknown')} · RSI: {policy_version}/{rsi_status} · alignment_quality: {process.get('alignment_quality', 'unknown')}")
    with st.container(border=True):
        st.markdown('<h4 class="ptitle">输入与多帧视觉证据</h4>', unsafe_allow_html=True)
        preview_left, preview_right = st.columns(2)
        with preview_left:
            media_type = str((demo.get("input") or {}).get("media_type", ""))
            if ss.get("src_bytes"):
                if media_type == "video":
                    st.video(ss.src_bytes)
                else:
                    st.image(ss.src_bytes, use_container_width=True)
            st.caption(f"{demo.get('input', {}).get('width', '—')} × {demo.get('input', {}).get('height', '—')} · {media_type or 'input'}")
        with preview_right:
            contact = visual.get("contact_sheet")
            contact_path = Path(str(contact)) if contact else None
            if contact_path and not contact_path.is_absolute():
                code_root = Path(__file__).resolve().parents[1]
                candidates = (code_root / contact_path, code_root.parent / contact_path)
                contact_path = next((item for item in candidates if item.exists()), contact_path)
            if contact_path and contact_path.exists():
                st.image(str(contact_path), caption="ROI 多帧 contact sheet", use_container_width=True)
            else:
                st.info("当前运行未生成可显示的 contact sheet")
            st.caption(f"帧序列：{visual.get('frame_ids', [])} · 关键帧：{visual.get('key_frame_ids', [])} · 丢失帧：{visual.get('missing_count', 0)}")
    trace = graph.get("trace", [])
    if trace:
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">Agent 协作轨迹</h4>', unsafe_allow_html=True)
            rows = [{"节点": event.get("node"), "状态": event.get("status"), "耗时(ms)": event.get("elapsed_ms"), "摘要": json.dumps({k: v for k, v in event.items() if k not in {"run_id", "thread_id", "node", "status", "elapsed_ms"}}, ensure_ascii=False)} for event in trace]
            st.dataframe(rows, use_container_width=True, hide_index=True)
    evidence_bundle = graph.get("evidence_bundle") or graph.get("result", {}).get("evidence_bundle", [])
    if evidence_bundle:
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">并行 Agent 状态</h4>', unsafe_allow_html=True)
            agent_columns = st.columns(min(5, max(1, len(evidence_bundle))))
            for column, item in zip(agent_columns, evidence_bundle):
                agent_id = str(item.get("agent_id", item.get("node", "agent")))
                label = {
                    "visual_review": "多帧视觉复核",
                    "evidence_quality": "证据完整性",
                    "knowledge_rag": "工艺知识 RAG",
                    "case_compare": "特殊案例比对",
                    "process_state": "Process State",
                }.get(agent_id, agent_id)
                status_value = str(item.get("status", "unknown"))
                with column:
                    st.metric(label, "完成" if status_value == "completed" else status_value)
                    st.caption(f"耗时 {item.get('elapsed_ms', '—')} ms · attempt {item.get('attempt', 1)}")
                    evidence_ids = item.get("evidence_ids") or item.get("output", {}).get("evidence_ids", [])
                    st.caption(f"证据 {len(evidence_ids) if isinstance(evidence_ids, list) else 0} 条")
            worker_rows = []
            for item in evidence_bundle:
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                worker_rows.append({
                    "Agent": item.get("agent_id", item.get("node", "agent")),
                    "策略": item.get("retry_policy_id", metadata.get("retry_policy_id", "—")),
                    "超时(s)": item.get("timeout_s", metadata.get("timeout_s", "—")),
                    "Attempt": item.get("attempt", metadata.get("attempt", 1)),
                    "重试": item.get("retry_count", metadata.get("retry_count", 0)),
                    "退避(s)": item.get("retry_delays_seconds", metadata.get("retry_delays_seconds", [])),
                    "错误": item.get("error_code", metadata.get("error_code")) or "—",
                })
            with st.expander("Worker 超时与重试策略"):
                st.dataframe(worker_rows, use_container_width=True, hide_index=True)
    pending = demo.get("human_review") or {}
    if pending.get("status") in {"waiting_review", "waiting_for_retake"}:
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">人工确认</h4>', unsafe_allow_html=True)
            st.caption("当前任务已暂停，审核结果会写入持久化审计记录。")
            amendment = st.text_area("amend 内容（可选）", key="modern_amendment")
            buttons = st.columns(4)
            for column, action, label in zip(buttons, ("approve", "amend", "reject", "request_retake"), ("通过", "修改并通过", "驳回", "请求补拍")):
                with column:
                    if st.button(label, key=f"modern_review_{action}", use_container_width=True):
                        amendments = {"operator_note": amendment} if action == "amend" and amendment else {}
                        if demo.get("schema_version") == "presentation-demo-1":
                            ss.modern_demo = _apply_presentation_review(demo, action, amendment)
                        elif demo.get("human_review_backend") == "langgraph_sqlite_interrupt":
                            ss.modern_demo = resume_demo_review(
                                runtime_root,
                                demo["run_id"],
                                action=action,
                                reviewer_id="streamlit_operator",
                                reason="web review",
                                amendments=amendments,
                            )
                        else:
                            service = HumanReviewService(runtime_root)
                            outcome = service.resume(demo["run_id"], action=action, reviewer_id="streamlit_operator", reason="web review", amendments=amendments)
                            demo["human_review"] = outcome
                            demo["graph"]["status"] = outcome.get("result", {}).get("status", outcome.get("status"))
                            ss.modern_demo = demo
                        st.rerun()
    with st.expander("视觉证据", expanded=True):
        st.json(visual)
    decision = pending.get("decision") or {}
    if (
        demo.get("human_review_backend") == "langgraph_sqlite_interrupt"
        and decision.get("action") == "request_retake"
    ):
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">补拍续检</h4>', unsafe_allow_html=True)
            st.caption(f"本次补拍将创建新的 run，并关联父任务 {demo.get('run_id')}。")
            replacement = st.file_uploader(
                "补拍图片或视频",
                type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
                key="modern_retake_upload",
            )
            if st.button("开始补拍续检", key="modern_start_retake", type="primary"):
                _run_modern_retake(demo, replacement)
                if not ss.get("modern_error"):
                    st.rerun()
    with st.expander("Process State Agent"):
        st.json(process)
        variables = ((process.get("variables") or {}).get("by_name") if isinstance(process, dict) else None)
        if isinstance(variables, dict):
            st.dataframe(
                [{"变量": name, "分组": value.get("group", ""), "均值": value.get("mean"), "趋势": value.get("slope_per_step"), "缺失": value.get("missing", False)} for name, value in variables.items()],
                use_container_width=True, hide_index=True,
            )
    with st.expander("JEV 路由依据、案例与工艺知识"):
        route_decision = demo.get("pre_route_decision") or graph.get("route_decision") or {}
        st.json({
            "route": route_decision.get("route", route),
            "router_backend": route_decision.get("router_backend", demo.get("router_backend", "deterministic_rule")),
            "router_version": route_decision.get("router_version", demo.get("router_version", "deterministic_alu_router_v1")),
            "reasons": route_decision.get("reasons", []),
            "confidence": route_decision.get("confidence"),
            "risk_level": route_decision.get("risk_level"),
            "gate_version": route_decision.get("gate_version"),
            "similar_case_ids": route_decision.get("similar_case_ids", []),
            "evidence_gaps": route_decision.get("evidence_gaps", []),
        })
        st.markdown("**案例匹配**")
        st.json(demo.get("cases", []))
        st.markdown("**RAG 引用**")
        knowledge_rows = demo.get("knowledge", [])
        if not knowledge_rows:
            st.info("本次任务没有返回知识引用")
        else:
            from agentic.platform.knowledge import VersionedKnowledgeBase

            code_root = Path(__file__).resolve().parents[1]
            knowledge_path = code_root / "agentic" / "knowledge" / "rules_v1.jsonl"
            knowledge_base = VersionedKnowledgeBase(knowledge_path)
            for index, citation in enumerate(knowledge_rows, start=1):
                if not isinstance(citation, dict):
                    continue
                source_id = str(citation.get("source_id", ""))
                rule = knowledge_base.get_rule(source_id)
                if rule is None:
                    st.warning(f"引用 {source_id or index} 不在当前知识包中")
                    continue
                st.markdown(f"**{rule.title}** · `{rule.rule_id}`")
                st.caption(f"来源：{rule.source} · 版本：{rule.source_version} · 生效日期：{rule.effective_date} · 风险：{rule.risk_level} · 检索分数：{citation.get('score', '—')}")
                st.caption(f"适用范围：{'、'.join(rule.applicability) or '未限定'} · 前置条件：{'；'.join(rule.preconditions) or '无'}")
                locator = str(citation.get("locator") or f"rule:{rule.rule_id}:paragraph:1")
                st.caption(f"定位：{locator}")
                st.write(rule.text)
                st.download_button(
                    "下载引用原文",
                    data=rule.text,
                    file_name=f"{rule.rule_id}.txt",
                    mime="text/plain",
                    key=f"knowledge_source_download_{index}_{rule.rule_id}",
                    help="下载当前版本的知识原文片段，保留 source_id、版本和定位信息。",
                )
    with st.expander("模型、知识、提示词与 Ticket 版本"):
        review_payload = graph.get("result", {}).get("review", graph.get("review", {}))
        st.json({
            "visual_model": visual.get("model_profile"),
            "llm_gateway": demo.get("llm_gateway", {}),
            "knowledge_version": demo.get("knowledge_version"),
            "prompt_version": demo.get("prompt_version", "diagnosis-aggregator-v2"),
            "rsi_runtime": demo.get("rsi_runtime", {
                "status": "disabled",
                "version_id": "base",
                "policy_hash": "builtin",
                "prompt_variant": demo.get("prompt_version", "diagnosis-aggregator-v2"),
                "knowledge_query_terms": [],
            }),
            "review_guard_version": review_payload.get("implementation", "deterministic_review_guard_v1") if isinstance(review_payload, dict) else "deterministic_review_guard_v1",
            "ticket": demo.get("ticket", graph.get("ticket", {})),
        })
    with st.expander("RSI 离线演化与评测素材"):
        presentation = demo.get("presentation") if isinstance(demo.get("presentation"), dict) else {}
        st.json(presentation or {
            "status": "flow_validation",
            "rsi": "candidate → replay → gate → human approval → publish → rollback",
            "evaluation": "visual/process/route/agent/system cards are reproducible summaries",
            "claim_policy": "结构验证，不代表模型精度或真实过热度测量",
        })
        evaluation_path, evaluation_report = _load_latest_evaluation_report()
        if evaluation_report:
            evaluation_freeze = evaluation_report.get("evaluation_freeze")
            if isinstance(evaluation_freeze, dict):
                st.markdown("**评测集冻结与研究结论资格**")
                frozen = bool(evaluation_freeze.get("frozen"))
                claimable = bool(evaluation_freeze.get("claimable"))
                freeze_status, claim_status = st.columns(2)
                freeze_status.metric("评测集冻结状态", "已冻结" if frozen else "未冻结")
                claim_status.metric("研究结论可声明", "是" if claimable else "否")
                reason = evaluation_freeze.get("claimability_reason")
                if reason:
                    st.caption(str(reason))

                manifest_details = evaluation_freeze.get("manifest_details", {})
                if isinstance(manifest_details, dict) and manifest_details:
                    freeze_rows = []
                    for name, details in sorted(manifest_details.items()):
                        if not isinstance(details, dict):
                            details = {}
                        record_count = details.get("record_count", 0)
                        valid_count = details.get("valid_record_count", 0)
                        try:
                            invalid_count = max(int(record_count) - int(valid_count), 0)
                        except (TypeError, ValueError):
                            invalid_count = "—"
                        errors = details.get("errors", [])
                        if not isinstance(errors, list):
                            errors = [str(errors)] if errors else []
                        invalid_reasons = details.get("invalid_error_counts", {})
                        if not isinstance(invalid_reasons, dict):
                            invalid_reasons = {}
                        freeze_rows.append({
                            "必需清单": name,
                            "状态": details.get("status", "unknown"),
                            "记录数": record_count,
                            "有效记录": valid_count,
                            "无效记录": invalid_count,
                            "字段/审核拒绝原因": "；".join(f"{reason}: {count}" for reason, count in sorted(invalid_reasons.items())) or "无",
                            "校验错误": "；".join(str(error) for error in errors) or "无",
                        })
                    st.dataframe(freeze_rows, use_container_width=True, hide_index=True)
                elif evaluation_freeze.get("missing_manifests") or evaluation_freeze.get("invalid_manifests"):
                    st.caption(
                        "未通过清单："
                        + "、".join(evaluation_freeze.get("missing_manifests", []) + evaluation_freeze.get("invalid_manifests", []))
                    )

            quality_eval = evaluation_report.get("image_quality_gate", {})
            process_eval = evaluation_report.get("process", {}).get("independent_baseline_evaluation", {})
            attribution = evaluation_report.get("process", {}).get("feature_attribution", {})
            retrieval_eval = evaluation_report.get("retrieval", {})
            knowledge_eval = retrieval_eval.get("knowledge", {})
            citation_eval = knowledge_eval.get("citation_integrity", {})
            case_eval = retrieval_eval.get("cases", {})
            knowledge_pack_eval = evaluation_report.get("knowledge_pack_validation", {})
            jev_eval = evaluation_report.get("jev", {})
            route_flow_eval = evaluation_report.get("route_flow", {})
            multi_agent_eval = evaluation_report.get("multi_agent", {})
            worker_eval = evaluation_report.get("worker_execution", {})
            system_eval = evaluation_report.get("system", {})
            input_integrity = evaluation_report.get("input_integrity", {})
            task_latency = system_eval.get("end_to_end_latency", {})
            node_latency = system_eval.get("node_latency", {})
            runtime_health_claimable = bool(system_eval.get("runtime_scope", {}).get("claimable") and system_eval.get("event_count", 0))
            def _percent(value):
                return f"{value:.0%}" if isinstance(value, (int, float)) else "—"

            transitions = route_flow_eval.get("transitions", {})
            if not isinstance(transitions, dict):
                transitions = {}
            transition_summary = "、".join(
                f"{name} {count} 次" for name, count in sorted(transitions.items())
            ) if transitions else "无路由转移记录"

            rows = [
                {"评测项": "质量门控失败样本", "结果": f"{quality_eval.get('failure_detected_count', 0)}/{quality_eval.get('failure_sample_count', 0)}", "性质": "合成规则回归", "可声明": "否"},
                {"评测项": "知识检索查询", "结果": f"{knowledge_eval.get('query_count', 0)} 条 · {_percent(knowledge_eval.get('fixture_pass_rate'))}", "性质": "demo fixture", "可声明": "否"},
                {"评测项": "知识引用字段完整性", "结果": f"{citation_eval.get('valid_citation_count', 0)}/{citation_eval.get('citation_count', 0)} · {_percent(citation_eval.get('pass_rate'))}", "性质": "source/version/locator fixture 检查", "可声明": "否 · 不含语义蕴含"},
                {"评测项": "工艺知识包结构", "结果": f"{knowledge_pack_eval.get('valid_record_count', 0)}/{knowledge_pack_eval.get('record_count', 0)}", "性质": knowledge_pack_eval.get("status", "not_run"), "可声明": "否 · 内容需专家审核"},
                {"评测项": "案例检索查询", "结果": f"{case_eval.get('query_count', 0)} 条 · {_percent(case_eval.get('fixture_pass_rate'))}", "性质": "demo fixture", "可声明": "否"},
                {"评测项": "JEV 路由回归", "结果": f"{jev_eval.get('fixture_count', 0)} 条 · {_percent(jev_eval.get('fixture_route_match_rate'))}", "性质": "安全 fixture", "可声明": "否"},
                {"评测项": "JEV 四路 fixture 召回", "结果": " / ".join(_percent(jev_eval.get(key)) for key in ("simple_fixture_recall", "complex_fixture_full_recall", "retake_fixture_recall", "human_fixture_recall")), "性质": "simple / full / retake / human", "可声明": "否 · policy fixture"},
                {"评测项": "simple/full 路径流转", "结果": f"{route_flow_eval.get('run_count', 0)} 次 · 升级 {route_flow_eval.get('upgrade_count', 0)} · {_percent(route_flow_eval.get('upgrade_rate'))} · {transition_summary}", "性质": "运行轨迹审计", "可声明": "否 · 流程验证"},
                {"评测项": "Worker 策略执行", "结果": f"{worker_eval.get('worker_count', 0)} 个 · 重试 {worker_eval.get('retry_total', 0)} · 超时 {worker_eval.get('timeout_worker_count', 0)} · 策略覆盖 {_percent(worker_eval.get('policy_coverage_rate'))}", "性质": "AgentResult 执行元数据", "可声明": "否 · 流程完整性"},
                {"评测项": "评测输入完整性", "结果": input_integrity.get("status", "not_run"), "性质": "JSONL 解析与范围检查", "可声明": "是" if input_integrity.get("claimable") else "否"},
                {"评测项": "节点失败 / 超时 / 重试", "结果": " / ".join(_percent(system_eval.get(key)) for key in ("node_failure_rate", "timeout_rate", "retry_rate")), "性质": "runtime 节点事件率", "可声明": "是" if runtime_health_claimable else "否 · scope 未限定"},
                {"评测项": "运行 Fallback", "结果": _percent(system_eval.get("fallback_rate")), "性质": "runtime 任务比例", "可声明": "是" if runtime_health_claimable else "否 · scope 未限定"},
                {"评测项": "任务端到端耗时 P50/P95/P99", "结果": " / ".join("—" if task_latency.get(key) is None else f"{task_latency[key]:.0f} ms" for key in ("p50_ms", "p95_ms", "p99_ms")), "性质": "显式 run_id 后可评估", "可声明": "是" if task_latency.get("claimable") else "否 · 当前范围混合"},
                {"评测项": "节点耗时 P50/P95/P99", "结果": " / ".join("—" if node_latency.get(key) is None else f"{node_latency[key]:.0f} ms" for key in ("p50_ms", "p95_ms", "p99_ms")), "性质": "节点事件统计", "可声明": "是" if node_latency.get("claimable") else "否 · 当前范围混合"},
            ]
            rows.append({
                "评测项": "Full-loop Agent 交接",
                "结果": f"{multi_agent_eval.get('present_worker_slots', 0)}/{multi_agent_eval.get('expected_worker_slots', 0)} 个 Agent · 失败 {multi_agent_eval.get('failed_worker_slots', 0)} · 来源链问题 {multi_agent_eval.get('source_lineage_violations', 0)}",
                "性质": multi_agent_eval.get("status", "not_run"),
                "可声明": "否 · 流程完整性",
            })
            for model_name, metrics in process_eval.get("models", {}).items():
                roc_auc = metrics.get("roc_auc")
                pr_auc = metrics.get("pr_auc")
                rows.append({
                    "评测项": f"过程基线 {model_name}",
                    "结果": f"ROC-AUC {roc_auc:.3f} · PR-AUC {pr_auc:.3f}" if isinstance(roc_auc, (int, float)) and isinstance(pr_auc, (int, float)) else "—",
                    "性质": f"{process_eval.get('status', 'experimental_baseline')} · 对齐 {evaluation_report.get('process', {}).get('alignment_quality', 'unknown')}",
                    "可声明": "否 · 独立过程任务",
                })
            for result in attribution.get("test_comparison", []):
                roc_auc = result.get("roc_auc")
                pr_auc = result.get("pr_auc")
                rows.append({
                    "评测项": f"过程输入 {result.get('candidate', 'candidate')} ({result.get('feature_count', '—')} 维)",
                    "结果": f"ROC-AUC {roc_auc:.3f} · PR-AUC {pr_auc:.3f}" if isinstance(roc_auc, (int, float)) and isinstance(pr_auc, (int, float)) else "—",
                    "性质": "单次实验 · 测试集最终对照",
                    "可声明": "否 · 实验基线",
                })
            st.caption(f"评测来源：{evaluation_path.name} · runtime 范围：{system_eval.get('runtime_scope', {}).get('scope', 'unknown')}")
            st.dataframe(rows, use_container_width=True, hide_index=True)
            delivery_path, delivery = _load_latest_delivery_snapshot()
            if delivery:
                st.caption(
                    f"交付快照：{delivery_path.name} · 状态 {delivery.get('status', 'unknown')} · "
                    f"视频 run {delivery.get('finished_run_count', 0)} · JEV fixture {delivery.get('route_fixture_count', '—')}"
                )
                st.json({
                    "status": delivery.get("status"),
                    "selected_run_ids": delivery.get("selected_run_ids", []),
                    "hitl_actions": delivery.get("hitl_actions", {}),
                    "rsi_runtime": delivery.get("rsi_runtime", {}),
                    "errors": delivery.get("errors", []),
                    "claims": delivery.get("claims", {}),
                })
            if attribution.get("validation_selected_features"):
                st.caption("验证集归因候选（相关性分析）：" + "、".join(attribution["validation_selected_features"]))
        else:
            st.info("尚未生成评测汇总。")
    with st.expander("诊断与复核 JSON"):
        st.json({"diagnosis": graph.get("result", {}).get("diagnosis", graph.get("diagnosis", {})), "review": graph.get("result", {}).get("review", graph.get("review", {})), "evidence_bundle": graph.get("result", {}).get("evidence_bundle", [])})
    with st.container(border=True):
        st.markdown('<h4 class="ptitle">报告归档与运行审计</h4>', unsafe_allow_html=True)
        service = HumanReviewService(_modern_runtime_root())
        report = None
        if demo.get("schema_version") == "presentation-demo-1":
            review_status = (demo.get("human_review") or {}).get("status")
            if review_status in {"completed", "waiting_for_retake"}:
                report = _presentation_report(demo)
        else:
            try:
                report = service.report(demo["run_id"])
            except Exception:
                report = None
        if report:
            audit = report.get("audit") or report.get("human_review") or {}
            st.caption(f"归档状态：{report.get('status', '—')} · 人工动作：{audit.get('action', 'pending')} · 审核人：{audit.get('reviewer_id', '—')}")
            if demo.get("schema_version") == "presentation-demo-1":
                exports = _presentation_report_exports(demo, report)
                report_json, report_md, report_html, report_png = exports["json"], exports["markdown"], exports["html"], exports["png"]
            else:
                report_json = service.export_report(demo["run_id"], "json")
                report_md = service.export_report(demo["run_id"], "markdown")
                report_html = service.export_report(demo["run_id"], "html")
                report_png = None
                png_path = _modern_runtime_root() / "presentation_exports" / f"{demo['run_id']}.png"
                try:
                    service.export_report(demo["run_id"], "png", destination=png_path)
                    if png_path.exists():
                        report_png = png_path.read_bytes()
                except Exception:
                    report_png = None
            ex1, ex2, ex3, ex4 = st.columns(4)
            with ex1:
                st.download_button("下载 JSON 报告", report_json, file_name=f"{demo['run_id']}.json", mime="application/json", key="modern_report_json", use_container_width=True)
            with ex2:
                st.download_button("下载 Markdown 报告", report_md, file_name=f"{demo['run_id']}.md", mime="text/markdown", key="modern_report_md", use_container_width=True)
            with ex3:
                st.download_button("下载 HTML 报告", report_html, file_name=f"{demo['run_id']}.html", mime="text/html", key="modern_report_html", use_container_width=True)
            with ex4:
                if report_png:
                    st.download_button("下载 PNG 报告", report_png, file_name=f"{demo['run_id']}.png", mime="image/png", key="modern_report_png", use_container_width=True)
                else:
                    st.caption("PNG 报告暂不可用")
            st.json({"run_id": demo.get("run_id"), "thread_id": demo.get("thread_id"), "route": report.get("route"), "review": audit, "report_id": (ticket.get("report_id") if isinstance(ticket, dict) else None)})
        else:
            st.info("人工确认完成后可在此导出归档报告。")


def page_agent():
    """Multi-agent collaboration page with graph, evidence, and node outputs."""
    if run_modern_demo is not None:
        _render_modern_agent_page()
        return
    ss = st.session_state
    result = ss.get("agent_result")
    st.markdown('<h4 class="ptitle big">多智能体协作流程</h4>', unsafe_allow_html=True)
    st.caption("该页面调用本地 LangGraph 工作流；默认由一次 Qwen3-VL 诊断和一次快速规则复核组成。")

    if run_diagnosis_trace is None:
        st.warning("当前环境未安装多智能体可选依赖，原有视觉分析页面仍可正常使用。")
        st.code("cd D:\\py_daima\\mars-agent\n"
                "D:\\py_daima\\.venvs\\mars-agent\\Scripts\\python.exe -m pip install -r Code\\agentic\\requirements.txt",
                language="powershell")
        return

    ctl1, ctl2, ctl3 = st.columns([1.5, 1.5, 5.0])
    with ctl1:
        if st.button("▶ 运行协作", type="primary", key="run_agent_page"):
            run_agent_diagnosis()
            st.rerun()
    with ctl2:
        if st.button("清除流程", key="clear_agent_page"):
            ss.agent_result = None
            ss.agent_error = None
            ss.agent_trace = []
            ss.agent_workflow = None
            ss.agent_source_sig = None
            st.rerun()
    with ctl3:
        if ss.get("src_name"):
            st.markdown(f"当前输入：`{ss.src_name}`", unsafe_allow_html=False)
        else:
            st.markdown("当前输入：尚未载入")

    with st.container(border=True):
        st.markdown('<h4 class="ptitle">协作状态图</h4>', unsafe_allow_html=True)
        st.plotly_chart(agent_flow_chart(result, ss.get("agent_trace")), use_container_width=True,
                         config={"displayModeBar": False})
        if result:
            status = result.get("status", "unknown")
            label = {"evidence_review_passed": "证据复核通过",
                     "needs_human_review": "需要人工复核",
                     "insufficient_visual_evidence": "视觉证据不足"}.get(status, status)
            st.info(f"最终门控：{label}。本流程不会自动下发加料、调压等工艺控制指令。")
            if result.get("run_id"):
                st.caption(f"运行 ID：`{result['run_id']}`。人工审核后可用 `python -m agentic.review` 记录决策。")
        else:
            st.caption("运行后，这里会显示每个 Agent 的成功、失败或待复核状态。")

    trace = ss.get("agent_trace", [])
    completed_trace = [event for event in trace if event.get("status") in {"completed", "error"}]
    if completed_trace:
        trace_rows = []
        labels = {
            "perception_agent": "视觉感知 Agent",
            "knowledge_agent": "工艺知识 Agent",
            "diagnosis_agent": "多模态诊断 Agent",
            "review_agent": "证据复核 Agent",
            "finalize": "人工复核门控",
        }
        for event in completed_trace:
            summary = event.get("summary") or {}
            trace_rows.append({
                "节点": labels.get(event.get("node"), event.get("node")),
                "状态": "完成" if event.get("status") == "completed" else "错误",
                "耗时(ms)": event.get("elapsed_ms", "—"),
                "摘要": json.dumps(summary, ensure_ascii=False),
            })
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">本次运行轨迹</h4>', unsafe_allow_html=True)
            st.dataframe(trace_rows, width="stretch", hide_index=True, height=218)

    if ss.get("agent_error"):
        st.error(f"流程在节点执行时中断：{ss.agent_error}")

    if not result:
        return

    visual = result.get("visual", {})
    diagnosis = result.get("diagnosis", {})
    review = result.get("review", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("视觉分类", visual.get("class_label", "未识别"))
    c2.metric("分类置信度", f"{float(visual.get('confidence', 0.0)) * 100:.1f}%")
    c3.metric("检索来源", len(result.get("knowledge_sources", [])))
    c4.metric("复核状态", "通过" if review.get("approved") else "待复核")

    st.markdown('<h4 class="ptitle">节点输出与证据</h4>', unsafe_allow_html=True)
    with st.expander("1 · 视觉取证 Agent：YOLO ROI + F-GMVAE-T", expanded=True):
        st.json(visual)
    with st.expander("2 · 知识检索 Agent：本地 Markdown / TXT"):
        st.json(result.get("knowledge_sources", []))
    with st.expander("3 · 诊断 Agent：Qwen3-VL 结构化 JSON", expanded=True):
        st.json(diagnosis)
    with st.expander("4 · 证据复核 Agent：是否允许通过", expanded=True):
        st.json(review)
    with st.expander("LangGraph 工作流定义（可复制到 Mermaid 工具）"):
        spec = ss.get("agent_workflow") or (workflow_spec() if workflow_spec else {})
        st.code(
            "graph LR\n"
            + "\n".join(
                f"    {node['id']}[{node['label']}]"
                for node in spec.get("nodes", [])
            )
            + "\n"
            + "\n".join(
                f"    {edge['source']} --> {edge['target']}"
                for edge in spec.get("edges", [])
            ),
            language="mermaid",
        )


def page_perf():
    ss = st.session_state
    results = ss.results or []

    left, right = st.columns([6.0, 4.0])

    with left:
        left_video_panel()

    # ---------------- 右：上=3D 流程图，下=四张性能曲线 ----------------
    with right:
        with st.container(border=True):
            st.markdown('<h4 class="ptitle big">铝电解槽 3D 结构剖视</h4>', unsafe_allow_html=True)
            st.plotly_chart(cell_3d(height=290, active=bool(results)),
                            use_container_width=True, config={"displayModeBar": False})

        specs, data, figs = perf_charts(results)
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">性能指标实时分析</h4>', unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)
            for (title, _color, _i), fig, col in zip(specs, figs, [r1c1, r1c2, r2c1, r2c2]):
                with col:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if results:
                # 压成一行 4 格，省掉原来 4 行 kv 的高度
                st.markdown(
                    '<div class="resgrid" style="grid-template-columns:repeat(4,1fr);">'
                    + f'<div class="rescell"><div class="rk">耗时 min/max</div>'
                      f'<div class="rv sm">{min(data[0])} / {max(data[0])} ms</div></div>'
                      f'<div class="rescell"><div class="rk">吞吐 min/max</div>'
                      f'<div class="rv sm">{min(data[1]):.1f} / {max(data[1]):.1f} FPS</div></div>'
                      f'<div class="rescell"><div class="rk">目标数合计</div>'
                      f'<div class="rv sm">{sum(data[2])} 个</div></div>'
                      f'<div class="rescell"><div class="rk">置信度平均</div>'
                      f'<div class="rv sm">{sum(data[3])/len(data[3]):.1f}%</div></div>'
                    + '</div>', unsafe_allow_html=True)


# ============================================================
# 2) 过程诊断 —— 工况 + AI 诊断
# ============================================================
def page_diagnose():
    ss = st.session_state
    results, summary = ss.results, ss.summary
    last = results[-1] if results else None

    left, right = st.columns([5.9, 4.1])
    with left:
        left_video_panel()

        if ss.ai_report:
            st.markdown('<div class="panel"><h4>🤖 专家诊断报告</h4>'
                        f'<div style="max-height:23vh;overflow-y:auto;white-space:pre-wrap;'
                        f'font-size:13px;line-height:1.65;color:#dbe9ff;">{ss.ai_report}</div></div>',
                        unsafe_allow_html=True)

    with right:
        code = (last or {}).get("cls_result", "NoTarget")
        conf = (last or {}).get("cls_conf", 0.0)
        color = STATE_COLOR.get(code, DIM)
        trend = (last or {}).get("trend", "")
        st.markdown(
            f'<div class="state-card" style="border-color:{color}66;box-shadow:inset 0 0 30px {color}22;">'
            f'<div class="state-title">实时工况判定</div>'
            f'<div class="state-value" style="color:{color};text-shadow:0 0 16px {color}77;">'
            f'{STATE_TEXT.get(code, code)}</div>'
            f'<div class="state-conf">置信度 <span class="num">{conf*100:.1f}%</span>'
            + (f' &nbsp;·&nbsp; <b style="color:#ffb74d">{trend}</b>' if trend else '')
            + '</div></div>', unsafe_allow_html=True)

        if last and last.get("alarm"):
            st.markdown('<div class="panel accent-hot">'
                        '<h4 style="color:#ff9a90;">⚠ 持续异常报警</h4>'
                        f'<div class="muted" style="color:#f0b8b0;">{CLASS_CN.get(code,"")} 持续 '
                        f'<span class="num">{last.get("alarm_secs",0)}</span> 秒且平均置信度 &gt; 90%</div>'
                        '</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>三分类置信度</h4>'
                    + prob_bars(last["cls_probs"] if last else [0, 0, 0]) + '</div>',
                    unsafe_allow_html=True)

        if last:
            st.markdown('<div class="panel"><h4>实测特征（ROI 内计算）</h4>' + kv_html([
                ("平均灰度 μ", last["gray_mean"]),
                ("对比度 σ", last["contrast"]),
                ("饱和度 S", last["saturation"]),
                ("角点数", last["corners"]),
                ("结壳速率 V_crust", last["crust_speed"]),
                ("角点速率 V_node", last["node_speed"]),
                ("ROI 尺寸", str(last.get("roi_size") or "—")),
            ]) + '</div>', unsafe_allow_html=True)

        if summary:
            counts = summary.get("counts", {})
            st.markdown('<div class="panel"><h4>本次采样统计</h4>' + kv_html([
                ("有效判定帧", f'{summary.get("classified",0)} / {summary.get("frames",0)}'),
                ("偏高 / 正常 / 偏低",
                 f'{counts.get("H",0)} / {counts.get("N",0)} / {counts.get("C",0)}'),
                ("是否触发报警", '<span class="badge badge-warn">是</span>' if summary.get("alarm")
                 else '<span class="badge badge-ok">否</span>'),
            ]) + '</div>', unsafe_allow_html=True)

        if results and len(results) > 1:
            with st.container(border=True):
                st.markdown('<h4 class="ptitle">平均灰度 μ 趋势</h4>', unsafe_allow_html=True)
                st.plotly_chart(chart_series(results, "gray_mean"), use_container_width=True,
                                config={"displayModeBar": False})
            with st.container(border=True):
                st.markdown('<h4 class="ptitle">动态速率趋势</h4>', unsafe_allow_html=True)
                st.plotly_chart(chart_multi(results, [
                    ("V_crust", "crust_speed", "#e08a5a", "solid"),
                    ("V_node", "node_speed", CYAN, "dot")]), use_container_width=True,
                    config={"displayModeBar": False})


# ============================================================
# 3) 加药分析 —— AlF3 规则推算
# ============================================================
def page_dosing():
    ss = st.session_state
    results, summary = ss.results, ss.summary

    if not results:
        no_data_hint()
        st.markdown('<div class="panel"><h4>规则说明</h4>'
                    '<div class="muted">过热度偏高 → 增加 AlF3 加料以降分子比、降温；'
                    '过热度偏低 → 减少/暂停加料。幅度由稳定判定置信度与异常帧占比共同决定。'
                    '该结论为规则推算，非实测加料量。</div></div>', unsafe_allow_html=True)
        return

    d = dosing_rules.analyze_dosing(results, summary)
    if not d.get("ready"):
        no_data_hint()
        return

    stat_cards([
        ("建议动作", d["action"].split(" ")[0], d["color"]),
        ("建议幅度", d["delta_text"], d["color"]),
        ("推算强度", f'{d["score"]:+.0f}', d["color"]),
        ("风险等级", d["severity"], {"高": "#c03030", "中": "#ffb020", "低": "#3ddc84"}[d["severity"]]),
    ])

    left, right = st.columns([5.3, 4.7])

    with left:
        left_video_panel(scale=0.5, with_results=False)   # 左侧固定视频（跨页面保留）
        st.markdown(f'<div class="panel" style="border-color:{d["color"]}66;">'
                    f'<h4 style="color:{d["color"]};">加料建议</h4>'
                    f'<div style="font-size:21px;font-weight:900;color:{d["color"]};'
                    f'text-shadow:0 0 14px {d["color"]}66;margin:4px 0 6px 0;">{d["action"]}</div>'
                    f'<div class="muted">{d["reason"]}</div></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<h4 class="ptitle">建议强度随采样帧演变</h4>', unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(
                y=d["series"], mode="lines+markers",
                line=dict(color=CYAN, width=2), marker=dict(size=4, color=CYAN),
                fill="tozeroy", fillcolor="rgba(0,234,255,.08)"))
            fig.add_hline(y=25, line=dict(color="#ffb020", width=1, dash="dot"))
            fig.add_hline(y=-25, line=dict(color="#4d8dff", width=1, dash="dot"))
            st.plotly_chart(_layout(fig, 165), use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown('<div class="muted">正值=需增加加料，负值=需减少；虚线为 ±25 的维持区间边界</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>判定依据（全部为实测/推算值）</h4>'
                    + "".join(f'<div class="kv"><span class="k">·</span>'
                              f'<span class="v" style="font-weight:500;text-align:right;">{b}</span></div>'
                              for b in d["basis"])
                    + '</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><h4>操作建议</h4>'
                    + "".join(f'<div class="kv"><span class="k">▸</span>'
                              f'<span class="v" style="font-weight:500;text-align:right;">{a}</span></div>'
                              for a in d["advice"])
                    + '</div>', unsafe_allow_html=True)

        with st.container(border=True):
            frac = d["frac"]
            st.markdown('<h4 class="ptitle">本次帧状态分布</h4>'
                        + prob_bars([frac["C"], frac["H"], frac["N"]]), unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>参考区间（行业常规，非实测）</h4>'
                    + kv_html([(n, f'<span class="num">{v}</span> <span class="muted">{s}</span>')
                               for n, v, s in d["reference"]])
                    + '</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="panel" style="border-color:#7a5a10;">'
                    f'<h4 style="color:#ffb020;">⚠ 重要声明</h4>'
                    f'<div class="muted" style="line-height:1.6;">{d["disclaimer"]}</div></div>',
                    unsafe_allow_html=True)


# ============================================================
# 4) 历史管理
# ============================================================
def _render_modern_review_history():
    st.divider()
    st.subheader("多智能体审核记录")
    if HumanReviewService is None:
        st.info("多智能体审核存储当前不可用。")
        return

    service = HumanReviewService(_modern_runtime_root())
    events = service.history(limit=1000)
    run_filter, route_filter, status_filter, model_filter, date_filter = st.columns([2.0, 1.1, 1.3, 1.3, 1.2])
    run_query = run_filter.text_input("run_id", key="modern_history_run_filter").strip().lower()
    routes = sorted({str(row.get("route", "")) for row in events if row.get("route")})
    route = route_filter.selectbox("诊断路由", ["全部", *routes], key="modern_history_route_filter")
    statuses = sorted({str(row.get("review_status", row.get("status", ""))) for row in events if row.get("review_status", row.get("status"))})
    review_status = status_filter.selectbox("审核结果", ["全部", *statuses], key="modern_history_status_filter")
    model_profiles = sorted({str(row.get("model_profile", "")) for row in events if row.get("model_profile")})
    model_profile = model_filter.selectbox("模型 Profile", ["全部", *model_profiles], key="modern_history_model_filter")
    date_query = date_filter.text_input("日期", key="modern_history_date_filter", placeholder="YYYY-MM-DD").strip()

    visible = [row for row in events
               if (not run_query or run_query in str(row.get("run_id", "")).lower())
               and (route == "全部" or row.get("route") == route)
               and (review_status == "全部" or row.get("review_status", row.get("status")) == review_status)
               and (model_profile == "全部" or row.get("model_profile") == model_profile)
               and (not date_query or str(row.get("decided_at", "")).startswith(date_query))]
    st.caption(f"{len(visible)} 条审核记录")
    if not visible:
        st.info("当前筛选条件下没有多智能体审核记录。")
        return

    display_rows = [{
        "时间": row.get("decided_at", ""),
        "run_id": row.get("run_id", ""),
        "路由": row.get("route", ""),
        "风险": row.get("risk_level", "unknown"),
        "审核动作": row.get("action", ""),
        "审核结果": row.get("review_status", row.get("status", "")),
        "审核人": row.get("reviewer_id", ""),
        "模型": row.get("model_profile", ""),
    } for row in reversed(visible)]
    st.dataframe(display_rows, width="stretch", hide_index=True)

    selected = st.selectbox(
        "查看诊断报告",
        list(range(len(visible))),
        key="modern_history_report_pick",
        format_func=lambda index: f"{visible[index].get('run_id', '')} · {visible[index].get('action', '')} · {visible[index].get('decided_at', '')}",
    )
    run_id = str(visible[selected].get("run_id", ""))
    try:
        report = service.report(run_id)
        with st.expander("报告与审核证据", expanded=False):
            st.json(report)
            json_report = service.export_report(run_id, "json")
            markdown_report = service.export_report(run_id, "markdown")
            html_report = service.export_report(run_id, "html")
            download_json, download_markdown, download_html = st.columns(3)
            download_json.download_button("JSON", json_report, file_name=f"{run_id}.json", mime="application/json", key="modern_history_report_json")
            download_markdown.download_button("Markdown", markdown_report, file_name=f"{run_id}.md", mime="text/markdown", key="modern_history_report_md")
            download_html.download_button("HTML", html_report, file_name=f"{run_id}.html", mime="text/html", key="modern_history_report_html")
    except Exception as exc:
        st.warning(f"审核记录存在，但关联报告暂不可读取：{exc}")


def _render_history_page_contents():
    records = history_store.load_records()
    s = history_store.stats(records)

    with st.container(key="historykpis"):
        stat_cards([
            ("总记录", s["total"], CYAN),
            ("今日新增", s["today"], "#8ba4ff"),
            ("报警次数", s["alarms"], "#ff9a90"),
            ("含AI报告", s["with_report"], "#a58bff"),
            ("最近一次", str(s["last_time"])[-8:] if s["total"] else "—", "#ffb020"),
        ])

    left, right = st.columns([5.6, 4.4])

    with left:
        f1, f2, f3, f4 = st.columns([2, 2.2, 1.2, 1.2])
        with f1:
            sel = st.selectbox("状态筛选", ["全部", "偏高", "正常", "偏低", "报警"], key="hist_filter")
        with f2:
            kw = st.text_input("搜索槽号/来源", "", key="hist_kw", placeholder="如 9306")
        with f3:
            st.download_button("⬇ CSV", data=history_store.to_csv(records).encode("utf-8-sig"),
                               file_name=f"fireeye_history_{datetime.now():%Y%m%d_%H%M}.csv",
                               mime="text/csv", key="hist_dl", use_container_width=True)
        with f4:
            if st.button("🗑 清空", key="hist_clear", use_container_width=True):
                history_store.clear_records()
                st.rerun()

        view = records
        if sel == "偏高":
            view = [r for r in view if r.get("result") == "H"]
        elif sel == "正常":
            view = [r for r in view if r.get("result") == "N"]
        elif sel == "偏低":
            view = [r for r in view if r.get("result") == "C"]
        elif sel == "报警":
            view = [r for r in view if r.get("alarm")]
        if kw.strip():
            k = kw.strip().lower()
            view = [r for r in view if k in str(r.get("cell", "")).lower()
                    or k in str(r.get("source", "")).lower()]

        df = history_store.to_dataframe(view)
        if df.empty:
            st.markdown('<div class="panel"><div class="muted">没有符合条件的记录。'
                        '到「过程诊断」上传图片/视频并点「启动分析」即可产生记录。</div></div>',
                        unsafe_allow_html=True)
        else:
            with st.container(key="historytable"):
                st.dataframe(df, width="stretch", height=300, hide_index=True)
            cards = []
            for row in df.to_dict(orient="records"):
                time_label = html.escape(str(row.get("时间", "—") or "—"))
                status = html.escape(str(row.get("状态", row.get("代码", "—")) or "—"))
                confidence = row.get("置信度")
                confidence_text = f"{float(confidence or 0) * 100:.1f}%"
                fields = (
                    ("槽号", row.get("槽号", "—")),
                    ("来源", row.get("来源", "—")),
                    ("置信度", confidence_text),
                    ("灰度 / 对比度", f'{row.get("平均灰度", "—")} / {row.get("对比度", "—")}'),
                    ("饱和度", row.get("饱和度", "—")),
                    ("报警", row.get("报警", "否")),
                )
                field_parts = []
                for label, value in fields:
                    display_value = "—" if value is None or value == "" else value
                    field_parts.append(
                        f"<div><span>{html.escape(label)}</span><br>"
                        f"<b>{html.escape(str(display_value))}</b></div>"
                    )
                cards.append(
                    "<article class='history-record-card'>"
                    f"<div class='history-record-head'><strong>{time_label}</strong>"
                    f"<span class='history-record-status'>{status}</span></div>"
                    f"<div class='history-record-fields'>{''.join(field_parts)}</div></article>"
                )
            st.markdown("<div class='history-mobile-list'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

    with right:
        if not records:
            no_data_hint("暂无历史记录。")
        else:
            with st.container(border=True):
                st.markdown('<h4 class="ptitle">状态分布</h4>', unsafe_allow_html=True)
                st.plotly_chart(chart_donut(s["counts"]), use_container_width=True,
                                config={"displayModeBar": False})

            with st.container(border=True):
                st.markdown('<h4 class="ptitle">记录详情</h4>', unsafe_allow_html=True)
                if view:
                    idx = st.selectbox("选择记录", list(range(len(view))), key="hist_pick",
                                       format_func=lambda i: f'{view[i].get("time","")} · '
                                                             f'{view[i].get("cell","")} · '
                                                             f'{view[i].get("result_cn") or view[i].get("result","")}')
                    r = view[idx]
                    if r.get("thumb"):
                        st.markdown(f'<img src="{r["thumb"]}" style="width:100%;border-radius:2px;'
                                    f'border:1px solid #1b2a6b;"/>', unsafe_allow_html=True)
                    st.markdown(kv_html([
                        ("时间", r.get("time", "")),
                        ("来源", f'<span style="font-size:11px;">{r.get("source","")}</span>'),
                        ("槽号", r.get("cell", "")),
                        ("状态", f'{r.get("result_cn","")}（{r.get("result","")}）'),
                        ("置信度", f'{float(r.get("conf") or 0)*100:.1f}%'),
                        ("μ / σ / S", f'{r.get("gray_mean")} / {r.get("contrast")} / {r.get("saturation")}'),
                        ("V_crust / V_node", f'{r.get("crust_speed")} / {r.get("node_speed")}'),
                        ("采样帧", r.get("frames")),
                        ("推算加药", (r.get("dosing") or {}).get("action", "—")),
                    ]), unsafe_allow_html=True)
                    if r.get("ai_report"):
                        st.markdown('<div class="muted" style="margin-top:6px;">AI 报告摘录</div>'
                                    f'<div style="max-height:18vh;overflow-y:auto;font-size:12px;'
                                    f'line-height:1.55;color:#d8c9ff;white-space:pre-wrap;">'
                                    f'{r["ai_report"][:1200]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="muted">左侧筛选后无可选记录</div>', unsafe_allow_html=True)

    _render_modern_review_history()


def page_history():
    with st.container(key="historypage"):
        _render_history_page_contents()


# ============================================================
# 5) 系统设置
# ============================================================
def page_settings():
    ss = st.session_state
    perf = ss.analyzer.perf()
    stats = history_store.stats()
    avg_ms = perf["avg_infer_ms"] or 0.0
    fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

    # ---- 性能总览（原先在「性能监控」页，按需求整合到这里）----
    with st.container(border=True):
        st.markdown('<h4 class="ptitle">性能指标总览</h4>', unsafe_allow_html=True)
        stat_cards([
            ("平均单帧耗时", f"{avg_ms} ms", CYAN),
            ("等效吞吐", f"{fps:.1f} FPS" if fps else "—", "#3ddc84" if fps else DIM),
            ("累计分析帧", f"{perf['frames']}", "#8ba4ff"),
            ("推理设备", f"{str(perf['device']).upper()}", "#ffb020"),
            ("历史记录", f"{stats['total']}", "#a58bff"),
        ])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="panel"><h4>大模型后端</h4>'
                    f'<div class="num" style="font-size:12px;color:#bfe6ff;">{llm_backend_summary()}</div>'
                    '<div class="muted" style="margin-top:7px;">API Key 写在项目根目录的 '
                    '<span class="num">.env</span>（DASHSCOPE_API_KEY），'
                    '或改 <span class="num">config/llm_api.json</span>。</div></div>',
                    unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<h4 class="ptitle">分析参数</h4>', unsafe_allow_html=True)
            st.session_state.sample_frames = int(st.slider(
                "视频采样帧数（越多趋势越细，耗时线性增加）", 6, 60,
                int(st.session_state.sample_frames), step=2, key="set_frames"))

        st.markdown('<div class="panel"><h4>报警判定规则</h4>'
                    '<div class="muted" style="line-height:1.65;">三分类为偏高/偏低且持续 &gt; 10 秒、'
                    '其间平均置信度 &gt; 90% 时触发报警；分类采用 5 帧滑动窗口平滑 + '
                    '连续 5 次同类别才切换的迟滞策略，抑制抖动。</div></div>', unsafe_allow_html=True)

    with c2:
        from algo_web_bridge import RESNET_WEIGHT, YOLO_WEIGHT
        rows = []
        for name, path in (("YOLO best.pt", YOLO_WEIGHT), ("DC-MMGAT .pth", RESNET_WEIGHT)):
            exists = os.path.isfile(path)
            badge = ('<span class="badge badge-ok">就位</span>' if exists
                     else '<span class="badge badge-warn">缺失</span>')
            size = f"{os.path.getsize(path)/1048576:.1f} MB" if exists else "—"
            rows.append((name, f'{badge} <span class="num">{size}</span>'))
        st.markdown('<div class="panel"><h4>模型权重</h4>' + kv_html(rows) +
                    f'<div class="muted" style="margin-top:6px;">推理设备 '
                    f'<span class="num">{get_device()}</span>，权重全部就位：'
                    f'{"是" if models_ready() else "否"}</div></div>', unsafe_allow_html=True)

        s = history_store.stats()
        st.markdown('<div class="panel"><h4>历史数据</h4>'
                    + kv_html([("存储位置", f'<span style="font-size:11px;">{history_store.HISTORY_PATH}</span>'),
                               ("记录数 / 上限", f'{s["total"]} / {history_store.MAX_RECORDS}')])
                    + '</div>', unsafe_allow_html=True)
        if st.button("🗑 清空全部历史", key="set_clear_hist", use_container_width=True):
            history_store.clear_records()
            st.success("已清空")


# ============================================================
# 顶部标题 + 顶部导航
# ============================================================
def top_header():
    c1, c2 = st.columns([8.0, 3.0])
    with c1:
        st.markdown(
            '<div class="app-title-wrap">'
            '<div class="app-title-bar"></div>'
            '<div class="app-title">基于<span class="accent">YOLO</span>火眼识别与'
            '多模态大模型<span class="accent">铝电解诊断系统</span></div>'
            '</div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="hdr-right">{datetime.now():%Y-%m-%d %H:%M:%S}</div>',
                    unsafe_allow_html=True)


def top_nav():
    """顶部导航：下划线标签（tab 风格），选中项底部亮条 + 白字。"""
    with st.container(key="tabbar"):
        cols = st.columns(len(PAGES))
        for (key, label), col in zip(PAGES, cols):
            with col:
                active = st.session_state.page == key
                if st.button(label, key=f"nav_{key}",
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    navigate(key)


# ============================================================
# 主入口
# ============================================================
ANALYSIS_PAGES = ("perf", "diagnose", "dosing", "agent")


def main():
    top_header()
    top_nav()

    page = st.session_state.page
    # 控制条现在**所有页面都渲染**：
    # ① 上传组件的状态不会被 Streamlit 清理，载入的视频跨页面一直都在；
    # ② 任何页面都能看到当前载入的源，也能随时换源。
    control_bar()

    # 整页半透明外框：用 st.container(key=...) 生成 .st-key-pageframe 供 CSS 定尺寸
    with st.container(key="pageframe"):
        if page == "perf":
            page_perf()
        elif page == "diagnose":
            page_diagnose()
        elif page == "dosing":
            page_dosing()
        elif page == "agent":
            page_agent()
        elif page == "history":
            page_history()
        else:
            page_settings()


main()
