# Code 目录说明

现代主线是 `agentic/`：它实现铝电解火眼多智能体诊断、JEV 路由、LangGraph fan-out/fan-in、Process State Agent、RAG、HITL、RSI 和评测。启动与发布说明请先阅读仓库根目录 [`README.md`](../README.md) 以及 [`docs/发布整理说明.md`](docs/发布整理说明.md)。

## 运行

```powershell
$py = 'D:\py_daima\.venvs\mars-agent\Scripts\python.exe'
& $py -m pytest agentic/tests -q
streamlit run web_interface/streamlit_app.py --server.port 8503
```

核心入口：

- `agentic/scripts/demo_run.py`：图片/视频现代诊断演示；
- `agentic/scripts/route_smoke.py`：四路 JEV fixture 和 LangGraph 回放；
- `agentic/evaluation/run_all.py`：质量、过程、路由、协作、引用、冻结和系统评测汇总；
- `web_interface/streamlit_app.py`：现代工作台与人工审核页面。

## 兼容模块

`web_interface/algo_fireeye.py`、`qt_interface/` 和 `model/` 保留原始 YOLO/分类系统入口，用于兼容和基线对照；它们不是当前多智能体主链路。现代入口不会把旧版单帧结果写成多帧 F-GMVAE-T 效果。

## 数据、模型和声明边界

原始时序数据、视频、图片帧和运行缓存不进仓库，见 [`data/README.md`](data/README.md)。基础权重仍在 `weights/`；新增 checkpoint 不应提交。当前评测中的 fixture、结构验证和流程轨迹不能替代专家标注下的视觉/路由效果或真实过热度测量。
