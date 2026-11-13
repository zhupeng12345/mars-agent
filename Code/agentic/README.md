# 铝电解火眼多智能体诊断系统

这是一个本地可回放、可评测、人工可审核的视觉-知识多智能体工作流：

```text
图片/短视频 → 质量门控 → 多帧 YOLO ROI + F-GMVAE-T
14维过程数据 → 独立 Process State Agent
视觉/过程/知识/案例证据 → Laya-JEV 诊断路径
simple_loop 或 full_loop → 证据复核 → 人工 interrupt/resume → 报告与案例候选
```

视觉状态标签 C/N/H 是火眼视觉证据，过程状态是独立先验；在没有可靠视频时间戳或槽位映射时，系统会显式记录 `alignment_quality=unknown`，不会把过程变量伪装成同步温度或过热度测量。系统不输出缺少温度、成分和初晶温度实测依据的数值过热度，也不自动下发生产控制指令。

## 主要模块

- `models/`：ROI 序列、GRU/TCN/Transformer、F-GMVAE-T、ConvNeXt-Tiny/轻量 ViT 可选空间编码器。
- `agents/`：图像质量、视觉取证、独立 Process State 和确定性证据复核。
- `platform/jev/`：JEV 四路路由、Laya/HTTP 可选适配、校准和安全规则。
- `orchestration/diagnosis_graph.py`：LangGraph `Send` fan-out/fan-in、有限升级轮次和人工暂停恢复。
- `platform/tickets.py`：Source/Thread/Ticket/Assignment/Worker 可审计交接账本。
- `rsi/`：离线候选、冻结评测、质量/安全/效率 gate、人工发布与回滚。

## 本地运行

统一使用：

```powershell
cd D:\py_daima\mars-agent\Code
$py = 'D:\py_daima\.venvs\mars-agent\Scripts\python.exe'
& $py -m pytest agentic/tests -q
& $py -m agentic.scripts.demo_run --input data\video\正常.mp4 --llm-profile fake_json
```

`fake_json` 用于无网络回归；`ollama_local` 调用本地 `qwen3-vl:latest`；`qwen_api` 使用环境变量配置的 OpenAI 兼容 Qwen API；`fallback_human` 保留人工路径。运行记录由 `MARS_AGENT_RUNTIME_DIR` 指定，默认写入 `Code/runtime_data/`。

现代入口默认使用 `--jev-backend deterministic_rule`，也可显式选择 `--jev-backend laya_adapter`。Laya adapter 指向 `typed-decisions`，只在确定性门控选择 `full_loop` 时提出路径建议；首次实际推理会加载模型权重。

网页工作台：

```powershell
& $py -m streamlit run web_interface\streamlit_app.py --server.port 8502
```

打开“智能体协作”页可查看输入视频、ROI contact sheet、JEV 原因、并行节点、Process State、RAG/案例引用、Ticket、模型/知识/提示词版本和四种人工动作。

## 结果边界

当前仓库已完成工程流程和结构验证。视频专家窗口标签、GPU 时序训练、视频-过程可靠对齐以及云端 Qwen API key 仍是外部实验条件；相关结果会标记为 `structure_validation_only` 或 `experimental`，不写成泛化精度或真实过热度测量结论。

更多阶段验收、第三方复用边界和 PPT 资产来源见 `Code/docs/铝电解多智能体完整实现执行计划.md`、`Code/docs/FINAL_IMPLEMENTATION_REPORT.md` 和 `Code/docs/third_party_manifest.md`。
