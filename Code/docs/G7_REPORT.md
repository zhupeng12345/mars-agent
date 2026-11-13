# G7 阶段报告：端到端演示与交付资产

- 状态：通过流程验收
- 真实演示：两个视频完成媒体清单、质量门控、多帧 ROI、14 维 Process State、JEV 路由、full-loop 并行和 `waiting_review` 持久化。
- 资产：`report_assets/multi_agent_v2` 含 15 张真实 contact sheet 及 JSON sidecar；资产校验 15/15 通过；`ppt_asset_manifest.json` 覆盖 25 页。
- 最终回归：当前工作区最终回归 59 个 pytest 通过（另有 1 个 Transformer warning 和 7 个 subtests），compileall 通过，route fixture 5/5，评测总入口通过。
- 交付边界：当前 CPU 模型为 smoke/流程验证；没有专家视频标签时不宣称视觉 Macro-F1，也不宣称真实过热度测量精度；云端 Qwen API/Ollama 是否可用取决于本机服务和密钥。
- 入口：`python -m agentic.run --input <image-or-video>` 默认走现代多帧/JEV/HITL 流程，`--mode legacy` 保留旧单帧入口；Streamlit 已在 `http://localhost:8502` 启动并通过健康检查。

## 2026-11-08 当前复核补充

- 最新真实视频运行：`runtime_data/demo_trained_current.json`，使用 10 轮图像 checkpoint 的逐帧概率池化，8 帧 ROI，5 个并行 worker，聚合模型节点和 `waiting_review` 审核态。
- 最新全量测试为 `78 passed`；JEV fixture `5/5`；PPT sidecar 尺寸已从实际图像字节校验，`runtime_data/assets.final.latest.json` 为 `15/15`。
- 真实 provider 结果按 `runtime_data/provider_smoke.json` 记录：本地 Ollama `qwen3-vl:latest` 返回结构化草案并进入人工授权，云端 Qwen 因缺少 API key 记录为 configuration error；两者都保留网关元数据。
- `runtime_data/demo_ollama_local_compact.json` 是真实 full_loop 聚合截图/数据源：视觉 source ID 被模型引用，复核 `pass`，流程状态 `waiting_review`；模型生成的建议仍按人工确认边界呈现。
