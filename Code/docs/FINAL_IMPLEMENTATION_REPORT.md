# 铝电解火眼多智能体最终实现审计

日期：2026-11-08

本版本已经形成可本地运行、可回放、可评测、可人工审核的多智能体工程闭环。主入口为 `python -m agentic.run --input <image-or-video>`，默认使用现代多帧/JEV/HITL 流程，历史单帧入口使用 `--mode legacy`。

## 已实现

| 模块 | 当前证据 |
|---|---|
| 协议与 Gateway | `G0_REPORT.md`；77 个测试通过；fake、Ollama、Qwen API 和人工 fallback profile |
| 媒体与质量门控 | 图片、视频、帧目录标准化；OpenCV 质量评分；`pass/retake/human`；质量失败时不会调用诊断模型 |
| 多帧视觉 | YOLO 逐帧 ROI、轨迹和缺失掩码；Conv/GRU/TCN/Transformer；本地 F-GMVAE-T 前向、损失、checkpoint replay 和训练入口 |
| Process State Agent | 14 个源变量、30 步历史窗口、2 分钟采样、日期切分、变化点和过程事件风险；视频映射缺失时明确 `alignment_quality=unknown` |
| JEV 编排 | 四路 `simple_loop/full_loop/retake/human`；规则、Laya adapter、HTTP adapter、校准接口；5 个冻结 fixture 全部通过 |
| 多 Agent 协作 | LangGraph `Send -> worker -> reducer fan-in -> aggregate -> review`；5 个并行 worker；失败节点保留结构化状态 |
| 知识与案例 | 版本化铝电解规则包、引用 ID、适用条件、空检索；案例 `candidate -> eval_only -> approved_special_case -> active -> deprecated` |
| HITL | LangGraph `interrupt/resume` 原生入口、SQLite/JSON checkpoint、重启恢复、四种人工动作、幂等审计和报告导出 |
| 评测 | 冻结 visual/process/route/knowledge/hitl 清单；JEV、证据、延迟、系统健康度和消融矩阵入口 |
| RSI | 候选 diff、谱系、冻结评估、安全/质量/效率 gate、人工批准、发布和回滚；禁止在线修改权重、规则和人工结论 |
| 汇报资产 | 真实视频运行 JSON、ROI contact sheet、尺寸已校验的 sidecar、25 页 PPT manifest；15/15 资产校验通过 |

## 真实运行证据

- `正常.mp4`、`偏高.mp4` 均完成解码、质量门控、多帧 ROI、独立 Process State、JEV 和待人工审核 checkpoint。
- 复杂路径 trace 包含 `full_loop_dispatch`、5 个 `full_loop_worker:*`、`full_loop_fanout`（`reducer_operator_add`）、诊断聚合、证据复核和 finalize。
- 当前 runtime 目录保存 `events.jsonl`、`runs.jsonl`、HITL 审计、评测 JSON/Markdown/CSV/PNG 和演示结果。

## 仍需外部条件的验收

以下内容已经有代码、数据协议和可复现实验入口，但不能在当前环境伪造为最终论文指标：

1. 视频没有专家级帧/窗口真值，因此视觉 Macro-F1、ECE、Brier 和完整消融保持 `structure_validation_only`。
2. F-GMVAE-T 目前完成 CPU smoke、前向、损失和训练管线；真实训练需要 NVIDIA GPU 与专家标签。
3. 过程数据与视频没有可靠时间/槽位映射，过程分支只作为独立状态先验，保持 `process_missing=video_mapping` 和 `alignment_quality=unknown`。
4. 本机 Ollama 服务和 `qwen3-vl:latest` 已完成一次短提示结构化视觉 smoke，返回草案并进入人工授权；较长聚合提示仍可能超时或返回空 content，网关会保留失败状态。未提供 Qwen API key，云端调用记录为 configuration error。
5. 当前系统输出视觉工况与风险辅助研判，不输出无温度、成分和初晶温度依据的数值过热度，也不生成生产控制指令。

## 一键回归

```powershell
cd D:\py_daima\mars-agent\Code
$py = 'D:\py_daima\.venvs\mars-agent\Scripts\python.exe'
& $py -m pytest agentic/tests -q
& $py -m agentic.scripts.inventory --output runtime_data/inventory.final.json
& $py -m agentic.scripts.smoke_test --simulate-no-model --output runtime_data/smoke.final.json
& $py -m agentic.scripts.freeze_evaluation --output runtime_data/frozen_evaluation.final.json
& $py -m agentic.evaluation.run_all
& $py -m agentic.scripts.demo_run --input data/video/正常.mp4 --output runtime_data/demo_normal.final.json
& $py -m agentic.scripts.demo_run --input data/video/偏高.mp4 --output runtime_data/demo_high.final.json
& $py -m agentic.scripts.validate_assets --directory report_assets/multi_agent_v2 --output runtime_data/assets.final.json
& $py -m agentic.scripts.rsi_defect_demo --output-dir runtime_data/rsi
& $py -m agentic.scripts.provider_smoke --output runtime_data/provider_smoke.json
```

当前最新回归：`pytest` 78 passed、JEV fixture 5/5、资产 15/15；视觉训练 checkpoint 的图像级测试 Macro-F1 为 0.7097，但因来源元数据缺失和视频无专家标签，仍标记为实验基线。
