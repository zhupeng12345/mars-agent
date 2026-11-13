# G0 阶段报告：协议与模型网关

## 阶段结论

P0.2 和 P0.3 初始验收快照通过，G0 的本地可复现部分通过。该报告形成时，主流程尚未迁移；当前主诊断图已接入统一 Gateway、`DiagnosisDraftPayload` 和路由后的多 Agent 聚合，阶段变更见执行计划及后续 implementation log。

## 协议

已实现 `InspectionTask`、`MediaManifest`、`EvidenceRef`、`AgentResult`、`VisualEvidence`、`ProcessStateSnapshot`、`InspectionFeatures`、`JevRouteDecision`、`DiagnosisDraft`、`ReviewDecision`、`TraceEvent`、`DiagnosisTicket` 和 `AgentAssignment`。

每类协议支持 `to_dict/from_dict/validate`，包含 schema、run/thread、时间、Agent 和 model/implementation 元数据。校验覆盖枚举、分数范围、来源 ID、同一运行线程的证据归属和私有推理字段拦截。

## Gateway profile

| Profile | Provider | 当前验收 |
|---|---|---|
| `fake_json` | fake | 已通过，结构化 JSON |
| `ollama_local` | Ollama `/api/chat` | 环境跳过，11434 未监听 |
| `qwen_api` | OpenAI-compatible | 无 `DASHSCOPE_API_KEY` 时返回 configuration_error |
| `fallback_human` | human | 已实现结构化 fallback |

诊断聚合使用 `DiagnosisDraftPayload` 作为独立模型输出契约，任务审计 ID 和证据归属由本地创建。Ollama 请求发送 JSON Schema；OpenAI 兼容 profile 默认发送 `json_object`，可对支持的模型显式切换 `structured_output_mode=json_schema`。所有 provider 输出仍执行本地 schema 校验，refusal、截断和非法 schema 均留有可审计错误状态。

## 验收

```powershell
cd D:\py_daima\mars-agent\Code
$py = 'D:\py_daima\.venvs\mars-agent\Scripts\python.exe'
& $py -m compileall -q agentic
& $py -m pytest agentic/tests -q
```

初始 G0 快照结果为 `11 passed`，编译通过。2026-11-09 Gateway 补充验证包括 fake/Ollama/OpenAI-compatible 请求构造、schema 错误、拒答/截断响应及完整 fake full-loop；当前全量回归为 `154 passed, 1 warning, 33 subtests`。pytest 已加入 `Code/agentic/requirements.txt`。本阶段没有 API key、模型权重或私有推理内容写入仓库；真实 Ollama 和 Qwen API 调用仍未验收。
