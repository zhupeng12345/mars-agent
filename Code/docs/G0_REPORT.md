# G0 阶段报告：协议与模型网关

## 结论

G0 的本地可复现部分通过。P0.2/P0.3 提供版本化 Agent 协议和统一 LLM Gateway；当前报告只覆盖协议和网关，不代表视觉模型精度或外部模型服务已经验收。

## 已验收

- 协议：`InspectionTask`、`MediaManifest`、`EvidenceRef`、`AgentResult`、`VisualEvidence`、`ProcessStateSnapshot`、`InspectionFeatures`、`JevRouteDecision`、`DiagnosisDraft`、`ReviewDecision`、`TraceEvent`、`DiagnosisTicket`、`AgentAssignment`。
- 每类协议支持 `to_dict/from_dict/validate`，并校验 schema、run/thread、来源 ID、枚举、分数范围及私有推理字段。
- Gateway profiles：`fake_json`、`ollama_local`、`qwen_api`、`fallback_human`。
- `fake_json` 结构化输出和缺失 Qwen API key 的 configuration error 已覆盖测试。

## 验收边界

- Ollama `localhost:11434` 未监听时标记为 skipped，不伪造本地模型结果。
- 没有 `DASHSCOPE_API_KEY` 时不调用云端 Qwen API，返回结构化 configuration error。
- 本阶段不产生视觉 Macro-F1、真实过热度测量或工艺控制效果结论。

## 命令与结果

```powershell
cd D:\py_daima\mars-agent\Code
$py = 'D:\py_daima\.venvs\mars-agent\Scripts\python.exe'
& $py -m compileall -q agentic
& $py -m pytest agentic/tests/test_p02_contracts.py agentic/tests/test_p03_gateway.py -q
```

历史 G0 工作区回归为 `59 passed`；当前全量回归见下方补充，历史 P0 记录保留在 [P0_REPORT.md](P0_REPORT.md)。
## 2026-11-08 当前复核补充

- 全量测试已更新为 `78 passed, 1 warning, 7 subtests passed`。
- Ollama 服务可连接且本机存在 `qwen3-vl:latest`；短提示真实视觉结构化 smoke 已返回 `DiagnosisDraft`，约 60.8 秒、464 tokens，确定性复核为 `pass` 但仍保留人工授权。
- Qwen API 记录为 `configuration_error/missing_api_key:DASHSCOPE_API_KEY`，未发起外网调用。
- `runtime_data/provider_smoke.json` 保存 provider、模型、耗时、重试和错误分类，不保存模型私有思维链。
- full_loop 的真实本地聚合运行 `runtime_data/demo_ollama_local_compact.json` 已完成：主长提示空响应后，受控短提示 fallback 返回结构化草案；主失败、fallback 耗时、token 和最终复核均写入网关元数据。
