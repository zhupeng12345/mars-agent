# Third-party reuse manifest

This file records the external projects used as design references. The local
implementation keeps the project license boundary explicit and does not copy
an external gateway or application wholesale.

| Project | License / pinned reference | Reused pattern | Local implementation | Boundary |
|---|---|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT; pinned commit `e5e37b5d70dbfd93c7006bd90a658c6179a95946`; API validated against installed `langgraph==1.2.20` | State graphs, `Send`, reducers, `interrupt`, `Command`, checkpointers and stream events | `Code/agentic/orchestration/` and workflow adapters | No EduAgent business state or private reasoning is copied |
| [Routapse](https://github.com/pooyaphoenix/Routapse) | MIT; pinned commit `f52977ec9eb7e9dbc533e626c1ad471d3862252a`; adapter design only | signals/rules, deterministic override, confidence fallback, route audit | `Code/agentic/platform/jev/` | No FastAPI/React/Redis gateway is imported |
| [Laya](https://pypi.org/project/laya/) | Apache-2.0; optional dependency | `predict`, `fit`, `calibrate`, abstain/unbound route semantics | `Code/agentic/platform/jev/laya_adapter.py` | Laya routes diagnostic loops; it does not classify fire-eye states |
| [RSIHub](https://github.com/simple-agent-lab/RSIHub) | Apache-2.0; pinned commit `d6554299c58bf0fe2db04ddda0277f9db67ea680`; offline evolution pattern | candidate lineage, evaluator/gate/publisher separation | `Code/agentic/rsi/` | Frozen evaluators, safety rules and human decisions remain immutable |
| [FlashRSI](https://github.com/RUC-NLPIR/FlashRSI) | Apache-2.0; pinned commit `1d3f4d3703cc7c90eba01968626d0873158f2e24`; trajectory/evaluation ledger pattern | train/gate/sealed split and usage summaries | `Code/agentic/rsi/records.py` | No external runtime is vendored |
| [DeepSeek SuperAgent prototype](https://github.com/deepseek-ai/dsh-superagent-prototype) | MIT; pinned commit `8c32a9dc5d72c033816c6a8110b37f40cf94fba4`; research prototype | Source/Thread/Eval/Ticket/Assignment/Executor task semantics | `Code/agentic/platform/tickets.py` | GitHub/Feishu connectors and TypeScript runtime are not copied |
| PyTorchVideo | Apache-2.0 | video tensor conventions and temporal encoder comparisons | `Code/agentic/models/` | Optional reference; no dependency required for the baseline |
| MMAction2 | Apache-2.0 | video benchmark and split conventions | `Code/agentic/evaluation/` | No model zoo weights are bundled without provenance |

The F-GMVAE-T implementation is a local reimplementation based on the cited
paper and thesis. No usable, license-clear official F-GMVAE implementation was
found, so the project must retain the paper citation and implementation notes.

Audit note (2026-11-08): the pinned references above are design/reference
inputs. No third-party model benchmark or unverified source code is presented
as a local training result. Local F-GMVAE-T metrics remain unavailable until
expert video labels and a reproducible GPU training run are available.
