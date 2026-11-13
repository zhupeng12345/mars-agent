# 实现阶段记录

## 2026-11-08 P5.3 Ablation matrix

- status: done (reproducible matrix and result provenance gate; no fabricated metrics)
- scope: `agentic/evaluation/ablation.py`, `agentic/scripts/run_ablations.py`, `agentic/tests/test_p5_ablations.py`, and `agentic/evaluation/run_all.py`
- comparisons: single/multi-frame, expert features, F-GMVAE-T, Process State, JEV, case gate, lexical/hybrid retrieval, HITL, and local/Qwen API profiles
- result policy: every numeric result must carry data version, split, seed, model and generated-at metadata; missing expert video labels, GPU runs or paired traces are reported as `structure_only`/`not_run`
- validation command: `D:\\py_daima\\.venvs\\mars-agent\\Scripts\\python.exe -m agentic.scripts.run_ablations --output runtime_data/ablation_report.json`
- boundary: the framework is complete; real visual ablation numbers await expert video labels and NVIDIA GPU execution

## 2026-11-08 visual training/evaluation boundary completion

- added `agentic/training/train_visual.py` and `train_fgmvae.py` for deterministic CPU forward/backward smoke and GPU-ready manifest training entry; the current manifest returns `blocked_no_human_video_labels` after the smoke step
- added `agentic/evaluation/visual_eval.py` for Macro-F1, confusion matrix, ECE and Brier when expert labels exist; unlabeled manifests remain `structure_validation_only`
- added `agentic/models/ood.py` novelty/energy features for JEV adapters
- validation: visual evaluation/training tests passed and `visual_training_smoke.json` records finite loss without claiming trained accuracy

## 2026-11-08 P7 End-to-end demo and report assets

- status: done (real media flow and auditable asset package; visual accuracy remains unclaimed without expert video labels)
- scope: `agentic/scripts/demo_run.py`, `agentic/scripts/validate_assets.py`, `agentic/scripts/build_ppt_asset_manifest.py`, `agentic/tests/test_p7_assets_demo.py`, `agentic/tests/test_p7_asset_manifest.py`, and the modern Streamlit agent page
- real runs: `正常.mp4` and `偏高.mp4` each decoded, passed image-quality gating, created multi-frame ROI evidence, ran the independent 14-variable process snapshot with `alignment_quality=unknown`, selected JEV routes, executed the full-loop five-agent fan-out where applicable, and persisted `waiting_review` HITL checkpoints
- assets: four contact sheets with JSON sidecars validate successfully; `ppt_asset_manifest.json` contains all 25 pages and maps each page to real runtime JSON/visual assets
- validation commands: `D:\\py_daima\\.venvs\\mars-agent\\Scripts\\python.exe -m agentic.scripts.validate_assets --directory report_assets/multi_agent_v2 --output runtime_data/asset_validation.json`; `... -m agentic.scripts.build_ppt_asset_manifest`
- boundary: generated model runs use CPU smoke weights unless a trained F-GMVAE-T checkpoint is provided; labels and reports are marked flow validation/experiment version
- reliability fix: JSON checkpoints and audit JSONL now use cross-process lock files plus atomic replacement; two concurrent real-video demos completed without a write collision
- entrypoint: `agentic.run` now defaults to the modern multi-frame/JEV/HITL flow; `--mode legacy` preserves the historical single-frame workflow.
- final validation: venv pytest `59 passed, 1 warning, 7 subtests`; compileall passed; route fixtures `5/5`; asset sidecars `7/7`; PPT manifest `25` pages; Streamlit `/_stcore/health` returned `ok` on port `8502`.

## 2026-11-08 P6.1/P6.2 RSI offline controlled evolution

- status: done (offline candidate/replay/gate/publish path; no online mutation)
- scope: `Code/agentic/rsi/records.py`, `candidate.py`, `evaluator.py`, `gate.py`, `lineage.py`, `publisher.py`, `cli.py`; `agentic/tests/test_p6_rsi.py`; `agentic/scripts/rsi_candidate_smoke.py`
- mutable surfaces: `prompts`, `jev_policies` (thresholds/templates only), `knowledge_queries`, and `report_templates`
- forbidden mutations: frozen/sealed sets, evaluator code, safety rules, model weights, production rules, and human conclusions
- lifecycle: append-only runtime inputs -> declarative candidate diff -> train/frozen/sealed replay -> quality/safety/efficiency gate -> expert approval -> versioned publication -> lineage rollback
- reference safety: candidate metadata and diff references can be checked against known source IDs; the smoke intentionally rejects an unknown reference and writes `runtime_data/rsi_candidate_smoke.p6.json`
- validation commands: `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m unittest agentic.tests.test_p6_rsi -v`; `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m unittest discover -s agentic/tests -q`; `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.rsi_candidate_smoke --output runtime_data/rsi_candidate_smoke.p6.json`; `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m compileall -q agentic/rsi`
- results: P6 tests 3 passed; full suite 44 passed; known-reference-error smoke returned `expected_reference_error`; CLI help exposes collect/candidate/replay/gate/approve/rollback
- boundary: publication changes only local RSI version snapshots and the active pointer; callers must explicitly integrate a published snapshot into a later deployment process

## 2026-11-08 P0.1
- 状态：done
- 目标：建立可复现的仓库资产清单、依赖基线和无模型 smoke test。
- 修改文件：
  - `Code/agentic/scripts/inventory.py`
  - `Code/agentic/scripts/smoke_test.py`
  - `Code/agentic/tests/test_p01_baseline.py`
  - `Code/agentic/scripts/__init__.py`
  - `Code/agentic/tests/__init__.py`
  - `.gitignore`
- 输入数据版本：当前工作区；`Code/data/video`、`Code/data/火眼分类图片帧`、`Code/data/v1`、`Code/agentic/knowledge`。
- 执行环境：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe`，Python 3.12.7；默认 `D:\py\python3\python.exe` 为 Python 3.9.5 且缺少 `cv2`，不能作为本项目运行解释器。
- 执行命令：
  - `python -m compileall -q agentic/scripts agentic/tests`
  - `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m unittest discover -s agentic/tests -q`
  - `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.inventory --output runtime_data/inventory.p01.json`
  - `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.smoke_test --simulate-no-model --output runtime_data/smoke.p01.json`
  - `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.run --help`
- 验收结果：通过。编译成功；4 个 unittest 全部通过；入口帮助正常返回 0；smoke 检查图片、视频、无效输入和无模型 fallback 均通过。
- 资产基线：火眼图片 6158 张（JPG）；视频 2 个（MP4）；`Code/data/v1` 51 个文件；知识文件 1 个；清单总计 6270 个文件、447012382 bytes。
- 失败样例：pytest 未安装（`No module named pytest`），未将其作为阻塞项，使用标准库 unittest 完成同等 P0.1 测试。
- 未解决问题：默认 Python 3.9 环境缺少视觉依赖；P0.1 未调用 Ollama 或云端 API，也未宣称模型推理完成。
- 下一步：P0.2 统一 Agent 协议，先定义并测试 `InspectionTask`、`EvidenceRef`、`AgentResult`、`JevRouteDecision`、`DiagnosisDraft`、`ReviewDecision` 和 `TraceEvent`。
# P0.2 Agent contracts (2026-11-08)

- status: done
- scope: `Code/agentic/platform/contracts.py` and `Code/agentic/tests/test_p02_contracts.py`
- implemented: versioned JSON contracts for media manifests, visual/process evidence, inspection features, JEV routing, diagnosis drafts/tickets, review decisions, assignments, agent results and trace events
- validation: enum and score bounds, required audit fields, source ID references, explicit `process_missing`/`alignment_quality`, same run/thread evidence lineage, and private reasoning field rejection
- compatibility: existing `InspectionTask`, `ReviewDecision`, `TraceEvent` positional constructors remain supported
- commands: `python -m compileall -q agentic/platform/contracts.py agentic/tests/test_p02_contracts.py`; `python -m unittest discover -s agentic/tests -q`
- result: 11 tests passed

## 2026-11-08 P1.1/P1.2

- status: done
- scope: media normalization and model-free image quality gate
- files: `Code/agentic/media/normalize.py`, `Code/agentic/agents/image_quality.py`, `Code/agentic/scripts/build_media_manifest.py`, `Code/agentic/tests/test_p1_media_quality.py`
- media support: image, video and extracted frame directory; deterministic SHA-256 media IDs; width/height/FPS/duration/frame count; file size and decoder errors
- quality output: blur, brightness, overexposure, occlusion, resolution, target presence and cross-frame consistency; versioned thresholds; `pass`/`retake`/`human` status
- validation: `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m unittest discover -s agentic/tests -q` (19 passed)
- real media smoke: both `Code/data/video/*.mp4` decoded successfully (720x1280, approximately 29.9 FPS, approximately 60.6 seconds, 1813/1814 frames); 12-frame quality sampling returned `pass`
- failure fixtures: black frame, overexposed frame, missing path and generated short video covered; invalid media routes to `retake`

## 2026-11-08 P0.3

- 状态：done（本地 Ollama smoke skipped：11434 未监听）
- 修改文件：`Code/agentic/platform/llm_gateway.py`、`Code/agentic/platform/__init__.py`、`Code/config/agent_profiles.json`、`Code/agentic/tests/test_p03_gateway.py`、`Code/agentic/requirements.txt`。
- 实现：`fake_json`、`ollama_local`、`qwen_api`、`fallback_human` 四类 profile；统一请求、图片编码、JSON 解码、schema 校验、错误分类、重试和 provider 元数据。
- 执行命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q`；`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m compileall -q agentic`。
- 验收结果：通过，6 个测试通过，完整测试集合 11 个测试通过，compileall 通过；fake provider 返回结构化 JSON，缺失 Qwen API key 返回 configuration_error，不发起网络请求。
- 环境记录：本机没有监听 Ollama `localhost:11434`，因此未声称本地模型推理完成；待 Ollama 启动后执行实际 vision smoke。
- 下一步：P1.1 媒体标准化与 P1.2 图像质量 Agent。

## 2026-11-08 P1.1/P1.2/P1.3

- 状态：done
- 修改文件：`Code/agentic/media/normalize.py`、`Code/agentic/agents/image_quality.py`、`Code/agentic/agents/review_guard.py`、`Code/agentic/scripts/build_media_manifest.py`、对应测试。
- 媒体验收：图片、视频、帧目录均生成 `media_id`、SHA-256、尺寸、FPS、时长、帧数和解码错误；空目录、坏路径、损坏媒体进入结构化错误。
- 质量验收：OpenCV 指标包含模糊、亮度、过曝、遮挡、分辨率、目标存在和跨帧一致性，输出 `pass/retake/human`，阈值版本为 `image-quality-1.0`。
- 诊断边界：证据来源、无测量过热度数值、工艺控制指令和降级草稿由确定性守门拦截。
- 真实命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.build_media_manifest data/video --output runtime_data/p1_media.jsonl --run-id p1-real`。
- 真实结果：两个 MP4 均解码成功，分辨率 720x1280，约 29.9 FPS，时长约 60.6 秒，帧数 1813/1814；采样质量检查通过。
- 测试结果：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q`，19 tests passed；compileall 通过。
- 下一步：P1.5 多帧 ROI 序列与视觉模型、P1.6 Process State Agent。

## 2026-11-08 P1.5/P1.6

- 状态：done（真实视频流程/结构验证；无人工视频真值）
- 多帧实现：ROISequence、YOLO 逐帧适配、轨迹/缺失 mask、contact sheet、Conv/GRU/TCN/Transformer 时序编码器、专家特征接口、F-GMVAE-T 最小前向与损失。
- 真实视觉产物：`Code/data/evaluation/visual_manifest.jsonl`，两个视频各生成 8 帧窗口和 ROI/contact sheet；6158 张分类图片记录为弱标签，未进入专家真值集。
- 过程实现：`Code/data/evaluation/process_manifest.json`、`process_state_metrics.json`；`X_seq=[5744,30,14]`，历史 30 步、采样 2 分钟、未来 15 步，日期切分 3952/965/827；视频映射为 unavailable，alignment unknown。
- 验收命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q`，36 tests passed；多帧合成 forward/loss、ROI 丢帧 mask、过程 manifest 和 baseline 评估均通过。
- 指标边界：过程 baseline test ROC-AUC 已记录为数据集事件风险结果；视觉无人工视频真值，只报告流程/结构验证。
- 下一步：P2 JEV 四路路由、simple_loop 和 full_loop 协同。

## 2026-11-08 P2.1-P2.4

- 状态：done（规则/受控并行实现）
- 修改文件：`Code/agentic/platform/jev/`、`Code/agentic/orchestration/diagnosis_graph.py`、`Code/data/evaluation/route_manifest.jsonl`、`Code/agentic/scripts/route_smoke.py`。
- JEV 行为：质量失败进入 retake，高风险/冲突进入 full_loop，新颖性/低置信进入 human，证据充分且适用案例进入 simple_loop；路由不输出 C/N/H 或数值过热度。
- full_loop：受控线程池并行执行视觉复核、证据质量、知识 RAG、案例比对和过程状态 worker；失败 Agent 保留结构化错误，聚合器继续运行。
- 验收命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.route_smoke --output runtime_data/route_smoke.json`；`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q`。
- 验收结果：五个 fixture 路由正确，完整测试集合通过；trace 显示路由、循环、聚合、复核和 finalize。
- 复用边界：Laya 与外部 JEV 通过可选 adapter 接入，规则优先；未安装或 abstain 时进入 human fallback。
- 下一步：P3 版本化领域 RAG 和特殊案例生命周期。

## 2026-11-08 P3.1-P3.3

- 状态：done（词面/版本化 RAG 与受控案例库）
- 知识实现：`KnowledgeRule` 和 `VersionedKnowledgeBase` 支持 rule_id、来源版本、生效日期、前置条件、适用范围、风险级别和允许输出；空检索返回空结果。
- 案例实现：`SpecialCaseMemory` 支持 candidate → eval_only → approved_special_case → active → deprecated；active 必须有人工审核和冻结回归通过，适用条件参与检索过滤。
- 知识包：`Code/agentic/knowledge/rules_v1.jsonl`，首批 3 条铝电解视觉/过热度/证据边界规则。
- 验收命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.build_knowledge_pack`；`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q`。
- 验收结果：39 tests passed，compileall 通过；版本检索、空检索、案例状态转移和 active 回归门控通过。
- 已知边界：当前仍是本地词面检索，BGE-M3/Reranker/Milvus 作为同一接口的后续适配器；案例相似度使用可审计的数值特征距离。
- 下一步：P4 interrupt/resume、人工审核工作台和历史报告。

## 2026-11-08 P5.1-P5.2（基础评测框架）

- 状态：done（冻结清单与结构化 smoke；完整视觉消融待 GPU/专家视频标签）
- 新增：`agentic/evaluation/metrics.py`、`agentic/evaluation/run_all.py`、`agentic/scripts/freeze_evaluation.py`、`Code/data/evaluation/knowledge_queries.jsonl`、`hitl_cases.jsonl`。
- 指标：JEV 路由准确率/简单路径召回/复杂进入 full 召回/危险样本 simple 漏放，证据链完整率/引用存在率，事件耗时 P50/P95/P99。
- 评测边界：视觉 manifest 没有人工视频真值，报告明确为 `structure_validation_only`；过程分支保留 `alignment_quality=unknown`；不生成真实过热度测量精度。
- 验收命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.freeze_evaluation`；`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.evaluation.run_all`；pytest。
- 验收结果：冻结文件生成，JEV fixture replay 通过，41 tests passed。
- 下一步：P4 完成后补 HITL 指标、真实运行日志指标和 P5.3 消融矩阵。

## 2026-11-08 P1.6 Process State Agent

- status: done (independent process branch; no visual-process fusion claim)
- scope: `Code/agentic/agents/process_state.py`, `Code/agentic/scripts/build_process_manifest.py`, `Code/agentic/evaluation/process_state.py`, `Code/agentic/scripts/evaluate_process_state.py`, and `Code/agentic/tests/test_p1_process_state.py`
- implemented: audit manifest for `Code/data/v1/step2_ew_202111/dataset.npz`/`split.json`, 14-variable groups, 30-step/2-minute window contract, date split checks, source hashes, baseline metric/prediction lineage, transparent window statistics and trends, change-point records, process regime/risk/tags, and validated `ProcessStateSnapshot` output
- process boundary: process variables retain source names and are marked as proxies; no field is renamed to chemistry or measured superheat. Without an auditable video mapping, snapshots set `alignment_quality=unknown` and include `video_mapping` in `process_missing`; no video classification improvement is reported.
- real commands: `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.build_process_manifest --data-root data/v1 --output data/evaluation/process_manifest.json`; `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.scripts.evaluate_process_state --data-root data/v1 --output data/evaluation/process_state_metrics.json`; real state smoke writes `runtime_data/process_state_smoke.p1.json`.
- real results: manifest audited 5744 samples with `X_seq=(5744,30,14)`, `X_stat=(5744,70)`, `L=30`, `H=15`, 2-minute median sampling and train/val/test counts 3952/965/827. Recomputed test ROC-AUC was 0.5268 (1DCNN), 0.5419 (CNN), and 0.4371 (LSTM); these are independent process-event metrics, not video metrics.
- tests: `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m unittest agentic.tests.test_p1_process_state -v` (5 passed); full suite and compileall were run after integration.
- next step: P2 JEV integration may consume a process snapshot only when a runtime process window and an explicit alignment quality are present; this task does not start that stage.

## 2026-11-08 P1.5 Multi-frame visual evidence and F-GMVAE-T

- status: done (flow/structure validation; no human video truth, so no generalization accuracy claim)
- scope: `Code/agentic/models/visual_types.py`, `temporal.py`, `visual_baselines.py`, `roi_sequence.py`, `losses.py`, `fgmvae_temporal.py`; `Code/agentic/scripts/build_visual_windows.py`; `Code/agentic/agents/visual_evidence.py`; `Code/agentic/tests/test_p15_multiframe.py`.
- data contract: `ROISequence.frames=[B,T,3,H,W]`, boxes, frame/quality masks, source IDs and weak labels; reuse platform `VisualEvidence`/`EvidenceRef` for video-window, key-frame, uncertainty, temporal-consistency and missing-mask fields.
- temporal/model: CPU Conv, GRU, TCN and Transformer encoders share `[B,T,D] -> [B,D]`; F-GMVAE-T includes a light spatial encoder, temporal encoder, mixture-Gaussian prior, reparameterization, KL, fuzzy-label and temporal-consistency losses, plus probability/risk/uncertainty/key-frame outputs.
- window pipeline: `build_visual_windows` scans videos and fire-eye image directories, keeps fixed video-group windows, runs per-frame YOLO/light tracking/ROI crop, and writes frame/ROI/overlay/contact sheets plus UTF-8 `visual_manifest.jsonl`. Human labels remain empty; directory/filename labels are recorded only as `weak_source_label` with `label_status=unlabeled/weak`.
- real media validation: both MP4s produced one complete eight-frame window; YOLO backend was active, each window had eight ROIs, `missing_count=0`, and distinct video IDs/groups. The image manifest contains 6158 weak-label records. Artifacts: `Code/data/evaluation/visual_manifest.jsonl` and `Code/data/evaluation/visual_windows/`.
- synthetic/model validation: mask handling, missing feature-guide/empty-prior path, mixture weights summing to one, finite KL/uncertainty, loss backpropagation, all four temporal profiles and contact-sheet rendering passed.
- commands: `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest Code/agentic/tests -q` (36 passed); compileall for the new model/agent/script modules passed.
- boundary: without human video truth, manifest/model output is reported as flow/structure validation only; the main workflow was not changed, and `visual_evidence.py` is an explicit adapter for later node integration.

## 2026-11-08 P4.1-P4.3

- status: done (local durable HITL service and compatible graph/UI helpers)
- implemented: `agentic/platform/memory.py` now provides atomic JSON and standard-library SQLite checkpoint stores; `agentic/platform/hitl.py` persists waiting reviews, supports restart-safe idempotent resume, validates only `approve`/`amend`/`reject`/`request_retake`, and writes de-duplicated audit events.
- graph integration: `diagnosis_graph.py` accepts an optional LangGraph checkpointer and `human_review` interrupt node; environments without LangGraph retain a sequential fallback. `run_diagnosis_graph(..., hitl_service=...)` creates a durable pending review.
- reports/history: `HumanReviewService.history()` filters audit rows by run, route, risk, status and model profile; `export_report()` renders JSON, Markdown or HTML. `agentic.hitl_cli` exposes pending/resume/history/export commands and `platform.hitl_ui` provides Streamlit-compatible callbacks.
- validation: `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest Code/agentic/tests/test_p4_hitl.py Code/agentic/tests/test_diagnosis_graph.py -q` passed (8 tests, 7 subtests); tested service restart, repeated resume, invalid action, all four actions, graph pause and report formats; compileall passed.
- known boundary: the default JSON audit log uses process-local locks; callers requiring multi-process checkpoint transactions can inject `SQLiteCheckpointStore`.
- next step: integrate the pending-review helpers into the existing Streamlit agent page without changing its current layout.

## 2026-11-08 Evaluation delivery audit

- status: done (delivery audit and reproducibility fixes)
- fixes: `agentic.evaluation.run_all` now accepts the plan's no-argument invocation and writes `runtime_data/evaluation_report.json` by default; `freeze_evaluation` inventories all five frozen manifests, records SHA-256/record counts, creates the process JSONL alias, and fails explicitly when a required manifest is missing.
- runtime evidence: modern `demo_run` now writes graph `TraceEvent` rows to `events.jsonl` and a run result to `runs.jsonl`, honoring `MARS_AGENT_RUNTIME_DIR`; system P50/P95/P99 and evidence metrics can therefore be regenerated from real demo traces.
- documentation audit: corrected pinned third-party URLs/commits for LangGraph, Routapse, RSIHub, FlashRSI and DeepSeek SuperAgent in `docs/third_party_manifest.md`; corrected G7's final pytest count to 57.
- boundary: visual ablation remains `structure_only` without expert video labels/GPU; process alignment remains `unknown`; no true superheat measurement claim is introduced.
- metrics: `evaluation_report.json` now separates completed run results from `running` records and reports citation support/no-source conclusion ratios plus operational failure, timeout, retry, fallback, structured-handoff and HITL adoption rates; empty denominators remain `null`.
- report artifacts: the no-argument evaluation command writes `evaluation_report.md`, `evaluation_report_metrics.csv`, `evaluation_report_summary.png`, and `evaluation_report_failures.jsonl` beside the JSON report; the graph is a fixture summary and does not claim visual accuracy.

## 2026-11-08 P2.4 LangGraph Send fan-out/fan-in audit

## 2026-11-08 final implementation audit

- status: engineering workflow complete; external model and label gates remain explicit
- changes: quality-gate short circuit prevents diagnostic provider calls for `retake`/`human`; modern demo computes and records a low-cost JEV pre-route before Gateway invocation; knowledge, case lifecycle, Gateway metadata and graph evidence share the same run/thread
- final validation: `pytest agentic/tests -q` -> 59 passed, 1 warning, 7 subtests; route fixtures 5/5; frozen manifests 5/5; asset sidecars 7/7; evaluation artifacts generated; LangGraph interrupt/resume smoke approved; Streamlit health `ok`
- evidence: two real videos completed media normalization, quality gate, multi-frame ROI, independent 14-variable Process State, JEV, full-loop worker trace and durable `waiting_review` checkpoints
- boundary: no expert video labels or NVIDIA GPU were available, so visual metrics and F-GMVAE-T training remain `structure_validation_only`; video/process alignment remains `unknown`; Ollama/Qwen API live calls require the corresponding service/key

- status: done
- scope: `Code/agentic/orchestration/diagnosis_graph.py` and `Code/agentic/tests/test_diagnosis_graph.py`
- implementation: LangGraph path now uses `Send("full_loop_worker", ...)` for one task per specialist and `Annotated[List[...], operator.add]` as the fan-in reducer; `full_loop_join` emits child/failure counts after the dynamic barrier. The no-LangGraph path keeps the bounded ThreadPoolExecutor fallback.
- checkpoint safety: worker callables stay in a short-lived `(run_id, thread_id)` registry; Send payloads and graph state carry worker IDs and JSON-safe context, preventing function objects from entering durable checkpoints.
- validation: `D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q` -> 57 passed, 1 warning, 7 subtests; full-loop trace showed `full_loop_dispatch`, three `full_loop_worker:*` nodes, `full_loop_fanout` with `fanin=reducer_operator_add`, and failed worker evidence retained for aggregation.

## 2026-11-08 当前执行记录：模型训练、RSI 闸门和最终回归

- 命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m agentic.training.train_image_classifier --epochs 10 --device cpu --output-dir runtime_data/visual_training`。
- 结果：训练 checkpoint 与图像级评测写入 `runtime_data/visual_training`；测试集 `n=927`、accuracy `0.7346`、Macro-F1 `0.7097`、ECE `0.2828`、Brier `0.3525`。由于无来源元数据且发生不安全 item fallback，报告明确 `leakage_risk=high`，该结果只作实验基线。
- 命令：`python -m agentic.scripts.demo_run --input data/video/正常.mp4 --output runtime_data/demo_trained_current.json --assets-dir report_assets/multi_agent_v2`。
- 结果：真实视频 8 帧 ROI，`framewise_probability_pooling`，5 个 full-loop worker，聚合节点和人工 `waiting_review` 均产生审计记录。
- 命令：`python -m agentic.scripts.rsi_defect_demo --output-dir runtime_data/rsi`。
- 结果：已知缺陷“缺少 source citation”完成 candidate → 三段 replay → gate → 人工批准 → publish → rollback，`gate_accepted=true`、`rollback_verified=true`；空切分现在拒绝通过。
- 命令：`python -m agentic.scripts.provider_smoke --output runtime_data/provider_smoke.json`。
- 结果：Ollama `qwen3-vl:latest` 返回结构化视觉草案，约 60.8 秒、464 tokens，确定性复核为 pass 但仍等待人工；Qwen API 因缺少 `DASHSCOPE_API_KEY` 返回 configuration error。两种结果均保留网关元数据。
- 结果：`demo_ollama_local_compact.json` 完成真实 full_loop 聚合；长提示主请求空响应后，短上下文 fallback 返回 Qwen3-VL 草案，模型 source ID 进入证据链，复核 pass，流程停在 `waiting_review`。
- 命令：`python -m pytest agentic/tests -q`、`python -m agentic.evaluation.run_all --output runtime_data/evaluation_current.json`、`python -m agentic.scripts.route_smoke --output runtime_data/route_current.json`、`python -m agentic.scripts.validate_assets --output runtime_data/assets_current.json`。
- 结果：`78 passed, 1 warning, 7 subtests`；JEV `5/5`；15 张 PPT contact sheet 资产尺寸校验通过；评测报告完整生成。

## 2026-11-08 E0/E1 链路回归与命名收口

- 统一运行时标识为 `laya_jev_router_v1` / `laya-jev-policy-0.1`；`JevRouter` 保留兼容入口，新增 `LayaJevRouter` 命名入口。JEV 在文档和界面中表示参考的证据驱动路径选择思想，Laya-JEV 表示基于 Laya 适配的项目路由器。
- E0：使用 `demo_run` 分别运行 `data/video/正常.mp4` 和 `data/video/偏高.mp4`；两次均完成媒体标准化、多帧 ROI、Process State（`alignment_quality=unknown`）、Laya-JEV `full_loop`、5 个并行 worker、诊断聚合、证据复核、人工 `approve` 和报告导出。
- E0 固定验收：`route_smoke` 5/5、Laya 校准数据集 5 条、`evaluation.run_all` 生成指标/失败样例/图表、`validate_assets` 17 张图通过；新增原生 SQLite checkpointer 重启恢复测试。
- E1：Streamlit 运行于 `http://localhost:8503`，`/_stcore/health` 返回 `ok`；页面显示 Laya-JEV 路由、Ticket 阶段、并行 Agent、HITL 审核、报告归档和下载入口。
- 全量测试结果：`84 passed, 1 warning, 10 subtests`。未宣称视觉准确率、真实过热度测量精度或视频-过程融合增益；这些进入 E2，等待专家标注、GPU 和可靠对齐数据。
- F-GMVAE-T 补强：`FeatureLabelGenerator` 将专家静态/动态描述生成 fuzzy mixture guide，并以 log-prior 注入混合先验；新增归一化、梯度回传和专家特征敏感性测试，P1.5 仍标记为 `structure_validation_only`。
- Process State E2 预备 smoke：`train_process --model cnn --epochs 1 --device cpu` 生成 `runtime_data/process_baseline_e0/process_cnn_report.json`，ROC-AUC `0.4572`，标记为 `experimental_baseline`，不作为视频分类或真实过热度指标。
- 依赖收口：`Code/agentic/requirements.txt` 增加 `langgraph-checkpoint-sqlite>=3.1,<4`，与已验证的原生 SQLite interrupt/resume 测试保持一致。
- 前端复核：重启 `streamlit run web_interface/streamlit_app.py --server.port 8503 --server.headless true` 后，`http://localhost:8503/_stcore/health` 返回 `ok`；当前服务对应本轮代码。
- Laya 适配：安装 `laya==0.4.29`，按新版 `Router.predict` 字典 questions/`answers.route.choice` 契约更新 `LayaAdapter`，新增 fake backend 契约测试；未下载领域权重，真实路由拟合仍标记为 E2 实验项。
- P7 资产重建：重新运行 `build_ppt_asset_manifest`，清单覆盖 25 页、26 个运行 JSON 和 17 张真实 contact sheet；`validate_assets` 结果为 `17/17`，页面状态仍按 `flow_validation`/`experimental` 边界记录。
- 追加验证：原生 LangGraph `InMemorySaver` interrupt 返回中断对象，`Command(resume=approve)` 恢复为 `approved`；P4 专项 6 项测试与 7 个 subtests 通过。
- CLI/UI：`agentic.run` 与 Streamlit 现代工作台现在都可选择 `fake_json`、`ollama_local`、`qwen_api` 或 `fallback_human`；JEV 仍在选择 full_loop 后才调用聚合 profile。

## 2026-11-08 文档术语与执行节奏校准

- 修改文件：`Code/docs/铝电解多智能体全局规划.md`、`Code/docs/铝电解多智能体完整实现执行计划.md`。
- 术语边界：JEV 固定表述为参考的证据驱动路径选择思想；Laya 固定表述为复用的开源路由基座；Laya-JEV 固定表述为面向铝电解火眼的领域适配路由器；完整系统为 Laya-JEV + 确定性安全门控 + LangGraph 编排 + 人工审核。
- 执行顺序：新增 E0 完整链路、E1 前端工作台、E2 增强优化三个阶段及勾选状态；E0 未完成端到端回归前，不以页面截图或模块数量替代验收。
- 事实边界：补充没有专家路由标注、GPU、Qwen API key 或可靠视频-过程对齐时的 `flow_validation`、`experimental`、`blocked_external` 状态，避免把 Laya fit、视觉指标和真实过热度写成已完成结果。

## 2026-11-09 PPT 演示版快速补全 T0-T6

- 状态：flow_validation complete；本阶段目标是汇报链路完整，不替代论文级训练和效果评测。
- T0/T1：新增 `agentic/scripts/presentation_demo.py`，生成 `runtime_data/presentation_demo/` 下四个固定场景：`stable_case → simple_loop`、`conflict_high_risk → full_loop`、`bad_quality → retake`、`ood_unknown_case → human`。full_loop trace 含视觉复核、证据完整性、工艺知识 RAG、特殊案例比对和 Process State 五类 Agent；所有 payload 明确禁止指标和真实过热度声明。
- 演示查询：新增 `data/evaluation/knowledge_queries_demo.jsonl` 和 `case_queries_demo.jsonl`，各 10 条 `demo_fixture` 查询，供 PPT 知识检索和案例生命周期页面使用；不写入冻结研究评测集。
- T2/T5：新增 `agentic/scripts/build_ppt_visual_assets.py`，生成 25 张 16:9 结构验证 PNG 及 JSON sidecar；重建 `ppt_asset_manifest.json`，25 页均绑定 page-specific image/source，并保留真实视频 contact sheet 和四路 canonical trace。
- T3：Streamlit 现代工作台增加演示案例选择、预生成 trace 加载、五类 Agent 状态卡、RSI/评测素材面板、路由 backend/version 展示和 PNG 报告下载。presentation fixture 的四个人工动作写入 `runtime_data/presentation_demo/review_audit.jsonl`，不依赖真实生产 checkpointer。
- T6 验收命令：`python -m agentic.scripts.presentation_demo` 返回 4 cases；`pytest agentic/tests/test_presentation_demo.py -q` 返回 `2 passed`；P7 资产测试返回 `4 passed`；`build_ppt_visual_assets` 生成 25 张图；`validate_assets --output runtime_data/presentation_asset_validation.json` 返回 `42/42 valid`；Streamlit AppTest 加载 bad-quality 和 full-loop trace 均无异常，full-loop 页面显示 5 个 Agent 状态卡；`/_stcore/health` 返回 `ok`。
- 边界：结构图、四路 trace、10+10 查询和 RSI 卡片均是演示素材；仍不代表视觉精度、路由模型准确率、视频-过程融合增益或真实过热度测量。
- 补强：演示夹具完成人工动作后，现代工作台现在可直接导出 JSON、Markdown、HTML 和 PNG 演示报告；报告保留 `flow_validation` 标记，不写入真实生产审核服务。
- 最终复核：全量 `pytest agentic/tests -q` 为 `87 passed, 1 warning, 10 subtests`；`route_smoke` 为 `5/5`；PPT 资产为 `42/42 valid`、`25/25` 页覆盖；`evaluation.run_all` 生成结构验证报告；Streamlit AppTest 成功加载 full_loop 五 Agent 并完成 stable_case 通过后的四种报告下载入口检查。

## 2026-11-09 E2 可本地验收的评测补齐

- 代码：新增 `agentic/evaluation/image_quality.py` 与 `agentic/scripts/evaluate_quality_gate.py`。确定性合成图覆盖清晰目标、模糊、欠曝、过曝、低分辨率和目标缺失；逐样本保存 SHA-256、质量状态、问题标签和阈值版本。
- 质量门控结果：`evaluate_quality_gate` 6 个夹具中 5 个失败样本全部进入 `retake/human`，检出率 `1.0`；清晰样本误补拍数 `0`。该结果类型为 `synthetic_quality_gate_regression`，不代表现场图像准确率或阈值已标定。
- 时延评测：`evaluation.metrics` 新增 `end_to_end_latency_metrics`，按 `run_id` 配对 `started_at/finished_at`，解析 ISO-8601 时区和 `Z`，排除不完整、非法、负时长记录；总报告分开写 `system.end_to_end_latency` 与 `system.node_latency`。未显式选择运行 ID 的历史混合统计保持不可声明。
- 过程评测：使用严格命令 `python -m agentic.scripts.evaluate_process_state --data-root data/v1 --output runtime_data/process_eval.final.json`，日期净化测试集 `827` 窗口；1DCNN/CNN/LSTM ROC-AUC 分别为 `0.527/0.542/0.437`，PR-AUC 为 `0.561/0.588/0.480`，状态仍为 `experimental_baseline`，`alignment_quality=unknown`。
- 检索评测：知识与案例各 10 条 demo fixture 行为回归均通过 `10/10`；明确标记为 fixture regression，不作为领域召回率或引用支持率。
- 汇总命令：`python -m agentic.evaluation.run_all --root . --runtime-dir runtime_data --process-eval runtime_data/process_eval.final.json --output runtime_data/evaluation.e2_20261109.json`。报告拆分节点/任务时延，并保留 `runtime_scope=unscoped_history_mixed, claimable=false`。
- RSI / 冻结状态：`rsi_defect_demo` 完成候选门控与回滚验证；当前冻结清单六项全部因标签/数量/来源字段不合格而拒绝冻结（其中 `case_queries.jsonl` 缺失），这是预期的 fail-closed 结果。
- 验证：全量 `pytest agentic/tests -q` 为 `118 passed, 1 warning, 16 subtests`；`compileall` 与 `git diff --check` 通过。新增测试覆盖合成图像门控、时区换算、零时长、缺失/非法时戳、按运行 ID 的端到端与节点耗时分离，以及显式过程报告并入总评测。
- 未解决：专家审核视觉/路由标签、真实视频分组评测、GPU 训练、可靠视频-过程对齐、BGE-M3/Reranker/Milvus、网页原生 LangGraph interrupt 浏览器验收、冻结集、完整消融和 RSI 发布后运行时切换仍未完成；不得将结构/fixture 结果写作准确率或性能提升。

## 2026-11-09 P1.6 过程变量归因与工作台评测联动

- checkpoint 契约：`train_process` 现在输出 `process-checkpoint-2`、`model_config`、`temporal_contract`、`model_profile` 和训练集归一化参数；新增 `feature_names` 子集训练和 `evaluate=False` 训练选项。`ProcessCheckpointPredictor` 可直接加载完整 14 维及子集 checkpoint，测试验证预测输入列名和次序。
- 归因实现：新增 `agentic/evaluation/process_feature_attribution.py` 与 `agentic/scripts/evaluate_process_feature_attribution.py`。对验证集按窗口置换单变量轨迹和同组变量，基于 PR-AUC 排序选 8 个候选；测试集只对 14 维、规划紧凑 8 维和验证集选择 8 维做一次最终对照。
- 真实数据试跑：命令 `python -m agentic.scripts.evaluate_process_feature_attribution --dataset data/v1/step2_ew_202111/dataset.npz --split data/v1/step2_ew_202111/split.json --model cnn --epochs 1 --seed 7 --repetitions 3 --output-dir runtime_data/process_feature_attribution` 成功，测试集 827 窗口。验证集选出的 8 个字段为 `potVolt`、`baseFeedingInter`、`wavingDelta`、`settingVoltMax`、`settingVoltMin`、`feedingState`、`aeWaitInterval`、`andoeMoveState`。
- 单次实验对照：14 维 ROC-AUC/PR-AUC `0.408/0.475`；规划紧凑 8 维 `0.434/0.477`；验证集选出的 8 维 `0.358/0.444`。当前仅为单 seed、1 epoch 实验基线，选出的 8 维在测试集变差，不能据此缩减诊断输入或声称变量贡献因果关系。
- 工件：输出 JSON、验证集单变量/分组 CSV 和测试集对照 CSV；总评测 `run_all` 增加 `--process-attribution`，现代工作台显示候选字段、模型指标和声明边界。
- 评测边界：归因仅解释过程数据中未来 15 步事件标签的预测关联；过程数据仍没有视频时间映射，`video_fusion_evaluated=false`、`real_superheat_measurement=false`。
- 验证：checkpoint round-trip、8 维子集训练/推理、合成信号归因、列映射和确定性测试通过；Streamlit AppTest 加载 trace 后评测表可见且无异常；全量 `pytest agentic/tests -q` 为 `122 passed, 1 warning, 16 subtests`。
## 2026-11-09 P4 native web HITL integration

- status: flow_validation; native LangGraph SQLite interrupt/resume is now wired into the modern Streamlit run path.
- implementation: browser uploads are persisted under `MARS_AGENT_RUNTIME_DIR/modern_demos/media/<sha256>/<original-name>` so video-manifest basename matching still works. Native graph checkpoints use `graph_checkpoints.sqlite`; canonical demo payloads use `modern_demos/<run_id>.json`.
- restore: the workbench lists pending service checkpoints, reloads the canonical run payload and source media after a page refresh, and resumes the same graph `thread_id` with the selected allowlisted action.
- consistency: resume reconciles the LangGraph checkpoint with the idempotent HumanReviewService audit, TicketStore, runtime events, run status, and CaseMemory feedback. Same-action retries do not duplicate the audit, ticket completion, or human/finalize events; a conflicting action is rejected.
- regression: all four actions (`approve`, `amend`, `reject`, `request_retake`) pass native SQLite pause -> persisted reload -> resume -> retry tests. Ticket status and runtime completion are asserted.
- UI verification: Streamlit AppTest seeded a real pending SQLite run in a temporary runtime directory, loaded it through the pending-run control, approved it, and observed `approved` with zero script exceptions. Runtime storage honors `MARS_AGENT_RUNTIME_DIR`.
- commands: `python -m pytest agentic/tests/test_demo_native_hitl.py agentic/tests/test_modern_demo_integration.py agentic/tests/test_p4_hitl.py -q` (10 passed, 11 subtests before ledger assertions); `python -m pytest agentic/tests/test_demo_native_hitl.py -q` (2 passed, 4 subtests including a separate Python child process); current full suite after P2.4 specialist integration `python -m pytest agentic/tests -q` (128 passed, 1 warning, 20 subtests); `python -m compileall -q agentic web_interface`; `git diff --check`.
- remaining acceptance: Streamlit script-level interaction, separate process recovery and all four graph actions pass. A real browser screenshot/Playwright session, concurrent reviewer race test, and full report-download verification remain open. PNG report rendering emits missing-CJK-font warnings in this Linux verification environment.

## 2026-11-09 P2.4 Full-loop specialist analysis

- status: flow_validation; replaces the earlier default-worker context-copy wrappers with five deterministic, role-specific specialists under `agentic/agents/specialists.py`.
- `visual_review` checks class probability/label agreement, confidence, uncertainty, temporal consistency and source provenance; `evidence_quality` checks quality gate, target presence, frame provenance and source completeness.
- `knowledge_rag` independently queries the versioned rule pack with a domain retrieval query and returns only source/version/locator-bound passages. `case_compare` independently searches the approved active case memory using the current numeric feature contract and applicability filters.
- `process_state` reports its state/risk/tags and alignment boundary; with `alignment_quality=unknown` it sets `video_relation=independent_prior`. Missing inputs return explicit `abstain`/`missing` results.
- Each specialist result now populates the shared `AgentResult.evidence`/`evidence_ids` contract. The aggregation prompt receives role-specific findings, retrieval sources, applicability and abstain issues. Existing Send fan-out/fan-in, retry, timeout and worker failure isolation are unchanged.
- tests: four focused tests prove independent retrieval, provenance, safe abstain, five-role fan-in, aggregation prompt handoff and one-worker failure isolation. Two real repository videos (`data/video/正常.mp4`, `data/video/偏高.mp4`) ran end-to-end in isolated runtime directories: both chose `full_loop`, reached `waiting_review`, and produced five completed specialists; knowledge retrieval returned three versioned rules, no approved matching case was available, and Process State stayed `prior_only` with alignment unknown.
- validation at specialist integration: full `pytest agentic/tests -q` -> `128 passed, 1 warning, 20 subtests`; `compileall -q agentic web_interface`; `git diff --check`.
- boundary: deterministic analysis and real trace validation only; no expert-video accuracy, specialist agreement quality, real superheat measurement, or process/video fusion gain is claimed.

## 2026-11-09 P2.4 Multi-Agent Flow Integrity Evaluation

- Added `multi_agent_flow_metrics` to `agentic/evaluation/metrics.py` and included it in `evaluation.run_all` under `multi_agent`. It reports full-loop specialist coverage, completed/failed/missing worker slots, abstentions, and source-reference integrity. The metric is explicitly not an accuracy claim and returns `not_run` when the selected runtime cohort has no full-loop runs.
- Added report-level tests for complete/failed/abstaining workers, missing or mismatched per-agent evidence refs, no eligible full-loop cohort, and explicit run-id scoping.
- Ran `data/video/正常.mp4` and `data/video/偏高.mp4` with `--no-model` in an isolated runtime directory, then evaluated only those two explicit run IDs. Both selected `full_loop`; all 10 expected worker slots completed, there were 0 failed and 0 missing workers, both case specialists explicitly abstained, and all 6 source-lineage checks passed.
- Artifacts: `runtime_data/specialist_eval_20261109/evaluation_report.json`, `.md`, `_metrics.csv`, `_failures.jsonl`, `_summary.png`, and each run's JSON plus event/runtime records.
- Verification: focused P2.4/P5 tests `14 passed`; full suite `131 passed, 1 warning, 20 subtests`; `compileall -q agentic web_interface`; `git diff --check`.
- Interpretation boundary: this is real-video orchestration and provenance validation. The `--no-model` runs do not establish visual accuracy, expert routing quality, latency savings, or industrial diagnostic effectiveness.

## 2026-11-09 P4 Streamlit AppTest And Concurrent Review

- Added `agentic/tests/test_streamlit_native_hitl.py`. It creates a native SQLite pending run in an isolated runtime, loads it through the modern Streamlit page, exercises `approve`, `amend`, `reject`, and `request_retake`, and verifies the resulting graph status plus exactly one durable audit row.
- The same AppTest confirms JSON/Markdown/HTML download controls are present after review. Each service export contains the correct run ID and non-empty report content.
- Added a concurrent conflicting reviewer test to `test_demo_native_hitl.py`: approve and reject race on one run, one action commits, the other gets the already-resumed conflict, and audit stays at one decision.
- Verification: `pytest agentic/tests/test_demo_native_hitl.py agentic/tests/test_streamlit_native_hitl.py -q` -> `4 passed, 8 subtests`.
- Limitation: no Playwright/Selenium package or Chrome/Edge executable is available in this environment, so screenshots and browser-triggered downloads are not verified. AppTest is script-level UI integration.
- Follow-up: expanded the AppTest to assert the PNG download control and verify an exported PNG has a valid signature under each of the four review actions. Focused result: `1 passed, 4 subtests`.

## 2026-11-09 Final Regression After P2.4/P4/P6

- Full `pytest agentic/tests -q`: `138 passed, 1 warning, 24 subtests`. The warning is the existing PyTorch Transformer nested-tensor warning.
- `python -m compileall -q agentic web_interface` and `git diff --check` passed.
- Re-ran the explicit two-video `evaluation.run_all` report after the specialist lineage metric change. It reports two full-loop runs, 10/10 specialist slots completed, zero failures/missing workers, two expected case abstentions, and 6/6 source lineage checks.
- Re-ran the hardened RSI same-input defect demo; gate accepted and rollback verified.
- Remaining local acceptance: real browser screenshot/download testing (browser automation unavailable here). Remaining research acceptance: expert visual/routing labels, GPU F-GMVAE-T training, process/video alignment, sealed evaluation set, and online RSI runtime activation.

## 2026-11-09 Media Boundaries And Evaluation Integrity

- P1.1: media normalization accepts configurable `max_media_bytes` (4 GiB default) across `MediaNormalizer`, `normalize_media`, and `build_media_manifest`. Oversized input is rejected before content hashing or decoder startup. The manifest explicitly marks content hash as not computed, leaves content SHA empty, and carries a separate metadata identity fingerprint.
- G1: strengthened the bad-quality image integration test with a diagnostic `LLMGateway.chat` spy and explicit `visual.weights_status=not_called`; the test proves that neither the learned visual model nor the diagnostic provider runs after the quality gate selects retake/human. Focused integration tests: `2 passed`.
- Multi-file manifests retain duplicate rows while annotating repeated content and its first occurrence. Tests cover duplicate images, oversized sparse video, unsupported extension, corrupt MP4, and the quality-gate retake path.
- P5 ablation overrides fail closed unless metrics are non-empty finite numbers; provenance includes data version/split/seed/model/timestamp, a positive paired sample count, and existing paired-data/baseline/variant artifacts with matching SHA-256 digests. Invalid evidence remains `structure_only`.
- P5 multi-Agent lineage includes completed non-abstaining outputs with no sources as missing lineage; explicit abstentions remain outside that denominator.
- The regenerated two-video report initially exposed missing process-snapshot references. The Process State specialist now derives a stable `process-snapshot-<run_id>` source for each available run snapshot; `run_diagnosis_graph` injects the task run/thread IDs into the specialist context. This cites the independent prior snapshot and does not claim video alignment.
- Focused validation: media + ablation + evaluation tests `22 passed, 7 subtests`; case lifecycle tests `4 passed`.
- Re-ran `正常.mp4` and `偏高.mp4` in an isolated runtime directory, then evaluated explicit run IDs `demo-c5c2b4ad64ad` and `demo-331e4c5ff694`. Result: 10/10 specialist slots completed, 2 explicit case abstentions, 8 non-abstaining source lineage checks, 0 missing references, 0 violations, 100% lineage integrity. Outputs are in `runtime_data/evaluation_integrity_runs/`; visual remains `structure_validation_only`, and route fixture scores remain non-claimable model metrics.
- Regenerated nine-row ablation report at `runtime_data/evaluation_integrity_runs/ablation_report.json`: 0 claimable, 9 `structure_only`.
- Final validation: `pytest agentic/tests -q` -> `145 passed, 1 warning, 31 subtests in 31.12s`; `compileall -q agentic web_interface` and `git diff --check` passed. The warning is the existing PyTorch Transformer nested-tensor warning.
- Remaining evidence gates: expert labels for visual/route claims, GPU F-GMVAE-T training, video/process alignment, a valid sealed evaluation set, genuine paired ablation outputs, live RSI runtime publication, and browser screenshot/download QA. The evaluator checks evidence integrity; it does not create missing experiments.

## 2026-11-09 Laya-JEV Backend Provenance And Delivery Snapshot

- `JevRouteDecision` now has an enum-validated `router_backend`: `deterministic_rule`, `laya_adapter`, `laya_adapter_fallback`, or `safety_fallback`. Route trace records backend, configured router profile, gate version, confidence and reasons. Tests verify backend truth for rule-only, accepted adapter, low-confidence and exception cases, plus trace parity and invalid enum rejection.
- Laya is pinned as an optional dependency in `agentic/requirements.txt` (`>=0.4.29,<0.5`); the active environment reports `0.4.29`.
- Corrected the execution plan asset path to `ppt_asset_manifest.json` and corrected stale status checkboxes for GPU F-GMVAE-T training, expert Laya route fit/calibration and dataset freezing: implementation interfaces exist, but those outcome gates remain unchecked/external.
- Re-ran route fixtures: `5/5 passed`. Re-ran `正常.mp4` and `偏高.mp4` under `runtime_data/evaluation_integrity_runs/latest/`, then evaluated the explicit run IDs `demo-61326ebc28b7` and `demo-0577997fb67d`. Both reached `full_loop` and `waiting_review`; recorded backend was `deterministic_rule`. Across 2 runs, all 10 workers completed; two expected case abstentions occurred; all 8 non-abstaining source links were present with no violations. Visual remains `structure_validation_only`; route fixture metrics are not claimable model accuracy; all 9 ablations remain structure-only.
- Revalidated PPT assets: `validate_assets` reports `42/42` files valid and manifest coverage `25/25` pages. The 25 generated page PNGs are structural illustration assets labeled `flow_validation`, not browser screenshots.
- Updated the current execution-plan facts, blocked research checkboxes, audit P7 rows, and P0 resolution status. Older implementation-log entries remain chronological snapshots; later dated entries supersede their counts.
- Focused router/contract/graph suite: `24 passed`. Final full-suite result and compile/diff verification are recorded below after rerunning all tests.
- Final verification: `pytest agentic/tests -q` -> `146 passed, 1 warning, 31 subtests` (warning: existing PyTorch Transformer nested-tensor setting); `compileall -q agentic web_interface` and `git diff --check` passed.

## 2026-11-09 Structured Gateway And Presentation Demo Completion

- Added a model-facing `DiagnosisDraftPayload` contract separate from the run-bound `DiagnosisDraft`. The aggregator and provider smoke pass the same payload schema; the application adds run/thread IDs, evidence references and model metadata after generation.
- Ollama `/api/chat` now receives the output JSON Schema. OpenAI-compatible profiles use `json_object` by default, with opt-in `structured_output_mode=json_schema`; both paths still validate locally. Invalid schema output produces a non-retried `schema_error`; refusal, truncated finish, empty choices, and content-part arrays are handled explicitly. Fake provider now generates a payload conforming to the requested schema.
- The presentation-only review flow persists under `MARS_AGENT_RUNTIME_DIR/presentation_demo`, restores by run ID after a fresh Streamlit session, appends a single audit event, and creates isolated candidate-only memory for approved/amended eligible fixture runs. Reject/retake does not create a case.
- The RAG display resolves `source_id` only against the packaged `rules_v1.jsonl` knowledge base and displays title, source/version/date, applicability, preconditions, risk and text. It does not open source paths or locators from model/runtime records.
- Focused validation: Gateway, graph, presentation, and knowledge suites `29 passed, 2 subtests`; presentation Streamlit AppTest `5 passed`.
- Full validation: `pytest agentic/tests -q` -> `154 passed, 1 warning, 33 subtests`; `compileall -q agentic web_interface` and `git diff --check` passed. Warning remains the existing Transformer nested-tensor setting.
- Re-generated `runtime_data/evaluation_integrity_runs/latest/evaluation_report.explicit_after_gateway.json` with explicit run IDs `demo-61326ebc28b7` and `demo-0577997fb67d`: both matched; worker slots `10/10`, source lineage `8/8`, route fixtures `5/5`; visual status `structure_validation_only`; ablation status `0 claimable, 9 structure_only`. PPT assets validated `42/42`.
- A full-loop `fake_json` video run `demo-78f6417d46a0` completed payload validation and recorded a human approve action. Empty fake citations correctly caused deterministic `request_more_evidence`, so this run is a Gateway/schema smoke, not diagnosis validation.
- Remaining external/research checks: real Ollama/Qwen provider calls, Qwen API key, expert video labels, GPU F-GMVAE-T training, expert Laya route labels, process/video alignment, sealed evaluation, online RSI activation and browser screenshots.

## 2026-11-09 Optional Laya Runtime Selection And Explicit E2E Evaluation

- Added runtime selection for `deterministic_rule` and `laya_adapter` in `demo_run`, `agentic.run`, and the Streamlit modern-agent workbench. The Laya adapter explicitly calls the installed Laya `typed-decisions` checkpoint; construction is lazy and does not download weights until an inference request.
- The deterministic policy remains the first safety gate. Laya is consulted only when the policy selects `full_loop`; hard safety conditions reject a downgrade to `simple_loop`, and adapter errors take the human safety fallback. Trace/result metadata separates configured backend, actual backend, model profile, and gate version.
- The graph reuses a cached route only when its serialized feature snapshot, `run_id`, and `thread_id` match the current task. This prevents the demo pre-route and graph route node from invoking an optional model twice. Stale or invalid cached decisions are ignored.
- Added an explicit four-action histogram to system evaluation reports: `approve`, `amend`, `reject`, and `request_retake`.
- Ran both repository videos in a new isolated runtime with model-free visual smoke, `fake_json` aggregation, and the deterministic JEV backend. Both runs reached `full_loop` and `waiting_review`; five specialist slots were present/completed in both runs (10/10), and eight non-abstaining source-lineage checks passed (8/8). These runs validate flow, not visual classification or live Laya inference.
- Explicit-run evaluation artifact: `runtime_data/jev_wiring_e2e/evaluation_report.json`. It records two selected run IDs, latency `n=2` (P50 1757.186 ms, P95 1807.003 ms for this local run), JEV policy fixtures `5/5`, visual `structure_validation_only`, no human decision in the selected cohort, and nine ablations still `structure_only`.
- Validation: full suite `pytest agentic/tests -q` -> `158 passed, 1 warning, 33 subtests`. `compileall -q agentic web_interface`, `agentic.run --help`, route fixtures (`5/5`), and `git diff --check` passed. The warning is the existing PyTorch Transformer nested-tensor warning.
- Remaining evidence gates: actual Laya checkpoint inference is not claimed until weights are available and a live request is run; expert video labels, GPU F-GMVAE-T training, aligned process/video data, reviewed frozen evaluation sets, paired ablations, and browser screenshot QA remain open.
- Included the existing independent Process State test report and validation attribution in the latest dashboard-loadable evaluation: `runtime_data/evaluation.e2_20261109_jev_wiring.json`. It remains `experimental_baseline`, with `alignment_quality=unknown`; CNN ROC-AUC 0.542, PR-AUC 0.588, event recall 0.88 on 827 test windows describe future process-risk labels, not video classification or superheat measurement. The 8 selected process variables are validation attribution candidates and are not causal.
- Final full suite after route-fixture/UI metadata corrections: `159 passed, 1 warning, 33 subtests`. `compileall -q agentic web_interface` and `git diff --check` passed; Streamlit `/_stcore/health` returned `200 ok`.

## 2026-11-09 RSI Runtime Policy 接入与评测更新

- 新增 `agentic/platform/jev/policy.py` 和 `agentic/rsi/runtime_policy.py`。现代诊断只在显式设置 `MARS_AGENT_RSI_REGISTRY` 或 CLI `--jev-policy-registry` 时读取 RSI active snapshot；未设置时用内置 `base`。解析器校验版本内容 hash、发布 candidate/diff、完整 gate 和人工 approval，再验证 policy schema。拒绝 demo registry、未审批版本、未知字段、非有限值及越界值，并回退至内置策略。
- 当前首个运行时可变项仅为 `jev_policies.minimum_laya_confidence`，取值范围 `[0.7, 1.0]`，只影响可选 Laya adapter 的建议置信阈值。确定性质量/风险/证据安全门、C/N/H 视觉识别和人工授权不受 RSI 修改。
- `Publisher.publish()` 在激活版本前验证 JEV policy schema。JEV 决策和现代 run JSON 记录 policy version/hash；Streamlit 状态栏展示 RSI 版本及解析状态。`run_all` 新增 `rsi_runtime_policy` 轨迹审计，只统计已完成运行并区分未完成 start 记录，明确标记 accuracy/behavior gain 不可声明。
- 测试覆盖 active publish → 下一个版本生效 → rollback 恢复阈值、run_demo 读取并记录 active 版本、未配置/未发布时 base fallback、缺少人工 approval 和篡改 registry 时 fail-closed、越界 threshold 发布拒绝及硬安全门不可降级。
- 验证：该阶段代码第一次全量 `pytest agentic/tests -q` -> `165 passed, 1 warning, 33 subtests`；新增 P5 测试后的最新全量结果记录于下方 Final Regression。RSI/JEV/CLI/Streamlit 定向测试 `46 passed, 4 subtests`；`compileall -q agentic web_interface`、`git diff --check`、route fixture `5/5` 通过。
- 两段真实视频重新运行到隔离 runtime `runtime_data/rsi_runtime_e2e/`，run IDs 为 `rsi-route-normal` 和 `rsi-route-high`。显式 cohort 报告显示 2/2 完成运行携带 `base/builtin` 策略版本；full-loop worker 10/10 完成、来源链 8/8 完整。五条路由 fixture `5/5`，流程耗时本次 `n=2`、P50 `1848.172 ms`、P95 `1922.602 ms`，仅为流程验证。本地冻结评测仍为 `frozen=false`，六份研究评测 manifest 均未通过。
- 交付边界：RSI 的 JEV 阈值运行时切换和回滚由发布/gate 测试验证；真实专家路由收益尚未验证。当前没有把 RSI 接入任意 prompt、知识查询、视觉权重或安全规则，也没有宣称效果提升。真实视频标签、Laya 权重/专家路由集、GPU 训练、过程-视频映射和 sealed 专家评测仍待外部条件。

## 2026-11-09 P5 冻结样本与 Process State 多种子补充

- `freeze_evaluation` 为六类 manifest 增加稳定身份字段检查：visual `window_id`、process `sample_id/sample_idx/window_id`、route/HITL `case_id`、knowledge/case query `query_id`。同一主键重复会标记 `duplicate_identity` 并按无效行计数，不能用重复记录凑足最低数量。
- 当前 `evaluation_freeze.json` 仍返回 `frozen=false`，六份清单均未通过；新增校验没有改变任何样本标签或放宽专家标注要求。
- `train_process` 现在将训练输出与固定测试索引上的逐样本预测 CSV 一起保存，并通过同一指标函数报告 ROC-AUC、PR-AUC、Brier/ECE、事件召回和误报段数。仅在测试集算一次指标，不用测试标签做阈值拟合；预测保留 `sample_idx/y_true/y_prob/anchor_ts` 供独立复算。
- 新增 `agentic.scripts.evaluate_process_repeats`，默认对 CNN 使用 seeds `7,17,27,37,47`、CPU、1 epoch；多种子汇总要求相同 dataset SHA-256 与 purged test-index hash，否则拒绝合并。评测输出写 JSON 和 CSV，并接入 `evaluation.run_all --process-repeats`。
- 实验命令：`python -m agentic.scripts.evaluate_process_repeats --dataset data/v1/step2_ew_202111/dataset.npz --split data/v1/step2_ew_202111/split.json --output-dir runtime_data/process_repeats_e2_20261109 --model cnn --epochs 1 --batch-size 128 --device cpu --seeds 7 17 27 37 47`。
- 实际结果：5 个 seed、同一日期净化 test `n=827`；ROC-AUC `0.3984 ± 0.0059`，PR-AUC `0.4645 ± 0.0080`，event recall `0.9520 ± 0.0299`，预测阳性率 `0.8675 ± 0.0830`，point precision/recall/F1 `0.5201/0.8419/0.6416`，specificity `0.1029 ± 0.0665`，误报段 `0.20 ± 0.40`，Brier `0.2565 ± 0.0020`，ECE `0.0502 ± 0.0224`（均值 ± population std）。事件召回偏高与 86.7% 预测阳性率、低 specificity 并存，应解释为固定阈值下的告警覆盖/误报取舍；只对应独立过程未来风险标签，不能解释为视频分类、物理过热度测量或多模态融合收益。
- 工件：`runtime_data/process_repeats_e2_20261109/repeated_process_evaluation.json`、`repeated_process_evaluation_metrics.csv`、每 seed 的 checkpoint、逐样本 test prediction CSV 和训练报告。
- 定向测试：过程模型、重复实验和评测汇总 `28 passed, 3 subtests`；冻结脚本/RSI/JEV 相关已有定向测试通过。最终全量回归在下一条记录。

## 2026-11-09 本轮最终回归

- 最新 `pytest agentic/tests -q`：`170 passed, 1 warning, 33 subtests`。唯一 warning 是既有 Transformer nested-tensor 配置提示。
- `compileall -q agentic web_interface`、`git diff --check` 通过；JEV policy fixture `5/5`；PPT 资产 `42/42`。
- 报告 `runtime_data/evaluation.e2_20261109_complete.json` 包含显式双视频 cohort、JEV policy fixtures、多 Agent 完整性、RSI policy version audit 和五 seed Process State 报告；同目录生成 Markdown/CSV/失败样例/摘要 PNG。
- 冻结校验工件 `runtime_data/evaluation_freeze.e2_20261109.json` 为 `frozen=false`，六项研究 manifest 均不合格；这是当前专家标注与数据协议的真实状态。
- Process State 指标还输出固定阈值下的混淆计数、point Precision/Recall/F1、Specificity 和 predicted-positive rate。五种子结果里 event recall `0.952` 与阳性预测率 `0.867`、specificity `0.103` 同时出现，应视为偏宽告警阈值，不作为效果优势。
- 重新生成的 process predictions 使用 ISO 时间戳，保留 sample_idx 与时间窗口，可被现有过程评测器按固定 split 复算。Streamlit health `http://127.0.0.1:8503/_stcore/health` 返回 `ok`。该部分仅对应过程未来风险。

## 2026-11-09 P1.3 复核边界与 P4.3 报告一致性

- P1.3 审计发现：有效引用会掩盖对 `12 C` 的漏检，且非空占位字符串可错误满足实测条件。本轮扩大温度/过热度单位表达检测，并要求温度、成分、初晶温度三项均解析为有限数值；空来源注册表不能验证草案的自带引用。补充英文升压/加料控制表达测试和 fallback 人工路由测试。
- P1.3 验证：`pytest agentic/tests/test_p13_review_guard.py -q` -> `12 passed, 6 subtests`。有效来源的反例直接触发过热度拦截，NaN/Infinity 和缺失引用不能偶然通过。
- Streamlit 现代诊断的模型/知识版本区域现在显示 `review_guard_version`，与模型、提示词和知识版本一起回放。
- P4.3 审计发现人工 amend 后 Markdown/PNG 仍读取原始 draft。报告对象现提供 `final_diagnosis`，保留 `draft` 原文；Markdown 显示最终诊断和原始审计草案，PNG 使用最终诊断并优先检测系统中文字体。JSON/HTML 同样携带最终诊断。
- P4.3 验证：`pytest agentic/tests/test_p13_review_guard.py agentic/tests/test_p4_hitl.py agentic/tests/test_streamlit_native_hitl.py -q` -> `19 passed, 15 subtests`；修改后的摘要逐项覆盖 JSON、Markdown、HTML 和 PNG 绘制文本。
- 历史页现已并列加载现代 HITL 审核审计，支持 run_id、路由、审核结果和日期筛选，并提供选中 run 的 JSON/Markdown/HTML 报告下载；旧业务记录仍保留在原历史表中。页面级隔离 runtime 测试通过。
- 仍未完成：候选案例与最终报告的全字段一致性、review guard 版本在网页显示、真实浏览器截图/下载验收。
- 最终全量验证：`pytest agentic/tests -q` -> `177 passed, 1 warning, 39 subtests`；唯一 warning 是 `agentic/models/temporal.py` 的既有 PyTorch Transformer nested-tensor 配置提醒。`compileall -q agentic web_interface` 与 `git diff --check` 通过。

## 2026-11-09 P3.2/P3.3 引用和案例一致性

- RAG UI 改为使用真实 `citation.locator`，并提供安全的当前知识版本原文片段下载；不会把检索文本中的任意路径交给浏览器执行。
- 新增无依赖 `VersionedKnowledgeBase.search_bm25()`，知识 Agent、demo 初始检索和 10 条知识 fixture 评测切换到 BM25；返回结构保留 source/version/locator，定向检索回归 `10 passed`。
- 特殊案例存储新增 append-only 候选结论更新，检索按 case ID 取最新快照。现代 demo 人工 approve/amend 后同步候选诊断到 HITL 最终诊断，reject/request_retake 将暂存候选标记为 deprecated；presentation fixture 同样读取图结果诊断。
- 回归：P3 案例快照更新、demo native amend/reject/retake、一致性和历史页 UI 入口均通过；`test_streamlit_native_hitl.py` 当前 `2 passed, 4 subtests`。
- 真实专家相关性、BM25/BGE-M3/Reranker/Milvus 和浏览器级原文下载仍需外部数据或浏览器环境。
- 最终全量回归：`pytest agentic/tests -q` -> `179 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface` 与 `git diff --check` 通过。唯一 warning 为既有 Transformer nested-tensor 配置提示。

## 2026-11-09 协议边界一致性

- 统一 `ProcessStateSnapshot.process_missing` 为列表边界；诊断聚合器缺少过程快照时使用 `["process_snapshot"]`，新增布尔值拒绝测试。该修正只保证交接协议一致，不改变过程数据与视频之间的 `alignment_quality=unknown` 事实。
## 2026-11-10 RSI 受控提示词与知识查询运行时消费

- 扩展 `agentic/platform/jev/policy.py` 与 `agentic/rsi/runtime_policy.py`：RSI active snapshot 现在可声明 `prompts.variant` 和 `knowledge_queries.extra_terms`，但仅允许代码白名单提示词变体（`diagnosis-aggregator-v2`、`diagnosis-evidence-first-v1`）及最多 4 个、每项不超过 40 个字符的检索附加词。
- `demo_run` 在显式 registry 通过内容 hash、candidate、GateResult 和人工 approval 校验后，将附加词接入初始 BM25 与 Knowledge Agent 查询，将变体接入聚合器上下文，并记录 `prompt_version`、`rsi_runtime` 和运行时版本 hash。聚合器仍保留固定安全约束，RSI 不能注入任意 prompt 正文。
- 历史自由文本 prompt candidate 保持离线 lineage 兼容；运行时白名单解析失败时 fail-closed 使用 base。版本 hash 覆盖 JEV/查询词/提示词三类运行面，rollback 后一并恢复。
- 新增多 surface publish/consume、非法 prompt/query fail-closed 和 demo 消费回归；现有 P6 回归仍通过。
- 验证：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q` -> `182 passed, 1 warning, 39 subtests`；warning 为既有 Transformer nested-tensor 配置提示。`compileall -q agentic web_interface` 与 `git diff --check` 通过。
- 结论边界：本次证明配置接线、版本追踪和回滚，不证明提示词/RAG 质量提升；真实缺陷轨迹、sealed 专家集、真实 Laya checkpoint 行为和浏览器视觉 QA 仍待外部条件。
- `agentic.scripts.demo_run --help` 现提供 `--jev-policy-registry`，与 `agentic.run` 的同名参数一致；显式 registry 是运行时启用 RSI surface 的唯一 CLI 入口，避免自动发现离线 defect registry。
- 新增 `data/evaluation/route_annotation_template.jsonl`，固定专家路由标注协议字段并显式标记 `is_template=true`；它不会被当作研究标签或冻结样本，后续人工标注需复制后填写真实 case/source/reviewer。

- 本轮附加回归：`route_smoke --fixtures data/evaluation/route_manifest.jsonl --output runtime_data/route_smoke.e3_20261110.json --no-execute` -> `5/5 passed`；`freeze_evaluation` -> `frozen=false`，六类研究清单仍因专家审核标签/数量/字段不足而拒绝冻结；这与新增路由模板的 `is_template=true` 约束一致。
- 多 surface RSI 双视频演示使用显式 run IDs `rsi-surface-normal`、`rsi-surface-high`，active version `v-e378032a0506488e` 实际消费 `diagnosis-evidence-first-v1` 和 `thermal trend`；报告 `runtime_data/evaluation.e3_20261110_rsi_surface.json` 显式 cohort 为 `claimable=true`，2/2 带同一 policy hash、提示词变体和查询词集合。该报告仍将 `accuracy_claimable=false`、`behavior_gain_claimable=false`。
- 新增 `agentic.scripts.validate_delivery_snapshot`，对选定 run IDs 的完成记录、JEV fixture、显式评测 scope、runtime 输入完整性、HITL 决策和 RSI 版本进行 fail-closed 交付审计。`runtime_data/delivery_snapshot.e3_20261110.json` 已验证双视频 + `5/5` fixture + RSI active 版本，未要求时允许 HITL 仍处于待审核状态。
- 交付脚本回归通过；全量测试更新为 `188 passed, 1 warning, 39 subtests`。

## 2026-11-10 路由流转与评测页可见性

- 现代图结果新增 `initial_route`、`route_history` 和 `upgrade_count`；`evaluation.run_all` 增加 `route_flow` 审计，统计 simple/full/retake/human 的初始/最终路径、升级次数和流转率。该指标标记 `accuracy_claimable=false`、`efficiency_gain_claimable=false`。
- 双视频新 cohort `runtime_data/evaluation.e3_20261110_routeflow.json` 使用 run IDs `route-flow-normal`、`route-flow-high`，两条实际记录均为 `retake -> retake`、升级 0；这是当前质量门控结果，不能替代简单/复杂任务的专家路由收益评测。
- Streamlit 评测页现在自动发现 `evaluation.e3_*.json`，显示 `input_integrity`、route flow 和最新 delivery snapshot；这让坏 runtime 记录、评测范围和交付证据在页面可见。

## 2026-11-10 P5 评测输入状态审计

- `agentic/evaluation/run_all.py` 新增 `_load_optional_report`：缺失的视觉训练、Process State 独立评测、特征归因和多种子报告现在统一输出 `status=not_run`、`report_path` 和 `reason=file_missing`；坏 JSON 或非对象输出 `status=invalid_report`，不再用空对象掩盖缺失证据。
- 新增 P5 回归覆盖缺失和损坏可选报告；定向测试：`pytest agentic/tests/test_p5_evaluation.py -q` -> `15 passed`。
- 该修正只改善评测状态可读性和 fail-closed 口径，不增加任何算法指标。

## 2026-11-10 RSI 运行时证据前端展示

- 现代 Streamlit 工作台的“模型、知识、提示词与 Ticket 版本”区新增 `rsi_runtime` 展示，包含 status、version_id、policy_hash、prompt_variant 和 knowledge_query_terms，便于汇报和回放时确认实际消费的策略版本。
- AppTest 解析 `st.json` 版本面板并断言其与 demo run 的 RSI 运行时记录一致；`pytest agentic/tests/test_streamlit_native_hitl.py -q` -> `2 passed, 4 subtests`。
- 该展示只证明运行时配置可追溯，不代表 RSI 带来模型质量或效率提升。

## 2026-11-10 旧版入口边界标识

- `agentic.workflow` 和 `agentic.run --mode legacy` 结果现在带 `flow_version=legacy-single-frame` 与 `claims`，明确其未使用现代多帧视觉、Process State、JEV 路由、原生 LangGraph HITL 或 RSI runtime。
- 新增回归 `test_legacy_result_is_explicitly_excluded_from_modern_claims`；现代申报主链路仍由 `agentic.run --mode modern` 进入 `demo_run -> diagnosis_graph`。
- 全量验证：`pytest agentic/tests -q` -> `185 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface` 与 `git diff --check` 通过。

## 2026-11-10 P5 runtime JSONL 输入完整性

- `evaluation.run_all` 对 `events.jsonl`、`runs.jsonl` 和 `hitl_audit.jsonl` 收集坏 JSON/非对象行号；报告增加 `input_integrity.status` 与 `runtime_jsonl_parse_errors`，出现解析错误时 `claimable=false`，不静默丢弃记录后计算时延或流程指标。
- 新增回归验证坏 runtime 行 fail-closed；P5 定向测试 `16 passed`。该改动只提升评测输入完整性，不产生新算法结果。
- 最终显式 cohort 评测 `runtime_data/evaluation.e3_20261110_final.json`：run IDs `rsi-surface-normal`、`rsi-surface-high` 均匹配，runtime 输入完整性 `passed/claimable=true`；RSI 版本/哈希、提示词变体和查询词 2/2 一致，`accuracy_claimable=false`、`behavior_gain_claimable=false`。JEV fixture `5/5`，视觉状态仍为 `structure_validation_only`。

## 2026-11-10 simple_loop 升级回归与评测页验收

- 新增诊断图端到端回归：让有效的 `simple_loop` 任务携带缺少 `source_id` 的视觉上下文，实际确定性复核产生 `missing_traceable_source_citation`，LangGraph 将任务升级到 `full_loop`，保留 `initial_route=simple_loop`、`route_history=[simple_loop, full_loop]` 和 `upgrade_count=1`。
- 回归还验证升级后五类 worker 均执行；聚合结果因来源缺失仍进入 `request_more_evidence`，任务保留人工复核状态。该样例是合成的流程边界验证，不是视频分类结果或路由效果指标。
- Streamlit AppTest 新增 e3 评测页验收：读取 `evaluation.e3_*.json` 中 route flow 和输入完整性，并展示 `delivery_snapshot*.json` 的 run 与 JEV fixture 摘要。
- 验证命令：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests/test_diagnosis_graph.py -q` -> `10 passed`；评测页定向用例 -> `1 passed`；全量 `pytest agentic/tests -q` -> `190 passed, 1 warning, 39 subtests`。warning 仍为 PyTorch Transformer nested-tensor 提示；`compileall -q agentic web_interface` 与 `git diff --check` 通过。
- 真实视频 cohort 仍为两次 `retake -> retake`、升级 0。route flow 字段能审计真实轨迹和合成回归，但不代表 simple/full 效率收益；视觉标签、Laya 专家路由集、冻结评测与 GPU 训练仍待补齐。

## 2026-11-11 P2.4 worker 策略配置与路由事实校准

- 新增 `config/worker_policies.json`：每个 full-loop worker 可单独配置 timeout、重试上限、指数退避初值/上限和可重试异常类型；环境变量及任务上下文提供有限覆盖，所有值在执行边界内裁剪。
- AgentResult 元数据记录 `retry_policy_id`、`max_retries`、`retryable_errors`、退避参数、实际退避序列、timeout 和 attempt。测试覆盖配置文件策略加载、RuntimeError 限定重试、ValueError 不重试，以及上一轮的 timeout/失败结果保留。
- 该实现采用 LangGraph `Send` 并行 fan-out 和本地 worker 执行边界；没有直接挂载 LangGraph RetryPolicy，因为 worker 会将失败转成成功返回的结构化 AgentResult，图层无法看到原始异常。超时通过 `_cancel_event` 协作通知，不能强行终止无协作的同步阻塞代码。
- 修正逐项核对报告中两项过期陈述：`router_backend` 已进入协议/trace/UI；路由标注模板已存在，但没有专家标签。报告保留真实 Laya 推理仍待验证的状态。
- 验证：定向 `test_diagnosis_graph.py` 与 `test_presentation_demo.py` -> `18 passed`；全量 `pytest agentic/tests -q` -> `192 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface` 和 `git diff --check` 通过。唯一 warning 为 PyTorch Transformer nested-tensor 设置提醒。

## 2026-11-11 P2.4 worker 策略 E2E 与中文路径修复

- Windows 下补齐中文/非 ASCII 图片路径读取：新增统一 `read_image`，在 Unicode 路径走字节解码，并接入媒体质量、视觉序列、现代/兼容诊断和窗口构建入口。新增 Unicode 路径回归覆盖 manifest、质量门控与单图视觉序列。
- 在隔离 runtime 对真实目录中的 `cold_127.jpg` 运行现代 demo：图片质量通过，确定性 JEV 选择 `full_loop`，五个 worker 均完成并进入 `waiting_review`；每个 AgentResult 带独立 `retry_policy_id` 和实际 attempt。
- 对显式 run ID `p2-4-worker-policy-cold-127-unicode` 重建评测：`route_flow` 记录 1 次 `full_loop -> full_loop`；五个 worker present/completed `5/5`，failed/missing `0/0`，来源链检查 `4/4`、缺失/违规均为 0。该结果只验证一次运行和 worker 交接，不是路由准确率、诊断准确率或效率收益。
- 评测工件：`runtime_data/p2_4_worker_policy_e2e/evaluation.p2_4_worker_policy.json` 及同名前缀 Markdown/CSV/失败样例/PNG；运行 trace：`runtime_data/p2_4_worker_policy_e2e/demo_unicode_fixed.json`。
- 验证：全量 `pytest agentic/tests -q` -> `193 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface`、`git diff --check` 通过。唯一 warning 是现有 Transformer nested-tensor 配置提示。
- 后续：将 `simple_loop -> full_loop` 升级样例加入 PPT 演示目录；专家标注、真实 Laya 路由、冻结评测与 GPU 训练仍未完成。

## 2026-11-11 P2.3/P2.4/P5/P7 演示与执行评测补齐

- `presentation_demo` 新增第五个 `simple_upgrade` 场景。它直接运行 `run_diagnosis_graph`，保存诊断图完整 trace 与结果，断言 `route_history=[simple_loop, full_loop]`、`upgrade_count=1` 和五个 specialist 都到位；canonical 目录已重建，仍保留 `flow_validation` 标记。
- Streamlit 演示页载入该场景后显示循环历史和升级次数；Worker 超时/重试展开表显示 Agent 策略 ID、超时、attempt、重试次数、退避和错误。评测页新增 `worker_execution` 摘要。
- `evaluation.metrics.worker_execution_metrics` 和 `run_all` 统计策略覆盖、策略 ID 分布、重试、失败与超时；合成单测验证重试成功和超时失败计数，准确率/效率收益声明保持 false。单图真实目录样本 run `p2-4-worker-policy-cold-127-unicode` 的报告统计 worker `5`、策略覆盖 `100%`、重试/失败/超时为 `0`。
- `validate_assets` 现在解码 PNG/JPEG/WebP、比对 sidecar 实际像素尺寸，并验证每页 `image_assets`/`supporting_image_assets` 引用存在且可解码。修正两张 contact sheet sidecar 的旧尺寸后，验证结果 `42/42` 有效、25 页引用完整、235 个引用均已解析。
- 重建 `runtime_data/presentation_demo/` 和 PPT 资产 manifest：五个可加载场景（四路路由 + 一个 simple-to-full 升级）；25 页仍标为流程验证资产。
- 验证：全量 `pytest agentic/tests -q` -> `199 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface`、`git diff --check`、资产 validator 均通过；Streamlit `http://127.0.0.1:8503/_stcore/health` 返回 `ok`。
- 未完成：真实浏览器截图/下载验收仍受浏览器自动化环境限制；视觉专家标签、真实 Laya fit/calibration、冻结研究集与 GPU F-GMVAE-T 不在本次流程测试结论内。

## 2026-11-11 P5 冻结状态汇总与 P3 引用完整性

- `evaluation.run_all` 现在把 `freeze_evaluation.freeze(root)` 的结果并入总报告，逐份展示研究 manifest 状态、记录数、有效记录数、最小记录数、SHA-256 和校验错误，并汇总缺失/无效清单。
- 总评测区分 `frozen`、`missing_manifests`、`invalid_manifests` 和 `incomplete`。冻结只表示评测数据达到结构/来源门槛；总报告的研究指标 `claimable` 保持 false，直到结果确实在冻结清单上运行并具备对应运行来源。
- 知识检索回归增加引用机械完整性审计：source ID 必须能解析到知识包；来源、版本、有效日期、适用范围、locator 和检索分数必须与原规则一致。语义支持率显式保持 `not_evaluated_without_human_entailment_labels`。
- 定向验收：冻结/检索相关测试 `27 passed`，覆盖全缺失清单、含无效清单、结构完整临时清单（仍不声称指标可用），以及合法/篡改/未知来源引用。
- 全量验收：`pytest agentic/tests -q` -> `207 passed, 1 warning, 39 subtests`；`git diff --check` 通过。唯一 warning 为现有 PyTorch Transformer nested-tensor 配置提示。

## 2026-11-11 P3.1/P4.3/P5 最终集成验收

- 新增知识包校验命令 `python -m agentic.scripts.validate_knowledge_pack`：对原始 JSONL fail-closed 检查格式、重复 ID、必需字段、日期、风险等级和适用范围。当前知识包为 3/3 结构有效、0 条无效；报告标明 `claimable=false`。
- `request_retake` 前端现在要求替代媒体，并验证父任务持久化的审核决定和 audit；新建 child run，task、run/runtime result 和子审核 audit 保存 `parent_run_id` / `retake_of_run_id`，父 run 与审核行保持原状。历史筛选新增 model profile。
- 总评测集成研究冻结检查；当前 6 项清单 `incomplete`（`case_queries.jsonl` 缺失，5 份无效），并在 UI 显示各项记录/有效数量、错误、SHA 和冻结资格。10 条检索 fixture 的 citation field integrity 为 `10/10`；语义支持率仍需人工蕴含标签。
- 评测产物：`runtime_data/evaluation.e3_20261111_gap_fill.json`；冻结结果 `incomplete`，JEV fixture replay 与质量门控合成回归通过，视觉 `structure_validation_only`，未指定 run IDs 的历史 runtime 不可声明。
- `evaluation.run_all` 与 Streamlit 评测表增加 `knowledge_pack_validation` 和引用字段完整性行；当前汇总显示知识包 3/3 条结构有效、知识引用 fixture 10/10 字段完整，并明确不代表专家语义审核。
- 最终验证：`D:\py_daima\.venvs\mars-agent\Scripts\python.exe -m pytest agentic/tests -q` -> `209 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface`、`git diff --check`、`validate_assets`（42/42）通过。唯一 warning 为既有 Transformer nested-tensor 配置提示。
- 研究待办保持 blocked_external：专家视觉/路由标签、真实 Laya fit/calibration、GPU F-GMVAE-T 训练、合格的六类冻结清单和视频—过程数据可靠映射。

## 2026-11-11 演示评测补齐与浏览器视觉验收

- JEV policy fixture 汇总现在分别计算 `simple_loop`、`full_loop`、`retake`、`human` 召回；加入回归确保四种路径各自统计。五个固定 fixture 的路径匹配率及四路召回均为 `1.0`，结果仍标记为 fixture regression，`model_accuracy_claimable=false`。
- 新评测工件：`runtime_data/evaluation.e3_20261111_final.json` 及 `.md`、`_metrics.csv`、`_failures.jsonl`、`_summary.png`。质量门控合成失败样本 5/5 检出；视觉仍为 `structure_validation_only`；知识包结构 3/3、检索引用机械完整 10/10；冻结研究集 `incomplete`（`case_queries.jsonl` 缺失，其余五份无效）；runtime scope 混合历史、`claimable=false`。
- 五个固定场景实际执行 LangGraph route smoke，结果全部命中预期路由；`stable_case` 初始选择 simple，证据复核缺口触发升级并完整执行 full-loop workers。轨迹见 `runtime_data/route_smoke.e3_20261111_final.json`。
- Streamlit 评测表新增 simple/full/retake/human 四路召回行；AppTest 验证四个 100% 值以 policy fixture 性质显示，并标记为不可声明模型效果。
- Streamlit 诊断区将内部 process regime `demo_summary` 显示为“演示摘要”，路由流程状态显示为“待人工确认”；AppTest 检查展示标签。浏览器已验证桌面 1440×1100 和手机 390×844 页面无水平溢出，见 `runtime_data/streamlit_agent_status_desktop_localized_updated.png`、`runtime_data/streamlit_agent_status_mobile_390_localized_updated.png`。浏览器级审核交互、刷新恢复和下载仍未完成端到端验收。
- 最终验证：`pytest agentic/tests -q` -> `210 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface`、`git diff --check`、资产校验 42/42 通过；Streamlit health 返回 `ok`。warning 为既有 Transformer nested-tensor 配置提醒。

## 2026-11-13 发布整理与评测产物复核

- 发布边界整理：根目录新增项目入口 README、`Code/data/README.md` 和 `Code/docs/发布整理说明.md`；原始视频、`data/v1`、火眼帧目录、运行时 JSONL、课件压缩包和 PPT/Word 二进制被加入忽略规则，评测 manifest、知识包、代码和 PPT 资产 sidecar 保留在版本库。
- P5 冻结审计补充 `invalid_error_counts` 与代表性错误行，当前 `evaluation_freeze.status=incomplete` 仍明确记录 `case_queries.jsonl` 缺失以及视觉/过程/路由/HITL/知识清单缺少专家审核字段；该状态不会被演示 fixture 覆盖。
- P3 检索报告 CSV 改用标准 `csv.writer`，嵌套指标值可安全 round-trip；最新检索产物 10 条知识查询和 10 条案例查询的机械引用字段完整性为 `10/10`，语义支持仍需人工蕴含标签。
- Streamlit 评测表统一中文列名和状态标签；8504 实例已完成桌面 1440×1100 与手机 390×844 视觉验收，JEV 四路 fixture 行可见且各为 `100%` policy fixture regression。该结果不构成专家路由准确率。
- 最终发布回归：`pytest agentic/tests -q` -> `212 passed, 1 warning, 39 subtests`；`compileall -q agentic web_interface` 通过；`validate_assets` 为 `42/42`；`route_smoke` 为 `5/5`；总评测 `runtime_data/evaluation.publish.json` 生成成功并保持 `visual=structure_validation_only`、`evaluation_freeze=incomplete`、`claimable=false`；`git diff --check` 随后通过。
