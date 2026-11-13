# 铝电解火眼多智能体 LLM 增强与 Laya-JEV 路由升级方案

> 适用 PPT：`铝电解火眼多智能体诊断系统_章节重组版.pptx`
>
> 当前 PPT 为 34 张 slide。本文只规划新增和替换内容，不把实验方案提前表述为已经取得的指标。

## 1. 当前短板与升级目标

当前系统已经具备以下 LLM 运行骨架：

- `diagnosis_graph.py` 可以把视觉、知识、案例和过程状态压缩为结构化上下文，交给 `LLMGateway` 生成 `DiagnosisDraft`；
- Qwen-VL/文本模型可以通过本地 Ollama 或兼容 API 配置；
- `review_guard.py` 已经能够检查来源、数值过热度的实测依据和控制指令；
- `LayaAdapter`、`JevRouter` 和确定性安全规则已经形成 Laya-JEV 路由骨架；
- `rsi/evaluator.py`、`gate.py`、`publisher.py` 已经具备离线回放、质量/安全/效率闸门、人工批准、发布和回滚能力。

目前仍缺少三块能在汇报中清楚体现“大模型创新”的内容：

1. 没有面向铝电解诊断 JSON、证据引用和现场建议的 QLoRA/SFT 训练模块；
2. 聚合模型后面还没有独立的 LLM 质疑/Verifier Agent，现有复核主要是确定性规则；
3. JEV 的路由置信度、校准和学习型适配没有在 PPT 中展示清楚，容易被误解为简单阈值分流。

目标版本增加一条可执行链路：

```text
多源证据包
  → LLM 诊断聚合 Agent
  → 独立 Verifier/质疑 Agent
  → 证据回补与有限再规划
  → 确定性安全复核
  → 人工确认
```

同时把 Laya-JEV 解释为：

```text
低成本特征
  → 确定性硬安全门
  → Laya 候选路由与 raw confidence
  → 温度校准
  → 安全覆盖
  → simple_loop / full_loop / retake / human
```

这里的“思维链”采用可审计的证据步骤摘要，不保存或展示模型私有隐藏思维链。每一个结论步骤必须绑定 `EvidenceRef`，对外输出 `rationale_summary`、`evidence_ids` 和 `uncertainty`。

## 2. 参考研究与开源工程

### 2.1 可迁移的研究思路

| 来源 | 可借鉴思想 | 在本项目中的落地 |
|---|---|---|
| [LoRA](https://arxiv.org/abs/2106.09685) | 冻结主模型，只训练低秩适配器 | 聚合 Agent 和 Verifier 分别维护领域 LoRA，减少显存和训练成本 |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 4-bit NF4 基础模型 + LoRA 微调 | 首版用 3B/7B 级 Qwen-VL 或文本模型做本地领域 SFT |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 生成、检索、批评交替进行 | 将“生成草案→证据质疑→回补证据”改造成最多两轮的受控循环 |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | 用过程级验证器检查中间推理步骤 | 不验证自由 CoT，而验证“结论主张→EvidenceRef”的支持关系 |
| [Corrective RAG](https://arxiv.org/abs/2401.15884) | 检索结果先评估，再采取补检索或修正动作 | Verifier 输出 `missing_source_ids`，只回补缺失的视觉、知识或案例 Agent |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 冷启动 SFT 后再用可验证奖励做强化学习 | 第一阶段采用 SFT；只有证据奖励稳定后才试 DPO/ORPO/GRPO |
| [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) | 视觉语言模型处理图像、视频和文档 | 强模型负责复杂视觉理解，领域训练聚焦结构化诊断输出和证据对齐 |

### 2.2 可复用的 Laya-JEV 工程思路

[Jev Model Router](https://github.com/zerone-agents/jev-model-router) 的工程启发是：模型能力卡、可编辑路由提示、结构化路由记录、离线固定集回放和模型失败 fallback 应该分开治理。它是通用文本模型路由器，不能直接当作火眼分类器。

本项目已经在 `Code/agentic/platform/jev/` 中完成领域适配：

- `features.py` 归一化图像质量、视觉置信度、跨帧一致性、新颖度、风险、知识覆盖和案例相似度；
- `rules.py::safety_route` 执行四路硬规则；
- `laya_adapter.py` 调用 `Router.predict(state, questions=...)`，读取 `answers.route.choice` 与置信度，并执行温度校准；
- `router.py::JevRouter` 只有在基础规则判定为 `full_loop` 时才接受 Laya 候选，且 `confidence >= 0.7`；硬安全条件禁止候选把任务降级到 `simple_loop`，异常或 abstain 进入 `human`。

## 3. 诊断聚合 Agent：从提示词调用升级为领域后训练模型

### 3.1 输入和输出

聚合 Agent 不直接接收没有组织的长文本，而是接收统一的 `EvidenceBundle`：

```json
{
  "visual_evidence": {
    "class_probs": {"C": 0.12, "N": 0.70, "H": 0.18},
    "uncertainty": 0.21,
    "frame_consistency": 0.88,
    "evidence_ids": ["ve-001", "ve-004"]
  },
  "process_state": {
    "process_regime": "transition",
    "event_risk": 0.48,
    "alignment_quality": "unknown"
  },
  "knowledge_hits": [{"source_id": "rule-4.3", "version": "v2"}],
  "case_matches": [{"case_id": "case-017", "similarity": 0.91}],
  "route_decision": {"route": "full_loop", "confidence": 0.78}
}
```

模型输出固定为 `DiagnosisDraft`，最少包含：

```json
{
  "diagnosis_class": "C|N|H|unknown",
  "risk_level": "low|medium|high|unknown",
  "findings": [],
  "evidence_claims": [
    {"claim": "...", "evidence_ids": ["ve-001", "rule-4.3"]}
  ],
  "inspection_items": [],
  "recommendations": [],
  "uncertainty": 0.0,
  "open_questions": [],
  "source_ids": []
}
```

`evidence_claims` 是结构化推理摘要，不是自由 CoT。每条 claim 必须能回指 `EvidenceRef`；没有温度、成分和初晶温度实测依据时，模型只能输出状态研判和检查建议，不能生成数值过热度。

### 3.2 聚合 Agent 的模型和后训练方案

建议采用“强教师 + 本地适配器 + 结构化约束”的组合：

1. **教师生成**：用较强 Qwen-VL API 或本地大模型根据证据包生成初稿、证据映射和不确定性说明；
2. **专家修订**：领域人员只修订诊断等级、依据、风险和现场检查项，保留 `source_id`；
3. **QLoRA/SFT**：冻结 Qwen-VL 主模型，使用 4-bit NF4 加载，训练聚合 Agent 的 LoRA；优先训练语言层和输出结构，视觉 backbone 首版冻结；
4. **结构化损失**：JSON 字段交叉熵 + schema 解析成功奖励 + EvidenceRef 存在性约束 + claim/evidence 支持关系奖励；
5. **偏好优化**：将“专家修改后的草案”作为 chosen，将证据缺失、越权动作或引用错误的草案作为 rejected，后续使用 DPO/ORPO；
6. **蒸馏**：将强 Qwen API 的短证据摘要蒸馏给本地小模型，减少 full_loop 的调用成本。

第一阶段不建议直接使用 GRPO。GRPO 只有在 `evidence_support`、`schema_valid`、`safety_pass` 等奖励可以稳定自动计算时才进入实验，否则容易把格式投机当成诊断能力。

推荐的首轮适配器配置是：基础模型 4-bit NF4，LoRA rank 从 16/32 起步，优先挂载 `q_proj/k_proj/v_proj/o_proj`，视觉编码器冻结；聚合 Agent 和 Verifier 使用不同 adapter 和不同 system prompt。若显存不足，Verifier 先使用文本 Qwen 小模型，只接收结构化证据与 ROI 描述，不重复读取整段视频。若只能调用云端 API，则用教师蒸馏和领域 prompt 版本化替代本地权重微调。

建议新增文件：

```text
Code/agentic/training/build_trace_dataset.py
Code/agentic/training/sft_diagnosis.py
Code/agentic/training/sft_verifier.py
Code/agentic/agents/critic_agent.py
Code/agentic/platform/evidence_graph.py
```

### 3.3 训练样本格式

所有样本按 `task_id/video_id/scene_id` 分组切分，禁止把同一视频的相邻帧随机分到 train 和 test。推荐保留数据来源和标签阶段：`weak`、`expert_review`、`expert_gold`。

```json
{
  "task_id": "T-001",
  "evidence_bundle": "...",
  "route_decision": "full_loop",
  "draft_before_review": "...",
  "expert_draft": "...",
  "rationale_summary": [
    {"step": "visual_observation", "evidence_ids": ["ve-001"]},
    {"step": "risk_assessment", "evidence_ids": ["rule-4.3", "case-017"]}
  ],
  "label_stage": "expert_gold"
}
```

没有真实专家诊断轨迹时，先用规则、知识手册和已有人工结论生成弱标注，再人工抽样修订。弱标注可用于预热，申报中的训练效果只能使用 `expert_gold` 子集报告。

## 4. 证据质疑/Verifier Agent

### 4.1 职责分离

Verifier 不读取聚合模型隐藏思维链，也不重新生成一篇长报告。它只读取：

```text
DiagnosisDraft + EvidenceRef 列表 + 已知来源清单 + 任务安全策略
```

输出：

```json
{
  "verdict": "pass|amend|request_more_evidence|human_review",
  "support_score": 0.0,
  "missing_source_ids": [],
  "contradictions": [],
  "numeric_input_ok": true,
  "action_authorization": "allowed|human_required|blocked",
  "issues": []
}
```

### 4.2 双层复核和有限回补

```text
aggregate_llm
  ↓
critic_llm（支持关系、冲突、缺失来源、表达边界）
  ↓
review_guard（确定性来源/数值/控制指令检查）
  ├─ pass → human_review
  ├─ request_more_evidence → evidence_repair_router
  │      → 只回补缺失的视觉/知识/案例/过程 Agent
  │      → aggregate_llm（最多 2 轮）
  └─ human_review → 人工确认
```

Verifier 训练分两步：

- SFT：使用“正确通过、应补证、应驳回、应人工”的标注样本；
- 偏好优化：构造“有来源支持的草案 > 语言流畅但无来源的草案”“保守请求补证 > 虚构数值过热度”的偏好对。

高风险动作由确定性规则优先拦截；Verifier 失败、超时或输出无法解析时，进入人工复核，不允许静默通过。

### 4.3 受控证据推理伪代码

```python
bundle = collect_evidence(task)
draft = aggregate_llm(bundle, schema=DiagnosisDraft)
for round_id in range(2):
    critique = critic_llm(draft, bundle, schema=VerifierResult)
    guard = review_guard(draft, bundle)
    if guard.pass_ and critique.verdict == "pass":
        break
    if critique.verdict != "request_more_evidence":
        return human_review(draft, critique, guard)
    bundle = repair_missing_evidence(critique.missing_source_ids, task)
    draft = aggregate_llm(bundle, schema=DiagnosisDraft)
return human_review(draft, critique, guard)
```

这里的 `range(2)` 是工程上的最大回补轮数，不是让模型无限自我反思。系统日志只记录 `round_id`、问题码、来源 ID、模型版本和耗时。

## 5. Laya-JEV 如何快速得到置信度并分流

### 5.1 低成本特征层

JEV 前置阶段不调用完整诊断 LLM，而是复用已经得到的轻量证据：

| 特征组 | 示例 |
|---|---|
| 质量 | 清晰度、曝光、遮挡、目标存在、ROI 覆盖 |
| 视觉置信 | C/N/H 最大概率、归一化熵、关键帧一致性、时序一致性 |
| 新颖度 | embedding 距离、energy/OOD 分数、与已审核案例的距离 |
| 证据覆盖 | 知识命中数、有效来源数、案例适用条件、来源冲突 |
| 风险 | 视觉风险、过程事件风险、控制动作请求、结论冲突 |
| 上下文 | 过程 `alignment_quality`、任务模式、历史改判和补拍记录 |

得到的置信度有两个不同含义：

- `visual_confidence`：视觉模型对 C/N/H 观察结果的置信度；
- `route_confidence`：Laya-JEV 对诊断工作量的置信度。

二者不能混用。JEV 不输出火眼类别，也不把 route confidence 解释为摄氏度过热度精度。

### 5.2 当前规则和 Laya 候选的顺序

当前 `safety_route` 的顺序应在 PPT 中完整画出：

```text
retake / decoder_error → retake
quality=human → human
target_missing → retake
novelty ≥ 0.85 或 visual_confidence < 0.45 → human
risk ≥ 0.75 或 evidence_conflict ≥ 0.45 → full_loop
knowledge_coverage < 0.34 → full_loop
case_similarity ≥ 0.88 且案例适用且风险 < 0.45 → simple_loop
visual_confidence ≥ 0.82 且案例相似度 ≥ 0.70 且低风险 → simple_loop
其他 → full_loop
```

只有基础规则已经判定为 `full_loop` 时，`LayaAdapter` 才提出候选路径：

```python
features = build_inspection_features(evidence)
base_route, reasons = safety_route(features)

if base_route == "full_loop":
    proposal = laya.predict(features.to_dict())
    raw_conf = proposal.get("raw_confidence", proposal.get("confidence", 0.0))
    proposal_conf = apply_temperature(raw_conf, calibration.temperature)
    if proposal.get("route") in ROUTES and proposal_conf >= 0.70:
        route = safety_gate(proposal.get("route"), features)
    else:
        route = base_route
else:
    route = base_route

return route, calibrated_confidence(route), reasons
```

`calibrated_confidence` 使用冻结 `route_manifest.jsonl` 做温度缩放，并报告 Brier score/ECE。Laya 发生异常、abstain 或输出非法值时转 `human`；硬门控禁止任何候选把高风险任务降级为 `simple_loop`。

### 5.3 Laya-JEV 的训练与评测

训练顺序：

1. 规则弱标注生成 `retake/simple/full/human` 候选；
2. 领域专家复核 `expert_route` 和 `route_reason`；
3. 按 `video_id/scene_id/task_id` 分组切分，训练/校准 Laya adapter；
4. 用温度校准或 isotonic regression 校准 route confidence；
5. 可选训练小型路由头/轻量分类器，规则作为最终否决层；
6. 通过冻结集回放后才接入在线流程。

必须报告：四路混淆矩阵、`simple_loop` 精确率、危险样本 simple 漏放率、复杂任务 full-loop 召回率、retake/human 召回率、ECE、Brier、P95 路由延迟和 Token 节省。平均延迟下降只有在诊断质量和安全指标不恶化时才成立。

## 6. 与现有代码的落地边界

| 模块 | 当前状态 | 本方案新增 |
|---|---|---|
| `diagnosis_graph.py` 聚合 | Qwen Gateway + `DiagnosisDraft` 结构化聚合 | `critic_llm`、`evidence_repair_router`、最多两轮回补 |
| `review_guard.py` | 确定性来源、数值和控制指令复核 | 与 Verifier 合并为双层复核，规则保留最终安全边界 |
| `platform/jev` | Laya adapter、温度校准、硬规则和四路输出 | 专家 route manifest、轻量适配训练、校准评测 |
| `models/fgmvae_temporal.py` | FGMVAE-T 结构/推理骨架 | 真实视频窗口标签训练和不确定性校准 |
| `agents/process_state.py` | 14 变量独立过程状态 Agent | 作为 EvidenceBundle 上下文输入，不静默对齐视频 |
| `rsi/` | 离线回放、闸门、人工发布和回滚 | 管理 prompt、Verifier、JEV policy 和阈值候选版本 |
| `agentic/training/` | 视觉、过程和 FGMVAE 训练脚本 | `build_trace_dataset.py`、`sft_diagnosis.py`、`sft_verifier.py` |

阶段划分：

```text
E2.1 证据轨迹数据集与 JSON Schema
E2.2 聚合 Agent QLoRA/SFT
E2.3 Verifier SFT + 确定性双层复核
E2.4 Laya-JEV route manifest + 温度校准
E2.5 证据回补循环接入 LangGraph
E2.6 DPO/ORPO 偏好优化与冻结集回放
E2.7 具备稳定可验证奖励后，再评估 GRPO
```

## 7. PPT 页位调整方案

当前文件有 34 张 slide，建议不增加页数，直接替换或扩展以下页面：

| 实际 slide | 现有内容 | 新内容 |
|---:|---|---|
| 19 | 模糊推理网络/多模态算法延伸 | 改为“领域 LLM 后训练与受控推理”，保留多模态论文公式到备份页 |
| 23 | 4.1 JEV 四路诊断编排 | 增加“低成本特征 → raw confidence → 温度校准 → 硬安全门 → 四路” |
| 24 | 4.2 JEV 规则基线与轻量模型 | 增加 route manifest、分组切分、校准、危险漏放率和延迟评测 |
| 29 | 4.7 诊断聚合与证据复核 | 改为“诊断聚合 Agent + 证据质疑 Agent”双栏协同图 |
| 32 | 5.1 RSI 受控离线演化 | 增加 LLM prompt、聚合 LoRA、Verifier 阈值和 JEV policy 的候选发布链 |

## 8. 图片生成提示词

以下提示词用于生成版式示意图。图片模型容易把中文和代码渲染错误，最终 PPT 应将代码和关键中文替换为 PowerPoint 可编辑文本。所有提示词都沿用当前模板：16:9、白底、大号红色标题、深蓝双横线、黄色强调条、浅蓝结论横幅、紫色描边内容框、深蓝正文、少量红色强调，不使用深色科技 UI、渐变背景、巨型机器人或营销海报风格。

### Prompt 1：slide 19，领域 LLM 后训练与受控推理

```text
16:9 Chinese academic defense PPT slide, strictly match an existing aluminium electrolysis thesis deck template: white background, large red title at the top, navy double horizontal rule with a yellow accent, pale blue rounded claim banner, purple outline panels, dark navy body text, red highlights, restrained flat academic style, no gradients, no 3D robot, no dark dashboard, no marketing poster.

Title: “3.2 领域LLM后训练与受控推理”
Claim banner: “领域数据让诊断大模型学习证据引用、结构化研判和安全拒答”

Create a clean left-to-right pipeline with four compact panels:
1. “领域轨迹数据” containing expert diagnosis records, visual evidence, 14-variable process state, process handbook, approved cases, verifier corrections;
2. “QLoRA / LoRA SFT” with a frozen Qwen-VL backbone and a small orange adapter block;
3. “证据结构化推理” with observe, evidence mapping, risk, action, uncertainty, JSON schema;
4. “Verifier偏好优化” with chosen expert-corrected draft, rejected unsupported draft, DPO/ORPO, frozen replay.

On the right, show two model cards: “诊断聚合 Agent” outputs DiagnosisDraft and “证据质疑 Agent” outputs pass / amend / request_more_evidence / human_review. Draw a feedback arrow from verifier to evidence repair and then back to aggregation, maximum two repair rounds. Bottom strip: “对外输出：rationale_summary · evidence_ids · uncertainty；不展示私有隐藏思维链”. Add a small orange label “E2实验方案 / 目标扩展”. Keep Chinese labels short and large; no fake metric numbers.
```

### Prompt 2：slide 23，Laya-JEV 置信度路由

```text
16:9 Chinese academic defense PPT slide in the same existing deck template: white background, large red title, navy double rule, yellow accent, pale blue claim banner, purple and blue outlined boxes, flat technical diagram, no gradients, no dark UI, no decorative robot.

Title: “4.1 Laya-JEV置信度路由”
Claim banner: “JEV选择诊断工作量，不输出C/N/H分类”

Draw a horizontal five-stage pipeline:
1. “低成本特征” with six compact chips: image quality, visual max probability and entropy, temporal consistency, OOD novelty, case similarity, knowledge coverage and risk;
2. “规则硬门” with red priority labels retake, human, target missing, high risk;
3. “Laya Router” showing a route choice question and raw confidence;
4. “温度校准” showing route_manifest, temperature scaling, Brier/ECE;
5. “安全覆盖” as a diamond that blocks unsafe downgrade.

After the diamond show four colored outputs: simple_loop, full_loop, retake, human. Add a red fallback arrow labelled “abstain / invalid / adapter error → human”. Include a small editable JSON card with route, confidence, reasons, gate_version. Use a small note: “visual_confidence ≠ route_confidence”. Make the routing order visually clear: deterministic safety gate first, Laya proposal only as an optional candidate.
```

### Prompt 3：slide 24，JEV 路由训练与评测

```text
16:9 Chinese academic defense PPT slide matching the existing thesis template: white background, red title, navy double line and yellow accent, pale blue claim banner, purple outline panels, clean academic flat design, no gradients, no dark dashboard, no invented numbers.

Title: “4.2 JEV规则基线、校准与轻量适配”
Claim banner: “先用可解释规则建立安全基线，再用冻结运行记录校准和学习路径边界”

Three-column layout:
Left column “规则基线”: quality gate, confidence threshold, risk/conflict threshold, approved case match, four routes.
Center column “数据与训练”: route_manifest JSONL, weak labels → expert_route, split by video/task/scene instead of random frames, Laya fit or lightweight adapter, temperature scaling.
Right column “评测”: four-route confusion matrix placeholder, dangerous simple-loop leakage rate, full-loop recall, retake/human recall, ECE, Brier, P95 routing latency, token savings. Use empty chart axes or neutral placeholders, never fabricate values.

Draw a red safety arrow from the right side back to “hard gate”. Add orange label “规则已实现；专家路由训练为E2扩展”. Keep each panel to 3–5 short lines.
```

### Prompt 4：slide 29，诊断聚合 Agent 与证据质疑 Agent

```text
16:9 Chinese academic defense PPT slide in the exact style of the existing deck: white background, red title, navy double horizontal rule with yellow accent, pale blue claim banner, purple and blue outline panels, high readability, flat academic diagram, no gradients, no 3D robot, no dense paragraphs.

Title: “4.7 诊断聚合Agent与证据质疑Agent”
Claim banner: “聚合模型形成研判草案，独立质疑模型检查证据支持关系”

Use a balanced two-column layout.
Left purple panel titled “Evidence Bundle → 诊断聚合 Agent”: five input cards VisualEvidence, ProcessStateSnapshot, KnowledgeHit, CaseMemory, JevRouteDecision flow into an orange “Qwen-VL Aggregator” card. Output a white JSON card labelled DiagnosisDraft with diagnosis_class, risk_level, evidence_claims, inspection_items, uncertainty.

Right red-purple panel titled “Verifier / 证据质疑 Agent”: input draft plus EvidenceRef list; five compact checks source validity, claim support, contradiction, numeric superheat inputs, action authorization. Output four pills pass, amend, request_more_evidence, human_review.

Draw a thin loop arrow from request_more_evidence to an “evidence repair router”, then back to only the missing visual/knowledge/case/process Agent. Mark “最多2轮回补 → 人工确认”. Bottom strip: “只保存 rationale_summary 与 evidence_ids，不保存私有隐藏思维链”. Add a small editable pseudo-code block of five lines, not an IDE screenshot.
```

### Prompt 5：slide 32，RSI 管理 LLM 和 JEV 候选版本

```text
16:9 Chinese academic defense PPT using the same existing template: white background, large red title, navy double rule, yellow accent, pale blue banner, purple dashed governance panel, flat academic style, no gradients, no dark UI, no marketing language.

Title: “5.1 RSI：LLM与JEV策略的受控离线演化”
Claim banner: “运行反馈产生候选版本，冻结评测和人工审批决定是否发布”

Draw a governance loop: audit logs and expert feedback → candidate generation for prompt / aggregator LoRA / verifier threshold / JEV policy → train/frozen/sealed replay sets → quality, safety, evidence and efficiency gates → expert approval → versioned registry → rollback and next-batch observation.

Use a red lock icon or label “offline only” at the publication gate. Add a small side note “禁止在线自动修改安全规则、人工授权边界和评测集”. Show three small version chips prompt_v, verifier_v, jev_policy_v. Bottom strip: “候选、评测、审批、发布、回滚均可审计”. No metric numbers and no claims of autonomous self-improvement in production.
```

## 9. 汇报时的统一表述

建议在 PPT 和答辩中统一使用以下状态语言：

- “现有运行骨架”：`diagnosis_graph.py`、`LayaAdapter`、确定性 `review_guard`、RSI 离线发布链已经存在并可回放；
- “E2 实验方案”：聚合 Agent QLoRA、Verifier SFT/偏好优化、证据回补循环、专家路由拟合和在线图接入；
- “尚需数据”：真实专家诊断轨迹、专家 route manifest、视频窗口级标签和稳定的证据支持标注；
- “不宣称”：不把弱标注、教师生成、图像级训练或非配对过程数据写成真实过热度测量指标。

最终要让听众看到的不是“增加了一个大模型”，而是：

```text
领域轨迹训练 → 结构化诊断聚合 → 独立证据质疑 → 有限回补 → 人工授权
                         ↑
                 Laya-JEV 先决定诊断工作量
```
