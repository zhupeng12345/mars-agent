# 铝电解火眼多智能体诊断系统：25页汇报大纲

> 目标：从项目自身的业务问题出发，完整讲清楚“视频证据 → 多帧视觉取证 → 过程状态先验 → JEV诊断编排 → 多 Agent 协同 → 人工确认 → RSI 受控演化”的闭环。
>
> 页面状态标记：`[现有实现]` 表示当前仓库已有可演示代码；`[目标实现]` 表示按本项目架构需要补齐的代码；`[基线数据]` 表示可直接引用的 `Code/data/v1` 结果。

> **2026-11-09 演示版口径**：四路 canonical trace、五类 full_loop Agent、25 张结构验证图、10+10 演示查询、RSI 离线卡片和 Streamlit 演示模式已经具备，统一标记为 `flow_validation`。它们用于讲清系统角色和运行路径，不代表视觉精度、Laya 专家路由准确率、视频-过程融合增益或真实过热度测量。PPT 中将能力分为“演示验证”“现有工程模块”“论文级待优化”三类，不再把演示素材写成模型效果。

## 叙事主线

```text
现场巡检任务
  → 代表帧/短视频采集
  → 图像质量与视觉取证
  → 多帧视觉分类 + 过程状态先验
  → JEV 判断诊断工作量
  → simple_loop 或 full_loop
  → 证据聚合与复核
  → 人工最终确认
  → 报告归档与特殊案例沉淀
  → RSI 离线评估、冻结、发布
```

## 25页逐页设计

### 第 1 页：项目封面

- **标题**：铝电解火眼视觉-知识多智能体诊断系统
- **副标题**：多帧视觉取证 · 过程状态先验 · JEV诊断编排 · Agent协同 · 人工闭环 · RSI受控演化
- **核心信息**：面向铝电解现场火眼巡检的证据驱动智能诊断系统。
- **版式**：左侧标题，右侧放火眼 ROI、Agent 流程和铝电解槽现场照片的组合视觉；底部放项目团队、日期、版本号。
- **图片素材**：`Code/data/火眼分类图片帧` 中抽取三类代表图，叠加 ROI 框和证据标签。
- **讲解重点**：系统的输出是可追溯的诊断意见、风险等级、复核问题和现场行动建议。

### 第 2 页：铝电解火眼巡检场景

- **要回答的问题**：现场人员面对什么信息，系统要帮助完成什么判断。
- **内容**：巡检视频/火眼图片、火眼区域、视觉异常征兆、工艺知识手册、现场复核事项、报告归档。
- **视觉布局**：用一张现场照片做背景，覆盖“采集—识别—研判—复核—处置建议”五个动作节点。
- **关键文案**：`从一段巡检视频形成一条有证据、有来源、有复核记录的诊断任务。`
- **数据说明**：当前主证据为短视频多帧或代表帧；过程变量作为工况上下文与先验输入。

### 第 3 页：诊断任务与系统输出

- **输入**：巡检视频/图片、任务说明、模型配置、可选过程变量窗口。
- **中间证据**：图像质量标签、ROI、目标类别、时序特征、过程状态、知识片段、相似案例。
- **输出**：火眼状态分类、风险等级、诊断依据、知识来源、现场检查建议、人工确认结果。
- **建议图形**：左侧输入卡片，中间证据链，右侧结构化报告卡片；报告卡片展示 `diagnosis / risk / evidence / actions / review` 五个字段。
- **结构化协议**：对应 `InspectionTask`、`EvidenceRef`、`AgentResult`、`ReviewDecision`。

### 第 4 页：系统目标与核心创新

- **四个目标**：视觉证据可靠、诊断路径可分流、Agent 协作可回放、结论进入人工闭环。
- **五项创新**：
  1. YOLO ROI 与多帧时序取证结合；
  2. F-GMVAE-T 融合静态/动态视觉特征与专家先验；
  3. Process State Agent 将非配对过程数据转化为工况上下文；
  4. JEV 按任务难度编排 `simple_loop/full_loop/retake/human`；
  5. 特殊案例记忆与 RSI 离线评估形成受控演化闭环。
- **建议图形**：五个创新点围绕“证据驱动多智能体诊断”中心圆环。
- **讲解重点**：JEV 管理诊断工作量，视觉模型负责识别，Agent 负责取证、检索、聚合和复核。

### 第 5 页：项目数据与证据源

- **视觉数据**：`Code/data/火眼分类图片帧`；视频抽帧、代表帧、ROI 标注、类别标签。
- **过程数据**：`Code/data/v1`；14 个过程变量，30 步历史窗口，2 分钟采样间隔，形成 60 分钟工况上下文。
- **知识数据**：`Code/agentic/knowledge` 下的过热度定义、工艺规则、SOP、风险边界。
- **案例数据**：已人工确认的任务、证据、结论、复核意见和版本信息。
- **建议图形**：四列数据源汇入统一 `InspectionTask`，每列标注“证据类型”和“进入 Agent”。
- **引用**：`Code/data/v1/README*`、`Code/agentic/knowledge/superheat_definition.md`。

### 第 6 页：任务数据组织与标注协议

- **任务对象**：`task_id`、`media_uri`、`frame_ids`、`roi`、`process_window`、`requested_mode`、`trace_id`。
- **视觉标注**：火眼 ROI、类别、可见征兆、图像质量、帧级置信度、视频级聚合标签。
- **过程状态标注**：14 个变量的窗口统计、趋势、波动、缺失率、工况状态标签。
- **诊断标注**：专家结论、风险级别、证据引用、处置建议、人工动作。
- **建议图形**：一条任务 JSON 记录拆成“媒体、视觉、过程、知识、复核”五个字段组。
- **代码位置**：`Code/agentic/platform/contracts.py` 中的协议类型和 `workflow.py` 的 `DiagnosisState`。

### 第 7 页：多智能体总体架构

- **主链路**：任务接入 → 质量门控 → 视觉取证 → JEV 编排 → 诊断循环 → 人工确认 → 报告归档。
- **协作层**：视觉复核 Agent、工艺知识 RAG Agent、特殊案例 Agent、Process State Agent 并行提供证据。
- **决策层**：诊断聚合 Agent、证据质疑/复核 Agent。
- **治理层**：统一协议、记忆、审计、模型网关、评测、RSI 发布控制。
- **建议图形**：采用三层架构图，不放长段落；每个 Agent 用“输入→输出”短标签。
- **必须突出**：Agent 名称、数据流方向、JEV 分流、人工确认、案例回流五类视觉元素。

### 第 8 页：在线任务状态机

- **状态**：`received → quality_check → visual_evidence → route_decision → diagnosis → evidence_review → human_review → archived`。
- **分支**：
  - `quality_check → request_retake`：质量问题触发补拍；
  - `route_decision → simple_loop`：证据充分、风险可控、相似案例明确；
  - `route_decision → full_loop`：证据冲突、风险较高、案例稀缺或不确定性高；
  - `route_decision → human`：直接进入人工复核。
- **恢复机制**：人工确认后可 `approve / amend / reject / request_retake`，节点事件写入 `trace_id`。
- **建议图形**：状态机节点用圆角矩形，JEV 分流用菱形，人工节点用红色边框。
- **代码位置**：`Code/agentic/orchestration/diagnosis_graph.py` 的 `build_diagnosis_graph` 已实现独立条件路由和可选人工 interrupt；旧 `workflow.py` 仍保留固定线性流程。代码展示和集成状态见 [PPT关键代码展示方案.md](./PPT关键代码展示方案.md)。

### 第 9 页：多 Agent 角色矩阵

用一页表格说明“谁负责什么”，每个 Agent 只保留输入、输出、实现方式三列。

| Agent | 输入 | 输出 | 实现方式 |
|---|---|---|---|
| 任务接入/安全门控 | 媒体、说明、配置 | 任务协议、策略档位 | 规则 + 协议 |
| 图像质量 Agent | 原始帧 | 清晰度、遮挡、曝光、分辨率 | OpenCV/质量模型 |
| 视觉取证 Agent | 合格帧/视频 | ROI、类别、置信度、时序特征 | YOLO + F-GMVAE-T |
| Process State Agent | 14变量窗口 | 工况状态、趋势、异常摘要 | 时序编码器 + 规则 |
| JEV 编排 Agent | 证据覆盖、风险、相似度 | 四路诊断路径 | 规则 + 轻量模型 |
| 知识 RAG Agent | 诊断问题 | 规则、来源、适用范围 | 向量检索 + 重排 |
| 案例记忆 Agent | 视觉/工况特征 | 已审核相似案例 | 版本化案例库 |
| 诊断聚合 Agent | 多源证据 | 结构化研判草案 | Qwen-VL/文本 LLM |
| 证据复核 Agent | 草案、证据链 | 冲突、缺口、越界项 | 规则 + 独立模型 |
| 人工确认/归档 Agent | 草案、复核问题 | 最终决策、报告、候选案例 | UI + 审计存储 |

- **建议图形**：矩阵右侧增加“串联 / 并行 / 回流”标识，体现角色之间的协作关系。
- **讲解重点**：视觉模型、工具型 Agent、LLM Agent 和人工节点共同构成完整系统。

### 第 10 页：短视频多帧视觉取证 Agent

- **流程**：视频解码 → 关键帧采样 → YOLO 逐帧 ROI → 跟踪/对齐 → 质量筛选 → 视频级证据聚合。
- **视觉输出**：`roi_sequence`、`frame_scores`、`class_probs`、`motion_features`、`evidence_refs`。
- **图形**：横向时间轴放 6 个抽取帧，每帧显示 ROI 框；下方连接一条“置信度/亮度/纹理”曲线。
- **代码片段**：展示目标实现的简化接口。

```python
def extract_visual_evidence(video_uri: str, sample_fps: float = 2.0):
    frames = sample_frames(video_uri, sample_fps)
    rois = yolo_detector.track(frames, target="fire_eye")
    aligned = align_roi_sequence(frames, rois)
    quality = quality_gate(aligned)
    return build_evidence(aligned, quality)
```

- **代码状态**：YOLO/代表帧入口已有；多帧跟踪、对齐和视频级聚合标记为 `[目标实现]`。
- **讲解重点**：每一帧都是可回看的证据，视频级结论来自帧级证据聚合。

### 第 11 页：静态、动态视觉特征与专家先验

- **静态特征**：ROI 亮度、色温、纹理、边缘、火焰形态、可见区域比例。
- **动态特征**：亮度变化、边缘位移、火焰面积变化、闪烁频率、帧间差分。
- **专家先验**：火眼状态定义、过热度征兆、类别边界和风险规则。
- **融合方式**：`static_encoder + temporal_encoder + expert_prior → fused_embedding → class_head`。
- **建议图形**：三路特征输入同一融合模块，右侧输出类别概率、证据摘要和不确定性。
- **讲解重点**：静态图像回答“看起来是什么”，动态序列回答“变化方式是什么”，专家先验约束“如何解释”。

### 第 12 页：F-GMVAE-T 多帧视觉分类器

- **方法名**：F-GMVAE-T（模糊标签高斯混合变分自编码器 + 时序编码）。
- **结构**：
  1. ConvNeXt-Tiny 或轻量 ViT 编码单帧 ROI；
  2. Temporal Transformer/TCN 编码帧序列；
  3. 高斯混合先验建模多模态状态；
  4. 模糊标签损失处理专家边界样本；
  5. 分类头输出 C/N/H 状态与置信度。
- **非配对过程数据用法**：过程变量不作为逐帧标签拼接，而是经过 Process State Agent 形成工况先验或训练期辅助任务。
- **建议图形**：`ROI序列 → 空间编码 → 时序编码 → GMVAE潜变量 → 分类/不确定性`。
- **代码骨架**：

```python
class FGMVAETemporal(nn.Module):
    def forward(self, roi_seq, expert_prior=None):
        z_frame = self.spatial_encoder(roi_seq)
        z_time = self.temporal_encoder(z_frame)
        mu, logvar, mix = self.gmvae(z_time, prior=expert_prior)
        return self.class_head(self.sample(mu, logvar)), {"mix": mix}
```

- **评测指标**：Macro-F1、风险召回率、ECE、视频级稳定性、帧数消融。

### 第 13 页：Process State Agent

- **定位**：将 14 个过程变量的历史窗口转成可解释工况上下文，供 JEV、RAG、案例检索和诊断聚合使用。
- **输入**：`X_seq=(5744, 30, 14)`，30 步 × 2 分钟，对应约 60 分钟历史窗口；可选 `X_stat=(5744, 70)`。
- **输出**：趋势、波动、变化点、缺失率、工况状态、变量重要性和 `process_evidence`。
- **变量分组**：电气变量、动态变量、加料变量、设备状态变量。
- **建议图形**：14 个变量按四组汇入时序编码器，再输出“稳定 / 波动 / 变化 / 数据质量”四类状态卡。
- **基线数据**：未来 15 步（约 30 分钟）事件风险标签用于过程分支的辅助评测。

```python
def process_state_agent(window_14x30):
    stats = summarize_window(window_14x30)
    z = temporal_encoder(window_14x30)
    state = state_head(z, stats)
    return {"state": state, "summary": stats, "evidence": make_refs(stats)}
```

### 第 14 页：过程状态先验进入诊断链路

- **运行时链路**：Process State Agent → JEV 路由证据、工艺知识检索查询、案例相似度特征、诊断聚合上下文。
- **训练时链路**：过程数据用于辅助状态学习、先验校准和风险评测；视觉任务仍以视频/图片证据作为主要观测。
- **四个作用**：
  - 为视觉结论补充工况背景；
  - 为 JEV 提供风险和不确定性信号；
  - 为 RAG 生成更精确的工艺问题；
  - 为案例检索提供时间窗口摘要。
- **建议图形**：过程数据分支用虚线表示，分别连接 JEV、RAG、案例 Agent、诊断聚合 Agent。
- **关键文案**：`非配对过程数据作为可解释先验与上下文证据，进入协作链路而非伪造帧级对应关系。`

### 第 15 页：JEV 四路诊断编排

- **JEV 的职责**：判断当前任务需要多大诊断工作量，选择诊断路径。
- **输入信号**：图像质量、视觉置信度、帧间一致性、过程风险、知识覆盖度、案例相似度、证据冲突。
- **四路输出**：
  - `simple_loop`：证据充分、风险可控、已有可靠规则/案例；
  - `full_loop`：证据冲突、复杂风险、知识缺口或相似度不足；
  - `retake`：视觉质量或 ROI 覆盖不足；
  - `human`：高风险、关键证据缺失或需专家直接判断。
- **建议图形**：证据评分卡 → JEV 菱形 → 四个彩色路径；旁边展示“路由原因”和“匹配案例”。
- **强调**：JEV 输出路径、难度、置信度、原因和证据引用，不替代视觉分类器。

### 第 16 页：JEV 规则基线与轻量模型

- **两阶段实现**：先用可解释规则上线，再用小样本路由模型学习边界。
- **规则特征**：质量分、视觉熵、帧一致性、过程风险、案例相似度、知识覆盖、历史改判率。
- **轻量模型候选**：LightGBM/XGBoost、MLP、校准逻辑回归；输出四路概率和置信度。
- **训练标签**：专家指定路径、完整流程所需证据数、人工升级记录、复核结果。
- **路由伪代码**：

```python
def route_task(signals):
    if signals.quality < Q_MIN:
        return Route("retake", reason="quality")
    if signals.risk >= R_HIGH or signals.conflict:
        return Route("full_loop", reason="risk_or_conflict")
    if signals.case_similarity >= S_CASE and signals.coverage >= C_MIN:
        return Route("simple_loop", reason="stable_case_match")
    return Route("human", reason="evidence_gap")
```

- **评测**：复杂任务进入 full_loop 的召回率、危险样本漏放率、人工改判率、平均诊断时延、平均 Token 消耗。

### 第 17 页：simple_loop 快速诊断

- **适用任务**：视觉证据稳定、规则命中明确、相似案例充分、风险处于可控范围。
- **Agent 链路**：案例/规则命中 → 轻量诊断摘要 Agent → 确定性证据检查 → 人工确认。
- **输出**：简短诊断结论、匹配证据、风险等级、现场检查项和升级原因预留字段。
- **升级出口**：证据检查发现冲突、覆盖不足或风险上升，转入 `full_loop`。
- **建议图形**：绿色横向流程，右下角放蓝色“升级 full_loop”回路箭头。
- **代码骨架**：

```python
def run_simple_loop(state):
    case = case_memory.match(state.visual, state.process)
    draft = light_diagnosis(case, state.knowledge)
    review = deterministic_review(draft, state.evidence)
    return upgrade_if_needed(state, draft, review)
```

- **讲解重点**：快速路径减少重复调用，把算力集中到真正需要协作的任务。

### 第 18 页：full_loop 多 Agent 并行协同

- **触发条件**：视觉不确定、证据冲突、风险较高、案例相似度低、工艺知识覆盖不足。
- **并行 Agent**：多帧视觉复核、图像质量与证据完整性、工艺知识 RAG、特殊案例比对、Process State Agent。
- **汇聚 Agent**：诊断聚合 Agent 统一处理结构化证据；证据质疑 Agent 检查引用、冲突和缺口。
- **返工回路**：复核发现缺失时回到相关视觉、知识或案例 Agent，设置最大轮次和超时。
- **建议图形**：fan-out/fan-in，五个 Agent 形成并行扇出，汇聚到聚合与复核两个节点。
- **代码骨架**：

```python
async def run_full_loop(state):
    results = await gather(
        visual_review(state), quality_review(state),
        process_state(state), knowledge_rag(state), case_memory(state)
    )
    draft = diagnosis_aggregate(results)
    review = evidence_review(draft, results)
    return revise_or_escalate(state, draft, review)
```

- **图中标签**：`fan-out`、`fan-in`、`evidence review`、`bounded retry`。

### 第 19 页：工艺知识 RAG Agent

- **知识层级**：过热度定义、火眼状态说明、工艺手册、规则前置条件、SOP、风险边界、处置建议。
- **检索链路**：诊断问题重写 → 混合检索 → 重排 → 规则适用性过滤 → 引用片段输出。
- **每条知识记录**：`rule_id`、`source`、`version`、`preconditions`、`applicability`、`risk_level`、`allowed_output`。
- **输出**：片段、来源、版本、适用条件、检索分数和证据引用 ID。
- **建议图形**：查询从诊断聚合 Agent 进入 RAG，右侧展示三条带来源的知识证据卡片。
- **现有接口**：`Code/agentic/platform/retrieval.py:26`、`:50`；后续可替换为 BGE-M3 + Milvus + Reranker。

```python
def retrieve_knowledge(question, state):
    query = rewrite_with_context(question, state.visual, state.process)
    hits = hybrid_search(query, filters={"domain": "aluminum_electrolysis"})
    return rerank_and_attach_citations(hits)
```

### 第 20 页：特殊案例记忆 Agent

- **案例来源**：人工确认后的任务、专家修订、复核冲突、补拍结果和最终报告。
- **案例生命周期**：`candidate → eval_only → approved_special_case → active → deprecated`。
- **检索特征**：ROI 视觉 embedding、时序特征、Process State 摘要、风险等级、知识命中和人工结论。
- **安全边界**：只有人工确认并通过评估的案例进入 JEV 可用检索库；候选案例先进入离线评估集。
- **建议图形**：案例从报告归档进入候选库，经过专家审核、冻结评测后进入正式案例库，再回到 JEV 和 simple_loop。
- **代码位置**：`Code/agentic/platform/runtime.py:55,88` 的运行记录和候选案例接口。
- **讲解重点**：每次任务都为后续诊断积累可复用的、可审计的领域经验。

### 第 21 页：诊断聚合与证据复核

- **诊断聚合 Agent**：汇总视觉、过程、知识、案例证据，生成固定 Schema 的研判草案。
- **输出字段**：状态、风险、置信度、证据引用、解释、现场检查项、建议动作、待确认问题。
- **证据复核 Agent**：检查证据是否支持结论、引用是否有效、来源是否适用、数值是否越界、Agent 间是否冲突。
- **复核结果**：`pass / amend / request_more_evidence / human_review`。
- **建议图形**：左侧四类证据，中央聚合 Agent，右侧复核 Agent；复核问题回箭头指向对应证据源。
- **现有节点**：`workflow.py:266` 诊断、`:341` 复核、`:399` 最终门控。

```python
draft = diagnosis_agent(evidence_bundle, output_schema=DiagnosisSchema)
review = evidence_reviewer(draft, evidence_bundle)
decision = apply_review_policy(draft, review)
```

### 第 22 页：人工确认、可视化与审计

- **人工操作**：确认归档、修改补充、驳回重做、请求补拍。
- **UI 展示**：任务视频/代表帧、ROI、Agent 时间线、证据来源、JEV 路由、诊断草案、复核问题和最终决定。
- **审计字段**：`run_id`、`thread_id`、节点开始/结束时间、模型版本、提示词版本、引用来源、Token、人工动作。
- **建议图形**：左侧 Streamlit 任务面板，右侧 Agent trace 时间线和人工决策面板。
- **现有页面**：`Code/web_interface/streamlit_app.py:893-1060` 已展示运行 ID、流程状态和诊断结果。
- **实现状态**：旧 Streamlit 主流程已有运行状态和事件展示；JEV 多 Agent 图包含独立的节点 trace 与可选 interrupt/checkpointer 路径，需在展示端完成端到端接入验收。

### 第 23 页：RSI 受控离线演化

- **RSI 定位**：对运行日志、人工反馈、路由结果和失败案例做离线分析，形成可审查的提示词、知识和路由策略候选。
- **闭环**：运行日志 → 问题与案例分析 → 生成候选改进 → 冻结评测 → 专家审核 → 版本发布 → 下一批任务观察。
- **可优化对象**：JEV 路由阈值、Agent 提示词、RAG 查询模板、案例索引、复核规则和报告模板。
- **治理闸门**：候选版本经过固定评测集、回归测试和人工批准后才进入线上配置。
- **建议图形**：紫色离线闭环，虚线连接在线审计和版本仓库；标出“候选—评测—审核—发布”。
- **评价信号**：路由改判率、复杂任务召回率、证据完整度、人工耗时、Token 消耗、知识引用有效率。

### 第 24 页：评测体系与消融实验

- **视觉质量**：ROI 检测 mAP、分类 Macro-F1、风险召回率、ECE、视频级稳定性。
- **诊断质量**：结论一致率、证据关联正确率、知识引用有效率、行动建议可执行率。
- **路由效率**：平均/P95 时延、平均 Token、simple_loop 占比、full_loop 复杂任务召回率、人工改判率。
- **安全与治理**：补拍召回率、越界结论拦截率、来源缺失率、审计字段完整度、案例版本可追溯率。
- **消融组**：单帧 vs 多帧；无专家先验 vs F-GMVAE-T；无过程状态 vs Process State Agent；无 JEV vs JEV；无案例库 vs 案例记忆；线性流程 vs fan-out/fan-in。
- **建议图形**：左侧指标树，右侧 6 组消融柱状图和路由混淆矩阵。
- **数据说明**：`Code/data/v1` 的训练/验证/测试划分和 CNN、1DCNN、LSTM 基线作为过程分支参考；视觉分类使用冻结的视频/图片评测集。

### 第 25 页：系统演示闭环与阶段成果

- **演示脚本**：
  1. 上传一段巡检视频；
  2. 展示关键帧和 ROI；
  3. 展示 Process State Agent 的工况摘要；
  4. JEV 输出路由及原因；
  5. 进入 simple_loop 或 full_loop；
  6. 展示并行 Agent 证据、聚合草案和复核问题；
  7. 人工确认并归档；
  8. 案例进入候选库，RSI 读取审计记录。
- **最终画面**：左侧视频与 ROI，中央多 Agent 协作时间线，右侧诊断报告和人工操作按钮。
- **阶段成果**：
  - `[现有实现]` 固定 LangGraph 流程、视觉/知识/诊断/复核/归档节点、Streamlit 展示、运行审计；
  - `[已有模块，待集成验收]` JEV 四路编排、full_loop 动态 fan-out、证据复核、可选 interrupt/resume；
  - `[目标实现]` 多帧 F-GMVAE-T、Process State Agent、RSI 受控发布；
  - `[评测交付]` 冻结视觉集、过程状态集、路由评测集、消融报告和可回放演示。
- **收束文案**：`让每一次火眼巡检都成为一条可解释、可复核、可积累的多智能体诊断记录。`

## 代码与素材准备清单

### 代码片段

1. `Code/agentic/workflow.py`：状态、节点、计时、追踪、LangGraph 构建。
2. `Code/agentic/platform/contracts.py`：任务、证据、Agent 结果、复核决策协议。
3. `Code/agentic/platform/retrieval.py`：知识检索接口和引用结构。
4. `Code/agentic/platform/runtime.py`：运行记录、候选案例和审计接口。
5. `Code/web_interface/streamlit_app.py`：流程图、运行状态、诊断结果、人工操作。
6. `[目标实现]`：多帧采样与跟踪、`FGMVAETemporal`、`ProcessStateAgent`、`route_task`、并行 fan-out/fan-in、`interrupt/resume`、RSI 发布门。

### 图片与图表

- `Code/data/火眼分类图片帧`：封面、ROI、多帧序列、类别样本、质量问题样本。
- `Code/data/v1`：14 变量窗口图、变量分组图、趋势/波动统计、过程基线结果。
- `Code/agentic/knowledge`：知识片段、规则卡片、引用链示意。
- 运行截图：Streamlit Agent 时间线、JEV 路由、并行证据、人工确认和归档结果。

## 制作约束

- 每页只保留一个主结论，正文控制在 3—6 个短要点。
- 总体架构、JEV 路由、full_loop 协作、RSI 闭环分别使用独立详解页，封面和总览页不堆实现细节。
- 代码使用可编辑文本，不贴 IDE 截图；每个代码框控制在 4—8 行，并标注“现有实现/目标实现”、源码路径和函数名。
- 代码只放在任务协议、状态路由、JEV、full_loop、证据复核和人工确认等详解页；封面、背景、总体架构、角色矩阵、RSI、评测和收束页保留图表与实物素材。
- 第 6、8、16、18、21、22 页的代码位置、摘录和口语解释见 [PPT关键代码展示方案.md](./PPT关键代码展示方案.md)。
- 图中统一使用颜色语义：蓝色表示视觉/在线 Agent，橙色表示 JEV，绿色表示 simple_loop，紫色表示 RSI，红色表示人工确认，灰色表示平台底座。
- 所有模型、数据集、指标和状态名称与仓库代码及评测报告保持一致。
