# PPT关键代码展示方案

LLM 后训练、诊断聚合/质疑 Agent 和 Laya-JEV 置信度路由的新增方案与图片生成提示词见：[LLM后训练_诊断聚合质疑与Laya-JEV路由升级方案.md](LLM后训练_诊断聚合质疑与Laya-JEV路由升级方案.md)。

> **演示版状态（2026-11-09）**：PPT 代码和结构图可以展示已运行的 `flow_validation` 链路，包括四路路由、五类 full_loop Agent、HITL 演示审计和 RSI 离线候选卡。涉及真实视频训练、专家路由 fit、混合检索和在线 RSI 的代码仍标为“论文级待优化”，不在现场表述为已验证效果。

## 展示原则

代码用于回答“这个模块如何工作”，不作为装饰。建议在 25 页中选择 7 个核心算法页和 3 个工程可信度页放代码，其余页面保留图片、现场材料和流程图。每个代码框控制在 4—8 行，使用可编辑文本，不贴 IDE 截图；旁边只配一句白话解释。

每个代码框左上角标状态，右下角标仓库路径和函数名：

- 蓝色 `现有实现`：代码已存在于当前仓库；如果尚未接入演示主流程，额外标注“独立模块/集成中”。
- 橙色 `目标实现`：方法设计或伪代码，明确标注“目标实现”，不当作运行结果。
- 不在一页放两段代码，不把代码压缩到无法阅读；若页面图片本身已满，缩小图片主体或替换底部次要说明区，给代码留出至少约四分之一页面宽度。
- 代码框字号建议 16—20 pt，最多 8 行；完整源文件放附录或仓库链接。

## 当前重组版 PPT 的代码策略

重组版 PPT 实际有 30 张 slide，其中 5 张是章节提纲页，正文页对应关系如下：

| 逻辑页号 | 实际 slide | 主题 |
|---:|---:|---|
| 2.1—2.5 | 7—11 | 数据基础与系统架构 |
| 3.1—3.5 | 13—17 | 多模态感知与分类算法 |
| 4.1—4.8 | 19—26 | JEV 诊断编排与多 Agent 协同 |
| 5.1—5.3 | 28—30 | RSI 受控演化、评测与演示 |

代码不要平均分布到 25 个正文页。建议采用“7 个核心算法页 + 3 个工程可信度页”的组合：核心算法页解释系统的创新方法，工程页证明任务协议、状态机和人工闭环能够落地。章节提纲页、背景页、案例结果页保留图示和讲解空间。

### 一、核心算法代码页（优先放置）

| 逻辑页号（slide） | 代码主题 | 状态标签 | 代码框位置 | 该页只讲一句话 |
|---:|---|---|---|---|
| 3.1（13） | 多帧 ROI 采样、跟踪与时序证据聚合 | 算法伪代码·目标实现 | 右下或右侧 35% | 视频被压缩成可对齐、可追溯的火眼证据序列 |
| 3.3（15） | F-GMVAE-T 时空编码与模糊混合先验 | 现有模型骨架·真实标签训练待补齐 | 右侧替换重复公式区 | 分类同时利用空间、时间和专家先验，并输出不确定性 |
| 3.4（16） | 14 变量过程窗口状态摘要 | 现有独立 Agent·视频映射保持未知 | 右侧过程数据区 | 非配对过程数据作为工况上下文，不冒充视频标签 |
| 4.1（19） | JEV 四路安全路由 | 现有规则模块·新图已接入，旧主流程集成中 | 路由图右侧 | JEV 决定诊断工作量，不负责 C/N/H 分类 |
| 4.2（20） | JEV 置信度校准与安全门控 | 现有校准+规则·学习适配器扩展 | 基线图右侧 | 先校准路径置信度，再由硬安全规则保留最终否决权 |
| 4.4（22） | full_loop 并行 fan-out/fan-in | 现有编排骨架·旧主流程集成中 | 并行协同图下方 | 复杂任务并行取证，再统一聚合和复核 |
| 4.7（25） | 证据来源、数值边界和越权动作复核 | 现有实现 | 复核图右侧 | 诊断结论必须能回到已知来源和可用输入 |
| 5.1（28） | RSI 候选生成、冻结评测与人工发布 | 现有离线模块·在线接入待定 | RSI 流程右侧或底部 | 运行反馈只产生候选版本，评测和审批后才发布 |

以下代码均为 PPT 可编辑文本，按“代码框 + 一句解释 + 状态标签 + 源码路径”制作。标注为“目标实现”的代码用于解释设计，不应在讲解中说成当前实测结果。

### 每页的具体落位

| 实际 slide | 建议占位 | 处理原图的方式 |
|---:|---|---|
| 13 | 右下角现有 `extract_visual_evidence` 区域，约占 30% 宽度 | 保留左侧多帧图，把原函数说明替换为 6 行代码 |
| 15 | 底部指标区上方或右下角，约占 32% 宽度 | 保留时空结构图，代码框替换重复公式/指标说明 |
| 16 | 右下角过程输出区，约占 28% 宽度 | 保留 14 变量曲线，替换 `X_seq:30×14` 旁的解释框 |
| 19 | JEV 菱形或四路结果旁的窄栏 | 代码只保留路由判断，四路标签仍用图形表达 |
| 20 | 基线图右侧，约占 35% 宽度 | 用校准+硬门控代码替换大段文字，保留 baseline 标题 |
| 22 | full_loop 图底部右侧，约占 35% 宽度 | 用代码替换 `evidence review / bounded retry` 说明 |
| 25 | 复核流程右下角，约占 30% 宽度 | 保留证据链，代码只展示守门条件 |
| 28 | RSI 图右下角，最多 4—6 行 | 图已经密集，代码作为小号流程注释，空间不足时移到附录 |

代码框统一使用白底、深蓝等宽字体和浅灰边框；页内只保留一段代码，中文解释控制在一行。伪代码页的右下角写“算法伪代码·目标/现有骨架”，真实实现页写文件路径和函数名。

#### 3.1（slide 13）：多帧 ROI 取证

```python
frames = sample_frames(video_uri, fps=2.0, max_frames=8)
rois, track_ids = yolo.track(frames, target="fire_eye")
roi_seq = align_by_track(frames, rois, track_ids)
quality = quality_gate(roi_seq)
features = extract_motion_features(roi_seq)
return aggregate_video_evidence(roi_seq, quality, features)
```

输入写在代码框上方：`巡检视频 / 火眼图片帧`；输出写在下方：`ROI 序列 · 轨迹 ID · 动态特征 · evidence_refs`。右下角标注：`目标实现｜YOLO 逐帧检测可复用，跟踪对齐与时序聚合待接入`。讲解时强调“先形成证据序列，再交给分类器”，不要把这段代码描述成已经完成的端到端视频模型。

#### 3.3（slide 15）：F-GMVAE-T 时空分类

```python
frame_features = spatial_encoder(frames)
temporal = temporal_encoder(frame_features, valid_mask)
mu, logvar = posterior(temporal)
mixture = gm_prior(temporal, expert_features)
z = sample_latent(mu, logvar)
class_probs = softmax(classifier(concat(temporal, z)))
return class_probs, uncertainty(class_probs, logvar), mixture
```

代码框旁配三个小标签：`空间表征`、`时序表征`、`模糊先验`。底部写：`现有模型骨架·真实标签训练待补齐`。讲解重点是“模型不仅给类别，还给置信度和不确定性，后续由 JEV 决定是否需要完整协同”。

代码框底部状态改为：`现有模型骨架｜Code/agentic/models/fgmvae_temporal.py · FGMVAETemporal；真实专家视频标签训练待补齐`。

#### 3.4（slide 16）：Process State Agent

```python
x, columns = load_window(window_14x30, shape=(30, 14))
stats = variable_stats(x, columns)
changes = detect_change_points(x, columns)
groups = score_groups(stats)                # 电气 / 波动 / 加料 / 设备
anomaly = (0.65 * mean(groups) + 0.25 * variability(x)
           + 0.10 * clip(len(changes) / 3))
regime = ("abnormal" if anomaly >= .72 else
          ("transition" if anomaly >= .42 or changes else "stable"))
alignment = check_mapping(video_mapping)   # 无映射则 unknown
return ProcessStateSnapshot(process_regime=regime,
    event_risk=anomaly, anomaly_score=anomaly,
    variables=stats, change_points=changes,
    alignment_quality=alignment)
```

代码框顶部放输入标签：`14 个过程变量 × 30 个时间步`；底部放输出标签：`process_regime · event_risk · change_points · EvidenceRef`。状态标成 `现有独立 Agent｜视频映射保持 unknown`。讲解时说明它提供工况先验和异常趋势，不能因为有过程数据就推断出与视频对应的真实时间标签。

代码框底部状态改为：`现有独立 Agent｜Code/agentic/agents/process_state.py · ProcessStateAgent；video_mapping 未提供时保持 unknown`。

#### 4.1（slide 19）：JEV 四路安全路由

```python
f = build_inspection_features(quality, visual, cases, knowledge, process)
if f.quality_status in {"retake", "human"}:
    return f.quality_status
if f.novelty >= .85 or f.confidence < .45:
    return "human"
if f.risk >= .75 or f.conflict >= .45:
    return "full_loop"
if f.coverage < .34:
    return "full_loop"
if f.case_sim >= .88 and f.case_applicable and f.risk < .45:
    return "simple_loop"
return "full_loop"
```

右下角放四个彩色结果标签：`simple_loop`、`full_loop`、`retake`、`human`。状态标成 `现有规则模块｜Code/agentic/platform/jev/rules.py · safety_route`。讲解句：风险、冲突或证据覆盖不足时增加诊断协同；相似案例和证据充分时走轻量路径；JEV 只决定路径，不输出火眼类别。

#### 4.2（slide 20）：JEV 置信度校准与安全门控

```python
rows = load_route_manifest("route_manifest.jsonl")
scores = [row["confidence"] for row in rows]
labels = [row["route"] == row["expected_route"] for row in rows]
tau = grid_search_temperature(scores, labels)  # minimize Brier
proposal = calibrate_confidence(raw_route_score, tau)
route = safety_route(features)                 # hard policy first
route = accept_adapter(route, proposal, hard_gates=policy_gates(features))
return route, tau, route_metrics(rows)
```

输入标签写成：`冻结运行记录 = 路由置信度 + 专家路径标签`；输出标签写成：`校准温度 · 路由置信度 · 路由评测`。状态标成 `现有校准+规则｜学习型路由器为扩展`。讲解时明确“校准和适配模块优化路径选择，安全规则负责阻止高风险样本降级到 simple_loop”。如果需要在答辩中展示 LightGBM，可作为旁注标记 `目标实现/未接入`。

#### 4.4（slide 22）：full_loop 并行协同

```python
workers = [visual_review, knowledge_rag,
           case_memory, process_state]
tasks = [Send("full_loop_worker", {"worker_id": fn.__name__,
          "state": state}) for fn in workers]
results = await run_with_timeout_retry(tasks, timeout=30, retries=1)
bundle = fan_in(results)
draft = diagnosis_aggregate(bundle)
return evidence_review(draft, bundle)
```

代码框左侧配箭头标签：`fan-out`、`join`、`aggregate`、`review`。状态标成 `独立编排模块·主流程集成中｜Code/agentic/orchestration/diagnosis_graph.py`。讲解时只解释并行派发、结果汇聚和证据复核三步，复杂的重试和超时参数放在源码，不要在图上展开。

#### 4.7（slide 25）：证据复核守门

```python
cited = set(draft.source_ids) | {ref.source_id for ref in draft.evidence}
missing = cited - set(known_source_ids)
if missing:
    issues.append(f"unknown_source:{sorted(missing)}")
if mentions_numeric_superheat(draft) and not has_measurements:
    issues.append("superheat_measurement_missing")
if contains_control_command(draft):
    issues.append("manual_authorization_required")
action = "pass" if not issues else "human_review"
return ReviewDecision(action, issues, evidence_ids=sorted(cited))
```

右侧用两个小标签解释检查项：`来源存在性`、`数值输入完整性`。状态标成 `现有实现｜Code/agentic/agents/review_guard.py · review_draft`。讲解时说明复核器检查的是证据链和表达边界，不能替代工艺专家做最终授权。

#### 5.1（slide 28）：RSI 离线受控演化

```python
candidates = analyze_failures_and_feedback(load_audit_events(run_ids))
for candidate in candidates:
    gate = offline_replay(candidate, frozen_sets)
    if not gate.accepted:
        continue
    publisher.approve(candidate, reviewer_id, "sealed replay passed", gate)
    version = publisher.publish(candidate, gate)
return version
```

代码框顶部标 `离线演化`，底部标 `候选 → 冻结评测 → 专家审批 → 版本发布`。状态标成 `现有离线模块·在线接入待定`。讲解时强调 RSI 是受控的案例、提示词和路由策略演化闭环，发布动作必须经过冻结集评测和人工审批。

代码框底部状态改为：`现有离线骨架｜Code/agentic/rsi/{candidate,evaluator,gate,publisher}.py；在线自动改写未开放`。

### 二、工程可信度代码页（次优先）

创新算法页完成后，再保留 3 个工程页。若页面空间不足，工程代码移到附录，不要压缩核心算法代码。

| 逻辑页号（实际 slide） | 代码主题 | 状态 | 听众应理解什么 |
|---:|---|---|---|
| 2.2（8） | `InspectionTask` 协议 | 现有实现 | 任务对象携带输入和运行追踪字段 |
| 2.4（10） | LangGraph 条件边 | 独立模块已实现 | 路由结果会决定图中的后续节点 |
| 4.8（26） | 人工 `interrupt/resume` | 独立模块代码已实现，端到端验收待确认 | 人工动作暂停、恢复并进入可审计闭环 |

JEV 规则、fan-out 和证据复核的代码已放在核心算法页，不再在工程页重复放置。每个工程页只保留一个代码框，并在角落标注源码路径和集成状态。

### 2.2（slide 8）：任务协议

放在任务协议图旁边，标题标注 `现有实现：任务对象`。

```python
@dataclass
class InspectionTask(Contract):
    input_path: str
    run_id: str = field(default_factory=lambda: f"run-{uuid4().hex[:12]}")
    tenant_id: str = "tenant_default"
    session_id: str = "default"
    operator_id: str = "local_operator"
    input_mode: str = InputMode.SINGLE_REPRESENTATIVE_FRAME.value
```

**讲解句**：当前任务对象已经记录输入路径、任务运行 ID、会话和输入模式；图中的 `media_uri/frame_ids/roi/process_window` 是目标任务协议示意，不是当前 `InspectionTask` 顶层字段。

**来源**：`Code/agentic/platform/contracts.py`，`InspectionTask`。

### 2.4（slide 10）：条件路由与图边

代码框标题标注 `现有实现：独立诊断编排图`。它展示 LangGraph 如何把 JEV 的路由结果连到后续节点。页面图中若包含质量门控、补拍、人工状态等目标流程，应标明当前主界面集成状态，不要仅凭这段代码暗示端到端页面已经接通。

```python
graph.add_conditional_edges(
    "route", _langgraph_route_target, route_map
)
graph.add_conditional_edges(
    "full_loop_dispatch", _full_send_targets
)
graph.add_edge("full_loop_worker", "full_loop_join")
graph.add_edge("full_loop_join", "aggregate")
graph.add_edge("aggregate", "review")
graph.add_conditional_edges("review", _review_next_target)
```

**讲解句**：JEV 节点给出路径后，条件边把任务送到相应分支；full_loop 的 worker 汇聚后再进入诊断聚合和复核。

**来源**：`Code/agentic/orchestration/diagnosis_graph.py`，`build_diagnosis_graph`。

#### 这一页的右侧落位

你提供的页面左侧状态机已经占据主要视觉区域，右侧留白可以直接做成“代码解释栏”，不需要重画状态机：

```text
页面宽度：约 1850 px
流程图区域：左侧约 0—1120 px
代码栏区域：右侧约 1160—1810 px
代码栏高度：从结论框下方到页脚说明框上方
```

代码栏建议按下面的层级制作：

1. 顶部放浅蓝色标题条：`关键代码｜LangGraph 条件边`；
2. 标题条右上角放小红标签：`独立编排模块`；
3. 中间放白底或极浅灰底代码框，使用深蓝色等宽字体；
4. 将前两行用浅橙色底纹标出，表示“路由分支”；
5. 将后两行用浅蓝色底纹标出，表示“汇聚与复核”；
6. 代码框下方放一条短解释：`JEV 路由 → 条件分支 → worker 汇聚 → 证据复核`。

代码栏不要使用黑色 IDE 截图、深色代码背景、过多语法高亮或完整的 `route_map` 字典。PowerPoint 中应使用可编辑文本，建议代码字号 16—18 pt，行距 1.15—1.25。

推荐直接放下面 6 行：

```python
graph.add_conditional_edges(
    "route", _langgraph_route_target, route_map
)
graph.add_conditional_edges(
    "full_loop_dispatch", _full_send_targets
)
graph.add_edge("full_loop_join", "aggregate")
graph.add_edge("aggregate", "review")
```

右下角加一行小号来源：

```text
diagnosis_graph.py · build_diagnosis_graph
```

页面讲解时可以指向代码栏说：

> 左边是任务状态如何流转，右边是这个流转在 LangGraph 中如何落地。第一行把 JEV 的路由结果接到不同分支，第二行把 full_loop 的多个 worker 派发出去，后两行把 worker 结果汇聚到诊断聚合和证据复核。代码展示的是编排关系，视觉分类和知识检索分别由对应 Agent 完成。

代码栏底部可以加一个很小的状态说明：

```text
代码模块：现有实现｜网页端完整接入：集成验收中
```

如果页面需要保持更简洁，可以不放这行状态说明，而在口头讲解时说明该代码来自独立诊断编排图。

### 4.8（slide 26）：人工确认暂停

代码框标题标注 `已实现节点代码：interrupt`，并在旁边注明该节点需在启用人工审核和 checkpointer 的运行中使用；当前网页端端到端暂停/恢复仍需验收。

```python
decision = langgraph_interrupt(payload)
values = dict(decision or {})
action = str(values.get("action", ""))
if action not in ALLOWED_HUMAN_ACTIONS:
    raise InvalidActionError(f"invalid human review action: {action!r}")
review_decision = ReviewDecision(
    run_id=state.get("run_id", ""),
    thread_id=state.get("thread_id", ""), action=action,
    reviewer_id=str(values.get("reviewer_id", "local_operator")),
)
```

**讲解句**：流程在人工确认节点暂停，恢复时只接受约定动作；决定与 run/thread 绑定，便于审计。

**来源**：`Code/agentic/orchestration/diagnosis_graph.py`，`_human_review_node`。

## 页面布局建议

- slide 13、15、16、19、20、22、25、28 是算法代码优先页；slide 8、10、26 是工程可信度代码页。每页只放一个代码框，优先替换图片下方重复说明或右侧次要留白。
- 如果当前生成图片铺满页面，保持原图主体缩至约 70%—75% 页面宽度，把代码作为独立可编辑栏加入；不要把 8 行代码缩成一条不可读的底注。
- 代码框下方只写一句解释，页脚写相对路径和函数名，例如 `diagnosis_graph.py · _full_send_targets`。
- 用色区分实现状态：蓝色“现有实现”，橙色“目标实现”；不要把伪代码截图放进“现有实现”页。
- 页面演讲时只解释代码中一条关键逻辑，完整细节留给源码或答疑。

## 验证与状态说明

本方案引用的是当前工作区源码。相关单测运行结果为 `21 passed, 1 failed, 1 skipped`；失败项 `test_full_loop_keeps_failed_worker` 断言 trace 中存在 `full_loop_dispatch`，而当前事件名称为 `full_loop_fanout`。因此 PPT 可展示 fan-out 代码路径，但不要把这组测试表述成全通过。独立图模块与现有 Streamlit 主流程的集成状态也应按实际演示版本标注。

## 最终制作顺序

1. 先完成 slide 13、15、16、19、22、25 六个代码框，覆盖“多帧取证 → 分类 → 过程先验 → JEV 分流 → 并行协同 → 证据复核”的主创新链。
2. 再决定 slide 20 是否加入校准代码；若时间有限，保留右侧的 `baseline → calibrated route → safety gate` 三段结构即可。
3. slide 28 的 RSI 代码只在图面仍有空间时加入，优先保证候选、冻结评测、人工审批和版本发布四个词清晰可见。
4. slide 8、10、26 的工程代码用于答辩追问或“系统如何落地”部分，不要抢占算法页的主体面积。
5. 所有代码框右下角写明“现有实现/独立骨架/目标扩展”之一，并附文件路径；演讲时先讲输入和输出，再解释一条关键判断。
