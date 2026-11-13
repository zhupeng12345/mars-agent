# 铝电解火眼多智能体 PPT 演示版快速补全执行清单

> 目标：先形成一套“内容完整、链路可演示、页面有素材、代码能讲解”的汇报版本。
> 当前策略：暂不追求论文级精度、GPU 训练、真实 Laya 专家路由效果和在线 RSI 生效；所有实验性结果必须标记为 `flow_validation`、`structure_validation` 或 `experimental`。

## 一、完成标准

演示版完成后，现场可以用一段视频完成以下展示：

```text
视频输入
→ 多帧采样与 ROI
→ 图像质量门控
→ 视觉证据 + Process State 摘要
→ Laya-JEV 四路诊断分流
→ simple_loop 或 full_loop
→ 多 Agent 协作时间线
→ 诊断聚合与证据复核
→ 人工 approve/amend/reject/request_retake
→ 报告归档与候选案例
→ RSI 离线候选演化
→ 评测卡片和审计记录
```

现场不需要证明模型准确率，但必须能回答：

1. 系统有哪些 Agent？
2. JEV/Laya 如何决定简单循环或完整循环？
3. 多个 Agent 如何并行协作和回流？
4. 人工在哪一步拥有最终授权？
5. 一次运行如何进入案例记忆、评测和 RSI 离线演化？

## 二、最高优先级：先做一条稳定的演示主链

### 任务 A：准备四个固定演示场景

建立 `Code/data/demo_cases/`，每个场景包含输入媒体、路由特征、预期路由、诊断草案和人工动作：

| 场景 | 路由 | 页面叙事 |
|---|---|---|
| `stable_case` | `simple_loop` | 视觉证据稳定、规则/案例命中、轻量诊断快速完成 |
| `conflict_high_risk` | `full_loop` | 证据冲突或风险升高，五个 Agent 并行取证 |
| `bad_quality` | `retake` | 模糊/过曝/目标缺失，请求补拍并返回输入层 |
| `ood_unknown_case` | `human` | 新颖性高或证据不足，直接进入人工复核 |

每个场景的 JSON 至少包含：

```json
{
  "case_id": "conflict_high_risk",
  "media": "data/video/偏高.mp4",
  "expected_route": "full_loop",
  "route_reason": "risk_or_evidence_conflict",
  "visual_status": "flow_validation",
  "process_status": "independent_prior",
  "alignment_quality": "unknown",
  "review_action": "approve"
}
```

验收：一条命令生成四个 `run_id`，每个 trace 都包含输入、路由、Agent 状态、人工动作、报告和候选案例记录。

### 任务 B：把路由后端显式显示出来

补齐 `JevRouteDecision` 和页面字段：

```text
router_backend: deterministic_rule | laya_adapter | laya_adapter_fallback | safety_fallback
router_version
gate_version
route_reason
confidence
```

PPT 现场要按 trace 标注实际 backend：当前真实视频流程回归使用 `deterministic_rule`；Laya adapter 是 CLI/网页端可选策略后端，只有显式选择并且任务进入 Laya 建议阶段才会调用。系统借鉴 JEV 的证据驱动路由思想，Laya 适配只负责路由策略接口，视觉 C/N/H 输出由视觉证据模块提供。

### 任务 C：让五个 full_loop Agent 具备独立的展示结果

即使暂时不追求真实模型效果，也要为每个 Agent 输出独立结构化结果：

| Agent | 演示输出 |
|---|---|
| 多帧视觉复核 Agent | 关键帧、ROI 覆盖率、帧间一致性 |
| 图像质量与证据完整性 Agent | 清晰度、目标存在性、证据缺口 |
| 工艺知识 RAG Agent | 规则 ID、来源、版本、适用条件 |
| 特殊案例 Agent | 匹配案例、相似度、状态、适用性 |
| Process State Agent | 14 变量分组、工况摘要、风险先验、对齐质量 |

每个 Agent 即使返回 `structure_validation`，也要有 `status`、`attempt`、`duration_ms`、`evidence_ids`、`error_code` 字段，便于在 PPT 上展示并行协作。

## 三、第二优先级：补齐页面 6–24 的可视素材

当前资产 manifest 有 25 页，但真实图片主要绑定在输入、视觉和端到端页面。演示版应为下列页面生成带 `flow_validation` 水印的结构图或代码图：

| 页面 | 必须补的素材 | 直接来源 |
|---|---|---|
| 6 | 标注/数据协议卡 | `contracts.py`、visual/process/route manifest |
| 7 | 总体架构图 | 四路主链 trace + Agent 角色表 |
| 8 | 任务状态机 | `received → evidence_check → route → review → archived` |
| 9 | Agent 角色矩阵 | `AgentResult`、`EvidenceRef`、`AgentAssignment` |
| 10 | 14 维过程变量分组 | `process_manifest.jsonl` |
| 11 | 非配对数据治理图 | `process_missing`、`alignment_quality=unknown` |
| 12 | F-GMVAE-T 结构图 | `fgmvae_temporal.py`、论文公式和模型配置 |
| 13 | 时序编码器对比图 | pooling / GRU / TCN / Transformer 代码接口 |
| 14 | JEV 四路路由图 | 四个 fixture 的 route decision |
| 15 | simple_loop 图 | 案例/规则 → 轻量诊断 → 复核 → 人工 |
| 16 | full_loop fan-out/fan-in 图 | 五个 Agent 的并行 trace |
| 17 | 工艺知识 RAG 卡 | 知识规则、来源、版本、引用 ID |
| 18 | 特殊案例生命周期图 | candidate → eval_only → approved → active |
| 19 | 诊断聚合/证据复核图 | `DiagnosisDraft`、review issues、evidence refs |
| 20 | LLM Gateway 图 | fake / Ollama / Qwen API / fallback profiles |
| 21 | interrupt/resume 图 | pause、thread_id、人工动作、resume |
| 22 | 审计和报告图 | JSON/Markdown/HTML、run_id、节点耗时 |
| 23 | RSI 离线闭环图 | trajectory → critique → candidate → gate → publish |
| 24 | 评测矩阵图 | 视觉、过程、路由、Agent、系统、消融 |

每张素材必须有同名 sidecar：

```json
{
  "asset_id": "ppt-page-16-full-loop",
  "page": 16,
  "title": "full_loop 多 Agent 并行协作",
  "source_files": ["runtime_data/demo_full_loop.json"],
  "run_id": "demo-full-loop-001",
  "status": "flow_validation",
  "claim_policy": "结构验证，不代表模型效果"
}
```

代码页允许使用结构化代码卡，不要求每个页面都是真实截图；但代码卡必须标注“现有实现”或“演示适配层”。

## 四、第三优先级：前端变成汇报工作台

在现有 Streamlit Agent 页面上增加一个“演示模式”，避免现场依赖长时间模型调用：

### 页面布局

```text
顶部：run_id / thread_id / route / route_reason / status
左栏：视频、代表帧、ROI contact sheet
中栏：JEV 路由卡 + 五个 Agent 并行状态
右栏：诊断草案 + 证据引用 + 复核问题
底部：人工动作 + 报告下载 + 候选案例 + RSI 记录
```

### 演示模式行为

1. 选择四个固定场景之一。
2. 一键加载预生成 trace，页面立即展示全链路。
3. 可以切换 `simple_loop/full_loop/retake/human`。
4. 可以点击四个人工动作并更新审计记录。
   演示决定需按 `run_id` 持久化；重新加载同一案例后恢复决定，重复点击不重复写审计。approve/amend 只沉淀到隔离的 `candidate` 案例库，reject/request_retake 不建案例。
5. 可以下载 JSON、Markdown、HTML；补充 PNG 下载按钮。
6. 页面显式显示 `flow_validation`，避免把演示 trace 当成真实精度。

这一步的价值高于继续训练一个没有真值的视频模型，因为 PPT 需要展示完整的角色协作和决策路径。

## 五、第四优先级：补齐“看起来完整”的评测和 RSI 内容

### 评测页采用结构验证数据

暂时不写准确率提升，页面展示以下内容即可：

```text
视觉：多帧 ROI、关键帧、模型输出 Schema、未标注状态
过程：14 变量、独立状态先验、alignment_quality=unknown
路由：四路 fixture 回放、路由原因、fallback
Agent：证据链完整率、引用存在率、节点失败/重试
系统：节点耗时、流程完成率、人工动作分布
消融：9 个实验组，标记 structure_only/not_run
```

### 知识和案例查询集

为了让第 17–18 页有内容，快速生成 10 条演示查询和 10 条案例查询，覆盖：

```text
有效命中、空检索、过期规则、条件不匹配、冲突规则、相似案例、低相似度、已废弃案例、人工修订案例、补拍案例
```

这些数据必须标记为 `demo_fixture`，不能写成真实评测集。

### RSI 页面

使用一个已知缺陷完成离线演示：

```text
运行记录中发现引用缺失
→ critique 生成候选修复
→ frozen replay
→ safety/quality/efficiency gate
→ 人工批准
→ 生成新版本
→ 展示 rollback
```

页面重点是治理链路，不是算法提升百分比。

## 六、暂时明确不投入时间的内容

以下内容放入论文级优化阶段，不阻塞 PPT 演示版：

- GPU 级 F-GMVAE-T 训练和视频 Macro-F1；
- 真实专家路由标签和 Laya fit；
- 视频与过程数据的真实时间/槽位对齐；
- BGE-M3、Milvus、Reranker 的工程化部署；
- 9 项消融的真实数值结果；
- RSI 发布版本在线改变 JEV/RAG/prompt runtime；
- P50/P95/P99 的干净性能基准；
- 机器人端部署、量化和剪枝。

## 七、建议执行顺序

```text
T0 生成四路固定 demo fixture、一条 canonical full_loop trace 和一条 simple-to-full 升级 trace
T1 补 router_backend、Agent 独立结果和统一 sidecar
T2 生成页面 6–24 的结构图/代码卡
T3 增加 Streamlit 演示模式、四路切换、升级轨迹、PNG 下载和状态卡
T4 生成 10+10 demo 查询/案例记录和 RSI 缺陷演示素材
T5 重建 ppt_asset_manifest，保证每页有 source 或 image_asset
T6 运行 validate_assets、五个演示场景、Streamlit AppTest 与 HTTP health 检查
T7 重新生成 PPT 讲解稿中的“现有实现/演示验证/论文级待优化”标签
```

## 八、演示版交付验收

- [x] 四路固定场景均能一键加载；另有一条由诊断图真实执行生成的 `simple_loop → full_loop` 升级 trace。
- [x] PPT 第 7、8、14、15、16、21、23、24、25 页绑定 canonical trace 或真实运行记录。
- [x] 第 9–13、17–20、22 页均有 16:9 结构验证图或代码卡，不再是空白页。
- [x] 五个 full_loop Agent 均有独立状态、耗时字段、attempt、策略 ID、超时/重试信息和 evidence_ids；工作台可展开查看 worker 执行策略。
- [x] 页面显示 Laya-JEV、确定性安全门控、LangGraph、人工审核和 RSI。
- [x] 所有页面素材有 JSON sidecar 和 `flow_validation`/`experimental` 状态。
- [x] 不使用高泄漏视觉基线宣称准确率，不使用 5 个 fixture 宣称路由模型准确率。
- [x] `presentation_demo`、P7 资产测试、`validate_assets` 和 Streamlit AppTest/健康检查全部通过；演示夹具审核后可下载 JSON/Markdown/HTML/PNG 报告；论文级 `demo_run` 指标仍按原计划单独标记。
- [x] 现代 CLI 和网页端提供确定性规则 / Laya adapter 选择；界面显示实际决策 backend/profile/gate。Laya adapter 指向 `typed-decisions`，只有实际调用时才加载权重。
- [x] 两段项目视频新隔离运行均到达 `full_loop` / `waiting_review`，专家协作位 `10/10` 完成、来源链 `8/8` 完整；本轮显式评测 `n=2`，任务耗时 P50 `1757.186 ms` / P95 `1807.003 ms`，结果属于 `flow_validation`。
- [x] 评测报告增加 approve/amend/reject/request_retake 四类人工动作计数；本轮选定 cohort 无人工动作，计数为 0。
- [x] 总评测与工作台展示 full-loop worker 策略覆盖、策略 ID、重试、失败和超时；以重试/超时 fixture 回归，不作为模型效果指标。
- [x] PPT 资产检查解码图像、核对 sidecar 像素尺寸并确认页面引用均可解析：42/42 图片有效、25 页覆盖、235 个引用解析、0 个悬空引用。
- [x] 人工请求补拍后可上传新图片/视频并启动子 run；task、runtime、结果和审核记录保留父子 run 关系，原父任务审核记录不被覆盖。
- [x] HITL 历史新增模型 profile 筛选；P3 知识包新增严格结构校验，当前 3/3 条规则通过结构检查。
- [x] 总评测和工作台展示研究冻结状态及每份 manifest 的记录数、错误和 SHA；当前 `incomplete`（1 缺失、5 无效），不会把 demo fixture 当作冻结研究集。
- [x] 检索 fixture 增加来源/版本/有效期/适用范围/locator/分数的机械引用完整性检查；10 条 fixture pass rate `1.0`，语义支持和研究指标仍不可声明。
- [x] 总评测 JSON 和 Streamlit 表格展示知识包结构校验 `3/3`、知识引用 fixture 字段完整 `10/10`；两者不作为领域正确率或语义支持率。
- [x] JEV fixture 总评测补齐 `simple_loop/full_loop/retake/human` 四路召回；五条固定回归均命中预期路径，仍标记为 policy fixture regression，不作为专家路由准确率。
- [x] 评测产物 `runtime_data/evaluation.e3_20261111_final.json` 及 Markdown/CSV/失败样例/PNG 已生成；视觉为 `structure_validation_only`，质量合成门控为 5/5，知识包结构为 3/3，引用机械完整性为 10/10。
- [x] 全量回归更新为 `212 passed, 1 warning, 39 subtests`；`compileall`、`git diff --check`、`validate_assets`（42/42）通过。唯一 warning 是既有 PyTorch Transformer nested-tensor 配置提示。
- [x] 五个固定场景实际执行 LangGraph route smoke，四路预期决策全部命中；输出节点轨迹，包含稳定初筛任务证据复核后升级 full_loop 的路径。产物：`runtime_data/route_smoke.e3_20261111_final.json`。
- [x] Streamlit 评测表展示四路独立 fixture 召回率，并将性质明确标为 policy fixture，避免只展示总命中率或误读为真实路由准确率。
- [x] 真实浏览器页面视觉验收：桌面 1440×1100 与手机 CSS 视口 390×844 均无页面横向溢出；历史记录卡片在手机可读。诊断页显示“待人工确认”和“演示摘要”。截图见 `runtime_data/streamlit_agent_status_desktop_localized_updated.png`、`runtime_data/streamlit_agent_status_mobile_390_localized_updated.png`。
- [x] 新增的 JEV 四路评测行已由 AppTest 和 8504 浏览器截图验证；桌面 1440×1100 截图 `runtime_data/streamlit_eval_jev_fixture_desktop_8504_updated.png` 显示四路行和值。结果仍标记为 policy fixture regression，不作为专家路由准确率。

### 发布前边界复核（2026-11-13）

- 评测冻结状态仍为 `incomplete`，冻结拒绝原因已在总评测和工作台逐份展示；不能用演示 fixture 填充研究冻结集。
- 运行时目录、原始视频、过程数据、课件压缩包和二进制 PPT/Word 已通过忽略规则排除；代码、评测 manifest、知识包、25 页结构验证资产和 sidecar 保留。
- 最终发布前以实际全量测试、资产校验和总评测命令输出为准更新本清单的测试计数。

### 后续论文级评测缺口（不得计为演示版通过）

- 实际 Laya checkpoint 推理和专家标注的路由评测尚未执行；当前真实视频记录仍来自 `deterministic_rule`。
- 6160 条视觉 manifest 记录无专家视觉标签，因此分类精度和视频多模态融合收益不可计算。
- 当前冻结评测清单尚未通过完整性审核；九项消融均为 `structure_only`。
- Process State 有独立时序 baseline，但 `alignment_quality=unknown`，不能解释为视频融合或过热度识别增益。
- 网页可加载评测报告 `runtime_data/evaluation.e2_20261109_jev_wiring.json` 已纳入该 baseline：CNN ROC-AUC `0.542`、PR-AUC `0.588`、event recall `0.88`、test windows `827`；性质为 `experimental_baseline`，目标是过程未来风险标签。
- 增加了固定日期切分下的五随机种子独立训练评测：`runtime_data/process_repeats_e2_20261109/repeated_process_evaluation.json`。单模型 CNN、每个 seed 1 epoch、测试窗口 `n=827`；ROC-AUC `0.398 ± 0.006`、PR-AUC `0.465 ± 0.008`、event recall `0.952 ± 0.030`、预测阳性率 `0.867 ± 0.083`、specificity `0.103 ± 0.066`、Brier `0.256 ± 0.002`、ECE `0.050 ± 0.022`。高事件召回伴随高阳性预测率和低 specificity，清楚显示当前固定阈值告警取舍；这是过程未来风险标签观察，不是视频融合收益。完整总表为 `runtime_data/evaluation.e2_20261109_complete.json`。
- 评测冻结脚本已增加样本主键去重；重复 ID 会计为无效行，不能靠重复样本达到最低条数。当前六份研究评测 manifest 仍全部未通过冻结，专家真值缺失状态不变。
- 本轮显式运行 cohort 为 `n=2`，可展示本机流程耗时并注明样本量，不能称为稳定性能基准；本轮已完成浏览器页面视觉/响应式截图，但浏览器级审核操作、刷新恢复和下载交互仍需独立端到端验收。
- [x] 演示审核状态在新页面会话中恢复；审核记录幂等；approve/amend 案例保持 candidate 且隔离；RAG 引用按本地 `source_id` 解析并展示来源版本与适用条件。

完成以上清单后，项目就具备“内容完整、链路完整、页面可讲、代码可展示”的 PPT 演示版本；之后再按照论文级要求补真值、训练、消融和性能评测。
