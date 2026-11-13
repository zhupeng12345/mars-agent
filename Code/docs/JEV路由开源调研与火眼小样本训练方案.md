# JEV 路由开源调研与火眼小样本训练方案

> 调研日期：2026-11-07
> 适用项目：铝电解火眼视觉识别与多智能体研判
> 目的：为 JEV 风格的任务分流选择可复用的开源代码，并确定当前小规模火眼数据的训练边界。

## 0. 命名边界（本项目最终口径）

本项目不把 JEV 作为自研模型名称。JEV 表示参考的证据驱动诊断路径选择思想；Laya 是可选的开源路由基座；在铝电解火眼任务上的适配实现统一称为 **Laya-JEV**。完整链路由 Laya-JEV、确定性安全门控、LangGraph 多 Agent 编排和人工审核组成。没有专家路由标注或可用的外部 Laya 包时，系统使用本地规则与兼容适配器完成流程验证，并将 Laya fit/准确率标记为待优化项。

## 1. 结论先行

GitHub 上没有找到一个可以直接替换为“铝电解火眼 JEV 诊断编排器”的通用项目。当前最接近的项目分成三类：

1. **LLM 请求路由**：`Routapse` 的 Laya/Jev 先判断请求应进入哪个模型或专家通道。它是文本路由器，不是视觉分类器，但网关、候选路由、置信度、规则覆盖、fallback 和 JSONL 审计方式可以借鉴。
2. **视觉早退/级联推理**：`earlyon`、Shallow-Deep Networks、LGViT 等项目让简单样本在浅层出口结束，困难样本继续完整推理。它们适合借鉴“先快后深”和阈值校准，不能直接把网络出口等同于本项目的多 Agent 工作流分支。
3. **拒答、OOD 与相似案例检索**：`pytorch-ood` 用于识别分布外或不确定样本，`FAISS` 用于特殊案例相似检索，二者更适合构成 JEV 的安全特征和记忆底座。

建议的实际组合是：

```text
YOLO 多帧 ROI + F-GMVAE-T + 图像质量指标
    → 轻量 JEV 诊断编排决策模型
    → 特殊案例相似度与 OOD 信号
    → simple_loop / full_loop / retake / human
    → LangGraph 多 Agent 流程
```

JEV 的目标是判断当前任务是否具备运行简单小循环的条件，以及是否必须进入完整大循环；它不是火眼分类器，不重新训练 C/N/H，也不直接预测摄氏度过热度。C/N/H、F-GMVAE-T 的风险和不确定性只是上游视觉 Agent 产生的证据字段。

## 2. 当前数据盘点

`Code/data/火眼分类图片帧` 当前是空目录，递归文件数为 0，目录中没有图片或隐藏文件。当前可见的视频数据是：

| 文件 | 规模 | 可用标签含义 |
|---|---:|---|
| `Code/data/video/正常.mp4` | 约 60.6 秒，720×1280，约 29.9 FPS，约 1813 帧 | 文件名提供“正常”弱标签 |
| `Code/data/video/偏高.mp4` | 约 60.6 秒，720×1280，约 29.9 FPS，约 1814 帧 | 文件名提供“偏高”弱标签 |

这两段视频不足以产生可靠的四路 JEV 监督标签，也不能把文件名直接当作真实过热度或 C/N/H 的专家真值。相邻视频帧高度相似，随机按帧切分会把同一场景泄漏到训练集和测试集，导致路由指标虚高。

项目另有单槽 14 维过程变量、窗口统计特征和事件标注，但当前无法与火眼视频建立帧级、区间级或槽级对应关系。建议由独立 Process State Agent 建模工况状态、过程事件风险和变化点，将结果作为工艺知识、特殊案例和 JEV 的过程先验。

当前 `Code/data/v1` 已形成过程状态建模基线：单槽 `8310`、2 分钟采样；每个样本使用 30 步（60 分钟）历史窗口，预测未来 15 步（30 分钟）的二分类过程事件状态。数据集包含 `X_seq=(5744,30,14)`、`X_stat=(5744,70)` 和 `y`，按日期切分为 train 3952、val 965、test 827。CNN、1D-CNN、LSTM 的测试 ROC-AUC 分别为 0.542、0.527、0.437，现阶段适合作为 Process State Agent 的流程基线和 JEV 过程先验，不作为火眼视频标签或视觉-过程融合指标。Process State Agent 使用电气状态、波动动态、加料操作和设备状态四组过程代理变量建模事件风险与工况变化。

建议的输入分组为：电气状态（`potVolt`、`potCurrent`、`settingVolt`、`settingVoltMin`、`settingVoltMax`）；波动动态（`fluctDelta`、`wavingDelta`、`slopeData`、`sumSlopeData`）；加料操作（`baseFeedingInter`、`actualFeedingInter`、`feedingState`）；设备状态（`aeWaitInterval`、`andoeMoveState`）。首版可用 `potVolt`、`potCurrent`、`fluctDelta`、`wavingDelta`、`slopeData`、`sumSlopeData`、`actualFeedingInter` 和 `feedingState` 组成紧凑输入，`feedingState` 与 `andoeMoveState` 按类别/事件上下文编码；其余变量作为边界和上下文特征。变量是否保留，以训练集 permutation importance/SHAP、时间滚动验证和工艺专家复核共同决定。

在开始训练前需要生成 `metadata.csv` 或等价 JSONL，至少包含以下视觉窗口和路由字段：

```text
task_id, video_id, window_id, frame_paths, frame_start, frame_end,
slot_id, capture_time, weak_source_label, cnh_label,
visual_model_version, quality_label, expert_route, risk_level,
route_reason, case_id
```

其中 `expert_route`、`risk_level` 和 `route_reason` 需要由领域人员审核；`weak_source_label` 只能记录视频文件名提供的弱监督信息。
如果运行时确实提供了可追溯的过程窗口，再额外记录 `process_state_id`、`process_window` 和 `alignment_quality`；历史上无法与视频映射的过程数据不能为了补齐字段而伪造这些值。

## 3. GitHub 项目筛选

### 3.1 优先复用：Routapse

- 仓库：[pooyaphoenix/Routapse](https://github.com/pooyaphoenix/Routapse)
- 许可证：MIT；仓库包含 `LICENSE`，调研时 GitHub API 显示约 5 stars。
- 定位：开源 LLM router。README 描述了 Laya/Jev 路由器、OpenAI-compatible gateway、候选 lane、规则信号、低置信度 fallback、Ollama/云端 provider 和请求日志。
- 可借鉴：`candidates` 结构、`label + confidence + signals` 输出、路由失败后的启发式 fallback、只调用被选中的下游模型、网关入口和 JSONL 运行记录。
- 不能直接复用：它的 Jev 输入是文本 prompt，`router/app/main.py` 的 `ClassifyIn` 只接收 `prompt` 和文本候选描述；它没有火眼图像质量、YOLO 多帧 ROI、F-GMVAE-T 特征或特殊案例向量检索。
- 适配方式：保留它的“路由协议/日志/网关”思想，把输入改为 `InspectionFeatures`，把 `prompt` 换成视觉时序和检索特征；不把 Routapse 的文本路由结果当作火眼分类结果。

### 3.2 优先借鉴：earlyon

- 仓库：[sohams25/earlyon](https://github.com/sohams25/earlyon)
- 许可证：MIT；面向 PyTorch 视觉模型，Python 3.10+、Torch 2.0+。
- 定位：给 ResNet/MobileNet/EfficientNet/ViT 增加中间分类头，支持每个出口独立校准、阈值选择、checkpoint 迁移、batch-1 分阶段 runtime 和公平 benchmark。
- 可借鉴：置信度阈值搜索、温度校准、简单样本早退、困难样本继续完整计算、估算计算量和分阶段运行记录。
- 不能直接复用：它的“出口”发生在单个神经网络内部，而本项目的 `simple_loop/full_loop` 是 LangGraph 诊断工作流分支；earlyon 目前也不是一个通用的 Agent 编排路由器。
- 适配方式：先把现有 ResNet18 保留为基线，在 F-GMVAE-T 的视觉时序输出和低成本证据特征之后增加诊断编排路由头；只有在后续确实需要降低视觉 backbone 推理时间时，再评估 earlyon 的中间出口。

### 3.3 研究参考：Shallow-Deep Networks、LGViT 与 MSDNet

- [Shallow-Deep-Networks](https://github.com/yigitcankaya/Shallow-Deep-Networks)：MIT，ICML 2019，适合理解简单/困难样本的浅层与深层出口。
- [LGViT](https://github.com/falcon-xu/LGViT)：MIT，ACM MM 2023，研究视觉 Transformer 动态早退；依赖较旧，建议只参考设计和评测。
- [MSDNet-Replication](https://github.com/barkinadiguzel/MSDNet-Replication)：MIT，结构容易阅读，但仓库定位是复现示例，不作为本项目生产依赖。
- [early-exit-cnn](https://github.com/Shikha-code36/early-exit-cnn)：MIT，使用 DQN 学习退出策略，数据域是 CIFAR-10；可参考动态退出思想，不建议直接迁移训练代码。

这些项目共同支持一个设计判断：JEV 的价值不在于再堆一个大模型，而在于用保守阈值把低风险、证据充分的任务送入简单小循环，把难例和风险任务送到完整大循环。

### 3.4 安全特征：pytorch-ood 与 SelectiveNet

- [pytorch-ood](https://github.com/kkirchheim/pytorch-ood)：Apache-2.0，提供 OOD 检测、置信度和拒答相关方法。可把 OOD score、energy、embedding 距离作为 JEV 的 `novelty_score` 或 full/human 信号。
- [SelectiveNet](https://github.com/geifmany/selectivenet)：仓库页面未明确给出可直接继承的标准许可证，不建议复制代码；可参考“带 reject option 的选择性分类”思想。

对火眼任务来说，拒答比强行分类更重要：模糊图像、无目标、未知工况、低置信度和证据冲突样本都应该有明确的 `abstain → full/human/retake` 行为。

### 3.5 训练和记忆底座

- [MMPRETRAIN](https://github.com/open-mmlab/mmpretrain)：Apache-2.0，提供分类模型、数据管线、配置、训练和评测框架。适合在样本增加后统一训练/评测，不建议为了当前两段视频马上迁移整个框架。
- [timm / pytorch-image-models](https://github.com/huggingface/pytorch-image-models)：Apache-2.0，提供 MobileNet、EfficientNet、ResNet、ConvNeXt 等 backbone。F-GMVAE-T 可优先用 ConvNeXt-Tiny，ResNet18/MobileNetV3 作为小样本或效率对照。
- [FAISS](https://github.com/facebookresearch/faiss)：MIT，适合建立已审核特殊案例的 embedding 相似检索。先用 F-GMVAE-T 的视觉时序嵌入和元数据过滤，案例规模足够后再上 FAISS 索引。
- [DINOv2](https://github.com/facebookresearch/dinov2)：Apache-2.0，可用于少样本视觉 embedding 和新颖度估计；模型较大，当前只作为后续对照，不作为第一版 JEV 依赖。

预训练权重、数据集和外部模型服务还要单独核对许可证。仓库许可证不自动覆盖第三方权重、火眼视频或现场数据。

### 3.6 不建议直接复制的项目

- [BranchyNet](https://github.com/kunglab/branchynet)：GitHub 页面未声明清晰的开源许可证；只能阅读，不直接复制代码。
- [SkipNet](https://github.com/ucbdrive/skipnet)：研究价值较高，但 GitHub 页面未声明清晰许可证，且实现面向 CNN 内部动态跳过，不适合直接并入当前 Agent 流程。

## 4. JEV 在本项目中的定义

本节的“JEV”均指参考思想；工程模块和汇报名称使用 **Laya-JEV**。代码中的 `JevRouter` 是兼容类名。当前现代入口默认选择 `deterministic_rule`；显式选择 `laya_adapter` 后，适配器才调用 Laya `typed-decisions`。两种后端共用铝电解确定性安全门，实际决策由哪个后端产生会写入 route trace。

Laya-JEV 是本项目的领域适配路由器，表示“任务难度与证据可信度驱动的诊断编排路由器”；其中 JEV 只指参考思想，不指代固定厂商模型。它可以由三层组成：

```text
JEV = 低成本视觉特征 + 特殊案例相似度/OOD + 诊断循环路由策略
```

它的决策对象是“调用哪些诊断 Agent、调用几轮、是否需要强模型和证据质疑”，不是“这张图片属于哪一个火眼类别”。建议保留两个语义层：上游视觉 Agent 输出 `cnh_label` 等观察结果；JEV 输出 `diagnostic_loop` 等编排决定。

推荐输入特征：

| 特征组 | 示例 |
|---|---|
| 图像质量 | Laplacian 模糊度、亮度、过曝比例、分辨率、遮挡、目标面积 |
| 多帧视觉/F-GMVAE-T | C/N/H 概率、风险、不确定性、跨帧一致性、动态征兆、关键证据帧、视觉嵌入距离 |
| YOLO | 是否检测到目标、逐帧 bbox、检测置信度、目标面积、ROI 对齐质量 |
| 案例 | top-k 相似度、案例标签一致性、槽型/工况适用性、案例版本状态 |
| 知识 | 关键规则命中数、有效来源数、证据覆盖率、来源冲突 |
| 过程上下文 | 仅在运行时可靠提供的 `ProcessStateSnapshot`、工况状态、异常分数、过程标签、`alignment_quality` |
| 历史 | 人工改判、补拍、失败、full 路径命中和报告修订记录 |

推荐输出：

```json
{
  "route": "simple_loop|full_loop|retake|human",
  "difficulty_score": 0.0,
  "novelty_score": 0.0,
  "confidence": 0.0,
  "risk_level": "low|medium|high",
  "similar_case_ids": [],
  "evidence_gaps": [],
  "reasons": [],
  "process_context_used": false,
  "alignment_quality": "unknown",
"model_profile": "deterministic_alu_router_v1",
"gate_version": "alu-jev-safety-0.1"
}
```

路由标签的定义：

- `simple_loop`：质量合格、低风险、证据充分、低成本结果稳定、可靠特殊案例或规则可解释；调用少量诊断节点即可形成报告。
- `full_loop`：低置信度、边界类别、陌生样本、无可靠案例、知识不足、模型冲突或需要多 Agent 取证；进入完整大循环。
- `retake`：模糊、过暗、过曝、遮挡、无目标或关键证据不可见，需要补拍。
- `human`：高风险、越权请求、严重冲突、分布外、规则无法判定或 JEV 自身异常。

现有 C/N/H 结果只能作为 JEV 输入特征，不能直接作为 `simple_loop/full_loop` 标签；视频文件名的“正常/偏高”也只能作为弱监督来源。

未与视频建立可靠映射的过程时序数据不能作为 JEV 在线输入。它们由 Process State Agent 独立提取工况状态和过程规则，只有运行时同时提供过程窗口且 `alignment_quality` 达到门槛时，才作为任务级上下文参与路由；否则保留为离线知识/策略库。

### 4.1 非配对过程数据与 JEV 边界

当前 14 维过程变量是独立的多变量时序集合，进入 Process State Agent 后输出工况状态、事件风险和变化点，再参与工艺知识、案例检索与 JEV 编排。

建议单独训练 **Process State Agent**：按日期、槽位或数据文件分组，提取稳定性、趋势、斜率、波动、变化点和缺失模式，输出过程状态和模糊标签，并写入版本化知识库、案例库和 JEV 策略库。推荐输出协议为：

```json
{
  "process_state_id": "ps-...",
  "process_regime": "stable|transition|abnormal|unknown",
  "anomaly_score": 0.0,
  "process_tags": [],
  "evidence_window": {"start": "...", "end": "..."},
  "alignment_quality": "exact|interval|segment|weak|unknown",
  "process_missing": true
}
```

JEV 只有在当前巡检任务同时携带可审计的过程窗口，且 `alignment_quality` 达到预先设定的门槛时，才把这个快照作为可选的任务级上下文；否则 JEV 只使用视觉窗口、案例和知识证据。即使发生晚融合，也应把过程快照标记为上下文证据，而不是把它重新解释为视频分类真值。

### 4.2 两条诊断循环

**简单小循环**用于低风险、可解释、已有可靠依据的任务：

```text
图像质量门控
  → YOLO 多帧 ROI + F-GMVAE-T 低成本取证
  → 特殊案例/关键规则快速命中
  → 轻量报告 Agent 或模板化摘要
  → 确定性证据检查
  → 人工确认/归档
```

**完整大循环**用于复杂、陌生、边界、冲突或高风险任务：

```text
图像质量门控
  → 多源视觉取证
  → 工艺知识 RAG || 特殊案例深度比对
  → Qwen-VL 诊断聚合
  → 证据质疑/独立复核
  → 人工确认/补拍/归档
```

简单小循环仍然经过安全检查和人工确认，不得因为 JEV 认为任务简单就绕过最终授权边界。完整大循环不是对每一帧都调用所有 Agent，而是只对 JEV 判断为复杂或风险高的任务启用更多证据源和更强模型。

## 5. 小规模数据训练路线

### 5.1 第一步：从视频生成可追溯清单

不要把约 3600 帧全部当成独立样本。先按视频、时间段、槽位和场景抽取短视频窗口，例如每个窗口 8/16 帧、stride 2/4，并保存原视频时间戳、窗口起止时间和质量标签。窗口内部可使用感知哈希去除完全重复片段；抽样策略需要用可视化检查确认，不是固定的最终标准。

数据切分必须按 `video_id / slot_id / date / scene_id` 分组：

```text
train：若干视频或场景组
val：未出现在 train 的视频/场景组
test：独立视频、槽位或新工况
```

目前只有两段视频，不能形成可信的 train/val/test。可以做“留一视频验证”用于调试，但结果只能写成开发观察，不能作为申报性能指标。

### 5.2 第二步：建立路由标签

初期采用“规则弱标注 + 专家复核”：

```text
retake：质量指标未通过或目标缺失
simple_loop：质量通过 + 初筛置信度高 + 结果稳定 + 低风险 + 有可靠规则/案例，少量诊断节点即可闭环
full_loop：边界置信度、无匹配案例、知识不足、OOD、类别冲突或人工需要详细证据
human：高风险、越权、严重冲突、无法解释或必须人工授权
```

弱标注只用于生成候选，最终 `expert_route` 必须由领域人员审阅并填写 `route_reason`。相似案例命中不能单独产生 `simple_loop` 标签；案例还必须满足场景、槽型、风险和版本适用条件。

### 5.3 第三步：模型顺序

按下面顺序逐步增加复杂度：

1. **规则基线**：使用 YOLO 多帧质量、F-GMVAE-T 置信度/一致性、图像质量和案例阈值，先跑通四路诊断编排状态机。
2. **表格路由器**：使用 Logistic Regression、Random Forest 或 HistGradientBoosting 融合上述特征，学习“简单小循环是否足够”的判断。小样本优先使用冻结视觉时序模型的输出，不从零训练 CNN。
3. **轻量诊断编排路由头**：使用 F-GMVAE-T 的冻结/部分微调视觉时序特征，加上质量、案例、证据和可选过程状态特征，预测 diagnostic-loop；C/N/H 只作为辅助证据，输出仍表示诊断编排决策。
4. **独立过程状态分支**：以 `Code/data/v1` 的 30 步历史 → 未来 15 步事件预警任务训练 Process State Agent，输出 `event_risk`、工况状态、变化点和过程标签；无可靠映射时只做知识/策略输入，不与视觉样本直接拼接。
5. **OOD/相似案例增强**：增加 embedding 距离、pytorch-ood 方法或 FAISS top-k 相似度，单独做消融。
6. **动态早退**：数据足够且视觉 backbone 成为瓶颈后，才考虑 earlyon 的中间出口；它不是当前第一版 JEV 的必要条件。

小样本训练应使用 class weight 或 focal loss 处理类别不平衡，并用 temperature scaling 或 isotonic regression 做概率校准。图像增强不能改变火眼颜色、结壳和黑色冷料等关键征兆，增强策略需要由领域人员检查。

### 5.4 特殊案例库

特殊案例必须分层管理：

```text
candidate → expert_review → eval_only / approved_special_case
approved_special_case → active / deprecated
```

每个正式案例保存原图/ROI 哈希、embedding、场景/槽型、可见征兆、风险、适用条件、人工结论、现场审核事项、知识来源、模型和版本信息。只有人工批准且通过冻结回归集的案例才能参与 JEV 快速路由；驳回、补拍、知识缺失和人工改判案例保留为负例，不得丢弃。

## 6. 评测方案

### 6.1 JEV 路由指标

- 四路路由混淆矩阵；
- `simple_loop` 的 Precision、Recall 和 coverage；
- 复杂样本进入 `full_loop` 的召回率；
- 高风险/危险样本误进入 `simple_loop` 的漏放率，作为最高优先级安全指标；
- `retake` 召回率和 `human` 升级准确率；
- ECE、Brier score 和阈值稳定性；
- 特殊案例 top-k 命中率、适用条件过滤准确率和人工改判率。

### 6.2 端到端收益

比较无 JEV 的固定完整大循环和有 JEV 的 simple_loop/full_loop 编排：

- 端到端 P50/P95 延迟；
- 首次响应时间；
- Qwen API 调用次数、Token 和成本；
- 诊断质量、证据链完整率、知识引用支持率；
- 人工复核时间、修改率和最终采纳率；
- 节点失败、重试、fallback 和补拍率。

只有在质量和安全指标不恶化的条件下，平均延迟/Token 下降才算 JEV 有效。需要至少做以下消融：

```text
固定全流程
固定规则分流
JEV 诊断循环路由
JEV + OOD
JEV + 特殊案例检索
JEV + 特殊案例适用条件门控
```

### 6.3 视觉时序与过程状态的独立评测

视觉分类本身需要单独比较：

```text
单帧 ResNet 基线
单帧 ConvNeXt/轻量 ViT
多帧视觉时序编码
多帧 + 静态/动态专家特征
F-GMVAE-T 完整融合
```

除 C/N/H 的 Macro-F1、风险等级一致率和校准误差外，还应报告跨帧一致性、关键证据帧命中率、模糊/无目标样本的拒答率以及窗口级而不是帧级的错误率。过程状态 Agent 另行报告工况状态 Macro-F1、异常检测 AUROC/AUPRC、变化点误差、缺失数据鲁棒性和规则审核一致率。过程数据没有可靠视频映射时，不报告样本级视觉-过程融合指标，也不把独立过程模型的结果写成视频分类精度。

JEV 的效率收益必须与上述视觉质量和过程状态安全指标分开呈现。推荐至少增加“视觉单帧 + JEV”和“视觉多帧/F-GMVAE-T + JEV”的对照，以及“视觉 + 独立过程状态上下文”的路由/解释/案例检索对照；后一项只在有可靠运行时过程快照时启用，不构成未配对数据的融合训练声明。

## 7. 推荐实施顺序

1. 把短视频窗口抽取、去重、元数据和视频级分组切分跑通；补齐人工 `expert_route` 和 `route_reason`。
2. 先实现规则版 JEV，不接大模型，用 YOLO 多帧 ROI、F-GMVAE-T 输出和图像质量信号验证四路状态机和日志。
3. 并行建立 Process State Agent 的独立数据集、状态标签和评测，不把未配对过程数据混入视觉训练集。
4. 建立特殊案例候选/审核/发布流程；数据量达到可检索规模后接 FAISS。
5. 用表格特征训练第一个轻量诊断编排路由器，做校准和危险样本漏放率评测；只有可靠运行时过程快照才作为可选上下文特征。
6. 再接入 Routapse 风格的统一路由协议和 fallback，并把 JEV 决策写入 LangGraph TraceEvent。
7. 需要进一步降低视觉模型耗时后，再评估 earlyon 的中间出口；不要一开始同时引入早退、OOD、向量库和新训练框架。

当前阶段不应报告“JEV 已训练完成”或“诊断编排准确率已提升”。准确表述是：已完成 JEV 诊断编排路由架构和开源复用调研，当前火眼帧目录为空、过程数据与视频未建立可靠映射，下一步需先完成短视频窗口同步、分组标注、F-GMVAE-T/Process State Agent 基线和规则版 JEV。

## 8. 许可证和数据合规

优先考虑 MIT、Apache-2.0 项目；复制代码时保留原始版权和许可证文件。MIT/Apache 许可证不代表第三方预训练权重、外部数据或 API 服务的授权自动成立。BranchyNet、SkipNet 和 SelectiveNet 页面未明确给出可直接继承的标准许可证，当前只阅读设计，不复制代码。火眼视频属于项目数据，训练、上传云端模型和写入案例库前要确认数据权限和脱敏边界。

## 9. 关联代码和文档

- 全局架构规划：`Code/docs/铝电解多智能体全局规划.md`
- Agent 工作流：`Code/agentic/workflow.py`
- 现有视觉模型入口：`Code/qt_interface/algo_fireeye.py`
- 现有权重：`Code/weights/best.pt`、`Code/weights/resnet18_fireeye_3cls.pth`
- 现有视频：`Code/data/video/正常.mp4`、`Code/data/video/偏高.mp4`
- 现有 API 客户端：`Code/llm_api_client.py`
