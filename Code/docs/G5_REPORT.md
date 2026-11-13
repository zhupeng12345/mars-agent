# G5 阶段报告：冻结评测与消融

- 状态：框架通过，真实视觉消融待标签/GPU。
- 实现：冻结 manifest、视觉标签感知指标、JEV/证据/延迟/系统健康指标、评测总入口、9 组消融矩阵和结果 provenance gate；健康指标包含失败、超时、重试、fallback、结构化交接和人工采纳/修改比例。`run_all` 同时输出 JSON、Markdown/CSV 表、PNG 汇总图和失败样例 JSONL。
- 当前回放：路由 fixture 通过；视觉报告为 `structure_validation_only`；消融为 0 completed、8 structure_only、1 not_run；全量测试 59 passed（另有 1 个 Transformer warning 和 7 个 subtests）。
- 规则：任何数值结果必须带数据版本、切分、随机种子、模型和生成时间，缺失条件不生成论文指标。

## 2026-11-08 当前复核补充

- 全量 pytest 更新为 `78 passed, 1 warning, 7 subtests passed`；`runtime_data/evaluation_current.json`、Markdown、CSV、PNG 和失败样例已重新生成。
- 过程分支仍是独立状态先验：测试集 1DCNN ROC-AUC `0.5268`、PR-AUC `0.5608`、事件 Recall `0.88`；与视频没有可审计时间映射，`alignment_quality=unknown`。
- 图像级监督结果单独列为 experimental baseline，不纳入视频 Macro-F1 或过热度测量结论。
