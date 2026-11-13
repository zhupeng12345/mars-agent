# G6 阶段报告：RSI 受控离线演化

- 状态：通过
- 实现：trajectory/feedback、候选 diff、train/frozen/sealed 回放、质量/安全/效率 gate、人工批准、版本谱系、发布和回滚 CLI。
- 允许变更：提示词、JEV 非安全阈值/模板、知识查询模板、报告模板。
- 禁止变更：冻结集、评测器、安全规则、权重、人工结论和生产控制配置。
- 验收：RSI 专项测试和已知引用错误 smoke 通过。

## 2026-11-08 当前复核补充

- `runtime_data/rsi/rsi_defect_demo.json` 完成已知缺陷“缺少 source citation”的离线闭环：candidate diff → train/frozen/sealed replay → quality/safety/efficiency gate → 人工批准 → publish → rollback，`gate_accepted=true`、`rollback_verified=true`。
- 回放层现在拒绝空切分；闸门要求三段 baseline/candidate 都有有效非空样本，避免空数据得到虚假的满分。
