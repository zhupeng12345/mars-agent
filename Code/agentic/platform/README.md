# 铝电解 Agent 平台适配层

这一层吸收了 `baiduwangpan/_extracted/EduAgent/source` 中可复用的工程模式：

- 统一请求/响应协议：`InspectionTask`、`AgentResult`、`EvidenceRef`；
- LangGraph 线程隔离：按租户、会话和运行 ID 创建 checkpointer 配置；
- 事件轨迹：每个 Agent 节点写入可回放的状态、耗时和摘要；
- Human-in-the-loop：模型只能产生候选案例，人工决策单独记录；
- 本地 JSONL 存储：先零基础设施跑通，后续可替换为 PostgreSQL/Redis/Milvus。

EduAgent 的 QA、考试、简历和面试业务没有复制进来。当前领域 Agent 仍由
`agentic/workflow.py` 提供，平台层只负责任务协议、记忆和审计，避免教育领域
状态污染铝电解诊断。
