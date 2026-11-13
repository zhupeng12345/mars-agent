# MARS-Agent：铝电解火眼多智能体诊断系统

这是项目的现代主线版本。系统以巡检图片/短视频为输入，经过质量门控、多帧视觉取证、独立过程状态先验、工艺知识与特殊案例检索，由 **JEV 参考思想 + Laya 可选适配器 + 确定性安全门控** 选择诊断路径，随后通过 LangGraph 完成多 Agent 协作、证据复核和人工确认。

## 当前主链路

```text
图片/视频
  -> 媒体标准化与质量门控
  -> 多帧 ROI / F-GMVAE-T 接口（CPU smoke）
  -> Process State Agent（14 维过程数据独立先验）
  -> 工艺知识 / 特殊案例检索
  -> JEV 路由：simple_loop | full_loop | retake | human
  -> LangGraph fan-out/fan-in 与证据复核
  -> interrupt/resume 人工审核
  -> 报告、审计、候选案例与 RSI 离线演化
```

代码入口在 [`Code/agentic`](Code/agentic)，网页工作台在 [`Code/web_interface/streamlit_app.py`](Code/web_interface/streamlit_app.py)。完整规划、阶段闸门和当前中断进度见：

- [`Code/docs/铝电解多智能体完整实现执行计划.md`](Code/docs/铝电解多智能体完整实现执行计划.md)
- [`Code/docs/铝电解多智能体全局规划.md`](Code/docs/铝电解多智能体全局规划.md)
- [`Code/docs/铝电解多智能体执行计划逐项核对报告.md`](Code/docs/铝电解多智能体执行计划逐项核对报告.md)
- [`Code/docs/发布整理说明.md`](Code/docs/发布整理说明.md)
- [`Code/docs/implementation_log.md`](Code/docs/implementation_log.md)

## 快速开始

```powershell
cd Code
$py = 'D:\py_daima\.venvs\mars-agent\Scripts\python.exe'
& $py -m pytest agentic/tests -q
& $py -m agentic.scripts.inventory --output runtime_data/inventory.json
& $py -m agentic.scripts.route_smoke --output runtime_data/route_smoke.json
& $py -m agentic.evaluation.run_all --output runtime_data/evaluation_report.json
streamlit run web_interface/streamlit_app.py --server.port 8503
```

当前本地验收基线：`212 passed, 1 warning, 39 subtests`。warning 是已有 PyTorch Transformer nested-tensor 配置提示，不影响测试通过。评测 fixture 和报告明确标记为流程/结构验证，不能当作专家标注下的视觉准确率、路由准确率或真实过热度测量精度。

## 配置模型

复制 [`Code/.env.example`](Code/.env.example) 为 `.env` 后填写 Qwen/DashScope Key，或使用本地 Ollama。模型 profile 和 OpenAI 兼容网关配置见 [`Code/config`](Code/config)。密钥文件、运行记录、原始数据和浏览器缓存均不会进入仓库。

## 数据与大文件策略

GitHub 发布包保留代码、知识规则、评测 schema/fixture、报告素材和规划文档；本地原始时序数据、视频、火眼帧图片、训练中间文件、运行缓存和教学资料被 `.gitignore` 排除。将数据放入本地 `Code/data/v1`、`Code/data/video` 或 `Code/data/火眼分类图片帧` 后，按照文档中的 manifest 命令重建即可。

基础视觉权重位于 `Code/weights`，因为旧版兼容视觉入口仍依赖它们；新增训练 checkpoint 不应直接提交。完整 F-GMVAE-T 训练、专家视觉/路由标签、视频-过程可靠对齐和冻结研究集仍是后续论文级工作。

## 状态边界

当前已经具备可本地回放的多 Agent 工程链路、四路 JEV fixture、人工审核和 RSI 受控离线流程。尚未完成且不会被流程 fixture 冒充的项目包括：GPU 级 F-GMVAE-T 训练、专家标注下的 Macro-F1/路由指标、真实 Laya fit/calibration、BGE-M3/Milvus/Reranker、完整消融和在线 RSI 质量收益。

## 许可证与第三方

复用范围和许可证记录在 [`Code/docs/third_party_manifest.md`](Code/docs/third_party_manifest.md)。本仓库用于研究与项目汇报，生产部署前需补齐数据授权、模型权重许可和工艺专家审核。
