# 数据目录说明

原始数据不随 GitHub 发布包提交。请将获得授权的本地数据放入：

- `v1/`：14 维过程时序数据与处理结果；
- `video/`：巡检视频；
- `火眼分类图片帧/`：图片帧或弱标注素材；
- `evaluation/`：已纳入版本控制的评测 schema、fixture 和标注模板。

`v1/`、`video/` 和图片帧目录已加入 `.gitignore`。没有视频时间戳/槽位映射时，Process State Agent 只能作为独立过程先验，输出 `alignment_quality=unknown`，不能作为视频标签。
