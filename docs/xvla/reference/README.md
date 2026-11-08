# 上游源码参考副本

`base_policy.py` 原样来自：

- 仓库：`RLinf/RLinf`
- commit：`1bf78980d9339cbc67497eddceb9a1e3a74e2acc`
- 路径：[`rlinf/models/embodiment/base_policy.py`](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/embodiment/base_policy.py)
- 许可：Apache-2.0，保留原文件头；本仓库 [LICENSE](../../../LICENSE) 包含许可全文。

副本用于离线阅读；它不属于本项目 `rlinf` 运行时，也没有把旧 worker 升级为新接口。BasePolicy 本身不是 `torch.nn.Module`，也没有实现概率、采样或权重加载。

规范化 LF 源文件的 SHA256、其他已核对上游文件和固定版本链接见 [upstream-reference.json](../upstream-reference.json)。修改接口方案时更新主文档，保持此参考副本与该 commit 一致；采用其他上游 commit 时显式重新审核并更新清单。
