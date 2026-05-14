# 当前项目不足与改进计划

这份文档用于诚实说明当前项目边界。面试时建议主动讲出这些点，这比被老师指出来更稳。

## 已完成

- 实现了轻量 no-CFG iMeanFlow action chunk policy。
- 实现了 `u` / `v` 双头网络、JVP 训练目标和少步采样器。
- 完成了 synthetic arm dataset、2D planar reaching demo、3D MuJoCo reaching demo。
- 搭建了 Franka Panda push-block viewer，用于后续真实机械臂风格任务。
- 在 companion FlowPolicy benchmark 中完成 Push-T lowdim iMeanFlow 训练评测。
- README 中展示了 iMeanFlow rollout、eval log、实机视频和 Flow Matching baseline。

## 主要不足

### 1. iMeanFlow 当前分数不如 Flow Matching

当前 iMeanFlow Push-T test mean score 是 `0.614`，Flow Matching baseline 是 `0.818`。这说明 iMeanFlow 实现和评测链路已经跑通，但还不能声称性能优于 Flow Matching。

合理解释：

- iMeanFlow 的目标是减少采样步数，不是天然保证同训练预算下分数更高。
- 当前只展示了一个训练结果，缺少系统调参和 ablation。
- 少步生成对训练目标、采样分布和模型容量更敏感。

### 2. 缺少 NFE 消融实验

目前 README 只写了 iMeanFlow `NFE=2`。还需要补：

| NFE | Test score | Inference time | Action smoothness |
| ---: | ---: | ---: | ---: |
| 1 | TODO | TODO | TODO |
| 2 | 0.614 | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |

这张表是后续最有价值的实验。

### 3. 真实机器人部分还只是定性展示

`assets/shiji.mp4` 是实机视频展示，但不是严格的真实机器人 benchmark。当前还没有：

- 多 episode 成功率；
- 失败案例统计；
- 真实控制频率记录；
- 真实动作误差记录；
- sim-to-real 对比。

### 4. 轻量仓库和 Push-T benchmark 代码还没有完全合并

这个仓库展示了轻量实现和结果；Push-T benchmark 的训练代码在 companion `flowpolicy` 项目中。后续最好把关键配置和说明统一到一个 `benchmarks/` 目录，或者在 README 中明确给出 companion repo 链接和 commit。

### 5. MuJoCo push-block 还不是完整训练 benchmark

当前 push-block 部分更接近：

- 场景搭建；
- scripted expert；
- viewer 调试；
- 单条 demonstration 保存。

还没有完成：

- 大规模 demonstration collection；
- dataset loader；
- closed-loop policy training；
- quantitative success metric。

## 面试时如何表述

建议这样说：

> 这个项目目前不是 SOTA claim，而是一个把 iMeanFlow 迁移到机器人动作生成的工程研究项目。我已经完成了方法实现、训练闭环、Push-T 评测和可视化展示。当前最主要的不足是 iMeanFlow 分数还低于 Flow Matching，并且缺少 NFE / latency / smoothness 的系统消融。下一步我会把这些指标补齐，证明少步生成在实时控制里的价值。

## 下一步计划

优先级从高到低：

1. 跑 `1/2/4/8` NFE 消融。
2. 记录推理时间，计算每个 action chunk 的 wall-clock latency。
3. 加 action smoothness metric，例如相邻动作差分平方和。
4. 统一 FlowPolicy benchmark 代码和本仓库说明。
5. 扩展 Franka / FR3 MuJoCo 数据采集，形成完整 push-block benchmark。
