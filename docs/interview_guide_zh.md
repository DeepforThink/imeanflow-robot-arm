# iMeanFlow Robotics 面试讲解手册

这份文档用于面试或项目答辩。建议按“项目目标 -> 算法原理 -> 代码实现 -> 实验结果 -> 局限与下一步”的顺序讲，不要一开始就陷入公式。

## 1. 一句话介绍

这个项目把 Flow Matching / Mean Flow 的生成建模思想迁移到机器人动作生成中，用 no-CFG 条件 iMeanFlow 从机器人观测生成未来一段可执行的 action chunk，并在轻量机械臂仿真、MuJoCo demo 和 Push-T lowdim benchmark 上做了验证。

## 2. 我到底做了什么

项目分成两层：

第一层是这个仓库里的轻量实现：

- `src/imeanflow_robotics/policy.py`：iMeanFlow loss、JVP 训练目标、少步采样、动作队列。
- `src/imeanflow_robotics/model.py`：observation-conditioned Transformer，包含 `u` 和 `v` 两个预测头。
- `src/imeanflow_robotics/data.py`：合成机械臂动作数据，用于快速验证训练闭环。
- `src/imeanflow_robotics/sim.py`：2D 平面机械臂 reaching 仿真。
- `src/imeanflow_robotics/sim3d.py`：3D reaching 数据和正运动学。
- `scripts/mujoco_3d_demo.py`：MuJoCo 3D 机械臂 reaching demo。
- `scripts/franka_push_block_viewer.py`：Franka Panda 桌面推块交互 viewer。

第二层是 companion FlowPolicy benchmark：

- 基于 Diffusion Policy 的 lowdim Push-T 训练和评测框架。
- 新增 Flow Matching policy，作为 baseline。
- 新增 no-CFG iMeanFlow policy，保持同样的数据、环境、normalizer、workspace 和 eval runner。
- 当前 iMeanFlow Push-T 评测结果已经导出到本仓库的 `eval_outputs/`，README 中展示了 rollout GIF 和实机视频预览。

## 3. 为什么机器人动作可以用生成模型

机器人策略不一定只输出单步动作。Diffusion Policy 和 pi0 一类工作说明，可以让策略一次生成未来一段动作序列，也就是 action chunk。这样做有三个好处：

1. 动作之间更连贯，不容易每一步都抖。
2. 策略可以提前规划短时轨迹。
3. 控制循环里只执行 chunk 的前几步，然后 receding horizon 地重新规划。

在这个项目中，action chunk 被看成一个高维向量：

```text
x = [a_1, a_2, ..., a_H]
```

模型学习从高斯噪声 `e` 变成专家动作 chunk `x` 的过程。

## 4. Flow Matching 原理

最简单的线性路径是：

```text
z_t = (1 - t) * x + t * e
```

其中：

- `x` 是真实动作 chunk；
- `e` 是高斯噪声；
- `t=0` 时是数据；
- `t=1` 时是噪声。

这条直线的速度是：

```text
v_target = e - x
```

Flow Matching 训练一个网络 `v_theta(z_t, t, obs)` 去预测这个速度。推理时从噪声出发，通过 Euler 积分往 `t=0` 走：

```text
z <- z - dt * v_theta(z, t, obs)
```

你可以这样讲：Flow Matching 不是一步一步预测去噪残差，而是直接学“应该往哪个方向流动”。

## 5. Mean Flow / iMeanFlow 原理

普通 Flow Matching 学的是某一个时刻的瞬时速度。问题是：如果推理时只走 1 到 2 步，瞬时速度的 Euler 误差会比较大。

Mean Flow 的想法是让模型学习一个时间区间上的平均速度：

```text
u(z_t, h, obs),  h = t - r
```

这里 `u` 不只是某个时刻的速度，而是希望表示从 `t` 到 `r` 这个区间上可以直接走大步的平均速度。

本项目里的 iMeanFlow 训练两个头：

```text
u(z_t, h, obs)      # 区间平均速度
v_hat(z_t, h, obs)  # 辅助瞬时速度
```

`v_hat` 在 `h=0` 时提供 JVP 的切向方向。训练目标写成：

```text
v_tangent = v_hat(z_t, h=0, obs)
dudt = JVP(u, direction=(v_tangent, dt=1))
V = u + h * stopgrad(dudt)
loss = ||V - (e - x)||^2 + lambda_v * ||v_hat - (e - x)||^2
```

直觉解释：

- `u` 负责让模型学会“大步走”；
- `v_hat` 负责提供局部瞬时速度方向；
- JVP 描述 `u` 沿着运动方向变化时会产生的修正；
- `stopgrad` 防止训练目标本身反向传播得过于复杂，保持优化稳定。

## 6. 为什么不做 CFG

CFG 需要两个前提：

1. 训练时有 unconditional branch；
2. 推理时同时算 conditional 和 unconditional，再做插值增强。

机器人动作生成和图像文本生成不同。机器人动作强依赖当前观测、关节状态、目标、历史动作或图像。如果没有真正训练 unconditional 分支，只在代码里放一个 `omega` 参数，就是表面功能。

所以本项目选择 no-CFG：

- 更诚实；
- 更符合当前 robot policy 的条件控制设定；
- 推理时少一次网络前向，更符合低延迟目标。

如果老师追问“以后能不能加 CFG”，回答是：可以，但必须在训练时加入 condition dropout，并显式训练 unconditional 分支，否则没有理论意义。

## 7. 代码细节怎么讲

### `policy.py`

重点讲四个函数：

- `sample_timesteps`：采样 `t >= r`，一部分样本设置 `r=t`，用于稳定局部速度学习。
- `compute_loss`：核心 iMeanFlow loss。
- `sample_action_chunk`：从高斯噪声出发，用少步 Euler 生成完整动作 chunk。
- `select_action`：只执行 chunk 的前几步，并用队列缓存，符合 receding-horizon 控制。

### `model.py`

模型输入包括：

- noisy action tokens；
- observation token；
- interval length `h` 的 time token。

输出是两个 action-space head：

- `u_out`：平均速度；
- `v_out`：辅助瞬时速度。

### Push-T benchmark 代码

真实 Push-T 分数来自 companion `flowpolicy` 项目，而不是这个轻量仓库的 synthetic demo。对应文件是：

```text
diffusion_policy/policy/imeanflow_unet_lowdim_policy.py
diffusion_policy/model/diffusion/conditional_unet1d_dual_head.py
diffusion_policy/config/train_imeanflow_unet_lowdim_workspace.yaml
```

这部分保持 Diffusion Policy 的 dataset、normalizer、workspace、env runner 和 eval 逻辑不变，只替换 policy objective 和 sampler。

## 8. 当前结果怎么讲

README 中展示的是当前训练得到的 iMeanFlow 结果：

```text
iMeanFlow lowdim Push-T
NFE: 2
test seeds: 50
test mean score: 0.614
train mean score: 0.544
```

Flow Matching baseline：

```text
test seeds: 50
test mean score: 0.818
```

正确解释方式：

- iMeanFlow 当前没有超过 Flow Matching；
- 它的意义是验证“少步 action generation”路径可以跑通；
- 下一步要做的是 NFE ablation、训练轮数、loss 权重、采样分布和动作平滑性对比；
- 不能把它包装成 SOTA。

## 9. 老师可能会问的问题

### Q1：你这个和 Diffusion Policy 的区别是什么？

Diffusion Policy 用多步 denoising 生成 action chunk；我的项目把动作生成写成 flow matching / mean flow 形式。核心区别在训练目标和采样器：我不是预测噪声或 denoising residual，而是预测从噪声到动作的速度场，iMeanFlow 进一步学习区间平均速度以减少 NFE。

### Q2：为什么 iMeanFlow 分数低于 Flow Matching，还有意义吗？

有意义，但不能夸大。当前结果说明实现和评测链路跑通了，少步采样能完成一部分任务。Flow Matching 分数更高，说明当前 iMeanFlow 还需要调参和更长训练。这个项目的价值在于把方法迁移到机器人动作生成，并建立可对比的实验框架。

### Q3：JVP 在这里到底干什么？

JVP 是计算 `u` 沿着运动方向变化的方向导数。因为我们希望 `u` 表示一个时间区间的平均速度，而不是普通瞬时速度，所以训练目标中要考虑 `u` 随着状态和时间变化产生的修正项。

### Q4：`u` 和 `v` 为什么要两个头？

`u` 是最终采样用的平均速度；`v` 是辅助瞬时速度，用来在 `h=0` 时提供 JVP 的切向方向，也用一个 MSE 辅助 loss 稳定训练。推理时主要用 `u`。

### Q5：为什么不用 CFG？

因为没有训练 unconditional branch。机器人动作本来就是强条件生成，盲目加 CFG 会让代码看起来像有功能，但实际没有对应训练依据。

### Q6：这个项目有哪些不足？

不足包括：

- iMeanFlow 当前 Push-T 分数还低于 Flow Matching；
- 没有完整 NFE ablation；
- 真实机器人视频是定性展示，不是大规模真实机器人评测；
- 轻量仓库和 Push-T benchmark 代码分在两个项目里，复现路径还可以继续统一；
- MuJoCo push-block 目前是 viewer / scripted prototype，不是完整闭环 benchmark。

## 10. 后续最该做什么

优先级最高的三件事：

1. 跑 iMeanFlow 的 NFE ablation：`1-step / 2-step / 4-step / 8-step`。
2. 和 Flow Matching 在相同训练步数、相同 seeds 下比较成功率、平滑性和推理时间。
3. 把 companion FlowPolicy benchmark 的关键 iMeanFlow 文件整理进一个清晰的分支或子目录，降低复现成本。

如果时间更多，再做：

- 改进 loss 权重和 `t, r` 采样分布；
- 引入 action smoothness metric；
- 加入 Franka / FR3 的更真实 MuJoCo 数据采集；
- 整理为 Hugging Face dataset / model card。
