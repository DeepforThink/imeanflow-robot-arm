# Flow Matching、Mean Flow 与 iMeanFlow 算法笔记

## 1. 从 diffusion 到 flow matching

Diffusion policy 通常把动作生成看作一个逐步去噪过程。训练时模型学习如何从带噪动作恢复干净动作，推理时从高斯噪声开始，多步迭代得到动作 chunk。

Flow Matching 换了一个角度：不显式模拟逐步加噪和反向去噪，而是在连续时间路径上学习速度场。

给定专家动作 chunk `x` 和高斯噪声 `e`，线性路径为：

```text
z_t = (1 - t) x + t e
```

速度目标为：

```text
v = dz_t / dt = e - x
```

训练目标：

```text
min || v_theta(z_t, t, obs) - (e - x) ||^2
```

推理时从 `z_1 = e` 出发，向 `z_0 = x` 积分：

```text
z_{t-dt} = z_t - dt * v_theta(z_t, t, obs)
```

## 2. Flow Matching 在机器人动作里的含义

在图像生成里，`x` 是图片；在机器人里，`x` 是未来动作序列：

```text
x = [a_1, a_2, ..., a_H]
```

条件 `obs` 可以是：

- 当前关节状态；
- 历史观测；
- 目标位置；
- 图像特征；
- 语言或任务指令。

因此 policy 学的是：

```text
given obs, transform noise into a plausible future action chunk
```

## 3. Mean Flow 的核心动机

普通 Flow Matching 的速度是瞬时速度。多步 Euler 积分时，每一步都重新估计速度，所以误差可以逐步修正。

但是如果希望一步或两步生成，瞬时速度不一定适合直接走大步。Mean Flow 的核心想法是学习区间平均速度：

```text
u(z_t, t, r, obs)
```

它表示从时间 `t` 到 `r` 这段区间上应该使用的平均速度。这样采样时可以直接做：

```text
z_r = z_t - (t-r) * u(z_t, t, r, obs)
```

这就是少步采样的来源。

## 4. iMeanFlow 的实现形式

本项目使用 `h = t - r` 表示区间长度，模型输入是：

```text
z_t, h, obs
```

模型输出两个 head：

```text
u     # interval-average velocity
v_hat # auxiliary instantaneous velocity
```

训练中：

1. 采样动作 `x` 和噪声 `e`；
2. 采样 `t >= r`；
3. 构造 `z_t = (1-t)x + t e`；
4. 用 `v_hat(z_t, h=0, obs)` 估计局部瞬时速度；
5. 用 `torch.func.jvp` 计算 `u` 沿局部运动方向的方向导数；
6. 构造 compound velocity；
7. 对齐 straight-line velocity `e - x`。

代码中的关键公式是：

```text
v_tangent = v_hat(z_t, h=0, obs)
dudt = JVP(u, direction=(v_tangent, dt=1))
V = u + h * stopgrad(dudt)
loss = || V - (e-x) ||^2 + lambda_v || v_hat - (e-x) ||^2
```

## 5. 为什么需要 `v_hat`

`u` 是最终推理使用的平均速度，但 JVP 需要一个方向。`v_hat` 提供这个局部运动方向，并通过辅助 MSE loss 保持它接近 Flow Matching 的瞬时速度目标。

可以把二者理解成：

- `v_hat`：告诉模型“当前这一点应该朝哪个方向流”；
- `u`：告诉模型“如果要跨一个时间区间，平均应该怎么走”。

## 6. 为什么要 `stopgrad`

compound target 里包含模型自己的 JVP。如果所有项都完整反传，目标会强依赖当前网络本身，优化会更复杂、更不稳定。

本项目对 JVP 修正项使用 detach：

```text
V = u + h * dudt.detach()
```

这样训练更接近“用当前估计构造一个修正目标，然后训练 `u` 去拟合它”。

## 7. 采样过程

推理时只使用 `u`：

```text
z = noise
for each interval:
    u, _ = model(z, h, obs)
    z = z - h * u
return z
```

如果 `num_inference_steps=2`，就是从 `t=1 -> 0.5 -> 0` 两步生成 action chunk。

## 8. 和 pi0 的关系

相同点：

- 都把机器人动作 chunk 看成生成对象；
- 都使用条件生成；
- 都受 flow matching / diffusion-style action generation 思路影响；
- 都不是简单的单步 MLP 行为克隆。

不同点：

- pi0 是大规模 vision-language-action policy，重点是跨任务、跨机器人泛化；
- 本项目是小型研究仓库，重点是 iMeanFlow 少步采样目标在机器人动作生成中的实现和验证；
- 本项目没有大规模视觉语言数据，也不声称复现 pi0。

## 9. 面试回答底线

不要说“我超过了 Flow Matching”。

应该说：

> 当前结果说明 iMeanFlow action generation 的代码路径、训练目标和 Push-T 评测已经跑通。它还没有超过 Flow Matching baseline，但它提供了少步采样的研究方向。下一步需要用 NFE、latency、smoothness 和 success rate 做系统对比，判断少步生成在实时控制里是否值得。
