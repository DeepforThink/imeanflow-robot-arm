# 面试讲解提纲

## 一句话介绍

这个项目把 Improved Mean Flow 的少步生成思想迁移到机械臂模仿学习中，用 1-2 次网络前向生成一段未来动作序列。

## 背景

机械臂策略常常不是只输出一个动作，而是输出未来一段 action chunk。pi0 和 Diffusion Policy 说明了动作序列可以通过从噪声到真实动作的生成模型来学习。普通 Flow Matching 稳定，但推理通常需要多步 Euler 积分。

## 我的改进点

我希望减少实时控制中的网络前向次数，所以没有只学习瞬时速度，而是学习时间区间上的平均速度。这样推理时可以从噪声动作直接跨较大的时间区间到可执行动作。

## 核心公式

```text
z_t = (1 - t) action + t noise
v_target = noise - action
h = t - r
V = u + h * stopgrad(JVP(u))
loss = ||V - v_target||^2 + ||v_head - v_target||^2
```

其中 `u` 是平均速度头，`v_head` 是辅助瞬时速度头。

## 为什么不做 CFG

CFG 来自图像生成，依赖 conditional/unconditional 两路预测。机械臂动作由观测强条件约束，pi0 的动作 Flow Matching 也不使用 CFG。没有 unconditional 训练时保留 CFG 参数只是表面功能，所以这个项目选择无 CFG 的条件 iMeanFlow。

## 项目价值

目标不是声称一定超过 pi0，而是验证一个清晰问题：能不能用 iMeanFlow 让 action chunk 生成从 10 步左右降低到 1-2 步，同时保持动作质量。

## 可以展示的代码点

- `src/imeanflow_robotics/policy.py`: 训练目标和少步采样。
- `src/imeanflow_robotics/model.py`: observation-conditioned transformer 和 u/v 双头。
- `src/imeanflow_robotics/data.py`: 合成机械臂数据，用于快速复现训练闭环。
- `tests/test_policy.py`: 检查 JVP loss、反向传播和推理形状。

