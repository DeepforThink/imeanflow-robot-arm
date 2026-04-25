# 面试讲解提纲

## 一句话介绍

这个项目把 Improved Mean Flow 的少步生成思想迁移到机械臂模仿学习中，用无 CFG 的条件生成模型，从机器人观测生成未来一段可执行的动作序列。

## 背景

机械臂策略通常不只输出单步动作，而是输出未来一段 `action chunk`。Diffusion Policy 和 pi0 说明了：动作序列可以看成从高斯噪声到真实动作轨迹的生成过程。

普通 Flow Matching 学习的是瞬时速度场，推理时一般需要多步 Euler 积分。iMeanFlow 的目标是学习时间区间上的平均速度，让采样器可以用更少的网络前向次数生成动作。

## 我的选择

我没有加入 CFG。原因是机器人动作已经由当前关节状态、目标、图像或任务指令强条件约束；如果没有真正训练 unconditional 分支，保留 CFG 参数只会制造表面功能。pi0 的动作生成也是条件 flow matching，不依赖 CFG。

## 核心公式

```text
z_t = (1 - t) action + t noise
v_target = noise - action
h = t - r
V = u + h * stopgrad(JVP(u))
loss = ||V - v_target||^2 + lambda_v ||v_head - v_target||^2
```

其中 `u` 是平均速度头，`v_head` 是辅助瞬时速度头。`v_head` 在 `h=0` 时提供 JVP 的切向方向。

## 仿真 Demo

项目里加入了一个轻量 2D 三关节平面机械臂环境：

- `PlanarArm2D` 提供正运动学和解析逆运动学目标；
- `PlanarReachDataset` 自动生成 reaching demonstration；
- `scripts/sim_demo.py` 训练 iMeanFlow 策略，并用 receding-horizon action chunk 控制机械臂；
- 输出 `assets/planar_arm_demo.png` 和 `assets/planar_arm_demo.gif`。

运行命令：

```bash
python scripts/sim_demo.py --train-steps 300
```

这个仿真不是物理引擎 benchmark，而是为了清楚展示：模型确实从观测和噪声生成动作 chunk，然后通过多轮重规划把机械臂末端推向目标。

## 可以展示的代码点

- `src/imeanflow_robotics/policy.py`：iMeanFlow loss、JVP 目标、少步采样、动作队列。
- `src/imeanflow_robotics/model.py`：observation-conditioned Transformer 和 `u/v` 双头。
- `src/imeanflow_robotics/sim.py`：平面机械臂、自动生成 demonstrations、rollout。
- `scripts/sim_demo.py`：训练、加载 checkpoint、渲染 GIF/PNG。
- `tests/test_policy.py` 和 `tests/test_sim.py`：保证 loss、采样、仿真接口形状正确。

## 诚实边界

我不会把它包装成 SOTA 机器人策略。这个仓库的价值是：把 iMeanFlow 动作生成的训练目标、采样逻辑和机械臂动作 chunk demo 做成一个可读、可复现、可扩展的小项目。下一步应该接入 LeRobot 数据集，比较 4-step iMeanFlow 和多步 Flow Matching 的速度、轨迹误差和成功率。
