# Method Notes

This repository implements a no-CFG conditional iMeanFlow policy for robotic
action chunk generation.

## Flow setup

For a demonstrated action chunk `x` and Gaussian action noise `e`, define:

```text
z_t = (1 - t) x + t e
v = e - x
```

The policy is conditioned on a robot observation vector `obs`. The backbone
predicts two action-space vector fields:

```text
u(z_t, h, obs)  # interval-average velocity
v_hat(z_t, h, obs)  # auxiliary instantaneous velocity
h = t - r
```

## iMeanFlow objective

The auxiliary `v_hat` at `h=0` provides the tangent direction for JVP:

```text
v_tangent = v_hat(z_t, h=0, obs)
dudt = JVP(u, direction=(v_tangent, dt=1, dr=0))
V = u + (t-r) stopgrad(dudt)
```

The training loss is:

```text
L = || V - (e-x) ||^2 + lambda_v || v_hat - (e-x) ||^2
```

This is designed to make large Euler steps work at inference time.

## Why no CFG?

Classifier-free guidance is useful in image generation when text/class
conditioning needs to be amplified. For robot actions, the state/image/task
condition is the actual control context. Keeping fake CFG parameters without an
unconditional branch would be misleading, so this implementation uses direct
conditional generation only.

