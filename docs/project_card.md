# Project Card

## Project name

iMeanFlow Robotics

## Scope

This repository implements a compact action-generation policy for robotic arms.
It is meant for research demonstration, interview discussion, and future
extension to real imitation-learning datasets.

## Claims

The repository demonstrates:

- a no-CFG conditional iMeanFlow training objective,
- u/v dual-head action velocity prediction,
- JVP-based compound velocity training,
- few-step Euler action chunk generation,
- a synthetic robotic arm dataset for reproducible smoke tests.

It does not claim:

- official iMeanFlow reproduction,
- state-of-the-art robot performance,
- validation on a real robot arm.

## Suggested evaluation

When connected to real demonstration data, evaluate:

- task success rate,
- action smoothness,
- inference latency,
- performance as a function of NFE,
- comparison with standard Flow Matching.

