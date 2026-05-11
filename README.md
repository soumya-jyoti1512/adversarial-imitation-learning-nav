# Autonomous Navigation via Adversarial Imitation Learning

---

## Overview

This project presents a framework for training a mobile robot to navigate cluttered environments without any hand-crafted reward functions. Rather than manually specifying what "good navigation" looks like, the robot learns directly from expert demonstrations through adversarial imitation learning.

Instead of designing reward functions, designed a discriminator network observe both expert and agent trajectories and learn to distinguish between them. The resulting signal guides the policy toward expert-like behavior extended with a lightweight hybrid reward that adds goal-progress and collision-avoidance terms to close the gap in high-constraint scenarios.

**Core algorithm stack:** `GAIL` + `SAC` + `Hybrid Reward Shaping`  
**Simulator:** Gazebo · **Sensor:** LiDAR (360°, 20 readings/timestep) · **Framework:** PyTorch + ROS

---

# Table of Contents

- [Architecture](#architecture)
  - [Policy Network](#1-policy-network--stochastic-actor)
  - [Critic Networks](#2-critic-networks--twin-q-networks)
  - [Adversarial Discriminator](#3-adversarial-discriminator--gail-component)
  - [Hybrid Reward Function](#4-hybrid-reward-function)
  - [Replay Buffers](#5-replay-buffers)
  - [Training Loop](#6-training-loop--six-step-iteration)
  - [Full System Architecture](#7-full-system-architecture)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion & Limitations](#discussion--limitations)
- [Future Work](#future-work)

---

# Architecture

The system combines three interlocking components, a stochastic actor-critic policy (SAC), an adversarial discriminator (GAIL), and a hybrid reward signal into an end-to-end imitation learning pipeline.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         FULL SYSTEM                                 │
│                                                                     │
│  Expert Buffer ──►┐                                                 │
│                   ├──► Discriminator ──► r_gail(s,a) ──►┐          │
│  Replay Buffer ──►┘                                     │          │
│                                                         ▼          │
│  Environment ──► State s ──► Actor ──► Action a    Hybrid Reward   │
│                                          │         r_total(s,a,s') │
│                                          ▼              │          │
│                                   Environment ──► s'    │          │
│                                          │              │          │
│                                          └──► Critics ◄─┘          │
│                                                   │                │
│                                                   └──► Actor Update│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Policy Network - Stochastic Actor

The actor is a stochastic Gaussian MLP that maps the robot's state to a distribution over continuous actions.

### State Vector (22-dimensional)

- 20 LiDAR distance readings sampled every 18° over a full 360° field of view
- Δx, Δy - relative displacement to the goal

### Output

- μ(s) - mean action vector
- log σ(s) - log standard deviation (clamped for numerical stability)

Actions are sampled using the **reparameterization trick** and passed through a Tanh squashing function to produce bounded outputs in `(-1, 1)`.

```text
State s ──► Shared MLP (ReLU) ──► μ(s) ─────────────────────────────►┐
                              └──► log σ(s) ──► exp ──► σ(s)         │
                                                         │           │
                                               N(μ,σ²) ◄─┘           │
                                                   │                 │
                                            Reparameterization       │
                                                   │                 │
                                              Tanh Squash            │
                                                   │                 │
                                              Action â               │
```

The actor is trained under the maximum entropy reinforcement learning framework.

---

## 2. Critic Networks - Twin Q-Networks

Value estimation uses two independent Q-networks operating in parallel to mitigate overestimation bias.

- Each critic takes `[state, action]` as input
- The minimum of the two estimates is used for Bellman targets
- Target critics are updated via **Polyak averaging**

### Bellman Target

```math
y = r_{total} + \gamma \cdot \left( \min Q_{target}(s', a') - \alpha \log \pi(a'|s') \right)
```

```text
[s || a] ──►┬──► Critic Q1 ──► Q1(s,a) ──►┐
            └──► Critic Q2 ──► Q2(s,a) ──►┤
                                            ▼
                                      min(Q1,Q2)
                                            │
                                      Bellman Target
```

---

## 3. Adversarial Discriminator - GAIL Component

The discriminator is an MLP that takes `(state, action)` pairs and predicts whether the transition came from the expert or the agent.

### Discriminator Loss

```math
L_D = - \mathbb{E}_{expert}[\log D(s,a)] - \mathbb{E}_{agent}[\log(1 - D(s,a))]
```

### Adversarial Reward

```math
r_{gail}(s,a) = -\log(1 - D(s,a))
```

```text
Expert transitions ──►┐
                      ├──► Discriminator D(s,a) ──► r_gail(s,a)
Agent transitions ───►┘
```

The reward signal becomes progressively harder to earn as the discriminator improves during training.

---

## 4. Hybrid Reward Function

Pure GAIL reward is powerful but incomplete: it encourages imitation but does not explicitly reward goal progress or obstacle avoidance.

To address this, a lightweight hybrid reward supplements the adversarial signal.

### Hybrid Reward

```math
r_{total}(s,a,s') = r_{gail}(s,a)
                  + \lambda_1 \cdot r_{goal}(s,s')
                  + \lambda_2 \cdot r_{collision}(s')
```

### Reward Components

| Term | Formula | Purpose |
|------|----------|----------|
| `r_gail(s,a)` | `−log(1 − D(s,a))` | Imitation learning signal |
| `r_goal(s,s')` | `d(s) − d(s')` | Goal progress |
| `r_collision(s')` | `−C if min(LiDAR) < ε else 0` | Obstacle avoidance |

### Recommended Hyperparameters

```text
λ₁ = 0.3
λ₂ = 0.5
C  = 1.0
ε  = 0.2 m
```

### Reward Flowchart

```text
Agent transition (s,a,s') ──►┐
                             ├──► Discriminator ──► r_gail
                             ├──► Goal Delta ─────► r_goal
                             └──► LiDAR Check ────► r_collision
                                                    │
                                                    ▼
                           r_total = r_gail + λ₁·r_goal + λ₂·r_collision
```

---

## 5. Replay Buffers

Two separate replay buffers are used:

### Agent Replay Buffer
Stores:
- `(s, a, s', done)`

Used for:
- SAC updates
- Random mini-batch sampling

### Expert Buffer (Read-Only)

Stores:
- Expert demonstration trajectories loaded from HDF5

Used exclusively for:
- Discriminator updates

---

## 6. Training Loop - Six-Step Iteration

Training alternates between environment interaction and optimization.

```text
┌────────────── Training Iteration ───────────────────────────────┐
│                                                                 │
│ Step 1 │ Policy Rollout                                         │
│ Step 2 │ Discriminator Update                                   │
│ Step 3 │ Hybrid Reward Computation                              │
│ Step 4 │ Critic Update                                          │
│ Step 5 │ Actor Update                                           │
│ Step 6 │ Target Network Update                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Objective

```math
J_\pi = \mathbb{E}[\alpha \log \pi(a|s) - \min(Q_1,Q_2)]
```

---

## 7. Full System Architecture

Training is organized into three functional layers.

### Layer 1 — Initialization

- Initialize actor
- Initialize twin critics
- Initialize target critics
- Initialize discriminator
- Load expert demonstrations

### Layer 2 — Episode Rollout

- Observe state `s`
- Actor samples action `a`
- Execute action in Gazebo
- Store `(s,a,s',done)` in replay buffer

### Layer 3 — Training Optimization

Execute:
1. Discriminator update
2. Reward computation
3. Critic update
4. Actor update
5. Target network update

The process repeats until convergence.

---

# Experimental Setup

| Parameter | Value |
|---|---|
| Algorithm | GAIL + SAC + Hybrid Reward |
| Simulator | Gazebo (ROS) |
| Environment Size | 5.4 m × 5.4 m |
| State Dimension | 22 |
| Action Dimension | 3 |
| Obstacles | Randomized cubes |
| Expert Episodes | 40 |
| Expert Data Format | HDF5 |
| Training Steps | 50,000 |

---

# Results

## Quantitative Results

| Metric | Value |
|---|---|
| Success Rate | 92% |
| Collision Rate | 11% |
| Avg Episode Length | 210 steps |
| Avg Distance to Goal | 0.32 m |

---

## Training Dynamics

- Episodic reward shows expected adversarial training volatility
- Discriminator loss converges from ~1.0 → ~0.3
- Navigation success rate increases steadily
- Heatmaps show consistent diagonal traversal behavior

---

## Qualitative Behavior

### Successful Runs

- Smooth obstacle avoidance
- Stable path corrections
- Efficient goal reaching

### Failure Modes

- Difficulty in narrow corridors
- Occasional stalling before collision
- Sparse expert coverage for constrained environments

---

# Discussion & Limitations

The GAIL+SAC framework successfully learns navigation without manual reward engineering.

### Key Improvements from Hybrid Reward

1. Better goal-directed behavior
2. Reduced collision hesitation
3. Preserved imitation learning characteristics

### Remaining Limitations

- Sparse demonstrations in narrow passages
- Hyperparameter sensitivity
- Static environment assumptions

---

# Future Work

- Richer expert demonstrations
- Dynamic obstacle environments
- Sim2real transfer
- Multi-agent navigation
- Adaptive reward weighting

---

