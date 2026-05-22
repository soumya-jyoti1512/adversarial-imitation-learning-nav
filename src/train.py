from __future__ import annotations

import argparse
import dataclasses
import random
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.algorithms.gail import GAILTrainer
from src.algorithms.sac import SACAgent
from src.buffers.expert_buffer import ExpertBuffer
from src.buffers.replay_buffer import ReplayBuffer
from src.envs.test_env import ToyNavEnv
from src.rewards.hybrid_reward import HybridReward
from src.utils.logger import Logger



# Configuration
@dataclass
class TrainConfig:
    # Run
    backend: str = "toy"           # "toy" or "gazebo"
    total_steps: int = 50_000
    seed: int = 42
    device: str = "cpu"            # set by main() after probing CUDA

    #Env
    env_size: float = 5.4
    v_max: float = 1.0
    omega_max: float = 1.5
    lidar_num_beams: int = 20
    lidar_max_range: float = 3.0
    max_episode_steps: int = 300

    #SAC
    state_dim: int = 22
    action_dim: int = 3
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    gamma: float = 0.99
    tau: float = 5e-3
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    alpha_init: float = 0.2
    automatic_entropy: bool = True

    #GAIL
    lr_disc: float = 3e-4
    r1_coeff: float = 10.0

    #Hybrid reward
    lambda_goal: float = 0.3
    lambda_collision: float = 0.5
    collision_penalty: float = 1.0
    collision_threshold: float = 0.2

    # Replay 
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 1_000     
    updates_per_step: int = 1      # SAC + GAIL updates per env step

    #Expert data
    expert_path: Optional[str] = None

    log_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_interval: int = 50         
    eval_interval: int = 5_000
    eval_episodes: int = 10
    checkpoint_interval: int = 10_000

   
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        cfg = cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = []
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                unknown.append(k)
        if unknown:
            warnings.warn(f"unknown config keys ignored: {unknown}")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)



# Helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(cfg: TrainConfig, seed: Optional[int] = None):
    if cfg.backend == "toy":
        return ToyNavEnv(
            env_size=cfg.env_size,
            lidar_num_beams=cfg.lidar_num_beams,
            lidar_max_range=cfg.lidar_max_range,
            v_max=cfg.v_max,
            omega_max=cfg.omega_max,
            max_steps=cfg.max_episode_steps,
            seed=seed,
        )
    elif cfg.backend == "gazebo":
        from src.envs.gazebo_env import GazeboEnv
        return GazeboEnv(
            env_size=cfg.env_size,
            lidar_num_beams=cfg.lidar_num_beams,
            lidar_max_range=cfg.lidar_max_range,
            v_max=cfg.v_max,
            omega_max=cfg.omega_max,
            max_steps=cfg.max_episode_steps,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown backend: {cfg.backend!r}")


def update_step(
    sac: SACAgent,
    gail: GAILTrainer,
    hybrid: HybridReward,
    replay: ReplayBuffer,
    expert: ExpertBuffer,
    batch_size: int,
) -> dict[str, float]:
    """One joint SAC + GAIL update on freshly sampled mini-batches."""
    agent_batch = replay.sample(batch_size)
    expert_batch = expert.sample(batch_size)

    # 1. Discriminator update.
    gail_m = gail.update(expert_batch, agent_batch)

    # 2. Hybrid reward computed AGAINST THE UPDATED D
    with torch.no_grad():
        r_gail = gail.compute_reward(agent_batch["state"], agent_batch["action"])
        rewards = hybrid.compute(
            agent_batch["state"], agent_batch["next_state"], r_gail
        )

    # 3. SAC update.
    sac_m = sac.update(agent_batch, rewards["r_total"])

    return {
        **{f"gail/{k}": v for k, v in gail_m.items()},
        **{f"sac/{k}":  v for k, v in sac_m.items()},
        "reward/r_gail":      float(rewards["r_gail"].mean()),
        "reward/r_goal":      float(rewards["r_goal"].mean()),
        "reward/r_collision": float(rewards["r_collision"].mean()),
        "reward/r_total":     float(rewards["r_total"].mean()),
    }


@torch.no_grad()
def evaluate(env, sac: SACAgent, num_episodes: int) -> dict[str, float]:
    """Deterministic rollouts. Returns aggregate metrics for the logger."""
    successes, collisions, lengths, returns, final_dists = [], [], [], [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_return, ep_steps = 0.0, 0
        while True:
            a = sac.act(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(a)
            ep_return += r
            ep_steps += 1
            if terminated or truncated:
                successes.append(float(info.get("reached_goal", False)))
                collisions.append(float(info.get("collided", False)))
                lengths.append(ep_steps)
                returns.append(ep_return)
                final_dists.append(info.get("dist_to_goal", float("nan")))
                break
    return {
        "eval/success_rate":  float(np.mean(successes)),
        "eval/collision_rate": float(np.mean(collisions)),
        "eval/avg_length":    float(np.mean(lengths)),
        "eval/avg_return":    float(np.mean(returns)),
        "eval/avg_final_dist": float(np.nanmean(final_dists)),
    }


def save_checkpoint(
    sac: SACAgent,
    gail: GAILTrainer,
    log_dir: Optional[str | Path],
    step: int,
) -> None:
    if log_dir is None:
        return
    ckpt_dir = Path(log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "sac": sac.state_dict(),
        "gail": gail.state_dict(),
    }
    torch.save(payload, ckpt_dir / f"step_{step:07d}.pt")
    torch.save(payload, ckpt_dir / "latest.pt")


# Main training loop
def train(cfg: TrainConfig) -> None:
    if cfg.expert_path is None:
        raise ValueError(
            "expert_path is required. Generate one with scripts/collect_expert.py."
        )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"[train] backend={cfg.backend}  device={device}  "
          f"total_steps={cfg.total_steps}  seed={cfg.seed}")

    #Components
    env = make_env(cfg, seed=cfg.seed)
    eval_env = make_env(cfg, seed=cfg.seed + 1_000)

    sac = SACAgent(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        action_scale=torch.tensor(
            [cfg.v_max, cfg.v_max, cfg.omega_max], dtype=torch.float32
        ),
        action_bias=0.0,
        hidden_dims=tuple(cfg.hidden_dims),
        gamma=cfg.gamma, tau=cfg.tau,
        lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic,
        lr_alpha=cfg.lr_alpha, alpha_init=cfg.alpha_init,
        automatic_entropy=cfg.automatic_entropy,
        device=device,
    )
    gail = GAILTrainer(
        state_dim=cfg.state_dim, action_dim=cfg.action_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        lr=cfg.lr_disc, r1_coeff=cfg.r1_coeff,
        device=device,
    )
    hybrid = HybridReward(
        lambda_goal=cfg.lambda_goal,
        lambda_collision=cfg.lambda_collision,
        collision_penalty=cfg.collision_penalty,
        collision_threshold=cfg.collision_threshold,
    )
    replay = ReplayBuffer(
        capacity=cfg.replay_capacity,
        state_dim=cfg.state_dim, action_dim=cfg.action_dim,
        device=device, seed=cfg.seed,
    )
    expert = ExpertBuffer(
        cfg.expert_path, device=device, seed=cfg.seed + 1,
    )
    print(f"[train] expert dataset: N={len(expert)}  episodes={expert.num_episodes}  "
          f"avg_len={expert.avg_episode_length:.1f}")

    with Logger(
        log_dir=cfg.log_dir,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
        config=cfg.to_dict(),
    ) as logger:
        try:
            _run_loop(env, eval_env, sac, gail, hybrid, replay, expert,
                      logger, cfg)
        except KeyboardInterrupt:
            print("\n[train] interrupted — saving final checkpoint...")
            save_checkpoint(sac, gail, cfg.log_dir, step=-1)
        finally:
            env.close() if hasattr(env, "close") else None
            eval_env.close() if hasattr(eval_env, "close") else None


def _run_loop(
    env, eval_env,
    sac: SACAgent, gail: GAILTrainer,
    hybrid: HybridReward,
    replay: ReplayBuffer, expert: ExpertBuffer,
    logger: Logger, cfg: TrainConfig,
) -> None:
    obs, _ = env.reset(seed=cfg.seed)
    ep_return = 0.0
    ep_steps = 0
    ep_count = 0

    success_window: deque = deque(maxlen=20)
    last_metrics: dict[str, float] = {}
    t_start = time.time()

    bar = tqdm(range(cfg.total_steps), desc="train", dynamic_ncols=True)
    for step in bar:
        #  1. Action
        if step < cfg.warmup_steps:
            action = env.action_space.sample()
        else:
            action = sac.act(obs, deterministic=False)

        #Env step
        obs_next, r_env, terminated, truncated, info = env.step(action)
        replay.add(obs, action, obs_next, terminated)
        ep_return += float(r_env)
        ep_steps += 1
        obs = obs_next

        # 2-6. Updates 
        if step >= cfg.warmup_steps and len(replay) >= cfg.batch_size:
            for _ in range(cfg.updates_per_step):
                last_metrics = update_step(
                    sac, gail, hybrid, replay, expert, cfg.batch_size
                )
            if step % cfg.log_interval == 0:
                logger.log(last_metrics, step=step)

        # Episode bookkeeping
        if terminated or truncated:
            ep_count += 1
            success_window.append(float(info.get("reached_goal", False)))
            ep_metrics = {
                "episode/return":       ep_return,
                "episode/length":       float(ep_steps),
                "episode/success_rate": float(np.mean(success_window)),
                "episode/dist_to_goal": float(info.get("dist_to_goal", float("nan"))),
                "episode/count":        float(ep_count),
            }
            logger.log(ep_metrics, step=step)
            ep_return, ep_steps = 0.0, 0
            obs, _ = env.reset()

        # Periodic eval
        if step > 0 and step % cfg.eval_interval == 0:
            eval_metrics = evaluate(eval_env, sac, cfg.eval_episodes)
            logger.log(eval_metrics, step=step)
            bar.set_postfix(
                success=f"{eval_metrics['eval/success_rate']:.2f}",
                d_exp=f"{last_metrics.get('gail/d_expert', float('nan')):.2f}",
                alpha=f"{last_metrics.get('sac/alpha', float('nan')):.3f}",
            )

        if step > 0 and step % cfg.checkpoint_interval == 0:
            save_checkpoint(sac, gail, cfg.log_dir, step)

    save_checkpoint(sac, gail, cfg.log_dir, cfg.total_steps)
    elapsed = time.time() - t_start
    print(f"[train] done. {cfg.total_steps} steps in {elapsed:.1f}s "
          f"({cfg.total_steps / max(elapsed, 1e-6):.0f} steps/s)")



# CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML file overriding TrainConfig defaults.")
    p.add_argument("--backend", choices=["toy", "gazebo"], default=None)
    p.add_argument("--expert-path", type=str, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="cpu, cuda, or cuda:N. Auto-detected if omitted.")
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    cfg = (TrainConfig.from_yaml(args.config) if args.config
           else TrainConfig())

    # Apply non-None CLI overrides.
    for field_name in (
        "backend", "expert_path", "total_steps", "seed", "device",
        "log_dir", "wandb_project", "wandb_run_name",
    ):
        cli_value = getattr(args, field_name.replace("_", "_"))
        if cli_value is not None:
            setattr(cfg, field_name, cli_value)

    if cfg.device == "cpu" and torch.cuda.is_available() and args.device is None:
        cfg.device = "cuda"

    train(cfg)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    import os
    import shutil
    import tempfile

    print("=" * 70)
    print(" smoke test: 500-step training run with synthetic expert data ")
    print("=" * 70)

    workdir = Path(tempfile.mkdtemp(prefix="train_smoke_"))
    expert_path = workdir / "expert.h5"
    log_dir = workdir / "runs"

    print(f"\n[1/3] generating synthetic expert at {expert_path} ...")
    env = ToyNavEnv(seed=0)
    states, actions, next_states, dones, ep_starts = [], [], [], [], []
    cur = 0
    for _ in range(3):
        o, _ = env.reset()
        ep_starts.append(cur)
        for _ in range(80):
            a = env.action_space.sample()
            o_next, _, term, trunc, _ = env.step(a)
            states.append(o); actions.append(a); next_states.append(o_next)
            dones.append([float(term)])
            cur += 1
            o = o_next
            if term or trunc:
                break
    ExpertBuffer.write_hdf5(
        expert_path,
        np.array(states), np.array(actions), np.array(next_states),
        np.array(dones), np.array(ep_starts, dtype=np.int64),
    )

    print(f"\n[2/3] running 500-step training loop ...")
    cfg = TrainConfig(
        backend="toy",
        total_steps=500,
        warmup_steps=50,
        batch_size=64,
        log_interval=20,
        eval_interval=200,
        eval_episodes=2,
        checkpoint_interval=200,
        replay_capacity=2_000,
        expert_path=str(expert_path),
        log_dir=str(log_dir),
        wandb_project=None,
        seed=0,
        device="cpu",
    )
    train(cfg)

    print(f"\n[3/3] verifying artifacts in {log_dir} ...")
    files = list(log_dir.rglob("*"))
    has_events = any("events" in f.name for f in files)
    has_config = any(f.name == "config.yaml" for f in files)
    has_ckpt = any(f.name.startswith("step_") and f.suffix == ".pt" for f in files)
    has_latest = any(f.name == "latest.pt" for f in files)
    print(f"  TB event file: {has_events}")
    print(f"  config.yaml:   {has_config}")
    print(f"  step ckpt:     {has_ckpt}")
    print(f"  latest.pt:     {has_latest}")
    assert has_events and has_config and has_ckpt and has_latest

    payload = torch.load(log_dir / "checkpoints" / "latest.pt",
                         map_location="cpu", weights_only=False)
    assert "sac" in payload and "gail" in payload and "step" in payload
    fresh_sac = SACAgent(
        state_dim=22, action_dim=3,
        action_scale=torch.tensor([1.0, 1.0, 1.5]),
    )
    fresh_sac.load_state_dict(payload["sac"])
    print(f"  checkpoint @ step {payload['step']}: load OK")

    shutil.rmtree(workdir, ignore_errors=True)
    print("\nall train smoke checks pass — full SAC + GAIL pipeline runs "
          "end-to-end and produces complete artifacts.")
