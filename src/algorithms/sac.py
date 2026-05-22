from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.networks.actor import GaussianActor
from src.networks.critic import TwinCritic


class SACAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float | Tensor | Sequence[float] = 1.0,
        action_bias: float | Tensor | Sequence[float] = 0.0,
        hidden_dims: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        tau: float = 5e-3,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        alpha_init: float = 0.2,
        target_entropy: float | None = None,
        automatic_entropy: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        if alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {alpha_init}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = torch.device(device)
        self.automatic_entropy = bool(automatic_entropy)

        #Networks
        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_scale=action_scale,
            action_bias=action_bias,
        ).to(self.device)
        self.critic = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        #Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(
            self.critic.online_parameters(), lr=lr_critic
        )

        #Automatic entropy tuning
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = float(target_entropy)

        self.log_alpha = torch.tensor(
            math.log(alpha_init),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.automatic_entropy,
        )
        self.alpha_opt: torch.optim.Optimizer | None = (
            torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            if self.automatic_entropy
            else None
        )

    # Convenience
    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray | Tensor,
        deterministic: bool = False,
    ) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        action = self.actor.act(state_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    # The four-step SAC update
    def update(
        self,
        batch: dict[str, Tensor],
        r_total: Tensor,
    ) -> dict[str, float]:
        s        = batch["state"]
        a        = batch["action"]
        s_next   = batch["next_state"]
        done     = batch["done"]

       
        # 1) Critic update
        with torch.no_grad():
            next_action, next_logp = self.actor(s_next)
            q_next = self.critic.q_target_min(s_next, next_action)
            y = r_total + self.gamma * (1.0 - done) * (
                q_next - self.alpha.detach() * next_logp
            )

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # 2) Actor update
        action_pi, logp_pi = self.actor(s)
        q_pi = self.critic.q_min(s, action_pi)
        actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # 3) Alpha update 
        if self.automatic_entropy:
            # logp_pi is detached — α should react to the policy's entropy
            # as a fixed quantity at this step.
            alpha_loss = -(
                self.log_alpha * (logp_pi.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]
            alpha_loss.backward()
            self.alpha_opt.step()                       # type: ignore[union-attr]
            alpha_loss_val = float(alpha_loss.detach())
        else:
            alpha_loss_val = 0.0

        # 4) Soft target update
        self.critic.soft_update(self.tau)

        with torch.no_grad():
            entropy_est = -logp_pi.mean().item() 
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss":  float(actor_loss.detach()),
            "alpha_loss":  alpha_loss_val,
            "alpha":       float(self.alpha.detach()),
            "q1_mean":     float(q1.mean().detach()),
            "q2_mean":     float(q2.mean().detach()),
            "y_mean":      float(y.mean().detach()),
            "entropy":     entropy_est,
            "r_mean":      float(r_total.mean().detach()),
        }

    # Checkpointing
    def state_dict(self) -> dict:
        sd = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        if self.alpha_opt is not None:
            sd["alpha_opt"] = self.alpha_opt.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.log_alpha.data.copy_(sd["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        if self.alpha_opt is not None and "alpha_opt" in sd:
            self.alpha_opt.load_state_dict(sd["alpha_opt"])



if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim, action_dim = 22, 3
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=torch.tensor([1.0, 1.0, 1.5]),
        action_bias=0.0,
        device="cpu",
    )

    s_np = np.random.randn(state_dim).astype(np.float32)
    a_np = agent.act(s_np)
    print(f"act() returned shape={a_np.shape}  dtype={a_np.dtype}  "
          f"range=[{a_np.min():+.3f}, {a_np.max():+.3f}]")
    assert a_np.shape == (action_dim,)
    assert a_np.dtype == np.float32

 
    B = 256
    batch = {
        "state":      torch.randn(B, state_dim),
        "action":     torch.randn(B, action_dim).clamp(-1.5, 1.5),
        "next_state": torch.randn(B, state_dim),
        "done":       (torch.rand(B, 1) < 0.02).float(), 
    }
    r_total = torch.randn(B, 1) * 0.5 + 0.5  

    print(f"\n{'step':>4}  {'critic':>8}  {'actor':>8}  {'alpha':>6}  "
          f"{'H':>6}  {'q1':>7}")
    prev_critic = None
    for step in range(50):
        metrics = agent.update(batch, r_total)
        if step % 10 == 0 or step == 49:
            print(f"{step:>4}  {metrics['critic_loss']:>8.4f}  "
                  f"{metrics['actor_loss']:>+8.4f}  "
                  f"{metrics['alpha']:>6.3f}  "
                  f"{metrics['entropy']:>+6.3f}  "
                  f"{metrics['q1_mean']:>+7.3f}")
        for k, v in metrics.items():
            assert math.isfinite(v), f"{k} went non-finite at step {step}: {v}"
        prev_critic = metrics["critic_loss"]

    assert agent.alpha.item() != 0.2, (
        f"alpha did not move from init — auto-tuning may be broken "
        f"(α = {agent.alpha.item()})"
    )
    print(f"\nalpha drifted from 0.200 → {agent.alpha.item():.4f}  "
          f"(auto-entropy tuning is active)")

    p_online = next(agent.critic.q1.parameters()).data
    p_target = next(agent.critic.q1_target.parameters()).data
    delta = (p_online - p_target).abs().mean().item()
    print(f"||q1 - q1_target||_1 mean = {delta:.5f}  "
          f"(non-zero confirms targets are lagging the online net)")
    assert delta > 0.0

    sd = agent.state_dict()
    fresh = SACAgent(
        state_dim=state_dim, action_dim=action_dim,
        action_scale=torch.tensor([1.0, 1.0, 1.5]),
    )
    fresh.load_state_dict(sd)
    a_new = fresh.act(s_np)
    a_old = agent.act(s_np)
  
    a_new_det = fresh.act(s_np, deterministic=True)
    a_old_det = agent.act(s_np, deterministic=True)
    assert np.allclose(a_new_det, a_old_det, atol=1e-6), (
        "deterministic action mismatched after checkpoint round-trip"
    )
    print("checkpoint round-trip preserved deterministic policy.")

    print("\nall checks pass.")
