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
            raise ValueError(f"gamma should be in between(0, 1], got {gamma}")
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

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(
            self.critic.online_parameters(), lr=lr_critic
        )

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

    def update(
        self,
        batch: dict[str, Tensor],
        r_total: Tensor,
    ) -> dict[str, float]:
        s        = batch["state"]
        a        = batch["action"]
        s_next   = batch["next_state"]
        done     = batch["done"]

       
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

        action_pi, logp_pi = self.actor(s)
        q_pi = self.critic.q_min(s, action_pi)
        actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.automatic_entropy:
            alpha_loss = -(
                self.log_alpha * (logp_pi.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad(set_to_none=True)  
            alpha_loss.backward()
            self.alpha_opt.step()                       
            alpha_loss_val = float(alpha_loss.detach())
        else:
            alpha_loss_val = 0.0

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
